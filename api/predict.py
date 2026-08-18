"""
Vercel Python Serverless Function
==================================
Competitive Coding Performance & Rating Trajectory Predictor
--------------------------------------------------------------
POST /api/predict
Body: {"handle": "<codeforces_handle>"}

Response (200):
{
  "handle": str,
  "currentRating": int,
  "maxRating": int,
  "rank": str,
  "ratingHistory": [{"contestId": int, "contestName": str, "date": str, "rating": int}, ...],
  "predictedRatings": [int, int, int, int, int],
  "predictedPeakRating": int,
  "tagAnalysis": [
     {"tag": str, "solvedCount": int, "avgDifficulty": float, "tpi": float, "cluster": "Proficient"|"Developing"|"Bottleneck"}
  ],
  "bottlenecks": [{"tag": str, "tpi": float, "solvedCount": int, "recommendation": str}, ...]
}

Errors are always returned as clean JSON with an "error" key and an
appropriate HTTP status code (400 for client/user input problems, 500 for
upstream/Codeforces API problems) -- unhandled exceptions never leak to the
client.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

CF_API_BASE = "https://codeforces.com/api"
REQUEST_TIMEOUT = 6  # seconds -- keeps us well inside Vercel's 10s hobby limit
MAX_SUBMISSIONS = 10000  # cap payload size / fetch time for very prolific users
DECAY_LAMBDA = 0.01  # time-decay rate applied to submission age in days
FUTURE_CONTESTS = 5

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


# --------------------------------------------------------------------------
# Codeforces API helpers
# --------------------------------------------------------------------------

class CodeforcesError(Exception):
    """Raised when the Codeforces API itself reports a failure (bad handle, etc.)."""


class UpstreamError(Exception):
    """Raised when we cannot reach / parse the Codeforces API at all."""


def cf_get(method, params):
    url = f"{CF_API_BASE}/{method}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise UpstreamError("Codeforces API timed out. Please try again shortly.") from exc
    except requests.exceptions.RequestException as exc:
        raise UpstreamError(f"Could not reach Codeforces API: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise UpstreamError("Codeforces API returned an unreadable response.") from exc

    if payload.get("status") != "OK":
        comment = payload.get("comment", "Unknown Codeforces API error.")
        raise CodeforcesError(comment)

    return payload["result"]


def fetch_user_info(handle):
    result = cf_get("user.info", {"handles": handle})
    if not result:
        raise CodeforcesError(f"Handle '{handle}' not found.")
    return result[0]


def fetch_rating_history(handle):
    # user.rating fails outright for handles that never entered a rated contest
    try:
        return cf_get("user.rating", {"handle": handle})
    except CodeforcesError:
        return []


def fetch_submissions(handle):
    try:
        result = cf_get("user.status", {"handle": handle, "from": 1, "count": MAX_SUBMISSIONS})
    except CodeforcesError:
        return []
    return result


# --------------------------------------------------------------------------
# Data science pipeline
# --------------------------------------------------------------------------

def time_decayed_weight(now_seconds, submission_seconds, lam=DECAY_LAMBDA):
    age_days = max(0.0, (now_seconds - submission_seconds) / 86400.0)
    return math.exp(-lam * age_days)


def compute_tag_stats(submissions, now_seconds):
    """
    Returns a dict: tag -> {
        "numerator": sum(w_i * I(AC_i) * R_i),
        "denominator": sum(w_i)  (over all attempts touching that tag),
        "solved_ratings": [R_i for each *unique solved problem* tagged k],
    }
    Only submissions whose problem carries a numeric 'rating' are considered,
    since TPI and difficulty are undefined without it.
    """
    tag_stats = {}
    seen_solved_problems = {}  # (contestId, index) -> tags, rating  (dedupe repeat ACs)

    for sub in submissions:
        problem = sub.get("problem", {})
        rating = problem.get("rating")
        tags = problem.get("tags", [])
        if rating is None or not tags:
            continue

        creation_ts = sub.get("creationTimeSeconds")
        if creation_ts is None:
            continue

        is_ac = sub.get("verdict") == "OK"
        w = time_decayed_weight(now_seconds, creation_ts)

        problem_key = (problem.get("contestId"), problem.get("index"))

        for tag in tags:
            stats = tag_stats.setdefault(tag, {"numerator": 0.0, "denominator": 0.0, "solved_ratings": {}})
            stats["denominator"] += w
            if is_ac:
                stats["numerator"] += w * rating
                # keep the max weight seen for this (problem, tag) pair so a
                # problem solved once is only counted once toward solved-count
                prev = stats["solved_ratings"].get(problem_key)
                if prev is None or w > prev:
                    stats["solved_ratings"][problem_key] = rating

    return tag_stats


def build_tag_analysis(tag_stats):
    """Compute TPI, solved counts, avg difficulty, then cluster with KMeans."""
    rows = []
    for tag, stats in tag_stats.items():
        solved = stats["solved_ratings"]
        if not solved:
            continue  # never solved anything with this tag -- nothing to recommend on
        solved_count = len(solved)
        avg_difficulty = float(np.mean(list(solved.values())))
        tpi = stats["numerator"] / stats["denominator"] if stats["denominator"] > 0 else 0.0
        rows.append({
            "tag": tag,
            "solvedCount": solved_count,
            "avgDifficulty": round(avg_difficulty, 1),
            "tpi": round(tpi, 1),
        })

    if not rows:
        return []

    n_clusters = min(3, len(rows))
    features = np.array([[r["solvedCount"], r["avgDifficulty"]] for r in rows], dtype=float)

    # Normalize features so solvedCount and avgDifficulty contribute comparably
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    normalized = (features - features.mean(axis=0)) / feature_std

    if n_clusters == 1:
        labels = np.zeros(len(rows), dtype=int)
        centroid_score = {0: np.mean([r["tpi"] for r in rows])}
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(normalized)
        # Score each cluster by its mean TPI (proficiency proxy) to order labels
        centroid_score = {}
        for c in range(n_clusters):
            member_tpis = [rows[i]["tpi"] for i in range(len(rows)) if labels[i] == c]
            centroid_score[c] = float(np.mean(member_tpis)) if member_tpis else 0.0

    ordered_clusters = sorted(centroid_score.keys(), key=lambda c: centroid_score[c])

    label_pool_3 = ["Bottleneck", "Developing", "Proficient"]
    if n_clusters == 3:
        cluster_to_label = {c: label_pool_3[i] for i, c in enumerate(ordered_clusters)}
    elif n_clusters == 2:
        cluster_to_label = {ordered_clusters[0]: "Bottleneck", ordered_clusters[1]: "Proficient"}
    else:
        cluster_to_label = {ordered_clusters[0]: "Developing"}

    for i, row in enumerate(rows):
        row["cluster"] = cluster_to_label[int(labels[i])]

    rows.sort(key=lambda r: r["tpi"])
    return rows


def build_bottlenecks(tag_analysis, top_n=3):
    bottlenecks = [r for r in tag_analysis if r["cluster"] == "Bottleneck"]
    if not bottlenecks:
        # fall back to the lowest-TPI topics overall if clustering found no
        # distinct bottleneck group (e.g. very few tags solved so far)
        bottlenecks = sorted(tag_analysis, key=lambda r: r["tpi"])
    bottlenecks = bottlenecks[:top_n]

    recommendations = []
    for b in bottlenecks:
        target_rating = int(round(b["avgDifficulty"] / 100.0) * 100) if b["avgDifficulty"] else 1200
        recommendations.append({
            "tag": b["tag"],
            "tpi": b["tpi"],
            "solvedCount": b["solvedCount"],
            "recommendation": (
                f"Practice ~5-10 more '{b['tag']}' problems around rating "
                f"{target_rating} to strengthen this topic."
            ),
        })
    return recommendations


def predict_rating_trajectory(rating_history):
    """Fit Ridge(alpha=1.0) on contest index -> rating, forecast next N contests."""
    ratings = np.array([c["newRating"] for c in rating_history], dtype=float)
    n = len(ratings)
    X = np.arange(n).reshape(-1, 1)

    model = Ridge(alpha=1.0)
    model.fit(X, ratings)

    future_X = np.arange(n, n + FUTURE_CONTESTS).reshape(-1, 1)
    future_preds = model.predict(future_X)

    # Ratings can't go below 0; keep forecasts sane.
    future_preds = np.clip(future_preds, 0, None)
    return [int(round(p)) for p in future_preds]


# --------------------------------------------------------------------------
# Request handling
# --------------------------------------------------------------------------

def analyze_handle(handle):
    now_seconds = datetime.now(timezone.utc).timestamp()

    # Validate the handle first (cheap call) so bad handles fail fast without
    # waiting on the much heavier submissions fetch below.
    info = fetch_user_info(handle)

    # rating_history and submissions are independent Codeforces API calls --
    # run them concurrently instead of back-to-back so total wait time is
    # bounded by the slower of the two, not their sum.
    with ThreadPoolExecutor(max_workers=2) as executor:
        rating_future = executor.submit(fetch_rating_history, handle)
        submissions_future = executor.submit(fetch_submissions, handle)
        rating_history = rating_future.result()
        submissions = submissions_future.result()

    if not rating_history:
        raise CodeforcesError(
            f"'{handle}' has no rated contest history yet. "
            "This tool needs at least one rated contest to build a trajectory."
        )

    predicted_ratings = predict_rating_trajectory(rating_history)
    current_rating = rating_history[-1]["newRating"]
    max_rating = max(c["newRating"] for c in rating_history)
    predicted_peak = max(max_rating, max(predicted_ratings) if predicted_ratings else max_rating)

    tag_stats = compute_tag_stats(submissions, now_seconds)
    tag_analysis = build_tag_analysis(tag_stats)
    bottlenecks = build_bottlenecks(tag_analysis)

    history_out = [
        {
            "contestId": c.get("contestId"),
            "contestName": c.get("contestName"),
            "date": datetime.fromtimestamp(c["ratingUpdateTimeSeconds"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "rating": c["newRating"],
        }
        for c in rating_history
    ]

    return {
        "handle": info.get("handle", handle),
        "currentRating": current_rating,
        "maxRating": max_rating,
        "rank": info.get("rank", "unrated"),
        "ratingHistory": history_out,
        "predictedRatings": predicted_ratings,
        "predictedPeakRating": predicted_peak,
        "tagAnalysis": tag_analysis,
        "bottlenecks": bottlenecks,
    }


class handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        # Simple health check / friendly hint for anyone hitting this via GET.
        self._send_json(200, {
            "message": "Send a POST request with JSON body {\"handle\": \"<codeforces_handle>\"} to analyze a user.",
        })

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            content_length = 0

        if content_length <= 0:
            self._send_json(400, {"error": "Empty request body. Expected JSON with a 'handle' field."})
            return

        try:
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return

        handle = (data.get("handle") or "").strip()
        if not handle:
            self._send_json(400, {"error": "Missing required field 'handle'."})
            return
        if not all(ch.isalnum() or ch in "_-" for ch in handle):
            self._send_json(400, {"error": "Handle contains invalid characters."})
            return

        try:
            result = analyze_handle(handle)
            self._send_json(200, result)
        except CodeforcesError as exc:
            self._send_json(400, {"error": str(exc)})
        except UpstreamError as exc:
            self._send_json(500, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- final safety net, never leak a traceback
            self._send_json(500, {"error": f"Unexpected server error: {exc}"})
