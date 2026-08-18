(function () {
  "use strict";

  const API_ENDPOINT = "/api/predict";

  const form = document.getElementById("analyze-form");
  const handleInput = document.getElementById("handle-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const btnLabel = analyzeBtn.querySelector(".btn-label");
  const errorBanner = document.getElementById("error-banner");
  const loadingState = document.getElementById("loading-state");
  const loadingMessage = document.getElementById("loading-message");
  const resultsSection = document.getElementById("results");

  const LOADING_MESSAGES = [
    "Fetching profile from Codeforces…",
    "Pulling rating history and submissions…",
    "Computing tag performance index…",
    "Training the rating trajectory model…",
    "Clustering topic proficiency…",
    "Almost there…",
  ];
  let loadingMessageTimer = null;

  const statCurrentRating = document.getElementById("stat-current-rating");
  const statRank = document.getElementById("stat-rank");
  const statMaxRating = document.getElementById("stat-max-rating");
  const statPredictedPeak = document.getElementById("stat-predicted-peak");
  const bottleneckList = document.getElementById("bottleneck-list");
  const tagClusters = document.getElementById("tag-clusters");

  let ratingChart = null;

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const handle = handleInput.value.trim();
    if (!handle) return;
    analyzeHandle(handle);
  });

  async function analyzeHandle(handle) {
    setLoading(true);
    hideError();
    hideResults();

    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handle: handle }),
      });

      let payload;
      try {
        payload = await response.json();
      } catch (parseErr) {
        throw new Error("Received an unexpected response from the server.");
      }

      if (!response.ok) {
        throw new Error(payload && payload.error ? payload.error : "Something went wrong while analyzing this handle.");
      }

      renderResults(payload);
    } catch (err) {
      showError(err.message || "Unable to analyze this handle right now. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    btnLabel.textContent = isLoading ? "Analyzing…" : "Analyze";
    loadingState.hidden = !isLoading;

    if (isLoading) {
      startLoadingMessageCycle();
    } else {
      stopLoadingMessageCycle();
    }
  }

  function startLoadingMessageCycle() {
    let index = 0;
    loadingMessage.textContent = LOADING_MESSAGES[0];
    loadingMessage.classList.remove("fade");

    stopLoadingMessageCycle();
    loadingMessageTimer = setInterval(function () {
      index = (index + 1) % LOADING_MESSAGES.length;
      loadingMessage.classList.add("fade");
      setTimeout(function () {
        loadingMessage.textContent = LOADING_MESSAGES[index];
        loadingMessage.classList.remove("fade");
      }, 250);
    }, 2200);
  }

  function stopLoadingMessageCycle() {
    if (loadingMessageTimer) {
      clearInterval(loadingMessageTimer);
      loadingMessageTimer = null;
    }
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function hideError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function hideResults() {
    resultsSection.hidden = true;
  }

  function renderResults(data) {
    statCurrentRating.textContent = formatNumber(data.currentRating);
    statRank.textContent = capitalize(data.rank || "unrated");
    statMaxRating.textContent = formatNumber(data.maxRating);
    statPredictedPeak.textContent = formatNumber(data.predictedPeakRating);

    renderChart(data);
    renderBottlenecks(data.bottlenecks || []);
    renderTagClusters(data.tagAnalysis || []);

    resultsSection.hidden = false;
  }

  function renderChart(data) {
    const history = data.ratingHistory || [];
    const predicted = data.predictedRatings || [];

    const historyLabels = history.map(function (c) { return c.date; });
    const historyValues = history.map(function (c) { return c.rating; });

    const predictedLabels = predicted.map(function (_, i) { return "Forecast +" + (i + 1); });
    const labels = historyLabels.concat(predictedLabels);

    // Actual line: real values, then nulls for the forecast region.
    const actualData = historyValues.concat(predicted.map(function () { return null; }));

    // Predicted line: nulls through history, then bridges from the last
    // actual rating into the forecasted points so the dashed line connects.
    const predictedData = history.map(function () { return null; });
    if (historyValues.length > 0) {
      predictedData[predictedData.length - 1] = historyValues[historyValues.length - 1];
    }
    predicted.forEach(function (val) { predictedData.push(val); });

    const ctx = document.getElementById("rating-chart").getContext("2d");

    if (ratingChart) {
      ratingChart.destroy();
    }

    ratingChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Historical Rating",
            data: actualData,
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.12)",
            pointRadius: 2.5,
            pointBackgroundColor: "#38bdf8",
            borderWidth: 2.5,
            tension: 0.25,
            spanGaps: false,
            fill: true,
          },
          {
            label: "Predicted Rating",
            data: predictedData,
            borderColor: "#fbbf24",
            backgroundColor: "rgba(251, 191, 36, 0.08)",
            pointRadius: 3,
            pointBackgroundColor: "#fbbf24",
            borderWidth: 2.5,
            borderDash: [6, 5],
            tension: 0.25,
            spanGaps: true,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1e293b",
            borderColor: "#2a3b52",
            borderWidth: 1,
            titleColor: "#e2e8f0",
            bodyColor: "#94a3b8",
            padding: 10,
          },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
            grid: { color: "rgba(148, 163, 184, 0.08)" },
          },
          y: {
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(148, 163, 184, 0.08)" },
          },
        },
      },
    });
  }

  function renderBottlenecks(bottlenecks) {
    bottleneckList.innerHTML = "";

    if (bottlenecks.length === 0) {
      const li = document.createElement("li");
      li.className = "bottleneck-item";
      li.textContent = "Not enough solved-problem data yet to identify bottlenecks.";
      bottleneckList.appendChild(li);
      return;
    }

    bottlenecks.forEach(function (b) {
      const li = document.createElement("li");
      li.className = "bottleneck-item";

      const header = document.createElement("div");
      header.className = "bottleneck-item-header";

      const tagEl = document.createElement("span");
      tagEl.className = "bottleneck-tag";
      tagEl.textContent = b.tag;

      const tpiEl = document.createElement("span");
      tpiEl.className = "bottleneck-tpi";
      tpiEl.textContent = "TPI " + b.tpi;

      header.appendChild(tagEl);
      header.appendChild(tpiEl);

      const rec = document.createElement("p");
      rec.className = "bottleneck-recommendation";
      rec.textContent = b.recommendation;

      li.appendChild(header);
      li.appendChild(rec);
      bottleneckList.appendChild(li);
    });
  }

  function renderTagClusters(tagAnalysis) {
    tagClusters.innerHTML = "";

    const order = ["Proficient", "Developing", "Bottleneck"];
    const grouped = { Proficient: [], Developing: [], Bottleneck: [] };

    tagAnalysis.forEach(function (t) {
      if (grouped[t.cluster]) {
        grouped[t.cluster].push(t);
      }
    });

    let anyRendered = false;

    order.forEach(function (clusterName) {
      const items = grouped[clusterName];
      if (!items || items.length === 0) return;
      anyRendered = true;

      const group = document.createElement("div");
      group.className = "cluster-group";

      const title = document.createElement("div");
      title.className = "cluster-group-title " + clusterName;

      const badge = document.createElement("span");
      badge.className = "cluster-badge " + clusterName;

      title.appendChild(badge);
      title.appendChild(document.createTextNode(clusterName));
      group.appendChild(title);

      const tagsWrap = document.createElement("div");
      tagsWrap.className = "cluster-tags";

      items.forEach(function (item) {
        const pill = document.createElement("span");
        pill.className = "cluster-tag-pill";
        pill.innerHTML = "<strong>" + escapeHtml(item.tag) + "</strong> &middot; " + item.solvedCount + " solved";
        tagsWrap.appendChild(pill);
      });

      group.appendChild(tagsWrap);
      tagClusters.appendChild(group);
    });

    if (!anyRendered) {
      const empty = document.createElement("p");
      empty.className = "panel-sub";
      empty.textContent = "Not enough tag data to cluster yet.";
      tagClusters.appendChild(empty);
    }
  }

  function formatNumber(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Math.round(n).toLocaleString();
  }

  function capitalize(str) {
    if (!str) return "";
    return str.replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
