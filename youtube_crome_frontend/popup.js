const DEFAULT_API_URL = "http://127.0.0.1:5001";

const elements = {
  apiUrl: document.getElementById("apiUrl"),
  limit: document.getElementById("limit"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  clearBtn: document.getElementById("clearBtn"),
  status: document.getElementById("status"),
  summary: document.getElementById("summary"),
  positiveCount: document.getElementById("positiveCount"),
  neutralCount: document.getElementById("neutralCount"),
  negativeCount: document.getElementById("negativeCount"),
  chartSection: document.getElementById("chartSection"),
  chartImage: document.getElementById("chartImage"),
  wordcloudSection: document.getElementById("wordcloudSection"),
  wordcloudImage: document.getElementById("wordcloudImage"),
  results: document.getElementById("results")
};

let chartUrl;
let wordcloudUrl;

document.addEventListener("DOMContentLoaded", init);
elements.analyzeBtn.addEventListener("click", analyzeCurrentVideo);
elements.clearBtn.addEventListener("click", clearResults);
elements.apiUrl.addEventListener("change", saveSettings);
elements.limit.addEventListener("change", saveSettings);

async function init() {
  const settings = await chrome.storage.local.get(["apiUrl", "limit"]);
  elements.apiUrl.value = settings.apiUrl || DEFAULT_API_URL;
  elements.limit.value = settings.limit || "50";
}

async function saveSettings() {
  await chrome.storage.local.set({
    apiUrl: normalizeApiUrl(elements.apiUrl.value),
    limit: elements.limit.value
  });
}

async function analyzeCurrentVideo() {
  setBusy(true);
  clearResults();
  setStatus("Checking current tab...");

  try {
    await saveSettings();
    const tab = await getActiveTab();
    if (!tab?.id || !isYouTubeVideoUrl(tab.url)) {
      throw new Error("Open a YouTube video page first.");
    }

    const limit = Number(elements.limit.value);
    setStatus("Collecting visible YouTube comments...");
    const comments = await collectComments(tab.id, limit);
    if (!comments.length) {
      throw new Error("No comments found. Scroll down until YouTube loads comments, then try again.");
    }

    setStatus(`Analyzing ${comments.length} comments...`);
    const apiUrl = normalizeApiUrl(elements.apiUrl.value);
    const predictions = await postJson(`${apiUrl}/predict_with_timestamps`, {
      comments: comments.map((item) => ({
        text: item.text,
        timestamp: item.timestamp || new Date().toISOString()
      }))
    });

    renderSummary(predictions);
    renderResults(predictions);
    await renderImages(apiUrl, predictions);
    setStatus(`Done. Analyzed ${predictions.length} comments.`);
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    setBusy(false);
  }
}

function clearResults() {
  revokeImageUrls();
  elements.summary.classList.add("hidden");
  elements.chartSection.classList.add("hidden");
  elements.wordcloudSection.classList.add("hidden");
  elements.positiveCount.textContent = "0";
  elements.neutralCount.textContent = "0";
  elements.negativeCount.textContent = "0";
  elements.results.innerHTML = '<div class="empty">No analysis yet.</div>';
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function isYouTubeVideoUrl(url = "") {
  try {
    const parsed = new URL(url);
    return /(^|\.)youtube\.com$/.test(parsed.hostname) && parsed.pathname === "/watch";
  } catch {
    return false;
  }
}

async function collectComments(tabId, limit) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId },
    args: [limit],
    func: scrapeYouTubeComments
  });
  return result?.result || [];
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof data === "object" ? data.error : data;
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return data;
}

async function fetchImage(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Image request failed: ${response.status}`);
  }
  return URL.createObjectURL(await response.blob());
}

function renderSummary(predictions) {
  const counts = countSentiments(predictions);
  elements.positiveCount.textContent = String(counts["1"]);
  elements.neutralCount.textContent = String(counts["0"]);
  elements.negativeCount.textContent = String(counts["-1"]);
  elements.summary.classList.remove("hidden");
}

function renderResults(predictions) {
  if (!predictions.length) {
    elements.results.innerHTML = '<div class="empty">No comments returned.</div>';
    return;
  }

  elements.results.innerHTML = predictions
    .map((item) => {
      const sentiment = Number(item.sentiment);
      const label = item.sentiment_label || sentimentLabel(sentiment);
      return `
        <div class="row">
          <span class="badge ${sentimentClass(sentiment)}">${escapeHtml(label)}</span>
          <p class="comment">${escapeHtml(item.comment)}</p>
        </div>
      `;
    })
    .join("");
}

async function renderImages(apiUrl, predictions) {
  const counts = countSentiments(predictions);
  const comments = predictions.map((item) => item.comment);

  chartUrl = await fetchImage(`${apiUrl}/generate_chart`, { sentiment_counts: counts });
  elements.chartImage.src = chartUrl;
  elements.chartSection.classList.remove("hidden");

  wordcloudUrl = await fetchImage(`${apiUrl}/generate_wordcloud`, { comments });
  elements.wordcloudImage.src = wordcloudUrl;
  elements.wordcloudSection.classList.remove("hidden");
}

function countSentiments(predictions) {
  return predictions.reduce(
    (counts, item) => {
      const key = String(item.sentiment);
      if (Object.prototype.hasOwnProperty.call(counts, key)) {
        counts[key] += 1;
      }
      return counts;
    },
    { "1": 0, "0": 0, "-1": 0 }
  );
}

function normalizeApiUrl(url) {
  return (url || DEFAULT_API_URL).trim().replace(/\/+$/, "");
}

function sentimentLabel(sentiment) {
  if (sentiment === 1) return "Positive";
  if (sentiment === 0) return "Neutral";
  if (sentiment === -1) return "Negative";
  return "Unknown";
}

function sentimentClass(sentiment) {
  if (sentiment === 1) return "positive";
  if (sentiment === 0) return "neutral";
  if (sentiment === -1) return "negative";
  return "neutral";
}

function setBusy(isBusy) {
  elements.analyzeBtn.disabled = isBusy;
  elements.clearBtn.disabled = isBusy;
  elements.apiUrl.disabled = isBusy;
  elements.limit.disabled = isBusy;
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.style.color = isError ? "#c3332f" : "#657184";
}

function revokeImageUrls() {
  if (chartUrl) URL.revokeObjectURL(chartUrl);
  if (wordcloudUrl) URL.revokeObjectURL(wordcloudUrl);
  chartUrl = undefined;
  wordcloudUrl = undefined;
  elements.chartImage.removeAttribute("src");
  elements.wordcloudImage.removeAttribute("src");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function scrapeYouTubeComments(limit) {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const commentsRoot = document.querySelector("ytd-comments");
  if (commentsRoot) {
    commentsRoot.scrollIntoView({ behavior: "auto", block: "start" });
  } else {
    window.scrollTo({ top: Math.floor(document.body.scrollHeight * 0.45), behavior: "auto" });
  }

  const readComments = () => {
    const nodes = Array.from(document.querySelectorAll("ytd-comment-thread-renderer"));
    const seen = new Set();
    const comments = [];

    for (const node of nodes) {
      const textNode = node.querySelector("#content-text");
      const text = textNode?.innerText?.replace(/\s+/g, " ").trim();
      if (!text || seen.has(text)) continue;

      seen.add(text);
      const timestamp =
        node.querySelector("#published-time-text a")?.innerText?.trim() ||
        node.querySelector("#published-time-text")?.innerText?.trim() ||
        new Date().toISOString();

      comments.push({ text, timestamp });
      if (comments.length >= limit) break;
    }

    return comments;
  };

  let comments = [];
  for (let attempt = 0; attempt < 8; attempt += 1) {
    comments = readComments();
    if (comments.length >= limit) break;
    window.scrollBy({ top: 900, behavior: "auto" });
    await sleep(550);
  }

  return comments.slice(0, limit);
}
