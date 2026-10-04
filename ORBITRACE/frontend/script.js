const API_BASE = "http://127.0.0.1:8001";

const state = {
  query: "",
  results: [],
  selected: null,
  map: null,
  marker: null,
  currentVisual: "overlay",
  locationCache: {}
};

const $ = (id) => document.getElementById(id);

function assetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path}`;
}

function formatNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}

function inferQueryIntent(query) {
  const q = String(query || "").toLowerCase();

  if (q.includes("industrial")) return "Industrial development";
  if (q.includes("road") || q.includes("infrastructure")) return "Infrastructure / corridor";
  if (q.includes("forest") || q.includes("vegetation")) return "Vegetation / land-cover";
  if (
    q.includes("urban") ||
    q.includes("building") ||
    q.includes("construction") ||
    q.includes("residential")
  ) return "Built-up / urban";
  if (q.includes("water")) return "Water / surface";

  return "General surface change";
}

function setEngineStatus(type, text) {
  const box = $("engineStatus");
  box.classList.remove("ready", "error");
  if (type) box.classList.add(type);
  $("engineStatusText").textContent = text;
}

async function loadStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/demo/status`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (!data.ready) throw new Error(data.error || "API not ready");

    setEngineStatus("ready", "AI engine ready");
    $("metricVectors").textContent = `${data.faiss_vectors} vectors`;
    $("metricAnalysis").textContent = `${data.rapid_change_results} ready`;

    const archiveCount = Math.round(Number(data.metadata_records || 160) / 2);
    $("metricArchive").textContent = `${archiveCount} AOIs`;
  } catch (error) {
    setEngineStatus("error", "API disconnected");
    console.error(error);
  }
}

function showLoading(visible) {
  $("loadingState").classList.toggle("hidden", !visible);
  $("searchButton").disabled = visible;
  $("searchButton").textContent = visible ? "Searching…" : "Search archive";
}

function showError(message = "") {
  $("errorState").classList.toggle("hidden", !message);
  $("errorText").textContent = message;
}

async function runSearch(queryOverride) {
  const input = $("searchInput");
  const query = String(queryOverride || input.value || "").trim();

  if (!query) {
    input.focus();
    return;
  }

  state.query = query;
  input.value = query;

  showError("");
  showLoading(true);
  $("querySummaryText").textContent = query;
  $("resultCount").textContent = "SEARCHING…";

  try {
    const params = new URLSearchParams({ q: query, top_k: "5" });
    const response = await fetch(`${API_BASE}/api/v1/demo/search?${params.toString()}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();
    state.results = Array.isArray(data.results) ? data.results : [];

    renderResults();
    $("resultCount").textContent = `${state.results.length} RESULTS`;

    if (state.results.length) {
      selectResult(0, false);
      $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      $("heroStatusTitle").textContent = "No matching AOI";
      $("heroStatusSub").textContent = "Try a broader satellite description.";
    }
  } catch (error) {
    console.error(error);
    state.results = [];
    renderResults();
    $("resultCount").textContent = "0 RESULTS";
    showError("Could not complete the search. Confirm the ORBITRACE API is running on port 8001.");
  } finally {
    showLoading(false);
  }
}

function renderResults() {
  const grid = $("resultsGrid");
  grid.innerHTML = "";

  state.results.forEach((result, index) => {
    const analysis = result.analysis || {};
    const card = document.createElement("article");
    card.className = "result-card";
    card.dataset.index = String(index);

    const preview = assetUrl(result.matched_scene?.preview_url || analysis.t1_preview_url);
    const scorePercent = (Number(result.score || 0) * 100).toFixed(1);
    const changeValue = analysis.change_percent == null ? "—" : `${Number(analysis.change_percent).toFixed(3)}%`;

    card.innerHTML = `
      <div class="result-image">
        <img src="${preview}" alt="Satellite preview for ${result.aoi_id}">
        <span class="rank-tag">#${result.rank}</span>
        <span class="ready-tag">${result.analysis_ready ? "ANALYSIS READY" : "RETRIEVAL ONLY"}</span>
      </div>
      <div class="result-info">
        <div class="result-score-row">
          <small>SEMANTIC MATCH</small>
          <strong>${scorePercent}%</strong>
        </div>
        <h3>${result.aoi_id}</h3>
        <p>${result.location_label || "Coordinates unavailable"}</p>
        <div class="result-meta">
          <span>${result.matched_scene?.temporal_role || "SCENE"} ${result.matched_scene?.date || ""}</span>
          <span>${analysis.change_level || "—"}</span>
          <span>${changeValue} change</span>
        </div>
      </div>
    `;

    card.addEventListener("click", () => selectResult(index, true));
    grid.appendChild(card);
  });
}

function offlineLocationHint(lat, lon) {
  const points = [
    {lat:23.34225, lon:113.53271, name:"Huangpu District, Guangdong, China"},
    {lat:36.33283, lon:-94.28467, name:"Northwest Arkansas, United States"},
    {lat:32.49123, lon:119.42139, name:"Jiangsu, China"},
    {lat:16.48876, lon:80.48582, name:"Andhra Pradesh, India"},
    {lat:32.60236, lon:-117.00440, name:"San Diego County, California, United States"},
    {lat:44.35528, lon:26.03760, name:"Bucharest metropolitan area, Romania"}
  ];
  let best=null,bestD=Infinity;
  for(const p of points){const d=Math.hypot(lat-p.lat,lon-p.lon);if(d<bestD){bestD=d;best=p;}}
  return best&&bestD<0.45?best.name:null;
}
async function resolveLocationName(result){
  const lat=Number(result?.coordinates?.lat), lon=Number(result?.coordinates?.lon);
  if(!Number.isFinite(lat)||!Number.isFinite(lon)) return "Location unavailable";
  const key=`${lat.toFixed(5)},${lon.toFixed(5)}`;
  if(state.locationCache[key]) return state.locationCache[key];
  const hint=offlineLocationHint(lat,lon); if(hint){state.locationCache[key]=hint;return hint;}
  try{
    const url=`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&localityLanguage=en`;
    const r=await fetch(url); if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    const parts=[d.city||d.locality,d.principalSubdivision,d.countryName].filter(Boolean);
    const name=[...new Set(parts)].join(", ")||result.location_label||key; state.locationCache[key]=name; return name;
  }catch(e){console.warn("Reverse geocode unavailable",e);return result.location_label||key;}
}

function selectResult(index, scrollToAnalysis = true) {
  const result = state.results[index];
  if (!result) return;

  state.selected = result;

  document.querySelectorAll(".result-card").forEach((card) => {
    card.classList.toggle("selected", Number(card.dataset.index) === index);
  });

  renderSelected(result);

  $("analysis").classList.remove("hidden");
  $("evidence").classList.remove("hidden");

  if (scrollToAnalysis) {
    $("analysis").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderSelected(result) {
  const analysis = result.analysis || {};
  const coords = result.coordinates || {};
  const lat = Number(coords.lat);
  const lon = Number(coords.lon);

  $("selectedRank").textContent = `#${result.rank}`;
  $("selectedAoi").textContent = result.aoi_id;
  $("selectedCoords").textContent = result.location_label || "Coordinates unavailable";
  $("selectedLocationName").textContent = "Resolving location…";
  $("semanticScore").textContent = Number(result.score || 0).toFixed(4);
  $("matchedScene").textContent = `${result.matched_scene?.temporal_role || "Scene"} ${result.matched_scene?.date || ""}`.trim();
  $("changedPixels").textContent = formatNumber(analysis.changed_pixels);
  $("stablePixels").textContent = formatNumber(analysis.stable_pixels);
  $("queryIntent").textContent = inferQueryIntent(state.query);

  $("t1Date").textContent = analysis.t1_date || "—";
  $("t2Date").textContent = analysis.t2_date || "—";

  const change = Number(analysis.change_percent);
  $("changePercent").textContent = Number.isFinite(change) ? `${change.toFixed(3)}%` : "—";
  $("changeInterpretation").textContent = analysis.interpretation || "No interpretation available.";
  $("analysisNote").textContent = analysis.note || "No analysis note available.";

  $("analysisModeBadge").textContent = String(analysis.analysis_mode || "ANALYSIS").replaceAll("_", " ");
  const level = String(analysis.change_level || "UNKNOWN").toUpperCase();
  $("changeLevelBadge").textContent = level;
  $("changeLevelBadge").className = `badge level ${level.toLowerCase()}`;

  $("analysisSubtitle").textContent =
    `${result.aoi_id} · ${analysis.t1_date || "T1"} → ${analysis.t2_date || "T2"}`;

  const t1 = assetUrl(analysis.t1_preview_url);
  const t2 = assetUrl(analysis.t2_preview_url);
  $("beforeImage").src = t1;
  $("afterImage").src = t2;

  state.currentVisual = "overlay";
  setEvidenceVisual("overlay");

  const heroPreview = assetUrl(result.matched_scene?.preview_url || analysis.t2_preview_url);
  if (heroPreview) {
    $("heroScene").src = heroPreview;
    $("heroScene").hidden = false;
  }

  $("heroStatusTitle").textContent =
    `${Number.isFinite(change) ? change.toFixed(3) + "% change" : "Analysis ready"} · ${level}`;
  $("heroStatusSub").textContent =
    `${result.aoi_id} · ${analysis.t1_date || "T1"} → ${analysis.t2_date || "T2"}`;

  $("mapAoi").textContent = result.aoi_id;
  $("mapCoords").textContent = result.location_label || "Coordinates unavailable";

  $("evidenceTitle").textContent = `Analysis for ${result.aoi_id}`;
  $("evQuery").textContent = state.query;
  $("evAoi").textContent = result.aoi_id;
  $("evPeriod").textContent = `${analysis.t1_date || "—"} → ${analysis.t2_date || "—"}`;
  $("evScore").textContent = Number(result.score || 0).toFixed(4);
  $("evMode").textContent = String(analysis.analysis_mode || "—").replaceAll("_", " ");
  $("evChange").textContent = Number.isFinite(change) ? `${change.toFixed(3)}%` : "—";
  $("evLevel").textContent = level;
  $("evSplit").textContent = result.split || "—";

  const engineName = analysis.analysis_mode === "RAPID_CHANGE_SCREENING" ? "Rapid Change Engine" : String(analysis.analysis_mode || "Change Engine").replaceAll("_", " ");
  $("activeChangeEngine").textContent = engineName;
  $("activeChangeEngineSub").textContent = analysis.building_specific ? "Building-specific model inference" : "Actual T1/T2 multispectral screening";

  $("technicalTrace").classList.remove("hidden");
  $("traceQuery").textContent = state.query;
  $("traceScore").textContent = Number(result.score || 0).toFixed(4);
  $("traceAoi").textContent = result.aoi_id;
  $("traceLocation").textContent = "Resolving…";
  $("traceT1").textContent = analysis.t1_date || "—";
  $("traceT2").textContent = analysis.t2_date || "—";
  $("traceEngine").textContent = engineName;
  $("traceChanged").textContent = formatNumber(analysis.changed_pixels);
  $("traceStable").textContent = formatNumber(analysis.stable_pixels);
  $("traceChangePct").textContent = Number.isFinite(change) ? `${change.toFixed(3)}%` : "—";
  $("traceLevel").textContent = level;
  $("traceIntent").textContent = inferQueryIntent(state.query);
  $("traceInterpretation").textContent = analysis.interpretation || "—";
  resolveLocationName(result).then((locationName)=>{if(state.selected?.aoi_id!==result.aoi_id)return;$("selectedLocationName").textContent=locationName;$("traceLocation").textContent=locationName;$("mapAoi").textContent=`${locationName} · ${result.aoi_id}`;});

  resetComparison();
  updateMap(Number.isFinite(lat) ? lat : null, Number.isFinite(lon) ? lon : null, result.aoi_id);
}

function syncBeforeImageWidth() {
  const comparison = $("comparison");
  const beforeImage = $("beforeImage");
  if (comparison && beforeImage) {
    beforeImage.style.width = `${comparison.clientWidth}px`;
  }
}

function resetComparison() {
  $("compareRange").value = "50";
  $("beforeClip").style.width = "50%";
  $("compareDivider").style.left = "50%";
  syncBeforeImageWidth();
}

function setEvidenceVisual(type) {
  const analysis = state.selected?.analysis || {};
  state.currentVisual = type;

  document.querySelectorAll(".visual-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === type);
  });

  if (type === "heatmap") {
    $("evidenceImage").src = assetUrl(analysis.heatmap_url);
    $("evidenceCaption").textContent = "Multispectral difference heatmap";
  } else {
    $("evidenceImage").src = assetUrl(analysis.overlay_url);
    $("evidenceCaption").textContent = "Detected surface-change overlay";
  }
}

function initMap() {
  if (typeof L === "undefined") return;

  state.map = L.map("map", { zoomControl: true }).setView([20, 0], 2);

  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Imagery © Esri",
      maxZoom: 18
    }
  ).addTo(state.map);
}

function updateMap(lat, lon, aoi) {
  if (!state.map || lat == null || lon == null) return;

  state.map.setView([lat, lon], 11);

  if (state.marker) {
    state.map.removeLayer(state.marker);
  }

  state.marker = L.marker([lat, lon])
    .addTo(state.map)
    .bindPopup(`<b>${aoi}</b><br>${lat.toFixed(5)}, ${lon.toFixed(5)}`)
    .openPopup();

  setTimeout(() => state.map.invalidateSize(), 120);
}

function evidencePayload() {
  const result = state.selected;
  if (!result) return null;

  return {
    system: "ORBITRACE",
    query: state.query,
    retrieval: {
      rank: result.rank,
      score: result.score,
      aoi_id: result.aoi_id,
      split: result.split,
      location: result.location_label,
      coordinates: result.coordinates,
      matched_scene: result.matched_scene
    },
    analysis: result.analysis
  };
}

function downloadEvidence() {
  const payload = evidencePayload();
  if (!payload) return;

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `ORBITRACE_${payload.retrieval.aoi_id}_evidence.json`;
  a.click();

  URL.revokeObjectURL(url);
}

async function copyEvidence() {
  const payload = evidencePayload();
  if (!payload) return;

  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    $("copyStatus").textContent = "Copied";
  } catch {
    $("copyStatus").textContent = "Copy unavailable";
  }

  setTimeout(() => {
    $("copyStatus").textContent = "";
  }, 1600);
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadStatus();

  $("searchButton").addEventListener("click", () => runSearch());
  $("searchInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearch();
  });

  document.querySelectorAll(".query-chip").forEach((button) => {
    button.addEventListener("click", () => {
      runSearch(button.dataset.query);
    });
  });

  $("compareRange").addEventListener("input", function () {
    $("beforeClip").style.width = `${this.value}%`;
    $("compareDivider").style.left = `${this.value}%`;
  });

  document.querySelectorAll(".visual-tab").forEach((button) => {
    button.addEventListener("click", () => setEvidenceVisual(button.dataset.view));
  });

  $("downloadEvidenceButton").addEventListener("click", downloadEvidence);
  $("copyEvidenceButton").addEventListener("click", copyEvidence);

  window.addEventListener("resize", syncBeforeImageWidth);
});

const hero = document.querySelector(".hero-premium");
const earthScene = document.querySelector(".earth-scene");

if (hero && earthScene) {
  hero.addEventListener("mousemove", (e) => {
    const rect = hero.getBoundingClientRect();
    const x = (e.clientX - rect.left - rect.width / 2) / rect.width;
    const y = (e.clientY - rect.top - rect.height / 2) / rect.height;

    earthScene.style.transform =
      `translate(${x * 12}px, ${y * 10}px)`;
  });

  hero.addEventListener("mouseleave", () => {
    earthScene.style.transform = "translate(0px, 0px)";
  });
}

document.querySelectorAll(".hero-suggestions button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const input = document.getElementById("searchInput");
    if (input) input.value = btn.textContent.trim();
  });
});
