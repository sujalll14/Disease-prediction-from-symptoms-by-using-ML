/**
 * script.js — Disease Prediction System
 * Frontend logic: symptom selection, API calls, Chart.js charts
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let allSymptoms   = [];
let modelInfo     = {};
let probChartObj  = null;
let accChartObj   = null;
let pieChartObj   = null;
let currentFilter = "all";

// ── Symptom tag categories ────────────────────────────────────────────────────
const tagMap = {
  all         : () => true,
  fever       : s => ["fever","chills","night_sweats","sweating","mild_fever"].includes(s),
  pain        : s => s.includes("pain") || s.includes("ache") || ["stiffness","swelling","joint_pain"].includes(s),
  respiratory : s => ["cough","shortness_of_breath","wheezing","chest_tightness","chest_pain","coughing_blood","congestion"].includes(s),
  digestive   : s => ["nausea","vomiting","diarrhea","bloating","indigestion","abdominal_pain","loss_of_appetite"].includes(s),
};

// ── Chart.js Defaults ─────────────────────────────────────────────────────────
Chart.defaults.color       = "#8b949e";
Chart.defaults.borderColor = "#30363d";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", init);

async function init() {
  try {
    const res  = await fetch("/api/info");
    const data = await res.json();

    allSymptoms = data.symptoms;
    modelInfo   = data.model_info;

    document.getElementById("modelBadge").textContent   = modelInfo.best_model + " · " + modelInfo.accuracy + "%";
    document.getElementById("statSymptoms").textContent = allSymptoms.length;
    document.getElementById("statAcc").textContent      = modelInfo.accuracy + "%";

    renderSymptoms(allSymptoms);
    renderAccChart(modelInfo.all_results);
    renderPieChart(modelInfo.diseases || []);

  } catch (err) {
    console.error("Init error:", err);
    showError("Could not connect to the Flask server. Make sure app.py is running on http://127.0.0.1:5000");
  }
}

// ── Render Symptom Checkboxes ─────────────────────────────────────────────────
function renderSymptoms(list) {
  const grid = document.getElementById("symptomsGrid");
  const fn   = tagMap[currentFilter] || tagMap.all;
  const filtered = list.filter(fn);

  if (filtered.length === 0) {
    grid.innerHTML = `<div style="grid-column:span 2;text-align:center;padding:24px;color:var(--sub2)">No symptoms found</div>`;
    return;
  }

  grid.innerHTML = filtered.map(s => {
    const label     = s.replace(/_/g, " ");
    const isChecked = isSymptomChecked(s);
    return `
      <div class="symptom-item ${isChecked ? "checked" : ""}" id="item_${s}" onclick="toggle('${s}')">
        <input type="checkbox" id="cb_${s}" ${isChecked ? "checked" : ""}
               onclick="event.stopPropagation(); toggle('${s}')">
        <label for="cb_${s}">${label}</label>
      </div>`;
  }).join("");
}

function isSymptomChecked(s) {
  const cb = document.getElementById("cb_" + s);
  return cb ? cb.checked : false;
}

function toggle(s) {
  const item = document.getElementById("item_" + s);
  const cb   = document.getElementById("cb_"   + s);
  if (!item || !cb) return;
  cb.checked = !cb.checked;
  item.classList.toggle("checked", cb.checked);
  updateSelectedCount();
}

function updateSelectedCount() {
  const n = document.querySelectorAll(".symptom-item.checked").length;
  document.getElementById("statSelected").textContent = n;
}

// ── Search ────────────────────────────────────────────────────────────────────
function filterSymptoms() {
  const q   = document.getElementById("searchBox").value.toLowerCase().trim();
  document.getElementById("clearSearch").style.display = q ? "block" : "none";
  const list = allSymptoms.filter(s => s.replace(/_/g, " ").includes(q));
  renderSymptoms(list);
}

function clearSearch() {
  document.getElementById("searchBox").value = "";
  document.getElementById("clearSearch").style.display = "none";
  renderSymptoms(allSymptoms);
}

// ── Tag Filter ────────────────────────────────────────────────────────────────
function filterByTag(btn, tag) {
  document.querySelectorAll(".tag").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  currentFilter = tag;
  renderSymptoms(allSymptoms);
}

// ── Clear All ─────────────────────────────────────────────────────────────────
function clearAll() {
  document.querySelectorAll(".symptom-item").forEach(el => el.classList.remove("checked"));
  document.querySelectorAll(".symptom-item input").forEach(cb => cb.checked = false);
  updateSelectedCount();
  document.getElementById("idleState").style.display     = "block";
  document.getElementById("resultContent").style.display = "none";
  if (probChartObj) { probChartObj.destroy(); probChartObj = null; }
  const pe = document.getElementById("probEmpty");
  if (pe) pe.style.display = "flex";
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function predict() {
  const payload = {};
  allSymptoms.forEach(s => { payload[s] = isSymptomChecked(s) ? 1 : 0; });

  const selectedCount = Object.values(payload).reduce((a, b) => a + b, 0);
  if (selectedCount === 0) {
    alert("Please select at least one symptom before predicting.");
    return;
  }

  showOverlay(true);

  try {
    const res = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Server returned " + res.status);

    const data = await res.json();
    showResult(data);
    if (data.top_predictions && data.top_predictions.length) {
      renderProbChart(data.top_predictions);
    }
  } catch (err) {
    showError("Prediction failed: " + err.message);
  } finally {
    showOverlay(false);
  }
}

// ── Show Result ───────────────────────────────────────────────────────────────
function showResult(data) {
  const conf     = Math.round(data.confidence * 100);
  const disease  = data.predicted_disease || "Unknown";
  const advice   = data.advice || "Please consult a qualified medical professional.";
  const modelTag = data.model_used || "";

  const color = conf >= 75 ? "var(--green)" : conf >= 50 ? "var(--yellow)" : "var(--red)";

  const html = `
    <div class="result-top">
      <div style="flex:1">
        <div class="result-label">Predicted Disease</div>
        <div class="disease-name" style="color:${color}">${disease}</div>
        <div class="model-tag">🤖 ${modelTag}</div>
      </div>
      <div class="confidence-section">
        <div class="conf-label">
          <span>Confidence Score</span>
          <span class="conf-pct" style="color:${color}">${conf}%</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" id="confFill"
               style="width:0%;background:linear-gradient(90deg,${color},${color}88)"></div>
        </div>
      </div>
    </div>
    <div class="advice-box">
      <strong>💡 Medical Advice:</strong> ${advice}
    </div>`;

  document.getElementById("idleState").style.display     = "none";
  document.getElementById("resultContent").style.display = "block";
  document.getElementById("resultContent").innerHTML     = html;

  setTimeout(() => {
    const fill = document.getElementById("confFill");
    if (fill) fill.style.width = conf + "%";
  }, 120);
}

function showError(msg) {
  document.getElementById("idleState").style.display     = "none";
  document.getElementById("resultContent").style.display = "block";
  document.getElementById("resultContent").innerHTML =
    `<div style="color:var(--red);text-align:center;padding:20px">❌ ${msg}</div>`;
}

// ── Probability Bar Chart ─────────────────────────────────────────────────────
function renderProbChart(preds) {
  const pe = document.getElementById("probEmpty");
  if (pe) pe.style.display = "none";
  if (probChartObj) probChartObj.destroy();

  const labels = preds.map(p => p.disease);
  const vals   = preds.map(p => Math.round(p.probability * 100));
  const colors = vals.map((_, i) => i === 0 ? "rgba(88,166,255,0.9)" : "rgba(88,166,255,0.32)");

  const ctx = document.getElementById("probChart").getContext("2d");
  probChartObj = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Probability %", data: vals, backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10 }, maxRotation: 30 }, grid: { color: "#21262d" } },
        y: { grid: { color: "#21262d" }, min: 0, max: 100, title: { display: true, text: "Probability (%)", color: "#6e7681", font: { size: 11 } } },
      },
    },
  });
}

// ── Model Accuracy Bar Chart ──────────────────────────────────────────────────
function renderAccChart(results) {
  if (accChartObj) accChartObj.destroy();

  const labels = Object.keys(results);
  const vals   = Object.values(results);
  const maxVal = Math.max(...vals);
  const colors = vals.map(v => v === maxVal ? "rgba(63,185,80,0.85)" : "rgba(88,166,255,0.4)");

  const ctx = document.getElementById("accChart").getContext("2d");
  accChartObj = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Accuracy %", data: vals, backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10 }, maxRotation: 30 }, grid: { color: "#21262d" } },
        y: { grid: { color: "#21262d" }, min: 0, max: 100, title: { display: true, text: "Accuracy (%)", color: "#6e7681", font: { size: 11 } } },
      },
    },
  });
}

// ── Disease Distribution Doughnut ─────────────────────────────────────────────
function renderPieChart(diseases) {
  if (!diseases || diseases.length === 0) return;
  if (pieChartObj) pieChartObj.destroy();

  const categories = {
    "Infectious"  : ["Flu","Common Cold","COVID-19","Typhoid","Malaria","Dengue","Tuberculosis","Chickenpox","Measles"],
    "Respiratory" : ["Pneumonia","Asthma"],
    "Chronic"     : ["Diabetes","Hypertension","Arthritis","Anemia"],
    "Digestive"   : ["Gastritis","Appendicitis","Jaundice","Kidney_Stone"],
    "Neurological": ["Migraine"],
  };

  const catCounts = {};
  diseases.forEach(d => {
    let found = false;
    for (const [cat, list] of Object.entries(categories)) {
      if (list.includes(d)) { catCounts[cat] = (catCounts[cat] || 0) + 1; found = true; break; }
    }
    if (!found) catCounts["Other"] = (catCounts["Other"] || 0) + 1;
  });

  const palette = ["#58a6ff","#3fb950","#e3b341","#f85149","#a371f7","#79c0ff"];
  const ctx     = document.getElementById("pieChart").getContext("2d");

  pieChartObj = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: Object.keys(catCounts),
      datasets: [{ data: Object.values(catCounts), backgroundColor: palette, borderColor: "#161b22", borderWidth: 3, hoverOffset: 8 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "62%",
      plugins: {
        legend: { position: "right", labels: { padding: 16, usePointStyle: true, pointStyleWidth: 10, font: { size: 12 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed} disease${ctx.parsed > 1 ? "s" : ""}` } },
      },
    },
  });
}

// ── Overlay ───────────────────────────────────────────────────────────────────
function showOverlay(show) {
  document.getElementById("overlay").classList.toggle("active", show);
}
