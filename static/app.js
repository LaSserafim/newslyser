/**
 * NEWSLYSER // CLIENT TELEMETRY & ENGINE CONTROLLER
 * Zero-dependency pure JavaScript for mathematical charts and live interactive simulation.
 */

let historicalData = [];
let modelData = {};
let tomorrowMeta = {};
let macroSummary = {};

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  fetchInitialData();
  setupSimulationListeners();
});

function initClock() {
  const clockEl = document.getElementById("live-clock");
  function tick() {
    const now = new Date();
    if (clockEl) {
      clockEl.innerText = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
    }
  }
  tick();
  setInterval(tick, 1000);
}

async function fetchInitialData() {
  try {
    const [histRes, tomRes] = await Promise.all([
      fetch("/api/historical"),
      fetch("/api/tomorrow")
    ]);

    const histJson = await histRes.json();
    const tomJson = await tomRes.json();

    historicalData = histJson.events || [];
    modelData = histJson.models || {};
    tomorrowMeta = tomJson.event || {};
    macroSummary = tomJson.macro_snapshots || {};

    renderTomorrowBanner(tomorrowMeta);
    renderModelComparison(modelData);
    renderHistoricalTable(historicalData);
    renderComponentChart(macroSummary);
    renderScatterChart(historicalData, modelData);
    
    // Initial simulation with forecast
    runSimulation(tomorrowMeta.forecast_mom, tomorrowMeta.forecast_core_yoy);
  } catch (err) {
    console.error("Data load failed:", err);
  }
}

function renderTomorrowBanner(tom) {
  document.getElementById("tom-date").innerText = tom.date || "2026-11-10";
  document.getElementById("tom-time").innerText = tom.time || "08:30 AM EST";
  document.getElementById("tom-forecast-mom").innerText = tom.forecast_mom !== undefined ? `${tom.forecast_mom}%` : "0.3%";
  document.getElementById("tom-prev-mom").innerText = tom.prev_mom !== undefined ? `${tom.prev_mom}%` : "0.6%";
  document.getElementById("tom-forecast-core").innerText = tom.forecast_core_yoy !== undefined ? `${tom.forecast_core_yoy}%` : "2.8%";
  document.getElementById("tom-prev-core").innerText = tom.prev_core_yoy !== undefined ? `${tom.prev_core_yoy}%` : "2.6%";
  document.getElementById("tom-nowcast-mom").innerText = tom.cleveland_fed_nowcast_mom !== undefined ? `${tom.cleveland_fed_nowcast_mom}%` : "0.28%";
  document.getElementById("tom-nowcast-core").innerText = tom.cleveland_fed_nowcast_core_mom !== undefined ? `${tom.cleveland_fed_nowcast_core_mom}%` : "0.22%";
  
  // Set default form values
  const momInput = document.getElementById("sim-mom");
  const momRange = document.getElementById("sim-mom-range");
  const coreInput = document.getElementById("sim-core");
  const coreRange = document.getElementById("sim-core-range");
  
  if (momInput && tom.forecast_mom !== undefined) momInput.value = tom.forecast_mom;
  if (momRange && tom.forecast_mom !== undefined) momRange.value = tom.forecast_mom;
  if (coreInput && tom.forecast_core_yoy !== undefined) coreInput.value = tom.forecast_core_yoy;
  if (coreRange && tom.forecast_core_yoy !== undefined) coreRange.value = tom.forecast_core_yoy;
}

function setupSimulationListeners() {
  const momInput = document.getElementById("sim-mom");
  const momRange = document.getElementById("sim-mom-range");
  const coreInput = document.getElementById("sim-core");
  const coreRange = document.getElementById("sim-core-range");
  const btnDeepseek = document.getElementById("btn-deepseek");

  if (momInput && momRange) {
    momInput.addEventListener("input", (e) => {
      momRange.value = e.target.value;
      runSimulation(parseFloat(e.target.value), parseFloat(coreInput.value));
    });
    momRange.addEventListener("input", (e) => {
      momInput.value = e.target.value;
      runSimulation(parseFloat(e.target.value), parseFloat(coreInput.value));
    });
  }

  if (coreInput && coreRange) {
    coreInput.addEventListener("input", (e) => {
      coreRange.value = e.target.value;
      runSimulation(parseFloat(momInput.value), parseFloat(e.target.value));
    });
    coreRange.addEventListener("input", (e) => {
      coreInput.value = e.target.value;
      runSimulation(parseFloat(momInput.value), parseFloat(e.target.value));
    });
  }

  if (btnDeepseek) {
    btnDeepseek.addEventListener("click", () => {
      executeDeepSeekBriefing(parseFloat(momInput.value), parseFloat(coreInput.value));
    });
  }
}

async function runSimulation(actualMom, actualCore) {
  if (isNaN(actualMom) || isNaN(actualCore)) return;
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_mom: actualMom, actual_core_yoy: actualCore })
    });
    const sim = await res.json();
    renderSimulationOutputs(sim);
  } catch (err) {
    console.error("Simulation failed:", err);
  }
}

function renderSimulationOutputs(sim) {
  const s = sim.surprises;
  const f = sim.model_forecast;

  document.getElementById("out-hl-surprise").innerText = (s.headline_surprise_pp > 0 ? "+" : "") + s.headline_surprise_pp.toFixed(2) + " pp";
  document.getElementById("out-core-surprise").innerText = (s.core_surprise_pp > 0 ? "+" : "") + s.core_surprise_pp.toFixed(2) + " pp";
  
  const zEl = document.getElementById("out-z-comp");
  zEl.innerText = (s.z_score_composite > 0 ? "+" : "") + s.z_score_composite.toFixed(2) + " σ";
  zEl.style.color = Math.abs(s.z_score_composite) >= 1.5 ? "var(--accent-red)" : (Math.abs(s.z_score_composite) >= 1.0 ? "var(--accent-cyan)" : "var(--fg-muted)");

  const goldEl = document.getElementById("out-gold-ret");
  goldEl.innerText = (f.expected_gold_return_pct > 0 ? "+" : "") + f.expected_gold_return_pct.toFixed(2) + "%";
  goldEl.className = "stat-value " + (f.expected_gold_return_pct > 0 ? "val-up" : (f.expected_gold_return_pct < 0 ? "val-down" : "val-neutral"));

  const dxyEl = document.getElementById("out-dxy-ret");
  dxyEl.innerText = (f.expected_dxy_return_pct > 0 ? "+" : "") + f.expected_dxy_return_pct.toFixed(2) + "%";
  dxyEl.className = "stat-value " + (f.expected_dxy_return_pct > 0 ? "val-up" : "val-down");

  document.getElementById("out-2y-bps").innerText = (f.expected_2y_yield_shift_bps > 0 ? "+" : "") + f.expected_2y_yield_shift_bps.toFixed(1) + " bps";

  // Tactical Signal Box
  const sigBox = document.getElementById("signal-box");
  const sigTitle = document.getElementById("signal-title");
  const sigDesc = document.getElementById("signal-desc");

  sigTitle.innerText = f.signal;
  sigDesc.innerText = f.tactical_action;

  sigBox.className = "signal-box " + (
    f.signal.includes("SHORT") ? "signal-strong-short" :
    (f.signal.includes("LONG") ? "signal-strong-long" : "signal-neutral")
  );
}

async function executeDeepSeekBriefing(actualMom, actualCore) {
  const terminal = document.getElementById("deepseek-terminal");
  const btn = document.getElementById("btn-deepseek");
  
  btn.disabled = true;
  btn.innerText = "[ PROCESSING QUANT INTEL... ]";
  terminal.innerText = ">>> INITIATING DEEPSEEK V3 QUANTITATIVE MACRO REASONING PROTOCOL...\n>>> CONNECTING TO API ENGINE (api.deepseek.com)...\n>>> PARAMETERS: MOM=" + actualMom + "%, CORE_YOY=" + actualCore + "%\n>>> CALCULATING CROSS-ASSET TRANSMISSION VECTOR...\n\n";

  try {
    const res = await fetch("/api/deepseek-briefing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_mom: actualMom, actual_core_yoy: actualCore })
    });
    const data = await res.json();
    
    if (data.status === "success") {
      terminal.innerText = "=== DEEPSEEK V3 INSTITUTIONAL DESK BRIEFING // OCT 2026 CPI ===\n" + data.briefing;
    } else {
      terminal.innerText += "[ERROR]: " + JSON.stringify(data);
    }
  } catch (err) {
    terminal.innerText += "\n[FATAL ERROR]: Failed to reach DeepSeek API: " + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = "RUN DEEPSEEK MACRO BRIEFING";
  }
}

function renderModelComparison(m) {
  if (!m.baseline_ols || !m.refined_composite) return;

  const base = m.baseline_ols;
  const ref = m.refined_composite;
  const reg = m.regime_breakdown;

  document.getElementById("base-formula").innerText = base.formula;
  document.getElementById("base-r2").innerText = base.r2.toFixed(4);
  document.getElementById("base-slope").innerText = base.slope.toFixed(3);
  document.getElementById("base-acc").innerText = base.directional_accuracy_pct + "%";

  document.getElementById("ref-formula").innerText = ref.formula;
  document.getElementById("ref-r2").innerText = ref.r2.toFixed(4);
  document.getElementById("ref-slope").innerText = ref.slope.toFixed(3);
  document.getElementById("ref-p").innerText = ref.p_value < 0.001 ? "< 0.0001 (HC3)" : ref.p_value;
  document.getElementById("ref-corr").innerText = ref.corr_gold_dxy.toFixed(3);

  if (reg && reg.momentum_shock_zone) {
    const sEl = document.getElementById("shock-acc");
    if (sEl) sEl.innerText = reg.momentum_shock_zone.directional_accuracy_pct + "%";
    const nEl = document.getElementById("noise-acc");
    if (nEl) nEl.innerText = reg.noise_zone.directional_accuracy_pct + "%";
  }
}

function renderHistoricalTable(events) {
  const tbody = document.getElementById("hist-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  // Sort descending by date
  const sorted = [...events].sort((a, b) => new Date(b.date) - new Date(a.date));

  sorted.forEach(ev => {
    const tr = document.createElement("tr");

    const retClass = ev.gold_return_pct > 0 ? "val-up" : (ev.gold_return_pct < 0 ? "val-down" : "val-neutral");
    const dxyClass = ev.dxy_return_pct > 0 ? "val-up" : "val-down";
    const zColor = Math.abs(ev.z_score_composite) >= 1.5 ? "color:var(--accent-red);" : (Math.abs(ev.z_score_composite) >= 1.0 ? "color:var(--accent-cyan);" : "color:var(--fg-muted);");

    tr.innerHTML = `
      <td>${ev.date}</td>
      <td><strong>${ev.ref_month}</strong></td>
      <td>${ev.actual_mom.toFixed(1)}%</td>
      <td>${ev.forecast_mom.toFixed(1)}%</td>
      <td class="${ev.headline_surprise > 0 ? 'val-up' : (ev.headline_surprise < 0 ? 'val-down' : '')}">${(ev.headline_surprise > 0 ? '+' : '') + ev.headline_surprise.toFixed(2)}</td>
      <td>${ev.actual_core_yoy.toFixed(1)}%</td>
      <td>${ev.forecast_core_yoy.toFixed(1)}%</td>
      <td style="${zColor}"><strong>${(ev.z_score_composite > 0 ? '+' : '') + ev.z_score_composite.toFixed(2)}σ</strong></td>
      <td>$${ev.gold_open ? ev.gold_open.toFixed(1) : '-'}</td>
      <td class="${retClass}"><strong>${ev.gold_return_pct !== undefined ? (ev.gold_return_pct > 0 ? '+' : '') + ev.gold_return_pct.toFixed(2) + '%' : '-'}</strong></td>
      <td>${ev.gold_range_pct !== undefined ? ev.gold_range_pct.toFixed(2) + '%' : '-'}</td>
      <td class="${dxyClass}">${ev.dxy_return_pct !== undefined ? (ev.dxy_return_pct > 0 ? '+' : '') + ev.dxy_return_pct.toFixed(2) + '%' : '-'}</td>
      <td><span style="font-size:9px; padding:2px 4px; border:1px solid ${ev.regime === 'MOMENTUM_DRIFT' ? 'var(--accent-red)' : (ev.regime === 'MODERATE_SHOCK' ? 'var(--accent-cyan)' : 'var(--border-color)')};">${ev.regime}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderComponentChart(macro) {
  const container = document.getElementById("component-chart-container");
  if (!container || !macro.cpi_headline_idx) return;

  const hlTail = macro.cpi_headline_idx.history_tail || [];
  const coreTail = macro.cpi_core_idx.history_tail || [];
  const superTail = macro.cpi_supercore_idx.history_tail || [];

  const n = hlTail.length;
  if (n < 2) return;

  const w = container.clientWidth || 700;
  const h = 260;
  const padL = 45;
  const padR = 20;
  const padT = 20;
  const padB = 30;

  // Compute YoY or MoM indexed
  const hlVals = hlTail.map(d => d.val);
  const coreVals = coreTail.map(d => d.val);
  const superVals = superTail.map(d => d.val);

  const minV = Math.min(...hlVals, ...coreVals, ...superVals) * 0.98;
  const maxV = Math.max(...hlVals, ...coreVals, ...superVals) * 1.02;

  function getX(i) {
    return padL + (i / (n - 1)) * (w - padL - padR);
  }
  function getY(v) {
    return padT + (1 - (v - minV) / (maxV - minV)) * (h - padT - padB);
  }

  function makePath(vals) {
    return vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getY(v).toFixed(1)}`).join(' ');
  }

  const pHL = makePath(hlVals);
  const pCore = makePath(coreVals);
  const pSuper = makePath(superVals);

  // SVG String
  let svg = `
    <svg class="telemetry-svg" viewBox="0 0 ${w} ${h}">
      <!-- Grid lines -->
      <line x1="${padL}" y1="${getY(minV)}" x2="${w - padR}" y2="${getY(minV)}" stroke="#262626" stroke-width="1"/>
      <line x1="${padL}" y1="${getY((minV + maxV) / 2)}" x2="${w - padR}" y2="${getY((minV + maxV) / 2)}" stroke="#262626" stroke-width="1" stroke-dasharray="3,3"/>
      <line x1="${padL}" y1="${getY(maxV)}" x2="${w - padR}" y2="${getY(maxV)}" stroke="#262626" stroke-width="1"/>
      
      <!-- Axis labels -->
      <text x="${padL - 6}" y="${getY(maxV) + 4}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">${maxV.toFixed(0)}</text>
      <text x="${padL - 6}" y="${getY((minV + maxV) / 2) + 4}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">${((minV + maxV) / 2).toFixed(0)}</text>
      <text x="${padL - 6}" y="${getY(minV) - 2}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">${minV.toFixed(0)}</text>
      
      <!-- Dates -->
      <text x="${getX(0)}" y="${h - 8}" fill="#666" font-size="9" font-family="monospace">${hlTail[0].date.substring(2, 7)}</text>
      <text x="${getX(Math.floor(n/2))}" y="${h - 8}" fill="#666" font-size="9" text-anchor="middle" font-family="monospace">${hlTail[Math.floor(n/2)].date.substring(2, 7)}</text>
      <text x="${getX(n-1)}" y="${h - 8}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">${hlTail[n-1].date.substring(2, 7)}</text>

      <!-- Series Lines -->
      <path d="${pHL}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="2,2"/>
      <path d="${pCore}" fill="none" stroke="#eaeaea" stroke-width="2"/>
      <path d="${pSuper}" fill="none" stroke="#ff2a2a" stroke-width="2"/>

      <!-- Crosshair markers at latest -->
      <circle cx="${getX(n-1)}" cy="${getY(hlVals[n-1])}" r="3" fill="#888"/>
      <circle cx="${getX(n-1)}" cy="${getY(coreVals[n-1])}" r="3" fill="#eaeaea"/>
      <circle cx="${getX(n-1)}" cy="${getY(superVals[n-1])}" r="3" fill="#ff2a2a"/>
    </svg>
  `;
  container.innerHTML = svg;
}

function renderScatterChart(events, models) {
  const container = document.getElementById("scatter-chart-container");
  if (!container || events.length === 0) return;

  const valid = events.filter(e => e.z_score_composite !== undefined && e.gold_return_pct !== undefined);
  if (valid.length === 0) return;

  const w = container.clientWidth || 700;
  const h = 260;
  const padL = 45;
  const padR = 20;
  const padT = 20;
  const padB = 30;

  const minZ = -3.0;
  const maxZ = 3.0;
  const minR = -4.0;
  const maxR = 3.0;

  function getX(z) {
    return padL + ((z - minZ) / (maxZ - minZ)) * (w - padL - padR);
  }
  function getY(r) {
    return padT + (1 - (r - minR) / (maxR - minR)) * (h - padT - padB);
  }

  const cx0 = getX(0);
  const cy0 = getY(0);

  // Noise zone rect: |Z| < 1.0
  const nzLeft = getX(-1.0);
  const nzRight = getX(1.0);

  let pointsSvg = "";
  valid.forEach(e => {
    const cx = getX(e.z_score_composite);
    const cy = getY(e.gold_return_pct);
    const color = e.gold_return_pct > 0 ? "var(--accent-green)" : "var(--accent-red)";
    pointsSvg += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4" fill="${color}" stroke="#000" stroke-width="1" opacity="0.85"><title>${e.date}: Z=${e.z_score_composite}σ, Gold=${e.gold_return_pct}%</title></circle>`;
  });

  // Regression line
  const slope = models.refined_composite ? models.refined_composite.slope : -0.558;
  const intercept = models.refined_composite ? models.refined_composite.intercept : 0.148;
  const yZneg3 = intercept + slope * (-3.0);
  const yZpos3 = intercept + slope * (+3.0);

  const regLine = `<line x1="${getX(-3.0)}" y1="${getY(yZneg3)}" x2="${getX(3.0)}" y2="${getY(yZpos3)}" stroke="#00e5ff" stroke-width="2" stroke-dasharray="4,4"/>`;

  let svg = `
    <svg class="telemetry-svg" viewBox="0 0 ${w} ${h}">
      <!-- Noise zone shading -->
      <rect x="${nzLeft}" y="${padT}" width="${nzRight - nzLeft}" height="${h - padT - padB}" fill="#181818" opacity="0.4"/>
      <text x="${cx0}" y="${padT + 14}" fill="#444" font-size="9" text-anchor="middle" font-family="monospace">NOISE ZONE (|Z| &lt; 1.0σ)</text>

      <!-- Zero axes -->
      <line x1="${padL}" y1="${cy0}" x2="${w - padR}" y2="${cy0}" stroke="#444" stroke-width="1"/>
      <line x1="${cx0}" y1="${padT}" x2="${cx0}" y2="${h - padB}" stroke="#444" stroke-width="1"/>

      <!-- Labels -->
      <text x="${padL - 6}" y="${cy0 + 3}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">0.0%</text>
      <text x="${cx0}" y="${h - 8}" fill="#666" font-size="9" text-anchor="middle" font-family="monospace">0.0σ</text>
      <text x="${w - padR}" y="${h - 8}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">+3.0σ</text>
      <text x="${padL}" y="${h - 8}" fill="#666" font-size="9" text-anchor="start" font-family="monospace">-3.0σ</text>

      <!-- Regression line -->
      ${regLine}

      <!-- Scatter points -->
      ${pointsSvg}
    </svg>
  `;
  container.innerHTML = svg;
}
