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
  initLiveSyncButton();
  initGsapEntrances();
});

function initGsapEntrances() {
  if (typeof gsap === "undefined") return;
  gsap.from(".top-nav", { opacity: 0, y: -16, duration: 0.6, ease: "power2.out" });
  gsap.from(".ticker-item", { opacity: 0, y: -10, stagger: 0.04, duration: 0.6, delay: 0.15, ease: "power2.out" });
  gsap.from(".grid-box", { opacity: 0, y: 18, stagger: 0.08, duration: 0.7, delay: 0.25, ease: "power2.out" });
}

function animateNumber(el, targetVal, prefix="", suffix="", decimals=2, duration=0.8) {
  if (!el) return;
  if (typeof gsap === "undefined") {
    el.innerText = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
    return;
  }
  const rawCurrent = el.innerText.replace(/[^0-9.-]/g, "");
  const current = parseFloat(rawCurrent) || 0;
  const obj = { val: current };
  gsap.to(obj, {
    val: targetVal,
    duration: duration,
    ease: "power2.out",
    onUpdate: () => {
      el.innerText = `${prefix}${obj.val.toFixed(decimals)}${suffix}`;
    }
  });
}

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

    try { renderTomorrowBanner(tomorrowMeta); } catch(e) { console.error("banner:", e); }
    try { renderModelComparison(modelData); } catch(e) { console.error("models:", e); }
    try { renderHistoricalTable(historicalData); } catch(e) { console.error("table:", e); }
    try { await renderComponentChart(); } catch(e) { console.error("comp:", e); }
    try { renderScatterChart(historicalData, modelData); } catch(e) { console.error("scatter:", e); }
    try { await renderMonteCarloChart(1, 0.0); } catch(e) { console.error("mc:", e); }
    try { await renderPDFChart(); } catch(e) { console.error("pdf:", e); }
    try { initMonteCarloControls(); } catch(e) { console.error("mcControls:", e); }
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
}

function setupSimulationListeners() {
  const btnPreNews = document.getElementById("btn-run-prenewsmemo");
  const btnPostNews = document.getElementById("btn-run-postnewsmemo");

  if (btnPreNews) {
    btnPreNews.addEventListener("click", () => {
      executeDeepSeekMemo("pre_news");
    });
  }

  if (btnPostNews) {
    btnPostNews.addEventListener("click", () => {
      executeDeepSeekMemo("post_news");
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

async function executeDeepSeekMemo(mode, actualMom, actualCore) {
  const terminal = document.getElementById("deepseek-terminal");
  const btnPre = document.getElementById("btn-run-prenewsmemo");
  const btnPost = document.getElementById("btn-run-postnewsmemo");
  
  const activeBtn = mode === "pre_news" ? btnPre : btnPost;
  if (activeBtn) {
    activeBtn.disabled = true;
    activeBtn.innerText = "[ GENERATING MACRO MEMO... ]";
  }
  
  terminal.innerText = `>>> INITIATING DEEPSEEK V3 QUANTITATIVE MACRO ENGINE [MODE: ${mode.toUpperCase()}]...\n>>> CONNECTING TO API (api.deepseek.com)...\n>>> STRIPPING ALL AI FORMATTING SLOP (ZERO ASTERISKS / PURE TELEMETRY)...\n\n`;

  try {
    const res = await fetch("/api/deepseek-briefing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: mode,
        actual_mom: actualMom !== undefined ? actualMom : (tomorrowMeta.forecast_mom || 0.3),
        actual_core_yoy: actualCore !== undefined ? actualCore : (tomorrowMeta.forecast_core_yoy || 2.8)
      })
    });
    const data = await res.json();
    
    if (data.status === "success") {
      terminal.innerText = `=== DEEPSEEK V3 INSTITUTIONAL MACRO MEMO // ${mode.toUpperCase()} ===\n` + data.briefing;
    } else {
      terminal.innerText += "[ERROR]: " + JSON.stringify(data);
    }
  } catch (err) {
    terminal.innerText += "\n[FATAL ERROR]: Failed to reach DeepSeek API: " + err.message;
  } finally {
    if (activeBtn) {
      activeBtn.disabled = false;
      activeBtn.innerText = mode === "pre_news" ? "GENERATE PRE-NEWS MACRO BRIEFING" : "RUN CONFIRMED POST-NEWS MACRO BRIEFING";
    }
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
let currentMCHorizon = 1;
let currentMCZ = 0.0;

function initMonteCarloControls() {
  const btn1d = document.getElementById("mc-btn-1d");
  const btn5d = document.getElementById("mc-btn-5d");
  const btnZ0 = document.getElementById("mc-btn-z0");
  const btnZpos = document.getElementById("mc-btn-zpos");
  const btnZneg = document.getElementById("mc-btn-zneg");

  if (btn1d && btn5d) {
    btn1d.addEventListener("click", () => {
      currentMCHorizon = 1;
      btn1d.classList.add("active");
      btn5d.classList.remove("active");
      renderMonteCarloChart(currentMCHorizon, currentMCZ);
    });
    btn5d.addEventListener("click", () => {
      currentMCHorizon = 5;
      btn5d.classList.add("active");
      btn1d.classList.remove("active");
      renderMonteCarloChart(currentMCHorizon, currentMCZ);
    });
  }

  if (btnZ0 && btnZpos && btnZneg) {
    btnZ0.addEventListener("click", () => {
      currentMCZ = 0.0;
      btnZ0.classList.add("active");
      btnZpos.classList.remove("active");
      btnZneg.classList.remove("active");
      renderMonteCarloChart(currentMCHorizon, currentMCZ);
    });
    btnZpos.addEventListener("click", () => {
      currentMCZ = 1.33;
      btnZpos.classList.add("active");
      btnZ0.classList.remove("active");
      btnZneg.classList.remove("active");
      renderMonteCarloChart(currentMCHorizon, currentMCZ);
    });
    btnZneg.addEventListener("click", () => {
      currentMCZ = -1.33;
      btnZneg.classList.add("active");
      btnZ0.classList.remove("active");
      btnZpos.classList.remove("active");
      renderMonteCarloChart(currentMCHorizon, currentMCZ);
    });
  }
}

let mcDataCache = null;
let componentDataFull = [];
let compSeriesVisibility = { supercore: true, shelter: true, core: true, headline: true };
let compRangeSelection = "all";
let scatterRegimeFilter = "all";

async function renderMonteCarloChart(horizon, z) {
  const container = document.getElementById("mc-chart-container");
  const hudEl = document.getElementById("mc-hover-hud");
  if (!container) return;

  let mc = null;
  try {
    const res = await fetch(`/api/monte-carlo?horizon=${horizon}&z=${z}`);
    mc = await res.json();
    mcDataCache = mc;
  } catch (err) {
    console.warn("Monte Carlo fetch error:", err);
    return;
  }

  // Smooth counter updates via GSAP
  const pPosEl = document.getElementById("mc-p-pos");
  const pNegEl = document.getElementById("mc-p-neg");
  const medEl = document.getElementById("mc-median-price");
  const varEl = document.getElementById("mc-var-95");

  animateNumber(pPosEl, mc.metrics.p_positive_return_pct, "", "%", 2);
  animateNumber(pNegEl, mc.metrics.p_negative_return_pct, "", "%", 2);
  animateNumber(medEl, mc.metrics.median_terminal_price, "$", "", 2);
  animateNumber(varEl, mc.metrics.var_95_pct, (mc.metrics.var_95_pct > 0 ? "+" : ""), "%", 2);

  const w = container.clientWidth || 700;
  const h = 320;
  const padL = 52;
  const padR = 65;
  const padT = 25;
  const padB = 32;

  const steps = mc.steps || 24;
  const p5 = mc.percentiles.p5;
  const p25 = mc.percentiles.p25;
  const p50 = mc.percentiles.p50;
  const p75 = mc.percentiles.p75;
  const p95 = mc.percentiles.p95;

  const allVals = [...p5, ...p95];
  const minP = Math.min(...allVals) * 0.995;
  const maxP = Math.max(...allVals) * 1.005;

  function getX(i) {
    return padL + (i / steps) * (w - padL - padR);
  }
  function getY(p) {
    return padT + (1 - (p - minP) / (maxP - minP)) * (h - padT - padB);
  }

  // 90% and 50% percentile envelopes
  let path90 = `M ${getX(0)} ${getY(p95[0])} ` + p95.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
    p5.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

  let path50 = `M ${getX(0)} ${getY(p75[0])} ` + p75.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
    p25.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

  let pathMedian = p50.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");

  // Sample path traces
  let samplePathsSvg = "";
  mc.sample_traces.forEach((tr, idx) => {
    const d = tr.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");
    const op = idx < 3 ? "0.35" : (idx < 6 ? "0.22" : "0.12");
    samplePathsSvg += `<path d="${d}" fill="none" stroke="#999" stroke-width="1" opacity="${op}"/>`;
  });

  // T0 Anchor Baseline Indicator
  const baseY = getY(mc.current_price);
  const baseLine = `
    <line x1="${padL}" y1="${baseY}" x2="${w - padR}" y2="${baseY}" stroke="#ffaa00" stroke-width="1.2" stroke-dasharray="3,3"/>
    <text x="${padL + 6}" y="${baseY - 4}" fill="#ffaa00" font-size="8" font-family="monospace">T0 ANCHOR: $${mc.current_price.toFixed(1)}</text>
  `;

  // Upper / Lower Band Indicators at Terminal
  const upperY = getY(p75[steps]);
  const lowerY = getY(p25[steps]);
  const medianY = getY(p50[steps]);
  const indicatorLines = `
    <text x="${getX(steps) + 8}" y="${upperY + 3}" fill="#00e5ff" font-size="8" text-anchor="start" font-family="monospace">+1σ: $${p75[steps].toFixed(0)}</text>
    <text x="${getX(steps) + 8}" y="${medianY + 3}" fill="#00e5ff" font-size="8.5" font-weight="900" text-anchor="start" font-family="monospace">$${p50[steps].toFixed(0)}</text>
    <text x="${getX(steps) + 8}" y="${lowerY + 3}" fill="#ff2a2a" font-size="8" text-anchor="start" font-family="monospace">-1σ: $${p25[steps].toFixed(0)}</text>
  `;

  // Grid lines
  let gridSvg = "";
  const stepStride = steps > 10 ? 4 : 1;
  for (let step = 0; step <= steps; step += stepStride) {
    const x = getX(step);
    gridSvg += `
      <line x1="${x}" y1="${padT}" x2="${x}" y2="${h - padB}" stroke="#141414" stroke-width="1"/>
      <text x="${x}" y="${h - 8}" fill="#666" font-size="8" text-anchor="middle" font-family="monospace">${mc.timesteps[step] || ""}</text>
    `;
  }

  const priceTicks = [minP, (minP + maxP) / 2, maxP];
  priceTicks.forEach(pt => {
    const y = getY(pt);
    gridSvg += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#141414" stroke-width="1"/>
      <text x="${padL - 6}" y="${y + 3}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">$${pt.toFixed(0)}</text>
    `;
  });

  let svg = `
    <svg class="telemetry-svg" id="mc-svg" viewBox="0 0 ${w} ${h}">
      ${gridSvg}
      ${baseLine}
      <!-- 90% Confidence Envelope -->
      <path d="${path90}" fill="#161616" opacity="0.6"/>
      <!-- 50% Interquartile Core Band -->
      <path d="${path50}" fill="#222222" opacity="0.85"/>
      <!-- Stochastic Path Traces -->
      ${samplePathsSvg}
      <!-- Median Trajectory -->
      <path d="${pathMedian}" fill="none" stroke="#00e5ff" stroke-width="2.5"/>
      <circle cx="${getX(steps)}" cy="${getY(p50[steps])}" r="3.5" fill="#00e5ff"/>
      <text x="${getX(steps) - 6}" y="${getY(p50[steps]) - 6}" fill="#00e5ff" font-size="9" font-weight="900" text-anchor="end" font-family="monospace">MEDIAN: $${p50[steps].toFixed(1)}</text>
      ${indicatorLines}
      <!-- Interactive Hover Crosshair -->
      <line id="mc-crosshair" x1="0" y1="0" x2="0" y2="0" stroke="#00e5ff" stroke-width="1" stroke-dasharray="2,2" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  if (typeof gsap !== "undefined") {
    gsap.fromTo("#mc-svg path, #mc-svg line", { opacity: 0.2 }, { opacity: 1, duration: 0.5, ease: "power2.out" });
  }

  // Interactive Hover Inspection
  const svgEl = document.getElementById("mc-svg");
  const crosshair = document.getElementById("mc-crosshair");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = ev.clientX - rect.left;
      const clampedX = Math.max(padL, Math.min(w - padR, (mouseX / rect.width) * w));
      const stepIdx = Math.round(((clampedX - padL) / (w - padL - padR)) * steps);
      const safeIdx = Math.max(0, Math.min(steps, stepIdx));

      const medP = p50[safeIdx];
      const upP = p95[safeIdx];
      const lowP = p5[safeIdx];
      const retPct = ((medP - mc.current_price) / mc.current_price * 100).toFixed(2);

      hudEl.innerText = `${mc.timesteps[safeIdx]} | Median: $${medP.toFixed(1)} (+2σ: $${upP.toFixed(1)} / -2σ: $${lowP.toFixed(1)}) | Ret: ${retPct > 0 ? "+" : ""}${retPct}%`;

      if (crosshair) {
        crosshair.style.display = "block";
        crosshair.setAttribute("x1", getX(safeIdx));
        crosshair.setAttribute("x2", getX(safeIdx));
        crosshair.setAttribute("y1", padT);
        crosshair.setAttribute("y2", h - padB);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER TIMESTEP FOR ENVELOPE";
      if (crosshair) crosshair.style.display = "none";
    });
  }
}

async function renderPDFChart() {
  const container = document.getElementById("pdf-chart-container");
  const hudEl = document.getElementById("pdf-hover-hud");
  if (!container) return;

  let pdfData = null;
  try {
    const res = await fetch("/api/pdf-distribution");
    pdfData = await res.json();
  } catch (err) {
    console.warn("PDF fetch error:", err);
    return;
  }

  const pts = pdfData.points;
  const n = pts.length;
  if (n < 2) return;

  const w = container.clientWidth || 700;
  const h = 320;
  const padL = 45;
  const padR = 25;
  const padT = 30;
  const padB = 32;

  const minX = -0.35;
  const maxX = +0.35;
  const maxY = 5.2;

  function getX(x) {
    return padL + ((x - minX) / (maxX - minX)) * (w - padL - padR);
  }
  function getY(y) {
    return padT + (1 - (y / maxY)) * (h - padT - padB);
  }

  const pathPrior = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.prior_pdf).toFixed(1)}`).join(" ");
  const pathPost = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.posterior_pdf).toFixed(1)}`).join(" ");

  const xMiss = getX(-0.10);
  const xBeat = getX(+0.10);
  const yZero = getY(0);
  const x0 = getX(0);
  const xNowcast = getX(-0.02);
  const xPeak = getX(0.08);

  // Indicator milestone lines & labels
  const indicatorsSvg = `
    <!-- T0 Consensus Marker -->
    <line x1="${x0}" y1="${padT + 16}" x2="${x0}" y2="${yZero}" stroke="#888" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="${x0}" y="${padT + 12}" fill="#888" font-size="7.5" text-anchor="middle" font-family="monospace">T0: 0.00pp</text>

    <!-- Nowcast Marker -->
    <line x1="${xNowcast}" y1="${padT + 28}" x2="${xNowcast}" y2="${yZero}" stroke="#00e5ff" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="${xNowcast - 4}" y="${padT + 26}" fill="#00e5ff" font-size="7.5" text-anchor="end" font-family="monospace">NOWCAST -0.02pp</text>

    <!-- Energy Posterior Peak Marker -->
    <line x1="${xPeak}" y1="${padT + 16}" x2="${xPeak}" y2="${yZero}" stroke="#ff2a2a" stroke-width="1.5"/>
    <text x="${xPeak + 4}" y="${padT + 12}" fill="#ff2a2a" font-size="8" font-weight="900" text-anchor="start" font-family="monospace">PEAK +0.08pp</text>
  `;

  let svg = `
    <svg class="telemetry-svg" id="pdf-svg" viewBox="0 0 ${w} ${h}">
      <!-- Shaded Zones -->
      <rect x="${padL}" y="${padT}" width="${xMiss - padL}" height="${h - padT - padB}" fill="rgba(74, 246, 38, 0.07)"/>
      <rect x="${xMiss}" y="${padT}" width="${xBeat - xMiss}" height="${h - padT - padB}" fill="#141414" opacity="0.6"/>
      <rect x="${xBeat}" y="${padT}" width="${w - padR - xBeat}" height="${h - padT - padB}" fill="rgba(255, 42, 42, 0.08)"/>

      <line x1="${padL}" y1="${yZero}" x2="${w - padR}" y2="${yZero}" stroke="#333" stroke-width="1"/>
      <line x1="${xMiss}" y1="${padT}" x2="${xMiss}" y2="${h - padB}" stroke="#444" stroke-width="1" stroke-dasharray="1,2"/>
      <line x1="${xBeat}" y1="${padT}" x2="${xBeat}" y2="${h - padB}" stroke="#444" stroke-width="1" stroke-dasharray="1,2"/>

      ${indicatorsSvg}

      <text x="${padL}" y="${h - 8}" fill="#666" font-size="8" text-anchor="start" font-family="monospace">-0.35pp</text>
      <text x="${xMiss}" y="${h - 8}" fill="var(--accent-green)" font-size="8" text-anchor="middle" font-family="monospace">-0.10pp (MISS)</text>
      <text x="${x0}" y="${h - 8}" fill="#888" font-size="8" text-anchor="middle" font-family="monospace">0.00pp</text>
      <text x="${xBeat}" y="${h - 8}" fill="var(--accent-red)" font-size="8" text-anchor="middle" font-family="monospace">+0.10pp (BEAT)</text>
      <text x="${w - padR}" y="${h - 8}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">+0.35pp</text>

      <path d="${pathPrior}" fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="3,3"/>
      <path d="${pathPost}" fill="none" stroke="#ff2a2a" stroke-width="2.5"/>
      <circle cx="${xPeak}" cy="${getY(4.43)}" r="4" fill="#ff2a2a"/>

      <!-- Interactive Scanning Needle -->
      <line id="pdf-needle" x1="0" y1="0" x2="0" y2="0" stroke="#00e5ff" stroke-width="1.5" stroke-dasharray="2,2" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  if (typeof gsap !== "undefined") {
    gsap.fromTo("#pdf-svg path", { opacity: 0.2, scaleY: 0.3, transformOrigin: "bottom" }, { opacity: 1, scaleY: 1, stagger: 0.1, duration: 0.7, ease: "power2.out" });
  }

  const svgEl = document.getElementById("pdf-svg");
  const needle = document.getElementById("pdf-needle");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = ev.clientX - rect.left;
      const clampedX = Math.max(padL, Math.min(w - padR, (mouseX / rect.width) * w));
      const xVal = minX + ((clampedX - padL) / (w - padL - padR)) * (maxX - minX);
      
      let closest = pts[0];
      let minDiff = 999;
      pts.forEach(p => {
        const diff = Math.abs(p.x - xVal);
        if (diff < minDiff) { minDiff = diff; closest = p; }
      });

      const zScore = (closest.x / 0.075).toFixed(2);
      hudEl.innerText = `x: ${(closest.x > 0 ? "+" : "") + closest.x.toFixed(2)}pp (Z = ${zScore}σ) | Prior: ${closest.prior_pdf.toFixed(2)} | Post: ${closest.posterior_pdf.toFixed(2)} [${closest.zone}]`;

      if (needle) {
        needle.style.display = "block";
        needle.setAttribute("x1", clampedX);
        needle.setAttribute("x2", clampedX);
        needle.setAttribute("y1", padT);
        needle.setAttribute("y2", yZero);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER BELL CURVE";
      if (needle) needle.style.display = "none";
    });
  }
}

async function renderComponentChart() {
  const container = document.getElementById("component-chart-container");
  const hudEl = document.getElementById("comp-hover-hud");
  if (!container) return;

  if (componentDataFull.length === 0) {
    try {
      const res = await fetch("/api/yoy-timeline");
      componentDataFull = await res.json();
    } catch (e) {
      console.warn("Failed to fetch yoy-timeline:", e);
      return;
    }
  }

  // Filter by range
  let data = [...componentDataFull];
  if (compRangeSelection === "24") data = data.slice(-24);
  else if (compRangeSelection === "12") data = data.slice(-12);

  const n = data.length;
  if (n < 2) return;

  const w = container.clientWidth || 700;
  const h = 320;
  const padL = 45;
  const padR = 25;
  const padT = 25;
  const padB = 32;

  const minV = 1.5;
  const maxV = 8.5;

  function getX(i) {
    return padL + (i / (n - 1)) * (w - padL - padR);
  }
  function getY(v) {
    return padT + (1 - (v - minV) / (maxV - minV)) * (h - padT - padB);
  }

  function makePath(key) {
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getY(d[key]).toFixed(1)}`).join(' ');
  }

  const pHL = makePath("headline_yoy");
  const pCore = makePath("core_yoy");
  const pSuper = makePath("supercore_yoy");
  const pShelter = makePath("shelter_yoy");

  // Grid lines
  let gridLines = "";
  for (let val = 2.0; val <= 8.0; val += 1.0) {
    const y = getY(val);
    gridLines += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#161616" stroke-width="1"/>
      <text x="${padL - 6}" y="${y + 3}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">${val.toFixed(0)}%</text>
    `;
  }

  // Federal Reserve 2.0% Inflation Target Line Indicator
  const yTarget = getY(2.0);
  const targetLineSvg = `
    <line x1="${padL}" y1="${yTarget}" x2="${w - padR}" y2="${yTarget}" stroke="#4af626" stroke-width="1" stroke-dasharray="3,3" opacity="0.8"/>
    <text x="${padL + 6}" y="${yTarget - 4}" fill="#4af626" font-size="7.5" font-family="monospace">FED 2.0% TARGET</text>
  `;

  let dateTicks = "";
  const cleanIndices = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1];
  cleanIndices.forEach((idx, i) => {
    const x = getX(idx);
    const anchor = i === 0 ? "start" : (i === cleanIndices.length - 1 ? "end" : "middle");
    dateTicks += `
      <line x1="${x}" y1="${padT}" x2="${x}" y2="${h - padB}" stroke="#141414" stroke-width="1"/>
      <text x="${x}" y="${h - 8}" fill="#666" font-size="8" text-anchor="${anchor}" font-family="monospace">${data[idx].date}</text>
    `;
  });

  const lastD = data[n - 1];
  let svg = `
    <svg class="telemetry-svg" id="comp-svg" viewBox="0 0 ${w} ${h}">
      ${gridLines}
      ${targetLineSvg}
      ${dateTicks}
      <!-- Series paths with visibility toggle -->
      <path id="path-headline" d="${pHL}" fill="none" stroke="#888888" stroke-width="1.5" stroke-dasharray="3,3" style="display:${compSeriesVisibility.headline ? 'block':'none'};"/>
      <path id="path-core" d="${pCore}" fill="none" stroke="#eaeaea" stroke-width="2" style="display:${compSeriesVisibility.core ? 'block':'none'};"/>
      <path id="path-shelter" d="${pShelter}" fill="none" stroke="#00e5ff" stroke-width="2" style="display:${compSeriesVisibility.shelter ? 'block':'none'};"/>
      <path id="path-supercore" d="${pSuper}" fill="none" stroke="#ff2a2a" stroke-width="2.5" style="display:${compSeriesVisibility.supercore ? 'block':'none'};"/>

      <!-- End Callouts -->
      ${compSeriesVisibility.supercore ? `<circle cx="${getX(n-1)}" cy="${getY(lastD.supercore_yoy)}" r="3.5" fill="#ff2a2a"/>` : ''}
      ${compSeriesVisibility.shelter ? `<circle cx="${getX(n-1)}" cy="${getY(lastD.shelter_yoy)}" r="3" fill="#00e5ff"/>` : ''}
      ${compSeriesVisibility.core ? `<circle cx="${getX(n-1)}" cy="${getY(lastD.core_yoy)}" r="3" fill="#eaeaea"/>` : ''}

      <line id="comp-crosshair" x1="0" y1="0" x2="0" y2="0" stroke="#00e5ff" stroke-width="1" stroke-dasharray="2,2" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  if (typeof gsap !== "undefined") {
    gsap.fromTo("#comp-svg path", { opacity: 0, strokeDashoffset: 600, strokeDasharray: 600 }, { opacity: 1, strokeDashoffset: 0, stagger: 0.1, duration: 1.0, ease: "power2.out" });
  }

  initComponentControls();

  const svgEl = document.getElementById("comp-svg");
  const crosshair = document.getElementById("comp-crosshair");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = ev.clientX - rect.left;
      const clampedX = Math.max(padL, Math.min(w - padR, (mouseX / rect.width) * w));
      const idx = Math.round(((clampedX - padL) / (w - padL - padR)) * (n - 1));
      const d = data[idx];
      if (d) {
        hudEl.innerText = `${d.date}: Supercore ${d.supercore_yoy}% | Shelter ${d.shelter_yoy}% | Core ${d.core_yoy}% | Headline ${d.headline_yoy}%`;
        if (crosshair) {
          crosshair.style.display = "block";
          crosshair.setAttribute("x1", getX(idx));
          crosshair.setAttribute("x2", getX(idx));
          crosshair.setAttribute("y1", padT);
          crosshair.setAttribute("y2", h - padB);
        }
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER OVER TIMELINE";
      if (crosshair) crosshair.style.display = "none";
    });
  }
}

function initComponentControls() {
  // Series toggle buttons
  const pillSuper = document.getElementById("pill-supercore");
  const pillShelter = document.getElementById("pill-shelter");
  const pillCore = document.getElementById("pill-core");
  const pillHL = document.getElementById("pill-headline");

  if (pillSuper && !pillSuper.__wired) {
    pillSuper.__wired = true;
    pillSuper.addEventListener("click", () => {
      compSeriesVisibility.supercore = !compSeriesVisibility.supercore;
      pillSuper.classList.toggle("active", compSeriesVisibility.supercore);
      pillSuper.style.opacity = compSeriesVisibility.supercore ? "1" : "0.3";
      renderComponentChart();
    });
  }
  if (pillShelter && !pillShelter.__wired) {
    pillShelter.__wired = true;
    pillShelter.addEventListener("click", () => {
      compSeriesVisibility.shelter = !compSeriesVisibility.shelter;
      pillShelter.classList.toggle("active", compSeriesVisibility.shelter);
      pillShelter.style.opacity = compSeriesVisibility.shelter ? "1" : "0.3";
      renderComponentChart();
    });
  }
  if (pillCore && !pillCore.__wired) {
    pillCore.__wired = true;
    pillCore.addEventListener("click", () => {
      compSeriesVisibility.core = !compSeriesVisibility.core;
      pillCore.classList.toggle("active", compSeriesVisibility.core);
      pillCore.style.opacity = compSeriesVisibility.core ? "1" : "0.3";
      renderComponentChart();
    });
  }
  if (pillHL && !pillHL.__wired) {
    pillHL.__wired = true;
    pillHL.addEventListener("click", () => {
      compSeriesVisibility.headline = !compSeriesVisibility.headline;
      pillHL.classList.toggle("active", compSeriesVisibility.headline);
      pillHL.style.opacity = compSeriesVisibility.headline ? "1" : "0.3";
      renderComponentChart();
    });
  }

  // Range selection buttons
  const rAll = document.getElementById("range-all");
  const r24 = document.getElementById("range-24m");
  const r12 = document.getElementById("range-12m");

  if (rAll && !rAll.__wired) {
    rAll.__wired = true;
    rAll.addEventListener("click", () => {
      compRangeSelection = "all";
      rAll.classList.add("active");
      if (r24) r24.classList.remove("active");
      if (r12) r12.classList.remove("active");
      renderComponentChart();
    });
  }
  if (r24 && !r24.__wired) {
    r24.__wired = true;
    r24.addEventListener("click", () => {
      compRangeSelection = "24";
      r24.classList.add("active");
      if (rAll) rAll.classList.remove("active");
      if (r12) r12.classList.remove("active");
      renderComponentChart();
    });
  }
  if (r12 && !r12.__wired) {
    r12.__wired = true;
    r12.addEventListener("click", () => {
      compRangeSelection = "12";
      r12.classList.add("active");
      if (rAll) rAll.classList.remove("active");
      if (r24) r24.classList.remove("active");
      renderComponentChart();
    });
  }
}

function renderScatterChart(events, models) {
  const container = document.getElementById("scatter-chart-container");
  const hudEl = document.getElementById("scatter-hover-hud");
  if (!container || events.length === 0) return;

  let valid = events.filter(e => e.z_score_composite !== undefined && e.gold_return_pct !== undefined);
  if (scatterRegimeFilter === "shock") valid = valid.filter(e => e.regime !== "NOISE_ZONE");
  else if (scatterRegimeFilter === "noise") valid = valid.filter(e => e.regime === "NOISE_ZONE");

  if (valid.length === 0) return;

  const w = container.clientWidth || 700;
  const h = 320;

  const scatterW = Math.floor(w * 0.54);
  const vrW = w - scatterW;
  const padL = 40;
  const padR = 20;
  const padT = 25;
  const padB = 32;

  const minZ = -3.2;
  const maxZ = 3.2;
  const minR = -4.0;
  const maxR = 3.0;

  function getX(z) {
    return padL + ((z - minZ) / (maxZ - minZ)) * (scatterW - padL - padR);
  }
  function getY(r) {
    return padT + (1 - (r - minR) / (maxR - minR)) * (h - padT - padB);
  }

  const cx0 = getX(0);
  const cy0 = getY(0);
  const nzLeft = getX(-1.0);
  const nzRight = getX(1.0);

  let pointsSvg = "";
  valid.forEach((e, idx) => {
    const cx = getX(e.z_score_composite);
    const cy = getY(e.gold_return_pct);
    const color = e.gold_return_pct > 0 ? "var(--accent-green)" : "var(--accent-red)";
    
    let labelSvg = "";
    if (e.gold_return_pct <= -3.0 || e.gold_return_pct >= 1.9 || Math.abs(e.z_score_composite) >= 2.5) {
      labelSvg = `<text x="${cx}" y="${cy - 7}" fill="#aaa" font-size="7.5" font-weight="700" text-anchor="middle" font-family="monospace">${e.date.substring(2,7)}</text>`;
    }
    
    pointsSvg += `
      <circle class="scatter-dot" data-idx="${idx}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" fill="${color}" stroke="#000" stroke-width="1.2" opacity="0.88" style="cursor:pointer;"></circle>
      ${labelSvg}
    `;
  });

  const slope = models.refined_composite ? models.refined_composite.slope : -0.558;
  const intercept = models.refined_composite ? models.refined_composite.intercept : 0.148;
  const yZneg3 = intercept + slope * (-3.2);
  const yZpos3 = intercept + slope * (+3.2);
  const regLine = `<line x1="${getX(-3.2)}" y1="${getY(yZneg3)}" x2="${getX(3.2)}" y2="${getY(yZpos3)}" stroke="#00e5ff" stroke-width="2" stroke-dasharray="3,3"/>`;

  // Volatility Ratio Bars on Right side with ample width to prevent clipping
  const vrData = [
    { label: "VR(5m)", ratio: 5.42, color: "#ff2a2a" },
    { label: "VR(15m)", ratio: 3.65, color: "#00e5ff" },
    { label: "VR(30m)", ratio: 2.58, color: "#eaeaea" },
    { label: "VR(60m)", ratio: 1.84, color: "#888888" }
  ];
  const maxVR = 6.0;
  const vrStartX = scatterW + 18;
  const vrWidth = Math.max(50, w - vrStartX - 70); // 70px right margin guarantees '5.42x' is NEVER cut off
  let vrSvg = `
    <line x1="${scatterW}" y1="${padT - 10}" x2="${scatterW}" y2="${h - 10}" stroke="#222" stroke-width="1"/>
    <text x="${vrStartX}" y="${padT + 4}" fill="#888" font-size="8" font-weight="700" font-family="monospace">VOLATILITY RATIO VR(w)</text>
  `;

  vrData.forEach((item, idx) => {
    const barY = padT + 28 + idx * 52;
    const barLen = (item.ratio / maxVR) * vrWidth;
    vrSvg += `
      <text x="${vrStartX}" y="${barY + 11}" fill="#eaeaea" font-size="8.5" font-family="monospace">${item.label}</text>
      <rect class="vr-bar-rect" x="${vrStartX + 52}" y="${barY}" width="${barLen.toFixed(1)}" height="14" fill="${item.color}" opacity="0.9"/>
      <text x="${vrStartX + 58 + barLen}" y="${barY + 11}" fill="${item.color}" font-size="9" font-weight="900" font-family="monospace">${item.ratio.toFixed(2)}x</text>
    `;
  });

  let svg = `
    <svg class="telemetry-svg" id="scatter-svg" viewBox="0 0 ${w} ${h}">
      <rect x="${nzLeft}" y="${padT}" width="${nzRight - nzLeft}" height="${h - padT - padB}" fill="#161616" opacity="0.5"/>
      <text x="${cx0}" y="${padT + 12}" fill="#555" font-size="8" text-anchor="middle" font-family="monospace">NOISE (|Z| &lt; 1.0σ)</text>
      <line x1="${padL}" y1="${cy0}" x2="${scatterW - padR}" y2="${cy0}" stroke="#262626" stroke-width="1"/>
      <line x1="${cx0}" y1="${padT}" x2="${cx0}" y2="${h - padB}" stroke="#262626" stroke-width="1"/>
      <text x="${padL - 6}" y="${cy0 + 3}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">0.0%</text>
      <text x="${cx0}" y="${h - 8}" fill="#666" font-size="8" text-anchor="middle" font-family="monospace">0.0σ</text>
      <text x="${scatterW - padR}" y="${h - 8}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">+3.0σ</text>
      <text x="${padL}" y="${h - 8}" fill="#666" font-size="8" text-anchor="start" font-family="monospace">-3.0σ</text>
      ${regLine}
      ${pointsSvg}
      ${vrSvg}
    </svg>
  `;
  container.innerHTML = svg;

  if (typeof gsap !== "undefined") {
    gsap.fromTo(".scatter-dot", { scale: 0, transformOrigin: "center" }, { scale: 1, stagger: 0.012, duration: 0.5, ease: "back.out(2)" });
    gsap.fromTo(".vr-bar-rect", { scaleX: 0, transformOrigin: "left" }, { scaleX: 1, stagger: 0.08, duration: 0.7, ease: "power2.out" });
  }

  initScatterControls(events, models);

  if (hudEl) {
    const dots = container.querySelectorAll(".scatter-dot");
    dots.forEach(d => {
      d.addEventListener("mouseenter", () => {
        const idx = parseInt(d.getAttribute("data-idx"));
        const e = valid[idx];
        if (e) {
          hudEl.innerText = `${e.date} (${e.ref_month}): Z=${e.z_score_composite}σ | Gold: ${(e.gold_return_pct > 0 ? "+" : "") + e.gold_return_pct}% | DXY: ${(e.dxy_return_pct > 0 ? "+" : "") + e.dxy_return_pct}% [${e.regime}]`;
          if (typeof gsap !== "undefined") {
            gsap.to(d, { attr: { r: 7 }, duration: 0.2, ease: "power2.out" });
          }
        }
      });
      d.addEventListener("mouseleave", () => {
        hudEl.innerText = "HOVER DATA POINT";
        if (typeof gsap !== "undefined") {
          gsap.to(d, { attr: { r: 4.5 }, duration: 0.2, ease: "power2.out" });
        }
      });
    });
  }
}

function initScatterControls(events, models) {
  const btnAll = document.getElementById("sc-filt-all");
  const btnShock = document.getElementById("sc-filt-shock");
  const btnNoise = document.getElementById("sc-filt-noise");

  if (btnAll && !btnAll.__wired) {
    btnAll.__wired = true;
    btnAll.addEventListener("click", () => {
      scatterRegimeFilter = "all";
      btnAll.classList.add("active");
      if (btnShock) btnShock.classList.remove("active");
      if (btnNoise) btnNoise.classList.remove("active");
      renderScatterChart(events, models);
    });
  }
  if (btnShock && !btnShock.__wired) {
    btnShock.__wired = true;
    btnShock.addEventListener("click", () => {
      scatterRegimeFilter = "shock";
      btnShock.classList.add("active");
      if (btnAll) btnAll.classList.remove("active");
      if (btnNoise) btnNoise.classList.remove("active");
      renderScatterChart(events, models);
    });
  }
  if (btnNoise && !btnNoise.__wired) {
    btnNoise.__wired = true;
    btnNoise.addEventListener("click", () => {
      scatterRegimeFilter = "noise";
      btnNoise.classList.add("active");
      if (btnAll) btnAll.classList.remove("active");
      if (btnShock) btnShock.classList.remove("active");
      renderScatterChart(events, models);
    });
  }
}

// ==================== LIVE ON-DEMAND SYNC ENGINE ====================
let wave1Timer = null;
let lastReleasedTimestamp = null;

function initLiveSyncButton() {
  const btnSync = document.getElementById("btn-sync-live");
  if (btnSync) {
    btnSync.addEventListener("click", executeManualSync);
  }
}

async function executeManualSync() {
  const btnSync = document.getElementById("btn-sync-live");
  const statusSub = document.getElementById("sync-status-sub");
  const badge = document.getElementById("sync-status-badge");
  const tsEl = document.getElementById("sync-timestamp");

  if (btnSync) {
    btnSync.disabled = true;
    btnSync.innerText = "[ FETCHING LIVE DATA... ]";
  }
  if (statusSub) {
    statusSub.innerText = "Querying BLS/ForexFactory endpoints & live Yahoo Finance price books...";
  }

  try {
    const res = await fetch("/api/sync-live", { method: "POST" });
    const data = await res.json();

    if (tsEl && data.timestamp) {
      tsEl.innerText = data.timestamp;
    }

    // Update ticker prices if returned
    if (data.prices) {
      updateTickerPrices(data.prices);
    }

    if (data.status === "RELEASED") {
      if (badge) {
        badge.innerText = "● OFFICIAL RELEASE CAPTURED";
        badge.style.color = "var(--accent-red)";
      }
      if (statusSub) {
        statusSub.innerText = `OFFICIAL DATA CAPTURED AT ${data.timestamp}. Macro transmission model computed.`;
      }
      handleOfficialRelease(data);
    } else {
      if (badge) {
        badge.innerText = "● VERIFIED SYNC COMPLETED";
        badge.style.color = "var(--accent-green)";
      }
      if (statusSub) {
        statusSub.innerText = `[AWAITING RELEASE] Real-time prices updated at ${data.timestamp}. BLS release scheduled for Nov 10 @ 08:30 AM EST.`;
      }
    }
  } catch (err) {
    console.error("Sync failed:", err);
    if (statusSub) statusSub.innerText = "[SYNC ERROR] Failed to reach live feeds: " + err.message;
  } finally {
    if (btnSync) {
      btnSync.disabled = false;
      btnSync.innerText = "SYNC CURRENT LIVE DATA";
    }
  }
}

function updateTickerPrices(prices) {
  // Ticker bar items
  const items = document.querySelectorAll(".ticker-item");
  items.forEach(it => {
    const label = it.querySelector(".ticker-label")?.innerText || "";
    const valEl = it.querySelector(".ticker-val");
    if (!valEl) return;
    if (label.includes("XAUUSD") && prices["GC=F"]) {
      animateNumber(valEl, prices["GC=F"], "$", "", 2);
    } else if (label.includes("DOLLAR INDEX") && prices["DX-Y.NYB"]) {
      animateNumber(valEl, prices["DX-Y.NYB"], "", "", 2);
    } else if (label.includes("10-YEAR YIELD") && prices["^TNX"]) {
      animateNumber(valEl, prices["^TNX"], "", "%", 2);
    }
  });
}

function handleOfficialRelease(data) {
  const isNew = (lastReleasedTimestamp !== data.timestamp);
  lastReleasedTimestamp = data.timestamp;

  const alertBox = document.getElementById("live-alert-box");
  const alertHeadline = document.getElementById("live-alert-headline");
  const alertDetail = document.getElementById("live-alert-detail");

  if (alertBox) alertBox.style.display = "block";
  if (alertHeadline) {
    alertHeadline.innerText = `[ OFFICIAL CPI RELEASE CAPTURED & VERIFIED ]`;
  }
  if (alertDetail) {
    const s = data.simulation ? data.simulation.surprises : {};
    alertDetail.innerText = `ACTUAL HEADLINE: ${data.actual_mom.toFixed(2)}% | ACTUAL CORE YoY: ${data.actual_core_yoy.toFixed(2)}% | COMPOSITE Z: ${s.z_score_composite ? (s.z_score_composite > 0 ? "+" : "") + s.z_score_composite.toFixed(2) + "σ" : "-"}`;
  }

  // Toggle Option B from pending to active
  const pendingEl = document.getElementById("option-b-pending");
  const activeEl = document.getElementById("option-b-active");
  if (pendingEl) pendingEl.style.display = "none";
  if (activeEl) activeEl.style.display = "block";
  if (data.simulation) {
    renderSimulationOutputs(data.simulation);
  }

  // 60-Second Wave 1 Countdown Firewall
  if (!wave1Timer) {
    let secondsLeft = 60;
    const timerEl = document.getElementById("wave1-timer");
    if (timerEl) timerEl.innerText = `${secondsLeft}s`;

    wave1Timer = setInterval(() => {
      secondsLeft--;
      if (timerEl) {
        if (secondsLeft > 0) {
          timerEl.innerText = `${secondsLeft}s`;
        } else {
          timerEl.innerText = "WAVE 2 READY";
          timerEl.style.color = "var(--accent-green)";
          clearInterval(wave1Timer);
          wave1Timer = null;
        }
      }
    }, 1000);
  }

  if (isNew) {
    executeDeepSeekMemo("post_news", data.actual_mom, data.actual_core_yoy);
  }
}
