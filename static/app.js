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
    // Pre-news initial model state is pre-rendered in HTML
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

async function renderComponentChart() {
  const container = document.getElementById("component-chart-container");
  if (!container) return;

  let data = [];
  try {
    const res = await fetch("/api/yoy-timeline");
    data = await res.json();
  } catch (e) {
    console.warn("Failed to fetch yoy-timeline:", e);
    return;
  }

  const n = data.length;
  if (n < 2) return;

  const w = container.clientWidth || 700;
  const h = 260;
  const padL = 45;
  const padR = 25;
  const padT = 25;
  const padB = 30;

  const minV = 2.0;
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

  // Grid lines every 1%
  let gridLines = "";
  for (let val = 2.0; val <= 8.0; val += 1.0) {
    const y = getY(val);
    gridLines += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#1f1f1f" stroke-width="1"/>
      <text x="${padL - 6}" y="${y + 3}" fill="#666" font-size="9" text-anchor="end" font-family="monospace">${val.toFixed(0)}%</text>
    `;
  }

  // Date ticks on x-axis (every 6 months)
  let dateTicks = "";
  for (let i = 0; i < n; i += 6) {
    const x = getX(i);
    dateTicks += `
      <line x1="${x}" y1="${padT}" x2="${x}" y2="${h - padB}" stroke="#161616" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${x}" y="${h - 8}" fill="#666" font-size="8" text-anchor="middle" font-family="monospace">${data[i].date}</text>
    `;
  }
  // Final date tick
  dateTicks += `<text x="${getX(n-1)}" y="${h - 8}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">${data[n-1].date}</text>`;

  const lastD = data[n - 1];
  let svg = `
    <svg class="telemetry-svg" viewBox="0 0 ${w} ${h}">
      ${gridLines}
      ${dateTicks}

      <!-- Series Paths -->
      <path d="${pHL}" fill="none" stroke="#888888" stroke-width="1.5" stroke-dasharray="3,3"/>
      <path d="${pCore}" fill="none" stroke="#eaeaea" stroke-width="2"/>
      <path d="${pShelter}" fill="none" stroke="#00e5ff" stroke-width="2"/>
      <path d="${pSuper}" fill="none" stroke="#ff2a2a" stroke-width="2.5"/>

      <!-- End Callouts -->
      <circle cx="${getX(n-1)}" cy="${getY(lastD.supercore_yoy)}" r="3.5" fill="#ff2a2a"/>
      <text x="${getX(n-1) - 4}" y="${getY(lastD.supercore_yoy) - 6}" fill="#ff2a2a" font-size="9" font-weight="700" text-anchor="end" font-family="monospace">${lastD.supercore_yoy}%</text>

      <circle cx="${getX(n-1)}" cy="${getY(lastD.shelter_yoy)}" r="3" fill="#00e5ff"/>
      <circle cx="${getX(n-1)}" cy="${getY(lastD.core_yoy)}" r="3" fill="#eaeaea"/>
      <text x="${getX(n-1) - 4}" y="${getY(lastD.core_yoy) + 12}" fill="#eaeaea" font-size="9" font-weight="700" text-anchor="end" font-family="monospace">${lastD.core_yoy}%</text>
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

  // Split: Left 62% for Scatter OLS, Right 38% for Volatility Ratios VR(w)
  const scatterW = Math.floor(w * 0.62);
  const vrW = w - scatterW;

  const padL = 40;
  const padR = 20;
  const padT = 25;
  const padB = 30;

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
  valid.forEach(e => {
    const cx = getX(e.z_score_composite);
    const cy = getY(e.gold_return_pct);
    const color = e.gold_return_pct > 0 ? "var(--accent-green)" : "var(--accent-red)";
    
    // Annotate major shocks
    let labelSvg = "";
    if (e.gold_return_pct <= -3.0 || e.gold_return_pct >= 1.9 || Math.abs(e.z_score_composite) >= 2.5) {
      labelSvg = `<text x="${cx}" y="${cy - 6}" fill="#aaa" font-size="7.5" text-anchor="middle" font-family="monospace">${e.date.substring(2,7)}</text>`;
    }
    
    pointsSvg += `
      <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.5" fill="${color}" stroke="#000" stroke-width="1" opacity="0.85">
        <title>${e.date} (${e.ref_month}): Z=${e.z_score_composite}σ, Gold=${e.gold_return_pct}%</title>
      </circle>
      ${labelSvg}
    `;
  });

  // Regression line
  const slope = models.refined_composite ? models.refined_composite.slope : -0.558;
  const intercept = models.refined_composite ? models.refined_composite.intercept : 0.148;
  const yZneg3 = intercept + slope * (-3.2);
  const yZpos3 = intercept + slope * (+3.2);
  const regLine = `<line x1="${getX(-3.2)}" y1="${getY(yZneg3)}" x2="${getX(3.2)}" y2="${getY(yZpos3)}" stroke="#00e5ff" stroke-width="2" stroke-dasharray="3,3"/>`;

  // Volatility Ratio Bars on Right side
  const vrData = [
    { label: "VR(5m)", ratio: 5.42, color: "#ff2a2a" },
    { label: "VR(15m)", ratio: 3.65, color: "#00e5ff" },
    { label: "VR(30m)", ratio: 2.58, color: "#eaeaea" },
    { label: "VR(60m)", ratio: 1.84, color: "#888888" }
  ];
  const maxVR = 6.0;
  const vrStartX = scatterW + 15;
  const vrWidth = w - vrStartX - 15;

  let vrSvg = `
    <!-- Vertical divider -->
    <line x1="${scatterW}" y1="${padT - 10}" x2="${scatterW}" y2="${h - 10}" stroke="#262626" stroke-width="1"/>
    <text x="${vrStartX}" y="${padT - 2}" fill="#888" font-size="9" font-weight="700" font-family="monospace">REALIZED VOLATILITY RATIO VR(w)</text>
  `;

  vrData.forEach((item, idx) => {
    const barY = padT + 20 + idx * 42;
    const barLen = (item.ratio / maxVR) * vrWidth;
    vrSvg += `
      <text x="${vrStartX}" y="${barY + 11}" fill="#eaeaea" font-size="9" font-family="monospace">${item.label}</text>
      <rect x="${vrStartX + 55}" y="${barY}" width="${barLen.toFixed(1)}" height="14" fill="${item.color}" opacity="0.9"/>
      <text x="${vrStartX + 60 + barLen}" y="${barY + 11}" fill="${item.color}" font-size="9" font-weight="800" font-family="monospace">${item.ratio.toFixed(2)}x</text>
    `;
  });

  let svg = `
    <svg class="telemetry-svg" viewBox="0 0 ${w} ${h}">
      <!-- Left: Scatter Area -->
      <rect x="${nzLeft}" y="${padT}" width="${nzRight - nzLeft}" height="${h - padT - padB}" fill="#161616" opacity="0.5"/>
      <text x="${cx0}" y="${padT + 12}" fill="#555" font-size="8" text-anchor="middle" font-family="monospace">NOISE (|Z| &lt; 1.0σ)</text>

      <!-- Axes -->
      <line x1="${padL}" y1="${cy0}" x2="${scatterW - padR}" y2="${cy0}" stroke="#333" stroke-width="1"/>
      <line x1="${cx0}" y1="${padT}" x2="${cx0}" y2="${h - padB}" stroke="#333" stroke-width="1"/>

      <text x="${padL - 6}" y="${cy0 + 3}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">0.0%</text>
      <text x="${cx0}" y="${h - 8}" fill="#666" font-size="8" text-anchor="middle" font-family="monospace">0.0σ</text>
      <text x="${scatterW - padR}" y="${h - 8}" fill="#666" font-size="8" text-anchor="end" font-family="monospace">+3.0σ</text>
      <text x="${padL}" y="${h - 8}" fill="#666" font-size="8" text-anchor="start" font-family="monospace">-3.0σ</text>

      ${regLine}
      ${pointsSvg}

      <!-- Right: VR Ratios -->
      ${vrSvg}
    </svg>
  `;
  container.innerHTML = svg;
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
      valEl.innerText = "$" + prices["GC=F"].toFixed(2);
    } else if (label.includes("DOLLAR INDEX") && prices["DX-Y.NYB"]) {
      valEl.innerText = prices["DX-Y.NYB"].toFixed(2);
    } else if (label.includes("10-YEAR YIELD") && prices["^TNX"]) {
      valEl.innerText = prices["^TNX"].toFixed(2) + "%";
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
