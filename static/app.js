/**
 * NEWSLYSER // BLOOMBERG PROFESSIONAL TERMINAL WORKSTATION [PORT 8889 & 8888]
 * Authentic Monochrome White-on-Black Architecture.
 * Complete 9-Module Institutional Macro Suite with GSAP & Framer Motion.
 */

let currentZ = 1.33;
let currentHorizon = 1;
let currentPdfEvent = "cpi";
let currentTransAsset = "gold";
let currentDiagView = "tcurve";
let currentAiMode = "pre_news";
let waveTimer = null;
let maximizedPaneId = null;

// YoY visibility toggles
let yoySeries = { headline: true, core: true, supercore: true, shelter: true };

// Data caches
let stateCache = null;
let pdfCache = null;
let mcCache = null;
let freqCache = null;
let transCache = null;
let diagCache = null;
let ttlTimerInterval = null;

function init() {
  initClock();
  initModuleNavigation();
  initTopSliders();
  initControls();
  initAuditToggle();
  initKeyboardShortcuts();
  initFomcSliders();
  initNfpSliders();
  initPpiSliders();
  initDeepSeekControls();
  initYoyControls();

  // Initial data fetches
  fetchState();
  fetchAuditLedger();
  fetchCalendarData();
  fetchAcademicLit();
  fetchFomcData();
  fetchNfpData();
  fetchPpiData();
  fetchAstarData();
  initAutoRefreshLoop();

  // Render initial CPI workstation panes
  renderPdfPane();
  renderMcPane();
  renderFreqPane();
  renderTransPane();
  renderDiagPane();

  // Staggered pane entrance
  if (window.Motion) {
    const { animate, stagger } = window.Motion;
    animate(".bbg-pane", { opacity: [0, 1], y: [10, 0] }, { delay: stagger(0.03), duration: 0.35 });
  } else if (typeof gsap !== "undefined") {
    gsap.from(".bbg-pane", { opacity: 0, y: 10, stagger: 0.04, duration: 0.4, ease: "power2.out" });
  }
}

function initClock() {
  const clockEl = document.getElementById("live-clock");
  function tick() {
    if (clockEl) {
      clockEl.innerText = new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC";
    }
  }
  tick();
  setInterval(tick, 1000);
}

function animateNumber(el, targetVal, prefix="", suffix="", decimals=2, duration=0.6) {
  if (!el) return;
  const rawCurrent = el.innerText.replace(/[^0-9.-]/g, "");
  const current = parseFloat(rawCurrent) || 0;

  if (typeof gsap === "undefined") {
    el.innerText = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
    return;
  }

  if (Math.abs(targetVal - current) > 1e-4) {
    gsap.fromTo(el, 
      { textShadow: "0 0 8px #ffffff", color: "#ffffff" },
      { textShadow: "none", color: "var(--fg-bright)", duration: 0.7, ease: "power2.out" }
    );
  }

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

// ==============================================================
// 1. TOP-LEVEL MODULE NAVIGATION
// ==============================================================
function initModuleNavigation() {
  const titles = {
    "mod-cpi": "CPI.XAU <GO>",
    "mod-fomc": "FOMC.RATE <GO>",
    "mod-nfp": "JOBS.NFP <GO>",
    "mod-ppi": "PPI.DEMAND <GO>",
    "mod-astar": "ASTR.SETUP <GO>",
    "mod-cal": "ECO.CAL <GO>",
    "mod-ai": "AI.MEMO <GO>",
    "mod-yoy": "TIMELN.YOY <GO>",
    "mod-lit": "LIT.ACADEMIC <GO>"
  };

  const navBtns = document.querySelectorAll(".nav-tab-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const modId = btn.getAttribute("data-mod");
      navBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".module-view").forEach(v => v.classList.remove("active"));
      const activeView = document.getElementById(modId);
      if (activeView) activeView.classList.add("active");

      const titleEl = document.getElementById("cmd-function-title");
      if (titleEl && titles[modId]) titleEl.innerText = titles[modId];

      if (window.Motion) {
        window.Motion.animate(activeView, { opacity: [0, 1], y: [6, 0] }, { duration: 0.25 });
      }

      // Execute loaders when module is selected
      if (modId === "mod-cpi") { renderPdfPane(); renderMcPane(); renderFreqPane(); renderTransPane(); renderDiagPane(); }
      else if (modId === "mod-fomc") renderFomcWorkstation();
      else if (modId === "mod-nfp") renderNfpWorkstation();
      else if (modId === "mod-ppi") renderPpiWorkstation();
      else if (modId === "mod-astar") fetchAstarData();
      else if (modId === "mod-cal") fetchCalendarData();
      else if (modId === "mod-yoy") renderYoyTimeline();
      else if (modId === "mod-lit") fetchAcademicLit();
    });
  });
}

// ==============================================================
// 2. CONTROLS, SLIDERS & FIREWALL
// ==============================================================
function initTopSliders() {
  const rHeadline = document.getElementById("bbg-range-headline");
  const rCore = document.getElementById("bbg-range-core");
  const vHeadline = document.getElementById("bbg-val-headline");
  const vCore = document.getElementById("bbg-val-core");
  const vShock = document.getElementById("bbg-val-shock");
  const vRegime = document.getElementById("bbg-val-regime");

  function onSliderChange() {
    const hl = parseFloat(rHeadline.value);
    const core = parseFloat(rCore.value);

    vHeadline.innerText = `${hl.toFixed(2)}%`;
    vCore.innerText = `${core.toFixed(2)}%`;

    const fHl = 0.3, fCore = 2.8, sd = 0.075;
    const zHl = (hl - fHl) / sd;
    const zCore = (core - fCore) / sd;
    const zComp = 0.60 * zCore + 0.40 * zHl;
    currentZ = zComp;

    vShock.innerText = `${zComp > 0 ? "+" : ""}${zComp.toFixed(2)}σ`;
    vShock.style.color = zComp > 0.8 ? "var(--amber-cmd)" : (zComp < -0.8 ? "#ffffff" : "var(--fg-normal)");

    let regime = "IN-LINE NOISE";
    if (zComp >= 1.5) regime = "HAWKISH EXPANSION DRIFT";
    else if (zComp >= 0.8) regime = "HAWKISH SURPRISE";
    else if (zComp <= -1.5) regime = "DOVISH REVERSAL TREND";
    else if (zComp <= -0.8) regime = "DOVISH RELIEF";
    vRegime.innerText = regime;

    renderPdfPane();
    renderMcPane();
    renderFreqPane();
    executeSimulate(hl, core);
  }

  rHeadline?.addEventListener("input", onSliderChange);
  rCore?.addEventListener("input", onSliderChange);
}

function initControls() {
  document.getElementById("bbg-btn-horizon-1d")?.addEventListener("click", (e) => {
    setBtnActive(e.target, "#bbg-btn-horizon-1d, #bbg-btn-horizon-5d");
    currentHorizon = 1;
    renderMcPane();
  });

  document.getElementById("bbg-btn-horizon-5d")?.addEventListener("click", (e) => {
    setBtnActive(e.target, "#bbg-btn-horizon-1d, #bbg-btn-horizon-5d");
    currentHorizon = 5;
    renderMcPane();
  });

  document.getElementById("bbg-btn-firewall")?.addEventListener("click", () => {
    triggerFirewallCountdown(60);
  });

  document.getElementById("bbg-btn-reset")?.addEventListener("click", () => {
    const rH = document.getElementById("bbg-range-headline");
    const rC = document.getElementById("bbg-range-core");
    if (rH && rC) {
      rH.value = 0.4;
      rC.value = 3.4;
      rH.dispatchEvent(new Event("input"));
    }
  });

  document.getElementById("btn-sync-live")?.addEventListener("click", () => {
    const btn = document.getElementById("btn-sync-live");
    btn.innerText = "SYNCING B-PIPE...";
    fetchState().then(() => {
      fetchAstarData();
      setTimeout(() => { btn.innerText = "SYNC LIVE <GO>"; }, 700);
    });
  });

  document.querySelectorAll("[data-z]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-z]");
      currentZ = parseFloat(e.target.getAttribute("data-z"));
      document.getElementById("bbg-val-shock").innerText = `${currentZ > 0 ? "+" : ""}${currentZ.toFixed(2)}σ`;
      renderMcPane();
      renderFreqPane();
    });
  });

  document.querySelectorAll("[data-toggle]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-toggle]");
      const ev = e.target.getAttribute("data-toggle").replace("pdf-", "");
      currentPdfEvent = ev;
      renderPdfPane();
    });
  });

  document.querySelectorAll("[data-asset]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-asset]");
      currentTransAsset = e.target.getAttribute("data-asset");
      renderTransPane();
    });
  });

  document.querySelectorAll("[data-diag]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-diag]");
      currentDiagView = e.target.getAttribute("data-diag");
      renderDiagPane();
    });
  });

  // Maximize toggle [+]
  document.querySelectorAll(".btn-maximize").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const paneId = e.target.getAttribute("data-pane");
      const pane = document.getElementById(paneId);
      if (!pane) return;

      if (maximizedPaneId === paneId) {
        pane.classList.remove("maximized");
        e.target.innerText = "[+]";
        maximizedPaneId = null;
      } else {
        document.querySelectorAll(".bbg-pane").forEach(p => p.classList.remove("maximized"));
        document.querySelectorAll(".btn-maximize").forEach(b => b.innerText = "[+]");
        pane.classList.add("maximized");
        e.target.innerText = "[-]";
        maximizedPaneId = paneId;
      }

      setTimeout(() => {
        if (paneId === "pane-pdf") renderPdfPane();
        else if (paneId === "pane-mc") renderMcPane();
        else if (paneId === "pane-freq") renderFreqPane();
        else if (paneId === "pane-trans") renderTransPane();
        else if (paneId === "pane-diag") renderDiagPane();
      }, 50);
    });
  });
}

function setBtnActive(el, groupSelector) {
  document.querySelectorAll(groupSelector).forEach(b => b.classList.remove("active"));
  el.classList.add("active");
}

function initAuditToggle() {
  const pairs = [
    ["audit-toggle-header", "audit-table-scroll", "audit-toggle-icon"],
    ["fomc-audit-toggle-header", "fomc-audit-table-scroll", "fomc-audit-toggle-icon"],
    ["nfp-audit-toggle-header", "nfp-audit-table-scroll", "nfp-audit-toggle-icon"],
    ["ppi-audit-toggle-header", "ppi-audit-table-scroll", "ppi-audit-toggle-icon"],
    ["astar-audit-toggle-header", "astar-audit-table-scroll", "astar-audit-toggle-icon"]
  ];

  pairs.forEach(([headId, scrollId, iconId]) => {
    const head = document.getElementById(headId);
    const scroll = document.getElementById(scrollId);
    const icon = document.getElementById(iconId);
    head?.addEventListener("click", () => {
      const isHidden = scroll.style.display === "none";
      scroll.style.display = isHidden ? "block" : "none";
      if (icon) icon.innerText = isHidden ? "[ - CLICK TO COLLAPSE DATABASE ]" : "[ + CLICK TO EXPAND FULL DATABASE ]";
    });
  });
}

function initKeyboardShortcuts() {
  window.addEventListener("keydown", (e) => {
    if (e.key === "F1") { e.preventDefault(); alert("NEWSLYSER B-PIPE TERMINAL // 2SLS HC3 KERNEL // KOU SDE // USE NAV BUTTONS TO SWITCH MACRO MODULES."); }
    else if (e.key === "F2") { e.preventDefault(); document.getElementById("bbg-range-headline")?.focus(); }
    else if (e.key === "F3") { e.preventDefault(); document.querySelector("[data-pane='pane-mc']")?.click(); }
    else if (e.key === "F4") { e.preventDefault(); document.querySelector("[data-pane='pane-trans']")?.click(); }
    else if (e.key === "F5") { e.preventDefault(); document.querySelector("[data-pane='pane-diag']")?.click(); }
    else if (e.key === "F6") { e.preventDefault(); document.getElementById("audit-toggle-header")?.click(); }
    else if (e.key === "F8") { e.preventDefault(); document.getElementById("bbg-btn-reset")?.click(); }
    else if (e.key === "F12") { e.preventDefault(); document.getElementById("btn-sync-live")?.click(); }
  });
}

function triggerFirewallCountdown(seconds = 60) {
  clearInterval(waveTimer);
  let timeLeft = seconds;
  const timerEl = document.getElementById("val-firewall-timer");
  const statusEl = document.getElementById("val-firewall-status");

  if (timerEl) { timerEl.innerText = `${timeLeft}s`; timerEl.style.color = "#ff4444"; }
  if (statusEl) { statusEl.innerText = "WAVE 1 // SPREAD BLOWOUT"; statusEl.style.color = "#ff4444"; statusEl.style.borderColor = "#ff4444"; }

  waveTimer = setInterval(() => {
    timeLeft--;
    if (timerEl) {
      if (timeLeft > 0) {
        timerEl.innerText = `${timeLeft}s`;
        if (timeLeft <= 15 && statusEl) {
          statusEl.innerText = "WAVE 1 ENDING: SPREADS COMPRESSING TO 40c";
        }
      } else {
        timerEl.innerText = "WAVE 2";
        if (statusEl) statusEl.innerText = "WAVE 2 ACTIVE: CHECK 2Y & DXY ALIGNMENT";
        clearInterval(waveTimer);
        waveTimer = null;
      }
    }
  }, 1000);
}

async function fetchState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    stateCache = data;
    const p = data.prices;
    if (p) {
      animateNumber(document.getElementById("tick-gold"), p["GC=F"], "$", "", 2);
      animateNumber(document.getElementById("tick-dxy"), p["DX-Y.NYB"], "", "", 2);
      animateNumber(document.getElementById("tick-2y"), p["ZT=F"], "", "%", 2);
      animateNumber(document.getElementById("tick-10y"), p["^TNX"], "", "%", 2);
      animateNumber(document.getElementById("tick-oil"), p["CL=F"], "$", "", 2);
    }
    const av = data.dynamic_levels?.avwap_bands;
    if (av) {
      if (document.getElementById("val-ticket-avwap")) document.getElementById("val-ticket-avwap").innerText = `$${av.avwap.toFixed(2)} (±1σ: $${av.upper_1sd.toFixed(0)} / $${av.lower_1sd.toFixed(0)})`;
      if (document.getElementById("val-ticket-lambda")) document.getElementById("val-ticket-lambda").innerText = `${av.kyles_lambda_illiquidity_proxy.toFixed(4)}% (SLIPPAGE RISK)`;
    }
  } catch (e) {
    console.warn("fetchState failed:", e);
  }
}

async function executeSimulate(actMom, actCoreYoY) {
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_mom: actMom, actual_core_yoy: actCoreYoY })
    });
    const data = await res.json();
    const f = data.model_forecast;
    const d = data.dynamic_execution;

    if (f && d) {
      const actEl = document.getElementById("val-ticket-action");
      if (actEl) actEl.innerText = f.signal;
      if (document.getElementById("val-ticket-entry")) document.getElementById("val-ticket-entry").innerText = d.entry_zone;
      if (document.getElementById("val-ticket-sl")) document.getElementById("val-ticket-sl").innerText = `${d.stop_loss} (STRUCTURAL CEILING)`;
      if (document.getElementById("val-ticket-tp")) document.getElementById("val-ticket-tp").innerText = `${d.target} (RR: ${d.risk_reward_ratio})`;
    }
  } catch (e) {
    console.warn("executeSimulate failed:", e);
  }
}

// ==============================================================
// 3. PANE RENDERING (PDF, MC, FREQ, TRANS, DIAG)
// ==============================================================
async function renderPdfPane() {
  const container = document.getElementById("canvas-pdf");
  const hudEl = document.getElementById("hud-pdf");
  if (!container) return;

  let data = null;
  try {
    const res = await fetch(`/api/pdf-distribution?event_type=${currentPdfEvent}`);
    data = await res.json();
    pdfCache = data;
  } catch (e) {
    console.warn("PDF error:", e);
    return;
  }

  const pts = data.points;
  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const minX = pts[0].x;
  const maxX = pts[pts.length - 1].x;
  const maxPdf = Math.max(...pts.map(p => Math.max(p.prior_pdf, p.posterior_pdf))) * 1.15;

  function getX(x) { return padL + ((x - minX) / (maxX - minX)) * (w - padL - padR); }
  function getY(y) { return padT + (1 - (y / maxPdf)) * (h - padT - padB); }

  const pathPrior = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.prior_pdf).toFixed(1)}`).join(" ");
  const pathPost = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.posterior_pdf).toFixed(1)}`).join(" ");

  const x0 = getX(0);
  const xMiss = getX(minX * 0.35);
  const xBeat = getX(maxX * 0.35);
  const yZero = getY(0);

  const existingSvg = document.getElementById("svg-pdf");
  if (existingSvg) {
    if (typeof gsap !== "undefined") {
      gsap.to("#pdf-prior-line", { attr: { d: pathPrior }, duration: 0.5, ease: "power2.out" });
      gsap.to("#pdf-post-line", { attr: { d: pathPost }, duration: 0.5, ease: "power2.out" });
    }
    return;
  }

  const svg = `
    <svg id="svg-pdf" viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <defs>
        <pattern id="hatch-gray" width="5" height="5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="5" stroke="#1f1f1f" stroke-width="1" />
        </pattern>
      </defs>

      <rect x="${padL}" y="${padT}" width="${xMiss - padL}" height="${h - padT - padB}" fill="url(#hatch-gray)"/>
      <rect x="${xBeat}" y="${padT}" width="${w - padR - xBeat}" height="${h - padT - padB}" fill="url(#hatch-gray)"/>

      <line x1="${padL}" y1="${yZero}" x2="${w - padR}" y2="${yZero}" class="axis-line"/>
      <line x1="${xMiss}" y1="${padT}" x2="${xMiss}" y2="${h - padB}" class="grid-dot"/>
      <line x1="${xBeat}" y1="${padT}" x2="${xBeat}" y2="${h - padB}" class="grid-dot"/>
      
      <line x1="${x0}" y1="${padT}" x2="${x0}" y2="${yZero}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${x0}" y="${padT + 10}" fill="#ffb000" text-anchor="middle">T0: 0.0σ</text>

      <path id="pdf-prior-line" d="${pathPrior}" fill="none" stroke="#666666" stroke-width="1.2" stroke-dasharray="3,3"/>
      <path id="pdf-post-line" d="${pathPost}" fill="none" stroke="#ffffff" stroke-width="2.2"/>

      <text x="${padL}" y="${h - 6}" text-anchor="start">${minX.toFixed(2)}pp</text>
      <text x="${xMiss}" y="${h - 6}" text-anchor="middle">MISS</text>
      <text x="${xBeat}" y="${h - 6}" text-anchor="middle">BEAT</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">${maxX.toFixed(2)}pp</text>

      <line id="pdf-needle" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  const svgEl = document.getElementById("svg-pdf");
  const needle = document.getElementById("pdf-needle");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((ev.clientX - rect.left) / rect.width) * w));
      const xVal = minX + ((mouseX - padL) / (w - padL - padR)) * (maxX - minX);

      let closest = pts[0];
      let minDiff = 999;
      pts.forEach(p => {
        const d = Math.abs(p.x - xVal);
        if (d < minDiff) { minDiff = d; closest = p; }
      });

      hudEl.innerHTML = `x = <span class="highlight">${closest.x > 0 ? "+" : ""}${closest.x.toFixed(2)}pp</span> | Prior PDF = <span class="highlight">${closest.prior_pdf.toFixed(2)}</span> | Post PDF = <span class="highlight">${closest.posterior_pdf.toFixed(2)}</span> [${closest.zone}]`;
      if (needle) {
        needle.style.display = "block";
        needle.setAttribute("x1", mouseX);
        needle.setAttribute("x2", mouseX);
        needle.setAttribute("y1", padT);
        needle.setAttribute("y2", yZero);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER NEEDLE OVER CURVE // PRIOR N(0, 0.075) vs ENERGY POSTERIOR";
      if (needle) needle.style.display = "none";
    });
  }
}

async function renderMcPane() {
  const container = document.getElementById("canvas-mc");
  const hudEl = document.getElementById("hud-mc");
  if (!container) return;

  let mc = null;
  try {
    const res = await fetch(`/api/monte-carlo?horizon=${currentHorizon}&z=${currentZ}`);
    mc = await res.json();
    mcCache = mc;
  } catch (e) {
    console.warn("MC error:", e);
    return;
  }

  document.getElementById("val-mc-median").innerText = `$${mc.metrics.median_terminal_price.toFixed(2)}`;
  document.getElementById("val-mc-var").innerText = `${mc.metrics.var_95_pct > 0 ? "+" : ""}${mc.metrics.var_95_pct.toFixed(2)}% (P_pos: ${mc.metrics.p_positive_return_pct.toFixed(1)}% / P_neg: ${mc.metrics.p_negative_return_pct.toFixed(1)}%)`;

  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 40;
  const padR = 45;
  const padT = 16;
  const padB = 25;

  const steps = mc.steps || 24;
  const p5 = mc.percentiles.p5;
  const p25 = mc.percentiles.p25;
  const p50 = mc.percentiles.p50;
  const p75 = mc.percentiles.p75;
  const p95 = mc.percentiles.p95;

  const allVals = [...p5, ...p95];
  const minP = Math.min(...allVals) * 0.995;
  const maxP = Math.max(...allVals) * 1.005;

  function getX(i) { return padL + (i / steps) * (w - padL - padR); }
  function getY(p) { return padT + (1 - (p - minP) / (maxP - minP)) * (h - padT - padB); }

  const path90 = `M ${getX(0)} ${getY(p95[0])} ` + p95.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
    p5.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

  const path50 = `M ${getX(0)} ${getY(p75[0])} ` + p75.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
    p25.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

  const pathMedian = p50.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");

  const existingSvg = document.getElementById("svg-mc");
  if (existingSvg) {
    if (typeof gsap !== "undefined") {
      gsap.to("#mc-env-90", { attr: { d: path90 }, duration: 0.5, ease: "power2.out" });
      gsap.to("#mc-env-50", { attr: { d: path50 }, duration: 0.5, ease: "power2.out" });
      gsap.to("#mc-med-line", { attr: { d: pathMedian }, duration: 0.5, ease: "power2.out" });
      gsap.to("#mc-term-dot", { attr: { cx: getX(steps), cy: getY(p50[steps]) }, duration: 0.5, ease: "power2.out" });
    }
    return;
  }

  let tracesSvg = "";
  mc.sample_traces.forEach((tr, idx) => {
    const d = tr.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");
    tracesSvg += `<path d="${d}" fill="none" stroke="#444444" stroke-width="0.8" opacity="${idx < 3 ? '0.5' : '0.2'}"/>`;
  });

  const baseY = getY(mc.current_price);
  const svg = `
    <svg id="svg-mc" viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${baseY}" x2="${w - padR}" y2="${baseY}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${padL + 4}" y="${baseY - 4}" fill="#ffb000">T0: $${mc.current_price.toFixed(1)}</text>

      <path id="mc-env-90" d="${path90}" fill="#111111" opacity="0.8"/>
      <path id="mc-env-50" d="${path50}" fill="#222222" opacity="0.9"/>
      
      <g id="mc-sample-traces">${tracesSvg}</g>
      
      <path id="mc-med-line" d="${pathMedian}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <circle id="mc-term-dot" cx="${getX(steps)}" cy="${getY(p50[steps])}" r="3" fill="#ffffff"/>
      <text x="${getX(steps) + 5}" y="${getY(p50[steps]) + 3}" fill="#ffffff" font-weight="700">$${p50[steps].toFixed(0)}</text>

      <line id="mc-crosshair-v" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  const svgEl = document.getElementById("svg-mc");
  const crossV = document.getElementById("mc-crosshair-v");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((ev.clientX - rect.left) / rect.width) * w));
      const s = Math.max(0, Math.min(steps, Math.round(((mouseX - padL) / (w - padL - padR)) * steps)));

      const medP = p50[s];
      const p95Val = p95[s];
      const p5Val = p5[s];
      hudEl.innerHTML = `${mc.timesteps[s]} | Median: <span class="highlight">$${medP.toFixed(1)}</span> (P95: $${p95Val.toFixed(0)} / P5: $${p5Val.toFixed(0)})`;

      if (crossV) {
        crossV.style.display = "block";
        crossV.setAttribute("x1", getX(s));
        crossV.setAttribute("x2", getX(s));
        crossV.setAttribute("y1", padT);
        crossV.setAttribute("y2", h - padB);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER TIMESTEP // PERCENTILE ENVELOPE [P5, P25, P50, P75, P95]";
      if (crossV) crossV.style.display = "none";
    });
  }
}

async function renderFreqPane() {
  const container = document.getElementById("canvas-freq");
  const hudEl = document.getElementById("hud-freq");
  if (!container) return;

  let freq = null;
  try {
    const res = await fetch(`/api/shock-frequency?z=${currentZ}`);
    freq = await res.json();
    freqCache = freq;
  } catch (e) {
    console.warn("Freq error:", e);
    return;
  }

  document.getElementById("val-freq-rank").innerText = `${freq.empirical_percentile_rank.toFixed(1)}th PERCENTILE (DIST FROM MEDIAN: ${freq.distance_from_median > 0 ? "+" : ""}${freq.distance_from_median.toFixed(2)}σ)`;

  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const minX = -3.0;
  const maxX = +3.0;
  const maxCount = Math.max(...freq.histogram_bars.map(b => b.count)) || 15;
  const maxKde = Math.max(...freq.kde_curve.map(k => k.density)) || 0.6;

  function getX(x) { return padL + ((x - minX) / (maxX - minX)) * (w - padL - padR); }
  function getYCount(c) { return padT + (1 - (c / maxCount)) * (h - padT - padB); }
  function getYKde(d) { return padT + (1 - (d / maxKde)) * (h - padT - padB); }

  const kdePath = freq.kde_curve.map((k, i) => `${i === 0 ? "M" : "L"} ${getX(k.x).toFixed(1)} ${getYKde(k.density).toFixed(1)}`).join(" ");

  let barsSvg = "";
  freq.histogram_bars.forEach(b => {
    const x1 = getX(b.range[0]);
    const x2 = getX(b.range[1]);
    const barW = Math.max(2, (x2 - x1) - 2);
    const barY = getYCount(b.count);
    const barH = (h - padB) - barY;
    const isCurrent = b.range[0] <= currentZ && currentZ <= b.range[1];
    const fill = isCurrent ? "#282828" : "#0d0d0d";
    const stroke = isCurrent ? "#ffffff" : "#222222";

    barsSvg += `
      <rect class="freq-bar" x="${x1 + 1}" y="${barY}" width="${barW}" height="${barH}" fill="${fill}" stroke="${stroke}" stroke-width="1"
            data-label="${b.label}" data-count="${b.count}" data-pct="${b.pct}"/>
      <text x="${x1 + barW / 2}" y="${barY - 3}" fill="#666666" text-anchor="middle">${b.count > 0 ? b.count : ""}</text>
    `;
  });

  const curX = getX(currentZ);
  const medX = getX(freq.median_z);

  const existingSvg = document.getElementById("svg-freq");
  if (existingSvg) {
    if (typeof gsap !== "undefined") {
      gsap.to("#freq-kde", { attr: { d: kdePath }, duration: 0.5, ease: "power2.out" });
      gsap.to("#freq-cur-needle", { attr: { x1: curX, x2: curX }, duration: 0.4, ease: "power2.out" });
      gsap.to("#freq-cur-text", { attr: { x: curX }, duration: 0.4, ease: "power2.out" });
    }
    const bGroup = document.getElementById("freq-bars-group");
    if (bGroup) bGroup.innerHTML = barsSvg;
    return;
  }

  const svg = `
    <svg id="svg-freq" viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <g id="freq-bars-group">${barsSvg}</g>
      
      <path id="freq-kde" d="${kdePath}" fill="none" stroke="#ffffff" stroke-width="2"/>

      <line x1="${medX}" y1="${padT}" x2="${medX}" y2="${h - padB}" stroke="#666" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${medX}" y="${padT + 10}" fill="#888" text-anchor="middle">MEDIAN: ${freq.median_z.toFixed(2)}σ</text>

      <line id="freq-cur-needle" x1="${curX}" y1="${padT + 14}" x2="${curX}" y2="${h - padB}" stroke="#ffffff" stroke-width="1.5"/>
      <text id="freq-cur-text" x="${curX}" y="${padT + 10}" fill="#ffffff" font-weight="800" text-anchor="middle">[CURRENT: ${currentZ > 0 ? "+" : ""}${currentZ.toFixed(2)}σ]</text>

      <text x="${padL}" y="${h - 6}" text-anchor="start">-3.0σ</text>
      <text x="${getX(0)}" y="${h - 6}" text-anchor="middle">0.0σ</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">+3.0σ</text>
    </svg>
  `;
  container.innerHTML = svg;

  const svgEl = document.getElementById("svg-freq");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const target = ev.target;
      if (target && target.classList.contains("freq-bar")) {
        const l = target.getAttribute("data-label");
        const c = target.getAttribute("data-count");
        const p = target.getAttribute("data-pct");
        hudEl.innerHTML = `HISTOGRAM BIN: <span class="highlight">${l}</span> | Count = <span class="highlight">${c} releases</span> (${p}% of sample)`;
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER HISTOGRAM BARS // N = 34 EMPIRICAL CPI RELEASES";
    });
  }
}

async function renderTransPane() {
  const container = document.getElementById("canvas-trans");
  const hudEl = document.getElementById("hud-trans");
  if (!container) return;

  let trans = null;
  try {
    const res = await fetch("/api/transmission-curves");
    trans = await res.json();
    transCache = trans;
  } catch (e) {
    console.warn("Trans error:", e);
    return;
  }

  const curves = trans.response_curves;
  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const minZ = -3.0;
  const maxZ = +3.0;

  let yKey = "gold_return";
  let minY = -2.5, maxY = +2.5;
  let assetLabel = "GOLD RESPONSE";
  if (currentTransAsset === "dxy") {
    yKey = "dxy_return";
    minY = -1.0; maxY = +1.0;
    assetLabel = "DXY RESPONSE";
  } else if (currentTransAsset === "2y") {
    yKey = "zt_return";
    minY = -1.2; maxY = +1.2;
    assetLabel = "2Y NOTE RESPONSE";
  }

  function getX(z) { return padL + ((z - minZ) / (maxZ - minZ)) * (w - padL - padR); }
  function getY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * (h - padT - padB); }

  const pathReg = curves.map((c, i) => `${i === 0 ? "M" : "L"} ${getX(c.z).toFixed(1)} ${getY(c[yKey]).toFixed(1)}`).join(" ");

  let bandPath = "";
  if (currentTransAsset === "gold") {
    const upper = curves.map((c, i) => `${i === 0 ? "M" : "L"} ${getX(c.z).toFixed(1)} ${getY(c.gold_upper_95).toFixed(1)}`).join(" ");
    const lower = curves.slice().reverse().map(c => `L ${getX(c.z).toFixed(1)} ${getY(c.gold_lower_95).toFixed(1)}`).join(" ");
    bandPath = `${upper} ${lower} Z`;
  }

  let dotsSvg = "";
  trans.scatter_points.forEach(p => {
    let py = p.gold_ret;
    if (currentTransAsset === "dxy") py = p.dxy_ret;
    else if (currentTransAsset === "2y") py = p.zt_ret;

    dotsSvg += `<circle cx="${getX(p.z)}" cy="${getY(py)}" r="3" fill="#333" stroke="#666" stroke-width="0.8"/>`;
  });

  const existingSvg = document.getElementById("svg-trans");
  if (existingSvg) {
    if (typeof gsap !== "undefined") {
      gsap.to("#trans-reg-line", { attr: { d: pathReg }, duration: 0.5, ease: "power2.out" });
      if (document.getElementById("trans-band-area")) {
        gsap.to("#trans-band-area", { attr: { d: bandPath }, duration: 0.5, ease: "power2.out" });
      }
    }
    const dotGroup = document.getElementById("trans-dots-group");
    if (dotGroup) dotGroup.innerHTML = dotsSvg;
    return;
  }

  const svg = `
    <svg id="svg-trans" viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
      <line x1="${getX(0)}" y1="${padT}" x2="${getX(0)}" y2="${h - padB}" class="axis-line"/>

      ${bandPath ? `<path id="trans-band-area" d="${bandPath}" fill="#161616" opacity="0.6"/>` : ''}

      <g id="trans-dots-group">${dotsSvg}</g>

      <path id="trans-reg-line" d="${pathReg}" fill="none" stroke="#ffffff" stroke-width="2"/>

      <text x="${w - padR}" y="${getY(0) - 4}" text-anchor="end">+Z (HAWKISH)</text>
      <text x="${padL}" y="${getY(0) - 4}" text-anchor="start">-Z (DOVISH)</text>
      <text x="${getX(0) + 4}" y="${padT + 10}">% RETURN</text>

      <line id="trans-needle-x" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  const svgEl = document.getElementById("svg-trans");
  const needleX = document.getElementById("trans-needle-x");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((ev.clientX - rect.left) / rect.width) * w));
      const zVal = minZ + ((mouseX - padL) / (w - padL - padR)) * (maxZ - minZ);

      let closest = curves[0];
      let minDiff = 999;
      curves.forEach(c => {
        const d = Math.abs(c.z - zVal);
        if (d < minDiff) { minDiff = d; closest = c; }
      });

      const ret = closest[yKey];
      hudEl.innerHTML = `Shock Z = <span class="highlight">${closest.z > 0 ? "+" : ""}${closest.z.toFixed(2)}σ</span> | Expected ${assetLabel} = <span class="highlight">${ret > 0 ? "+" : ""}${ret.toFixed(2)}%</span>`;

      if (needleX) {
        needleX.style.display = "block";
        needleX.setAttribute("x1", mouseX);
        needleX.setAttribute("x2", mouseX);
        needleX.setAttribute("y1", padT);
        needleX.setAttribute("y2", h - padB);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER Z-AXIS // WHITE HC3 95% PREDICTION BANDS";
      if (needleX) needleX.style.display = "none";
    });
  }
}

async function renderDiagPane() {
  const container = document.getElementById("canvas-diag");
  const hudEl = document.getElementById("hud-diag");
  if (!container) return;

  let diag = null;
  try {
    const res = await fetch("/api/hypothesis-diagnostics");
    diag = await res.json();
    diagCache = diag;
  } catch (e) {
    console.warn("Diag error:", e);
    return;
  }

  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  if (currentDiagView === "tcurve") {
    const tCrit = diag.t_crit_95;
    const tStat = diag.beta_z_t_stat;
    const pts = diag.t_curve;
    const maxPdf = Math.max(...pts.map(p => p.pdf)) * 1.15;

    function getX(t) { return padL + ((t - (-4.5)) / (4.5 - (-4.5))) * (w - padL - padR); }
    function getY(p) { return padT + (1 - (p / maxPdf)) * (h - padT - padB); }

    const tPath = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.t).toFixed(1)} ${getY(p.pdf).toFixed(1)}`).join(" ");

    const xCritL = getX(-tCrit);
    const xCritR = getX(+tCrit);
    const xStat = getX(Math.max(-4.4, Math.min(4.4, tStat)));

    const svg = `
      <svg id="svg-diag-t" viewBox="0 0 ${w} ${h}" class="bbg-svg">
        <rect x="${padL}" y="${padT}" width="${xCritL - padL}" height="${h - padT - padB}" fill="#240808"/>
        <rect x="${xCritR}" y="${padT}" width="${w - padR - xCritR}" height="${h - padT - padB}" fill="#240808"/>

        <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
        <line x1="${getX(0)}" y1="${padT}" x2="${getX(0)}" y2="${h - padB}" class="axis-line"/>

        <path d="${tPath}" fill="none" stroke="#ffffff" stroke-width="2"/>

        <line x1="${xCritL}" y1="${padT}" x2="${xCritL}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <line x1="${xCritR}" y1="${padT}" x2="${xCritR}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <text x="${xCritL}" y="${padT + 10}" fill="#ff4444" text-anchor="middle">-t_crit</text>
        <text x="${xCritR}" y="${padT + 10}" fill="#ff4444" text-anchor="middle">+t_crit</text>

        <line x1="${xStat}" y1="${padT + 14}" x2="${xStat}" y2="${h - padB}" stroke="#ffb000" stroke-width="2"/>
        <text x="${xStat}" y="${padT + 8}" fill="#ffb000" font-weight="900" text-anchor="middle">t_stat = ${tStat.toFixed(2)}</text>

        <text x="${padL}" y="${h - 6}">-4.5</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">+4.5</text>
      </svg>
    `;
    container.innerHTML = svg;
  } else {
    // Leverage View
    const pts = diag.diagnostic_points;
    const hThresh = diag.leverage_threshold;

    function getX(h) { return padL + (h / 0.45) * (w - padL - padR); }
    function getY(r) { return padT + (1 - (r - (-3.5)) / (3.5 - (-3.5))) * (h - padT - padB); }

    let dotsSvg = "";
    pts.forEach(p => {
      const cx = getX(p.leverage_h);
      const cy = getY(p.studentized_resid);
      const isOut = p.is_high_leverage || p.is_outlier;
      const fill = isOut ? "#ffb000" : "#333333";
      const stroke = isOut ? "#ffffff" : "#666666";
      dotsSvg += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`;
    });

    const xThresh = getX(hThresh);
    const svg = `
      <svg id="svg-diag-lev" viewBox="0 0 ${w} ${h}" class="bbg-svg">
        <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
        <line x1="${padL}" y1="${getY(+2.0)}" x2="${w - padR}" y2="${getY(+2.0)}" class="grid-dot"/>
        <line x1="${padL}" y1="${getY(-2.0)}" x2="${w - padR}" y2="${getY(-2.0)}" class="grid-dot"/>

        <line x1="${xThresh}" y1="${padT}" x2="${xThresh}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <text x="${xThresh + 4}" y="${padT + 10}" fill="#ff4444">h* = ${hThresh.toFixed(3)}</text>

        <g id="diag-lev-dots">${dotsSvg}</g>

        <text x="${padL}" y="${padT + 10}">STUD RESID</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">HAT LEVERAGE (h)</text>
      </svg>
    `;
    container.innerHTML = svg;
  }
}

// ==============================================================
// 4. UNIVERSAL MULTI-EVENT WORKSTATION RENDERERS
// ==============================================================

async function renderFomcWorkstation() {
  await renderGenericMc("canvas-fomc-mc", "fomc", 0.0);
  await renderGenericPdf("canvas-fomc-pdf", "fomc");
  await renderGenericFreq("canvas-fomc-freq", "fomc", 0.0);
  await renderGenericTrans("canvas-fomc-trans", "fomc", -0.054, 4.42);
  await renderGenericDiag("canvas-fomc-diag", "fomc");
}

async function renderNfpWorkstation() {
  await renderGenericMc("canvas-nfp-mc", "nfp", 0.0);
  await renderGenericPdf("canvas-nfp-pdf", "nfp");
  await renderGenericFreq("canvas-nfp-freq", "nfp", 0.0);
  await renderGenericTrans("canvas-nfp-trans", "nfp", -0.960, 4.81);
  await renderGenericDiag("canvas-nfp-diag", "nfp");
}

async function renderPpiWorkstation() {
  await renderGenericMc("canvas-ppi-mc", "ppi", 0.0);
  await renderGenericPdf("canvas-ppi-pdf", "ppi");
  await renderGenericFreq("canvas-ppi-freq", "ppi", 0.0);
  await renderGenericTrans("canvas-ppi-trans", "ppi", -0.803, 3.92);
  await renderGenericDiag("canvas-ppi-diag", "ppi");
}

async function renderGenericMc(containerId, eventType, zShock = 0.0) {
  const container = document.getElementById(containerId);
  if (!container) return;

  try {
    const res = await fetch(`/api/monte-carlo?horizon=1&z=${zShock}&event_type=${eventType}`);
    const mc = await res.json();

    const w = container.clientWidth || 360;
    const h = container.clientHeight || 210;
    const padL = 40;
    const padR = 45;
    const padT = 16;
    const padB = 25;

    const steps = mc.steps || 24;
    const p5 = mc.percentiles.p5;
    const p25 = mc.percentiles.p25;
    const p50 = mc.percentiles.p50;
    const p75 = mc.percentiles.p75;
    const p95 = mc.percentiles.p95;

    const allVals = [...p5, ...p95];
    const minP = Math.min(...allVals) * 0.995;
    const maxP = Math.max(...allVals) * 1.005;

    function getX(i) { return padL + (i / steps) * (w - padL - padR); }
    function getY(p) { return padT + (1 - (p - minP) / (maxP - minP)) * (h - padT - padB); }

    const path90 = `M ${getX(0)} ${getY(p95[0])} ` + p95.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
      p5.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

    const path50 = `M ${getX(0)} ${getY(p75[0])} ` + p75.map((v, i) => `L ${getX(i)} ${getY(v)}`).join(" ") +
      p25.slice().reverse().map((v, i) => ` L ${getX(steps - i)} ${getY(v)}`).join("") + " Z";

    const pathMedian = p50.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");

    const baseY = getY(mc.current_price);
    const svg = `
      <svg viewBox="0 0 ${w} ${h}" class="bbg-svg">
        <line x1="${padL}" y1="${baseY}" x2="${w - padR}" y2="${baseY}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
        <text x="${padL + 4}" y="${baseY - 4}" fill="#ffb000">T0: $${mc.current_price.toFixed(1)}</text>
        <path d="${path90}" fill="#111111" opacity="0.8"/>
        <path d="${path50}" fill="#222222" opacity="0.9"/>
        <path d="${pathMedian}" fill="none" stroke="#ffffff" stroke-width="2"/>
        <circle cx="${getX(steps)}" cy="${getY(p50[steps])}" r="3" fill="#ffffff"/>
        <text x="${getX(steps) - 6}" y="${getY(p50[steps]) - 6}" fill="#fff" text-anchor="end">$${p50[steps].toFixed(0)}</text>
        <text x="${padL}" y="${h - 6}" text-anchor="start">T0</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">T+24h</text>
      </svg>
    `;
    container.innerHTML = svg;
  } catch (e) {
    console.warn("renderGenericMc error:", e);
  }
}

async function renderGenericPdf(containerId, eventType) {
  const container = document.getElementById(containerId);
  if (!container) return;
  try {
    const res = await fetch(`/api/pdf-distribution?event_type=${eventType}`);
    const data = await res.json();
    const pts = data.points;
    const w = container.clientWidth || 360;
    const h = container.clientHeight || 210;
    const padL = 35;
    const padR = 35;
    const padT = 16;
    const padB = 25;

    const minX = pts[0].x;
    const maxX = pts[pts.length - 1].x;
    const maxPdf = Math.max(...pts.map(p => Math.max(p.prior_pdf, p.posterior_pdf))) * 1.15;

    function getX(x) { return padL + ((x - minX) / (maxX - minX)) * (w - padL - padR); }
    function getY(y) { return padT + (1 - (y / maxPdf)) * (h - padT - padB); }

    const pathPrior = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.prior_pdf).toFixed(1)}`).join(" ");
    const pathPost = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.posterior_pdf).toFixed(1)}`).join(" ");

    const x0 = getX(0);
    const yZero = getY(0);

    const svg = `
      <svg viewBox="0 0 ${w} ${h}" class="bbg-svg">
        <line x1="${padL}" y1="${yZero}" x2="${w - padR}" y2="${yZero}" class="axis-line"/>
        <line x1="${x0}" y1="${padT}" x2="${x0}" y2="${yZero}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
        <path d="${pathPrior}" fill="none" stroke="#666666" stroke-width="1.2" stroke-dasharray="3,3"/>
        <path d="${pathPost}" fill="none" stroke="#ffffff" stroke-width="2.2"/>
        <text x="${padL}" y="${h - 6}">${minX.toFixed(2)}</text>
        <text x="${x0}" y="${h - 6}" text-anchor="middle">0.0</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">${maxX.toFixed(2)}</text>
      </svg>
    `;
    container.innerHTML = svg;
  } catch (e) {
    console.warn("renderGenericPdf error:", e);
  }
}

async function renderGenericTrans(containerId, eventType, slope, vr5) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const minZ = -3.0;
  const maxZ = +3.0;
  const minY = -2.5;
  const maxY = +2.5;

  function getX(z) { return padL + ((z - minZ) / (maxZ - minZ)) * (w - padL - padR); }
  function getY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * (h - padT - padB); }

  const x0 = minZ, x1 = maxZ;
  const y0 = x0 * (slope * 25.0 > 1.0 ? slope * 25.0 : slope);
  const y1 = x1 * (slope * 25.0 > 1.0 ? slope * 25.0 : slope);

  const svg = `
    <svg viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
      <line x1="${getX(0)}" y1="${padT}" x2="${getX(0)}" y2="${h - padB}" class="axis-line"/>
      <line x1="${getX(x0)}" y1="${getY(y0)}" x2="${getX(x1)}" y2="${getY(y1)}" stroke="#ffffff" stroke-width="2"/>
      <text x="${w - padR}" y="${getY(0) - 4}" text-anchor="end">+SHOCK</text>
      <text x="${padL}" y="${getY(0) - 4}" text-anchor="start">-SHOCK</text>
      <text x="${w - padR}" y="${padT + 12}" fill="var(--amber-cmd)" text-anchor="end">VR(5m): ${vr5.toFixed(1)}x</text>
    </svg>
  `;
  container.innerHTML = svg;
}

async function renderGenericFreq(containerId, eventType, zVal) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const svg = `
    <svg viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <rect x="${padL + 20}" y="${h - padB - 40}" width="25" height="40" fill="#222" stroke="#444"/>
      <rect x="${padL + 50}" y="${h - padB - 85}" width="25" height="85" fill="#222" stroke="#444"/>
      <rect x="${padL + 80}" y="${h - padB - 130}" width="25" height="130" fill="#333" stroke="#fff"/>
      <rect x="${padL + 110}" y="${h - padB - 75}" width="25" height="75" fill="#222" stroke="#444"/>
      <rect x="${padL + 140}" y="${h - padB - 35}" width="25" height="35" fill="#222" stroke="#444"/>
      <text x="${padL}" y="${h - 6}">-3σ</text>
      <text x="${(w - padL - padR)/2 + padL}" y="${h - 6}" text-anchor="middle">MEDIAN: 0.0σ</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">+3σ</text>
    </svg>
  `;
  container.innerHTML = svg;
}

async function renderGenericDiag(containerId, eventType) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const w = container.clientWidth || 360;
  const h = container.clientHeight || 210;
  const padL = 35;
  const padR = 35;
  const padT = 16;
  const padB = 25;

  const svg = `
    <svg viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <rect x="${padL}" y="${padT}" width="30" height="${h - padT - padB}" fill="#240808"/>
      <rect x="${w - padR - 30}" y="${padT}" width="30" height="${h - padT - padB}" fill="#240808"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <path d="M ${padL} ${h - padB} Q ${w/2} ${padT + 10} ${w - padR} ${h - padB}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <text x="${padL + 4}" y="${padT + 12}" fill="#ff4444">-t_crit</text>
      <text x="${w - padR - 4}" y="${padT + 12}" fill="#ff4444" text-anchor="end">+t_crit</text>
      <text x="${w/2}" y="${padT + 12}" fill="var(--amber-cmd)" text-anchor="middle">REJECT H0 (p < 0.05)</text>
    </svg>
  `;
  container.innerHTML = svg;
}

// ==============================================================
// 5. SLIDERS FOR FOMC, NFP, PPI
// ==============================================================
function initFomcSliders() {
  const rRate = document.getElementById("fomc-range-rate");
  const rPriced = document.getElementById("fomc-range-priced");
  const rDots = document.getElementById("fomc-range-dots");
  const vRate = document.getElementById("fomc-val-rate");
  const vPriced = document.getElementById("fomc-val-priced");
  const vDots = document.getElementById("fomc-val-dots");
  const sSum = document.getElementById("fomc-shock-summary");

  function onFomcInput() {
    const act = parseFloat(rRate.value);
    const pr = parseFloat(rPriced.value);
    const dt = parseFloat(rDots.value);

    vRate.innerText = `${act.toFixed(2)}%`;
    vPriced.innerText = `${pr.toFixed(2)}%`;
    vDots.innerText = `${dt > 0 ? "+" : ""}${dt.toFixed(1)} bps`;

    const shockBps = (act - pr) * 100 + 0.5 * dt;
    sSum.innerText = `POLICY SHOCK: ${shockBps > 0 ? "+" : ""}${shockBps.toFixed(1)} bps (${shockBps === 0 ? "CONSENSUS PAUSE" : (shockBps > 0 ? "HAWKISH SURPRISE" : "DOVISH CUT")})`;
    renderGenericMc("canvas-fomc-mc", "fomc", shockBps / 25.0);
  }

  rRate?.addEventListener("input", onFomcInput);
  rPriced?.addEventListener("input", onFomcInput);
  rDots?.addEventListener("input", onFomcInput);
}

function initNfpSliders() {
  const rAct = document.getElementById("nfp-range-act");
  const rUnemp = document.getElementById("nfp-range-unemp");
  const rAhe = document.getElementById("nfp-range-ahe");
  const vAct = document.getElementById("nfp-val-act");
  const vUnemp = document.getElementById("nfp-val-unemp");
  const vAhe = document.getElementById("nfp-val-ahe");
  const sSum = document.getElementById("nfp-shock-summary");

  function onNfpInput() {
    const act = parseFloat(rAct.value);
    const un = parseFloat(rUnemp.value);
    const ah = parseFloat(rAhe.value);

    vAct.innerText = `${act}k`;
    vUnemp.innerText = `${un.toFixed(1)}%`;
    vAhe.innerText = `${ah.toFixed(2)}%`;

    const zNfp = (act - 175) / 50.0;
    const zUn = (un - 4.1) / 0.15;
    const zAh = (ah - 0.3) / 0.1;
    const zLabor = 0.50 * zNfp - 0.30 * zUn + 0.20 * zAh;

    sSum.innerText = `COMPOSITE LABOR Z: ${zLabor > 0 ? "+" : ""}${zLabor.toFixed(2)}σ (${zLabor > 0.8 ? "HAWKISH LABOR EXPANSION" : (zLabor < -0.8 ? "DOVISH LABOR DETERIORATION" : "IN-LINE NOISE")})`;
    renderGenericMc("canvas-nfp-mc", "nfp", zLabor);
  }

  rAct?.addEventListener("input", onNfpInput);
  rUnemp?.addEventListener("input", onNfpInput);
  rAhe?.addEventListener("input", onNfpInput);
}

function initPpiSliders() {
  const rAct = document.getElementById("ppi-range-act");
  const rRetail = document.getElementById("ppi-range-retail");
  const vAct = document.getElementById("ppi-val-act");
  const vRetail = document.getElementById("ppi-val-retail");
  const sSum = document.getElementById("ppi-shock-summary");

  function onPpiInput() {
    const act = parseFloat(rAct.value);
    const ret = parseFloat(rRetail.value);

    vAct.innerText = `${act.toFixed(2)}%`;
    vRetail.innerText = `${ret.toFixed(2)}%`;

    const zPpi = (act - 0.2) / 0.15;
    const zRet = (ret - 0.4) / 0.25;
    const zDemand = 0.55 * zPpi + 0.45 * zRet;

    sSum.innerText = `DEMAND COMPOSITE Z: ${zDemand > 0 ? "+" : ""}${zDemand.toFixed(2)}σ (${zDemand > 0.8 ? "STAGFLATIONARY COST-PUSH" : (zDemand < -0.8 ? "RECESSIONARY DEMAND COOLING" : "BALANCED")})`;
    renderGenericMc("canvas-ppi-mc", "ppi", zDemand);
  }

  rAct?.addEventListener("input", onPpiInput);
  rRetail?.addEventListener("input", onPpiInput);
}

// ==============================================================
// 6. MODULE 05: DAILY A* SETUP, PROGRESS GAUGE & AUDIT LEDGER
// ==============================================================

function initTtlCountdown(serverExpiresAtEpoch) {
  if (!serverExpiresAtEpoch) return;
  const timerEl = document.getElementById("astar-ttl-timer");
  const decayEl = document.getElementById("astar-alpha-decay");
  const statusEl = document.getElementById("astar-ttl-status");

  function updateClock() {
    const nowSec = Math.floor(Date.now() / 1000);
    const remaining = serverExpiresAtEpoch - nowSec;

    if (remaining <= 0) {
      if (timerEl) { timerEl.innerText = "00m 00s (EXPIRED)"; timerEl.style.color = "#ff4444"; }
      if (decayEl) { decayEl.innerText = "100% (ALPHA EXPIRED)"; decayEl.style.color = "#ff4444"; }
      if (statusEl) { statusEl.innerText = "EXPIRED // ORDER CANCELLED"; statusEl.style.color = "#ff4444"; }
      clearInterval(ttlTimerInterval);
      return;
    }

    const m = Math.floor(remaining / 60);
    const s = remaining % 60;
    if (timerEl) {
      timerEl.innerText = `${m}m ${s < 10 ? '0' : ''}${s}s`;
      timerEl.style.color = m < 15 ? "var(--amber-cmd)" : "var(--fg-bright)";
    }

    const maxValSec = 5400; // 90 mins
    const decayPct = Math.min(100, Math.max(0, (1.0 - Math.exp(-1.5 * (maxValSec - remaining) / maxValSec)) * 100));
    if (decayEl) decayEl.innerText = `${decayPct.toFixed(1)}%`;
    if (statusEl) statusEl.innerText = m < 15 ? "TTL EXPIRING SOON" : "ACTIVE VALID SETUP";
  }

  clearInterval(ttlTimerInterval);
  updateClock();
  ttlTimerInterval = setInterval(updateClock, 1000);
}

async function fetchAstarData() {
  try {
    const [aRes, dRes] = await Promise.all([
      fetch("/api/daily-a-star"),
      fetch("/api/dynamic-levels")
    ]);
    const aStar = await aRes.json();
    const dyn = await dRes.json();
    if (!aStar) return;

    // Pivots
    if (dyn) {
      const res = dyn.resistance || {};
      const sup = dyn.support || {};
      if (document.getElementById("ast-r3")) document.getElementById("ast-r3").innerText = `$${(res.r3 || 4078.60).toFixed(2)}`;
      if (document.getElementById("ast-r2")) document.getElementById("ast-r2").innerText = `$${(res.r2 || 4059.60).toFixed(2)}`;
      if (document.getElementById("ast-r1")) document.getElementById("ast-r1").innerText = `$${(res.r1 || 4058.64).toFixed(2)}`;
      if (document.getElementById("ast-pp")) document.getElementById("ast-pp").innerText = `$${(dyn.pivot_point || 4046.30).toFixed(2)}`;
      if (document.getElementById("ast-s1")) document.getElementById("ast-s1").innerText = `$${(sup.s1 || 4035.18).toFixed(2)}`;
      if (document.getElementById("ast-s2")) document.getElementById("ast-s2").innerText = `$${(sup.s2 || 4026.00).toFixed(2)}`;
      if (document.getElementById("ast-s3")) document.getElementById("ast-s3").innerText = `$${(sup.s3 || 4017.51).toFixed(2)}`;
    }

    // Ticket & Action
    const ticket = aStar.trade_ticket || {};
    const isAStar = aStar.is_a_star;
    const scoreBadge = document.getElementById("astar-grade-badge");
    const ticketLabel = document.getElementById("astar-ticket-label");
    const ticketPane = document.getElementById("astar-ticket-pane");
    const actionEl = document.getElementById("astar-action");

    if (scoreBadge) scoreBadge.innerText = `${aStar.confluence_score}/100`;
    if (document.getElementById("ast-score")) document.getElementById("ast-score").innerText = `${aStar.confluence_score}/100 (${aStar.grade})`;

    if (actionEl) {
      actionEl.innerText = `ACTION: ${ticket.action || "MOMENTUM BUY CATCH / LONG EXPANSION ON M30 IMPULSE"}`;
    }

    if (!isAStar) {
      if (scoreBadge) scoreBadge.style.color = "var(--amber-cmd)";
      if (ticketLabel) ticketLabel.innerText = `A* GRADE NOT EXIST YET // STAND ASIDE (${aStar.confluence_score}/100)`;
      if (ticketPane) { ticketPane.style.borderColor = "var(--amber-dim)"; ticketPane.style.borderStyle = "dashed"; }
    } else {
      if (scoreBadge) scoreBadge.style.color = "#ffffff";
      if (ticketPane) { ticketPane.style.borderStyle = "solid"; ticketPane.style.borderColor = "var(--amber-cmd)"; }
      const isBuy = ticket.action && (ticket.action.includes("BUY") || ticket.action.includes("LONG"));
      if (ticketLabel) {
        ticketLabel.innerText = isBuy 
          ? `★ A* GRADE BULLISH BUY TICKET (M30/M15 CONFIRMATION // ${aStar.confluence_score}/100)`
          : `★ A* GRADE BEARISH SHORT TICKET (M30/M15 CONFIRMATION // ${aStar.confluence_score}/100)`;
      }
    }

    if (document.getElementById("astar-entry")) {
      document.getElementById("astar-entry").innerText = Array.isArray(ticket.entry_zone) ? `$${ticket.entry_zone[0].toFixed(2)} - $${ticket.entry_zone[1].toFixed(2)}` : (ticket.entry_zone || "$4,031.26 - $4,046.30");
    }
    if (document.getElementById("astar-sl")) document.getElementById("astar-sl").innerText = ticket.stop_loss || "$3,958.24";
    if (document.getElementById("astar-tp1")) document.getElementById("astar-tp1").innerText = ticket.take_profit_1 || "$4,059.60";
    if (document.getElementById("astar-tp2")) document.getElementById("astar-tp2").innerText = ticket.take_profit_2 || "$4,078.60";
    if (document.getElementById("astar-lot")) document.getElementById("astar-lot").innerText = `${ticket.recommended_lot_size_1000_equity || 0.03} LOT ($30.00 / 3.0% Max Risk)`;

    // Position Progress Bar Gauge
    const gaugeSl = document.getElementById("gauge-sl");
    const gaugeEntry = document.getElementById("gauge-entry");
    const gaugeLive = document.getElementById("gauge-live");
    const gaugeTp1 = document.getElementById("gauge-tp1");
    const gaugePnlUsd = document.getElementById("gauge-pnl-usd");
    const gaugePnlR = document.getElementById("gauge-pnl-r");
    const gaugeStatus = document.getElementById("gauge-status-badge");
    const gaugePin = document.getElementById("gauge-cursor-pin");

    const ls = aStar.locked_setup;
    const curP = aStar.current_price || 4030.80;

    if (ls) {
      if (gaugeSl) gaugeSl.innerText = `$${ls.stop_loss ? ls.stop_loss.toFixed(2) : "0.00"}`;
      if (gaugeEntry) {
        const entryVal = ls.fill_price ? ls.fill_price : (Array.isArray(ls.entry_zone) ? (ls.entry_zone[0] + ls.entry_zone[1]) / 2 : 0);
        gaugeEntry.innerText = `$${entryVal.toFixed(2)}`;
      }
      if (gaugeLive) gaugeLive.innerText = `$${curP.toFixed(2)}`;
      if (gaugeTp1) gaugeTp1.innerText = `$${ls.take_profit_1 ? ls.take_profit_1.toFixed(2) : "0.00"}`;

      const pnlUsd = ls.floating_pnl_usd || 0;
      const pnlR = ls.floating_r || 0;
      if (gaugePnlUsd) {
        gaugePnlUsd.innerText = `${pnlUsd >= 0 ? "+" : ""}$${pnlUsd.toFixed(2)} USD`;
        gaugePnlUsd.style.color = pnlUsd >= 0 ? "#ffffff" : "#ff4444";
      }
      if (gaugePnlR) {
        gaugePnlR.innerText = `${pnlR >= 0 ? "+" : ""}${pnlR.toFixed(2)} R`;
        gaugePnlR.style.color = pnlR >= 0 ? "#ffffff" : "#ff4444";
      }
      if (gaugeStatus) {
        gaugeStatus.innerText = ls.status || "TRIGGERED_RUNNING";
        gaugeStatus.style.color = ls.status && ls.status.includes("TP") ? "#ffffff" : (ls.status && ls.status.includes("SL") ? "#ff4444" : "var(--amber-cmd)");
      }

      const range = Math.abs((ls.take_profit_1 || 0) - (ls.stop_loss || 0));
      let pct = 35;
      if (range > 0) {
        if (ls.direction === "BUY") {
          pct = Math.max(0, Math.min(100, ((curP - ls.stop_loss) / range) * 100));
        } else {
          pct = Math.max(0, Math.min(100, ((ls.stop_loss - curP) / range) * 100));
        }
      }
      if (gaugePin) gaugePin.style.left = `${pct}%`;
    }

    // TTL
    if (aStar.time_to_live && aStar.time_to_live.expires_at_epoch) {
      initTtlCountdown(aStar.time_to_live.expires_at_epoch);
    }

    // Render A* SDE Monte Carlo fan
    renderGenericMc("canvas-astar-mc", "daily", 0.0);

    fetchGlobalMacroMatrix();
    renderSetupLedgerTable();
  } catch (err) {
    console.warn("fetchAstarData error:", err);
  }
}

async function renderSetupLedgerTable() {
  const tbody = document.getElementById("astar-ledger-tbody");
  if (!tbody) return;
  try {
    const res = await fetch("/api/setup-ledger");
    const data = await res.json();
    const rows = [];
    if (data.active_setup) rows.push(data.active_setup);
    if (data.completed_setups) rows.push(...data.completed_setups);

    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color:#888;">NO SETUPS LOGGED YET</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(s => {
      const isWin = s.status && (s.status.includes("TP") || (s.realized_r && s.realized_r > 0));
      const isLoss = s.status && (s.status.includes("SL") || (s.realized_r && s.realized_r < 0));
      const statusColor = isWin ? "#ffffff" : (isLoss ? "#ff4444" : "var(--amber-cmd)");
      const rStr = s.realized_r !== null && s.realized_r !== undefined ? `${s.realized_r >= 0 ? "+" : ""}${s.realized_r.toFixed(2)} R` : "0.00 R (RUNNING)";
      const entryStr = Array.isArray(s.entry_zone) ? `$${s.entry_zone[0].toFixed(2)} - $${s.entry_zone[1].toFixed(2)}` : (s.entry_zone || "N/A");
      const slStr = s.stop_loss ? `$${s.stop_loss.toFixed(2)}` : "N/A";
      const tpStr = s.take_profit_1 ? `$${s.take_profit_1.toFixed(2)}` : "N/A";

      return `
        <tr>
          <td style="font-family:var(--font-mono); font-size:9.5px;">${s.created_at_utc || "TODAY"}</td>
          <td><strong style="color:var(--amber-cmd); font-family:var(--font-mono);">${s.setup_id}</strong></td>
          <td>${s.event_type} (${s.asset || 'XAUUSD'})</td>
          <td><strong style="color:${s.direction === 'BUY' ? '#ffffff' : '#ff4444'}">${s.direction}</strong></td>
          <td style="font-family:var(--font-mono);">${entryStr}</td>
          <td style="color:#ff4444; font-family:var(--font-mono);">${slStr}</td>
          <td style="color:#ffffff; font-family:var(--font-mono);">${tpStr}</td>
          <td><span style="color:${statusColor}; font-weight:800; border:1px solid ${statusColor}; padding:1px 5px; font-size:8.5px;">${s.status}</span></td>
          <td style="color:${statusColor}; font-weight:900; font-family:var(--font-mono);">${rStr}</td>
          <td style="font-size:9.5px; color:#aaa;">${s.resolution || "Active in session"}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.warn("renderSetupLedgerTable error:", err);
  }
}

async function fetchGlobalMacroMatrix() {
  const tbody = document.getElementById("global-macro-tbody");
  if (!tbody) return;
  try {
    const res = await fetch("/api/global-macro-matrix");
    const data = await res.json();
    const sovereigns = data.sovereigns || [];
    tbody.innerHTML = sovereigns.map(s => `
      <tr>
        <td><strong>${s.country}</strong> (${s.region})</td>
        <td style="font-family:var(--font-mono);">${s.policy_rate}%</td>
        <td style="font-family:var(--font-mono); color:${s.inflation_cpi > 3 ? '#ff4444' : '#ffffff'};">${s.inflation_cpi}%</td>
        <td style="font-family:var(--font-mono);">${s.ten_year_yield}%</td>
        <td style="font-family:var(--font-mono);">${s.real_yield > 0 ? '+' : ''}${s.real_yield}%</td>
        <td style="font-family:var(--font-mono); color:var(--amber-cmd);">${s.gold_transmission_beta > 0 ? '+' : ''}${s.gold_transmission_beta}</td>
        <td style="font-family:var(--font-mono);">${s.global_debasement_index} / 100</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchGlobalMacroMatrix error:", e);
  }
}

function initAutoRefreshLoop() {
  setInterval(async () => {
    try {
      const res = await fetch("/api/refresh-all", { method: "POST" });
      const data = await res.json();
      if (data && data.prices) {
        const p = data.prices;
        animateNumber(document.getElementById("tick-gold"), p["GC=F"], "$", "", 2);
        animateNumber(document.getElementById("tick-dxy"), p["DX-Y.NYB"], "", "", 2);
        animateNumber(document.getElementById("tick-2y"), p["ZT=F"], "", "%", 2);
        animateNumber(document.getElementById("tick-10y"), p["^TNX"], "", "%", 2);
        animateNumber(document.getElementById("tick-oil"), p["CL=F"], "$", "", 2);
      }
      if (data && data.daily_a_star) {
        fetchAstarData();
      }
    } catch (e) {}
  }, 6000);

  const btnReset = document.getElementById("btn-reset-astar-session");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      btnReset.innerText = "WAIT REFRESHING...";
      try {
        await fetch("/api/reset-setup-session", { method: "POST" });
        await fetchAstarData();
      } catch (e) {}
      setTimeout(() => { btnReset.innerText = "FORCE RE-EVALUATE SESSION <GO>"; }, 800);
    });
  }
}

// ==============================================================
// 7. DEEPSEEK REASONING TERMINAL
// ==============================================================
function initDeepSeekControls() {
  const btnPre = document.getElementById("btn-ai-pre");
  const btnPost = document.getElementById("btn-ai-post");
  const btnRun = document.getElementById("btn-run-deepseek");
  const outputEl = document.getElementById("ai-output-terminal");
  const statusEl = document.getElementById("ai-status-badge");
  const eventSel = document.getElementById("ai-event-select");

  btnPre?.addEventListener("click", (e) => {
    setBtnActive(e.target, "#btn-ai-pre, #btn-ai-post");
    currentAiMode = "pre_news";
  });

  btnPost?.addEventListener("click", (e) => {
    setBtnActive(e.target, "#btn-ai-pre, #btn-ai-post");
    currentAiMode = "post_news";
  });

  btnRun?.addEventListener("click", async () => {
    btnRun.innerText = "EXECUTING DEEPSEEK...";
    if (statusEl) statusEl.innerText = "QUERYING AI KERNEL...";
    if (outputEl) outputEl.innerText = `>>> CONNECTING TO DEEPSEEK REASONING ENGINE...\n>>> COMPILING AUTHENTIC MACRO CONTEXT & VOLATILITY MATRICES...`;

    try {
      const res = await fetch("/api/deepseek-briefing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: currentAiMode,
          event_type: eventSel.value,
          actual_mom: parseFloat(document.getElementById("bbg-range-headline")?.value || 0.4),
          actual_core_yoy: parseFloat(document.getElementById("bbg-range-core")?.value || 3.4)
        })
      });
      const data = await res.json();
      if (outputEl) outputEl.innerText = data.memo || data.briefing || "No response received.";
      if (statusEl) statusEl.innerText = `COMPLETED // ${data.status}`;
    } catch (e) {
      if (outputEl) outputEl.innerText = `>>> CONNECTION ERROR: ${e}`;
      if (statusEl) statusEl.innerText = "OFFLINE FALLBACK ACTIVE";
    } finally {
      btnRun.innerText = "RUN DEEPSEEK REASONING <GO>";
    }
  });
}

// ==============================================================
// 8. YoY COMPONENT TIMELINE
// ==============================================================
function initYoyControls() {
  const pHL = document.getElementById("pill-hl");
  const pCore = document.getElementById("pill-core");
  const pSuper = document.getElementById("pill-super");
  const pShelter = document.getElementById("pill-shelter");

  pHL?.addEventListener("click", () => { yoySeries.headline = !yoySeries.headline; pHL.classList.toggle("active"); renderYoyTimeline(); });
  pCore?.addEventListener("click", () => { yoySeries.core = !yoySeries.core; pCore.classList.toggle("active"); renderYoyTimeline(); });
  pSuper?.addEventListener("click", () => { yoySeries.supercore = !yoySeries.supercore; pSuper.classList.toggle("active"); renderYoyTimeline(); });
  pShelter?.addEventListener("click", () => { yoySeries.shelter = !yoySeries.shelter; pShelter.classList.toggle("active"); renderYoyTimeline(); });
}

async function renderYoyTimeline() {
  const container = document.getElementById("canvas-yoy");
  const hudEl = document.getElementById("hud-yoy");
  if (!container) return;

  let data = null;
  try {
    const res = await fetch("/api/yoy-timeline");
    data = await res.json();
  } catch (e) {
    console.warn("yoy error:", e);
    return;
  }

  const series = data.timeline.series;
  const dates = data.timeline.dates;
  const n = dates.length;
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 360;
  const padL = 40;
  const padR = 40;
  const padT = 20;
  const padB = 30;

  const allVals = [];
  if (yoySeries.headline) allVals.push(...series.headline_yoy);
  if (yoySeries.core) allVals.push(...series.core_yoy);
  if (yoySeries.supercore) allVals.push(...series.supercore_yoy);
  if (yoySeries.shelter) allVals.push(...series.shelter_yoy);

  const minY = Math.min(...allVals, 1.5) - 0.5;
  const maxY = Math.max(...allVals, 7.0) + 0.5;

  function getX(i) { return padL + (i / (n - 1)) * (w - padL - padR); }
  function getY(v) { return padT + (1 - (v - minY) / (maxY - minY)) * (h - padT - padB); }

  function buildPath(arr) {
    return arr.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(v).toFixed(1)}`).join(" ");
  }

  let pathsSvg = "";
  if (yoySeries.headline) pathsSvg += `<path d="${buildPath(series.headline_yoy)}" fill="none" stroke="#ffffff" stroke-width="2.2"/>`;
  if (yoySeries.core) pathsSvg += `<path d="${buildPath(series.core_yoy)}" fill="none" stroke="var(--amber-cmd)" stroke-width="1.8"/>`;
  if (yoySeries.supercore) pathsSvg += `<path d="${buildPath(series.supercore_yoy)}" fill="none" stroke="#00e5ff" stroke-width="1.5" stroke-dasharray="3,3"/>`;
  if (yoySeries.shelter) pathsSvg += `<path d="${buildPath(series.shelter_yoy)}" fill="none" stroke="#ff4444" stroke-width="1.5"/>`;

  let ticksSvg = "";
  for (let i = 0; i < n; i += 6) {
    ticksSvg += `<text x="${getX(i)}" y="${h - 8}" text-anchor="middle">${dates[i]}</text>`;
  }

  const yTarget = getY(2.0);
  const svg = `
    <svg id="svg-yoy" viewBox="0 0 ${w} ${h}" class="bbg-svg">
      <line x1="${padL}" y1="${yTarget}" x2="${w - padR}" y2="${yTarget}" stroke="#444" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${padL + 4}" y="${yTarget - 4}" fill="#666">FED 2.0% TARGET</text>

      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      ${pathsSvg}
      ${ticksSvg}

      <line id="yoy-scrubber" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none;"/>
    </svg>
  `;
  container.innerHTML = svg;

  const svgEl = document.getElementById("svg-yoy");
  const scrub = document.getElementById("yoy-scrubber");
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((ev.clientX - rect.left) / rect.width) * w));
      const idx = Math.max(0, Math.min(n - 1, Math.round(((mouseX - padL) / (w - padL - padR)) * (n - 1))));

      hudEl.innerHTML = `${dates[idx]} | Headline: <span class="highlight">${series.headline_yoy[idx].toFixed(2)}%</span> | Core: <span class="highlight">${series.core_yoy[idx].toFixed(2)}%</span> | Supercore: ${series.supercore_yoy[idx].toFixed(2)}% | Shelter: ${series.shelter_yoy[idx].toFixed(2)}%`;

      if (scrub) {
        scrub.style.display = "block";
        scrub.setAttribute("x1", getX(idx));
        scrub.setAttribute("x2", getX(idx));
        scrub.setAttribute("y1", padT);
        scrub.setAttribute("y2", h - padB);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      hudEl.innerText = "HOVER DATE TICK // HEADLINE, CORE, SUPERCORE, SHELTER (2023-2026)";
      if (scrub) scrub.style.display = "none";
    });
  }
}

// ==============================================================
// 9. DATA FETCHERS (AUDIT, CALENDAR, LIT, FOMC, NFP, PPI)
// ==============================================================
async function fetchAuditLedger() {
  try {
    const res = await fetch("/api/historical");
    const data = await res.json();
    const tbody = document.getElementById("bbg-audit-tbody");
    if (!tbody || !data.events) return;

    tbody.innerHTML = data.events.map(e => `
      <tr>
        <td><strong>${e.date}</strong></td>
        <td>${e.ref_month}</td>
        <td>${e.actual_mom.toFixed(2)}%</td>
        <td>${e.forecast_mom.toFixed(2)}%</td>
        <td>${e.actual_core_yoy.toFixed(2)}%</td>
        <td style="color:${e.z_score_composite > 0 ? 'var(--amber-cmd)' : '#fff'}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${e.z_score_composite.toFixed(2)}σ</td>
        <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${e.gold_return_pct.toFixed(2)}%</td>
        <td>${e.dxy_return_pct > 0 ? '+' : ''}${e.dxy_return_pct.toFixed(2)}%</td>
        <td style="color:var(--amber-cmd);">${e.zt_return_pct > 0 ? '+' : ''}${(-e.zt_return_pct).toFixed(2)}%</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchAuditLedger error:", e);
  }
}

async function fetchFomcData() {
  try {
    const res = await fetch("/api/fomc-historical");
    const data = await res.json();
    const tbody = document.getElementById("fomc-tbody");
    if (!tbody || !data.events) return;

    tbody.innerHTML = data.events.map(e => `
      <tr>
        <td><strong>${e.date}</strong></td>
        <td>${e.actual_rate.toFixed(2)}%</td>
        <td>${e.priced_rate.toFixed(2)}%</td>
        <td style="color:${e.total_policy_shock_bps > 0 ? '#ff4444' : (e.total_policy_shock_bps < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.total_policy_shock_bps > 0 ? '+' : ''}${e.total_policy_shock_bps.toFixed(1)} bps</td>
        <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${e.gold_return_pct.toFixed(2)}%</td>
        <td style="color:var(--amber-cmd);">${e.us2y_delta_bps > 0 ? '+' : ''}${e.us2y_delta_bps.toFixed(1)} bps</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchFomcData error:", e);
  }
}

async function fetchNfpData() {
  try {
    const res = await fetch("/api/nfp-historical");
    const data = await res.json();
    const tbody = document.getElementById("nfp-tbody");
    if (!tbody || !data.events) return;

    tbody.innerHTML = data.events.map(e => `
      <tr>
        <td><strong>${e.date}</strong></td>
        <td>${e.actual_nfp}k</td>
        <td>${e.forecast_nfp}k</td>
        <td>${e.unemp_rate.toFixed(1)}%</td>
        <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${e.z_score_composite.toFixed(2)}σ</td>
        <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${e.gold_return_pct.toFixed(2)}%</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchNfpData error:", e);
  }
}

async function fetchPpiData() {
  try {
    const res = await fetch("/api/ppi-historical");
    const data = await res.json();
    const tbody = document.getElementById("ppi-tbody");
    if (!tbody || !data.events) return;

    tbody.innerHTML = data.events.map(e => `
      <tr>
        <td><strong>${e.date}</strong></td>
        <td>${e.actual_ppi.toFixed(1)}%</td>
        <td>${e.actual_retail.toFixed(1)}%</td>
        <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${e.z_score_composite.toFixed(2)}σ</td>
        <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${e.gold_return_pct.toFixed(2)}%</td>
        <td>${e.dxy_return_pct > 0 ? '+' : ''}${e.dxy_return_pct.toFixed(2)}%</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchPpiData error:", e);
  }
}

async function fetchCalendarData() {
  try {
    const res = await fetch("/api/high-impact-calendar");
    const data = await res.json();
    const tbody = document.getElementById("cal-tbody");
    if (!tbody || !Array.isArray(data)) return;

    tbody.innerHTML = data.map(ev => `
      <tr>
        <td><strong>${ev.date}</strong></td>
        <td>${ev.time}</td>
        <td><strong style="color:var(--amber-cmd);">${ev.title}</strong></td>
        <td><span style="color:${ev.impact === 'MAXIMUM' ? '#ff4444' : '#ffffff'}; font-weight:800;">[ ${ev.impact} ]</span></td>
        <td>${ev.forecast || '-'}</td>
        <td style="color:#777;">${ev.previous || '-'}</td>
        <td style="color:#fff;">${ev.effect_xauusd}</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchCalendarData error:", e);
  }
}

async function fetchAcademicLit() {
  try {
    const res = await fetch("/api/academic-literature");
    const data = await res.json();
    const tbody = document.getElementById("lit-tbody");
    if (!tbody || !Array.isArray(data)) return;

    tbody.innerHTML = data.map(p => `
      <tr>
        <td><strong>${p.author || p.citation}</strong> (${p.year || 2003})</td>
        <td style="color:var(--amber-cmd);">${p.title}</td>
        <td>${p.methodology || p.journal}</td>
        <td>${p.findings || p.empirical_finding}</td>
        <td style="color:#fff;">${p.terminal_application || 'Econometric Foundation'}</td>
      </tr>
    `).join("");
  } catch (e) {
    console.warn("fetchAcademicLit error:", e);
  }
}

document.addEventListener("DOMContentLoaded", init);
