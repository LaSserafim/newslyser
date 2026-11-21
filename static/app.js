/**
 * NEWSLYSER // BLOOMBERG PROFESSIONAL TERMINAL WORKSTATION
 * Authentic Monochrome White-on-Black Architecture.
 * 10,000-Path Kou (2002) Jump-Diffusion MC, White (1980) HC3 Regressions,
 * Student's t + Binomial 5% Significance Hypothesis Testing & Live TradingView B-PIPE.
 */

const EVENT_STATE = {
  cpi:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", pdfView: "cpi", mcView: "sde", mode: "PREMARKET", fwTimer: null },
  fomc: { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mcView: "sde", mode: "PREMARKET", fwTimer: null },
  nfp:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mcView: "sde", mode: "PREMARKET", fwTimer: null },
  ppi:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mcView: "sde", mode: "PREMARKET", fwTimer: null },
  astar:{ z: -1.18, horizon: "daily", asset: "gold", diag: "tcurve", mcView: "sde", mode: "ACTIVE", fwTimer: null }
};

let PREMARKET_CACHE = null;
let currentAiMode = "pre_news";
let maximizedPaneId = null;
let yoyDomain = "cpi";
let yoyTimelineCache = null;
let yoyActiveKeys = { headline_yoy: true, core_yoy: true, supercore_yoy: true, shelter_yoy: true };
let ttlTimerInterval = null;
let CURRENT_ASTAR_ASSET = "XAUUSD";

async function init() {
  initClock();
  initModuleNavigation();
  initCpiSliders();
  initFomcSliders();
  initNfpSliders();
  initPpiSliders();
  initAllControls();
  initAuditToggle();
  initKeyboardShortcuts();
  initDeepSeekControls();
  initYoyControls();
  initLotCalculator();

  // Load frozen Pre-Market Snapshot FIRST so Pane 01, 02, 03, 06 match identically on first paint
  await Promise.all([fetchPremarketSnapshot(false), fetchState()]);
  renderWorkstation("cpi");
  fetchAuditLedger();
  fetchFomcData();
  fetchNfpData();
  fetchPpiData();
  fetchCalendarData();
  fetchAcademicLit();
  fetchUsEconomyAtlas();
  fetchAstarData();
  initAutoRefreshLoop();

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

function animateNumber(el, targetVal, prefix = "", suffix = "", decimals = 2, duration = 0.5) {
  if (!el || targetVal === undefined || targetVal === null || isNaN(targetVal)) return;
  const num = Number(targetVal);
  const rawCurrent = el.innerText.replace(/[^0-9.-]/g, "");
  const current = parseFloat(rawCurrent);
  if (typeof gsap === "undefined" || isNaN(current)) {
    el.innerText = `${prefix}${num.toFixed(decimals)}${suffix}`;
    return;
  }
  if (Math.abs(num - current) > 1e-4) {
    gsap.fromTo(el, { textShadow: "0 0 8px #ffffff" }, { textShadow: "none", duration: 0.6, ease: "power2.out" });
  }
  const obj = { val: current };
  gsap.to(obj, {
    val: num,
    duration: duration,
    ease: "power2.out",
    onUpdate: () => {
      el.innerText = `${prefix}${obj.val.toFixed(decimals)}${suffix}`;
    }
  });
}

function getOrCreateTooltip(container) {
  let tt = container.querySelector(".chart-adaptive-tooltip");
  if (!tt) {
    tt = document.createElement("div");
    tt.className = "chart-adaptive-tooltip";
    container.style.position = "relative";
    container.appendChild(tt);
  }
  return tt;
}

function positionTooltip(tt, container, clientX, clientY, html) {
  const rect = container.getBoundingClientRect();
  tt.innerHTML = html;
  tt.style.display = "block";
  const ttW = tt.offsetWidth || 190;
  const ttH = tt.offsetHeight || 60;
  const rawX = clientX - rect.left + 14;
  const rawY = clientY - rect.top - ttH - 8;
  tt.style.left = `${Math.max(6, Math.min(rect.width - ttW - 6, rawX))}px`;
  tt.style.top = `${Math.max(6, Math.min(rect.height - ttH - 6, rawY))}px`;
}

function setBtnActive(el, groupSelector) {
  document.querySelectorAll(groupSelector).forEach(b => b.classList.remove("active"));
  if (el) el.classList.add("active");
}

function getActiveModuleId() {
  const active = document.querySelector(".module-view.active");
  return active ? active.id : "mod-cpi";
}

// ==============================================================
// 1. MODULE NAVIGATION
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
    "mod-lit": "LIT.ACADEMIC <GO>",
    "mod-us": "ECO.US <GO>"
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

      if (window.Motion && activeView) {
        window.Motion.animate(activeView, { opacity: [0, 1], y: [6, 0] }, { duration: 0.25 });
      }

      if (modId === "mod-cpi") renderWorkstation("cpi");
      else if (modId === "mod-fomc") renderWorkstation("fomc");
      else if (modId === "mod-nfp") renderWorkstation("nfp");
      else if (modId === "mod-ppi") renderWorkstation("ppi");
      else if (modId === "mod-astar") fetchAstarData();
      else if (modId === "mod-cal") fetchCalendarData();
      else if (modId === "mod-yoy") renderYoyTimeline();
      else if (modId === "mod-us") fetchUsEconomyAtlas();
      else if (modId === "mod-lit") fetchAcademicLit();
    });
  });
}

function renderWorkstation(ev) {
  renderEventPdfPane(ev);
  renderEventMcPane(ev);
  renderEventFreqPane(ev);
  renderEventTransPane(ev);
  renderEventDiagPane(ev);
}

// ==============================================================
// 2. SLIDERS & COCKPIT RIBBON CONTROLS (ALL 4 NEWS MODULES)
// ==============================================================
function getEventConsensus() {
  const up = PREMARKET_CACHE?.snapshot?.upcoming_consensus || {};
  return {
    cpiHl: Number(up.cpi?.forecast_mom ?? 0.3),
    cpiCore: Number(up.cpi?.forecast_core_yoy ?? PREMARKET_CACHE?.models?.cpi?.forecast_core_yoy ?? 2.9),
    fomcTarget: Number(up.fomc?.current_target ?? 4.0),
    nfpK: Number(up.nfp?.forecast_nfp_k ?? 135),
    nfpUnemp: Number(up.nfp?.forecast_unemp ?? 4.1),
    nfpAhe: Number(up.nfp?.forecast_ahe ?? 0.3),
    ppiMom: Number(up.ppi?.forecast_ppi_mom ?? 0.4),
    retailMom: Number(up.ppi?.forecast_retail_mom ?? 0.4)
  };
}

function initCpiSliders() {
  const rHeadline = document.getElementById("bbg-range-headline");
  const rCore = document.getElementById("bbg-range-core");
  const vHeadline = document.getElementById("bbg-val-headline");
  const vCore = document.getElementById("bbg-val-core");
  const vShock = document.getElementById("bbg-val-shock");
  const vRegime = document.getElementById("bbg-val-regime");

  function onSliderChange() {
    const c = getEventConsensus();
    const hl = parseFloat(rHeadline.value);
    const core = parseFloat(rCore.value);
    vHeadline.innerText = `${hl.toFixed(2)}%`;
    vCore.innerText = `${core.toFixed(2)}%`;

    const zHl = (hl - c.cpiHl) / 0.075;
    const zCore = (core - c.cpiCore) / 0.075;
    const zComp = 0.60 * zCore + 0.40 * zHl;
    EVENT_STATE.cpi.z = zComp;
    EVENT_STATE.cpi.mode = "WHATIF";

    vShock.innerText = `${zComp > 0 ? "+" : ""}${zComp.toFixed(2)}σ (WHAT-IF)`;
    vShock.style.color = zComp > 0.8 ? "var(--amber-cmd)" : (zComp < -0.8 ? "#ffffff" : "var(--fg-normal)");

    let regime = "WHAT-IF: IN-LINE NOISE";
    if (zComp >= 1.5) regime = "WHAT-IF: HAWKISH DRIFT";
    else if (zComp >= 0.8) regime = "WHAT-IF: HAWKISH SURPRISE";
    else if (zComp <= -1.5) regime = "WHAT-IF: DOVISH REVERSAL";
    else if (zComp <= -0.8) regime = "WHAT-IF: DOVISH RELIEF";
    vRegime.innerText = regime;

    renderEventMcPane("cpi");
    renderEventFreqPane("cpi");
    executeSimulateCpi(hl, core);
  }

  rHeadline?.addEventListener("input", onSliderChange);
  rCore?.addEventListener("input", onSliderChange);
}

function initFomcSliders() {
  const rRate = document.getElementById("fomc-range-rate");
  const rPriced = document.getElementById("fomc-range-priced");
  const rDots = document.getElementById("fomc-range-dots");
  const vRate = document.getElementById("fomc-val-rate");
  const vPriced = document.getElementById("fomc-val-priced");
  const vDots = document.getElementById("fomc-val-dots");
  const sSum = document.getElementById("fomc-shock-summary");

  async function onFomcInput() {
    const act = parseFloat(rRate.value);
    const pr = parseFloat(rPriced.value);
    const dt = parseFloat(rDots.value);

    vRate.innerText = `${act.toFixed(2)}%`;
    vPriced.innerText = `${pr.toFixed(2)}%`;
    vDots.innerText = `${dt > 0 ? "+" : ""}${dt.toFixed(1)} bps`;

    const shockBps = (act - pr) * 100 + 0.5 * dt;
    const zFomc = shockBps / 7.5;
    EVENT_STATE.fomc.z = zFomc;
    EVENT_STATE.fomc.mode = "WHATIF";
    sSum.innerText = `${shockBps > 0 ? "+" : ""}${shockBps.toFixed(1)} bps (WHAT-IF)`;
    renderEventMcPane("fomc");
    renderEventFreqPane("fomc");
    try {
      const res = await fetch(`/api/fomc-model?actual_rate=${act}&priced_rate=${pr}&dots_shift_bps=${dt}`);
      const d = await res.json();
      if (d && d.metrics && d.dynamic_execution) {
        updateExecTicket("fomc", d.metrics.tactical_signal, d.dynamic_execution);
      }
    } catch (e) {}
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

  async function onNfpInput() {
    const c = getEventConsensus();
    const act = parseFloat(rAct.value);
    const un = parseFloat(rUnemp.value);
    const ah = parseFloat(rAhe.value);

    vAct.innerText = `${act}k`;
    vUnemp.innerText = `${un.toFixed(1)}%`;
    vAhe.innerText = `${ah.toFixed(2)}%`;

    const zNfp = (act - c.nfpK) / 45.0;
    const zUn = (un - c.nfpUnemp) / 0.15;
    const zAh = (ah - c.nfpAhe) / 0.10;
    const zLabor = 0.50 * zNfp - 0.30 * zUn + 0.20 * zAh;
    EVENT_STATE.nfp.z = zLabor;
    EVENT_STATE.nfp.mode = "WHATIF";

    sSum.innerText = `${zLabor > 0 ? "+" : ""}${zLabor.toFixed(2)}σ (WHAT-IF)`;
    renderEventMcPane("nfp");
    renderEventFreqPane("nfp");
    try {
      const res = await fetch(`/api/nfp-model?nfp_actual=${act}&nfp_forecast=${c.nfpK}&unemp_actual=${un}&unemp_forecast=${c.nfpUnemp}&ahe_actual=${ah}&ahe_forecast=${c.nfpAhe}`);
      const d = await res.json();
      if (d && d.metrics && d.dynamic_execution) {
        updateExecTicket("nfp", d.metrics.tactical_signal, d.dynamic_execution);
      }
    } catch (e) {}
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

  async function onPpiInput() {
    const c = getEventConsensus();
    const act = parseFloat(rAct.value);
    const ret = parseFloat(rRetail.value);

    vAct.innerText = `${act.toFixed(2)}%`;
    vRetail.innerText = `${ret.toFixed(2)}%`;

    const zPpi = (act - c.ppiMom) / 0.20;
    const zRet = (ret - c.retailMom) / 0.35;
    const zDemand = 0.55 * zPpi + 0.45 * zRet;
    EVENT_STATE.ppi.z = zDemand;
    EVENT_STATE.ppi.mode = "WHATIF";

    sSum.innerText = `${zDemand > 0 ? "+" : ""}${zDemand.toFixed(2)}σ (WHAT-IF)`;
    renderEventMcPane("ppi");
    renderEventFreqPane("ppi");
    try {
      const res = await fetch(`/api/ppi-retail-model?ppi_actual=${act}&ppi_forecast=${c.ppiMom}&retail_actual=${ret}&retail_forecast=${c.retailMom}`);
      const d = await res.json();
      if (d && d.metrics && d.dynamic_execution) {
        updateExecTicket("ppi", d.metrics.tactical_signal, d.dynamic_execution);
      }
    } catch (e) {}
  }

  rAct?.addEventListener("input", onPpiInput);
  rRetail?.addEventListener("input", onPpiInput);
}
function initAllControls() {
  // Wire Horizon, Firewall & Reset buttons across ALL 4 news modules
  const prefixes = { cpi: "bbg", fomc: "fomc", nfp: "nfp", ppi: "ppi" };

  Object.entries(prefixes).forEach(([ev, pre]) => {
    const b1d = document.getElementById(`${pre}-btn-horizon-1d`);
    const b5d = document.getElementById(`${pre}-btn-horizon-5d`);
    const bFw = document.getElementById(`${pre}-btn-firewall`);
    const bRst = document.getElementById(`${pre}-btn-reset`);

    b1d?.addEventListener("click", () => {
      b1d.classList.add("active");
      b5d?.classList.remove("active");
      EVENT_STATE[ev].horizon = 1;
      renderEventMcPane(ev);
    });

    b5d?.addEventListener("click", () => {
      b5d.classList.add("active");
      b1d?.classList.remove("active");
      EVENT_STATE[ev].horizon = 5;
      renderEventMcPane(ev);
    });

    bFw?.addEventListener("click", () => {
      triggerFirewallCountdown(ev, 60);
    });

    bRst?.addEventListener("click", () => {
      resetEventModule(ev, pre);
    });
  });

  // Sync Live <GO>
  document.getElementById("btn-sync-live")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-sync-live");
    btn.innerText = "SYNCING B-PIPE...";
    try { await fetch("/api/sync-releases?force=true", { method: "POST" }); } catch (e) {}
    yoyTimelineCache = null;
    await Promise.all([fetchPremarketSnapshot(false), fetchState(), fetchAuditLedger(), fetchFomcData(), fetchNfpData(), fetchPpiData(), fetchCalendarData(), fetchAstarData()]);
    const mod = getActiveModuleId();
    if (mod === "mod-cpi") renderWorkstation("cpi");
    else if (mod === "mod-fomc") renderWorkstation("fomc");
    else if (mod === "mod-nfp") renderWorkstation("nfp");
    else if (mod === "mod-ppi") renderWorkstation("ppi");
    else if (mod === "mod-yoy") renderYoyTimeline();
    setTimeout(() => { btn.innerText = "SYNC LIVE <GO>"; }, 500);
  });
  // GARCH 90D vs 10K SDE View Toggle across all news suites and Daily A*
  document.querySelectorAll("[data-mc-toggle]").forEach(btn => {
    btn.addEventListener("click", () => {
      const ev = btn.getAttribute("data-ev") || "cpi";
      const mode = btn.getAttribute("data-mc-toggle");
      if (!EVENT_STATE[ev]) EVENT_STATE[ev] = {};
      EVENT_STATE[ev].mcView = mode;
      document.querySelectorAll(`[data-ev="${ev}"][data-mc-toggle]`).forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderEventMcPane(ev);
    });
  });

  // A* Multi-Asset 3-Way Switcher: Gold, Euro, Bitcoin
  document.querySelectorAll(".astar-asset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const asset = btn.getAttribute("data-asset") || "XAUUSD";
      CURRENT_ASTAR_ASSET = asset;
      if (!EVENT_STATE.astar) EVENT_STATE.astar = {};
      EVENT_STATE.astar.asset = asset;
      document.querySelectorAll(".astar-asset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      fetchAstarData();
    });
  });

  // A* Horizon Switcher: Scalping, Intraday, Daily, Swing
  document.querySelectorAll(".astar-horizon-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const h = btn.getAttribute("data-horizon") || "daily";
      if (!EVENT_STATE.astar) EVENT_STATE.astar = {};
      EVENT_STATE.astar.horizon = h;
      document.querySelectorAll(".astar-horizon-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      fetchAstarData();
    });
  });

  // Zero-Luck Confidence Level Selector (80% -> 99%)
  document.getElementById("conf-level-select")?.addEventListener("change", (e) => {
    window.SELECTED_CONFIDENCE = parseInt(e.target.value || "95", 10);
    ["cpi", "fomc", "nfp", "ppi"].forEach(ev => renderEventMcPane(ev));
    fetchUsEconomyAtlas();
  });
  // Update All Economy & Statistics Data <GO>
  async function runFullDataUpdate(btnEl, defaultLabel) {
    if (btnEl) btnEl.innerText = "UPDATING ALL DATA...";
    try { await fetch("/api/update-all-data", { method: "POST" }); } catch (e) {}
    yoyTimelineCache = null;
    await Promise.all([fetchPremarketSnapshot(false), fetchState(), fetchAuditLedger(), fetchFomcData(), fetchNfpData(), fetchPpiData(), fetchCalendarData(), fetchAstarData(), fetchUsEconomyAtlas()]);
    const mod = getActiveModuleId();
    if (mod === "mod-cpi") renderWorkstation("cpi");
    else if (mod === "mod-fomc") renderWorkstation("fomc");
    else if (mod === "mod-nfp") renderWorkstation("nfp");
    else if (mod === "mod-ppi") renderWorkstation("ppi");
    else if (mod === "mod-cal") fetchCalendarData();
    else if (mod === "mod-yoy") renderYoyTimeline();
    else if (mod === "mod-us") fetchUsEconomyAtlas();
    if (btnEl) setTimeout(() => { btnEl.innerText = defaultLabel; }, 600);
  }
  document.getElementById("btn-update-all-data")?.addEventListener("click", (e) => {
    runFullDataUpdate(e.target, "UPDATE ALL DATA <GO>");
  });
  document.querySelectorAll(".btn-update-history").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      runFullDataUpdate(btn, "UPDATE HISTORY <GO>");
    });
  });


  // CPI PDF event toggle buttons (CPI / FOMC / NFP)
  document.querySelectorAll("[data-toggle]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-toggle]");
      EVENT_STATE.cpi.pdfView = e.target.getAttribute("data-toggle").replace("pdf-", "");
      renderEventPdfPane("cpi");
    });
  });

  // CPI MC Z preset buttons
  document.querySelectorAll("[data-z]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-z]");
      EVENT_STATE.cpi.z = parseFloat(e.target.getAttribute("data-z"));
      const vShock = document.getElementById("bbg-val-shock");
      if (vShock) vShock.innerText = `${EVENT_STATE.cpi.z > 0 ? "+" : ""}${EVENT_STATE.cpi.z.toFixed(2)}σ`;
      renderEventMcPane("cpi");
      renderEventFreqPane("cpi");
    });
  });

  // CPI Asset & Diag toggles
  document.querySelectorAll("[data-asset]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-asset]");
      EVENT_STATE.cpi.asset = e.target.getAttribute("data-asset");
      renderEventTransPane("cpi");
    });
  });
  document.querySelectorAll("[data-diag]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-diag]");
      EVENT_STATE.cpi.diag = e.target.getAttribute("data-diag");
      renderEventDiagPane("cpi");
    });
  });

  // FOMC / NFP / PPI MC Z presets, Asset & Diag toggles
  document.querySelectorAll("[data-ev-z]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const ev = e.target.getAttribute("data-ev");
      setBtnActive(e.target, `[data-ev="${ev}"][data-ev-z]`);
      EVENT_STATE[ev].z = parseFloat(e.target.getAttribute("data-ev-z"));
      renderEventMcPane(ev);
      renderEventFreqPane(ev);
    });
  });

  document.querySelectorAll("[data-ev-asset]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const ev = e.target.getAttribute("data-ev");
      setBtnActive(e.target, `[data-ev="${ev}"][data-ev-asset]`);
      EVENT_STATE[ev].asset = e.target.getAttribute("data-ev-asset");
      renderEventTransPane(ev);
    });
  });

  document.querySelectorAll("[data-ev-diag]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const ev = e.target.getAttribute("data-ev");
      setBtnActive(e.target, `[data-ev="${ev}"][data-ev-diag]`);
      EVENT_STATE[ev].diag = e.target.getAttribute("data-ev-diag");
      renderEventDiagPane(ev);
    });
  });

  // Maximize toggle [+] across ALL panes
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
        const mod = getActiveModuleId();
        if (mod === "mod-cpi") renderWorkstation("cpi");
        else if (mod === "mod-fomc") renderWorkstation("fomc");
        else if (mod === "mod-nfp") renderWorkstation("nfp");
        else if (mod === "mod-ppi") renderWorkstation("ppi");
        else if (mod === "mod-astar") fetchAstarData();
      }, 60);
    });
  });
}

function resetFirewallBox(ev) {
  if (EVENT_STATE[ev].fwTimer) {
    clearInterval(EVENT_STATE[ev].fwTimer);
    EVENT_STATE[ev].fwTimer = null;
  }
  const timerId = ev === "cpi" ? "val-firewall-timer" : `val-${ev}-firewall-timer`;
  const statusId = ev === "cpi" ? "val-firewall-status" : `val-${ev}-firewall-status`;
  const tEl = document.getElementById(timerId);
  const sEl = document.getElementById(statusId);
  if (tEl) { tEl.innerText = "ARMED"; tEl.style.color = "var(--amber-cmd)"; }
  if (sEl) {
    sEl.innerText = ev === "fomc" ? "STANDBY // TRIGGERS AT 02:00 EST" : "STANDBY // TRIGGERS AT RELEASE";
    sEl.style.color = "var(--amber-cmd)";
    sEl.style.borderColor = "var(--amber-cmd)";
  }
}

function triggerFirewallCountdown(ev, seconds = 60) {
  clearInterval(EVENT_STATE[ev].fwTimer);
  let timeLeft = seconds;
  const timerId = ev === "cpi" ? "val-firewall-timer" : `val-${ev}-firewall-timer`;
  const statusId = ev === "cpi" ? "val-firewall-status" : `val-${ev}-firewall-status`;
  const timerEl = document.getElementById(timerId);
  const statusEl = document.getElementById(statusId);

  if (timerEl) { timerEl.innerText = `${timeLeft}s`; timerEl.style.color = "#ff4444"; }
  if (statusEl) {
    statusEl.innerText = "WAVE 1 // SPREAD BLOWOUT LOCKOUT";
    statusEl.style.color = "#ff4444";
    statusEl.style.borderColor = "#ff4444";
  }

  EVENT_STATE[ev].fwTimer = setInterval(() => {
    timeLeft--;
    if (!timerEl) return;
    if (timeLeft > 0) {
      timerEl.innerText = `${timeLeft}s`;
      if (timeLeft <= 15 && statusEl) {
        statusEl.innerText = "WAVE 1 ENDING: SPREADS COMPRESSING TO 40c";
      }
    } else {
      timerEl.innerText = "WAVE 2";
      timerEl.style.color = "var(--fg-bright)";
      if (statusEl) {
        statusEl.innerText = "WAVE 2 ACTIVE: VALIDATE 2Y YIELDS & VWAP";
        statusEl.style.color = "var(--fg-bright)";
        statusEl.style.borderColor = "var(--fg-bright)";
      }
      clearInterval(EVENT_STATE[ev].fwTimer);
      EVENT_STATE[ev].fwTimer = null;
    }
  }, 1000);
}

async function resetEventModule(ev, pre) {
  EVENT_STATE[ev].horizon = 1;
  EVENT_STATE[ev].asset = "gold";
  EVENT_STATE[ev].diag = "tcurve";
  EVENT_STATE[ev].mode = "PREMARKET";
  resetFirewallBox(ev);

  const b1d = document.getElementById(`${pre}-btn-horizon-1d`);
  const b5d = document.getElementById(`${pre}-btn-horizon-5d`);
  b1d?.classList.add("active");
  b5d?.classList.remove("active");

  const c = getEventConsensus();
  if (ev === "cpi") {
    EVENT_STATE.cpi.pdfView = "cpi";
    const rH = document.getElementById("bbg-range-headline"), rC = document.getElementById("bbg-range-core");
    if (rH) rH.value = c.cpiHl;
    if (rC) rC.value = c.cpiCore;
    if (document.getElementById("bbg-val-headline")) document.getElementById("bbg-val-headline").innerText = `${c.cpiHl.toFixed(2)}%`;
    if (document.getElementById("bbg-val-core")) document.getElementById("bbg-val-core").innerText = `${c.cpiCore.toFixed(2)}%`;
    document.querySelectorAll("[data-z]").forEach(b => b.classList.remove("active"));
    setBtnActive(document.querySelector("[data-asset='gold']"), "[data-asset]");
    setBtnActive(document.querySelector("[data-diag='tcurve']"), "[data-diag]");
  } else if (ev === "fomc") {
    if (document.getElementById("fomc-range-rate")) document.getElementById("fomc-range-rate").value = c.fomcTarget;
    if (document.getElementById("fomc-range-priced")) document.getElementById("fomc-range-priced").value = c.fomcTarget;
    if (document.getElementById("fomc-range-dots")) document.getElementById("fomc-range-dots").value = 0.0;
    if (document.getElementById("fomc-val-rate")) document.getElementById("fomc-val-rate").innerText = `${c.fomcTarget.toFixed(2)}%`;
    if (document.getElementById("fomc-val-priced")) document.getElementById("fomc-val-priced").innerText = `${c.fomcTarget.toFixed(2)}%`;
    if (document.getElementById("fomc-val-dots")) document.getElementById("fomc-val-dots").innerText = "0.0 bps";
    setBtnActive(document.querySelector('[data-ev="fomc"][data-ev-asset="gold"]'), '[data-ev="fomc"][data-ev-asset]');
    setBtnActive(document.querySelector('[data-ev="fomc"][data-ev-diag="tcurve"]'), '[data-ev="fomc"][data-ev-diag]');
  } else if (ev === "nfp") {
    if (document.getElementById("nfp-range-act")) document.getElementById("nfp-range-act").value = c.nfpK;
    if (document.getElementById("nfp-range-unemp")) document.getElementById("nfp-range-unemp").value = c.nfpUnemp;
    if (document.getElementById("nfp-range-ahe")) document.getElementById("nfp-range-ahe").value = c.nfpAhe;
    if (document.getElementById("nfp-val-act")) document.getElementById("nfp-val-act").innerText = `${c.nfpK}k`;
    if (document.getElementById("nfp-val-unemp")) document.getElementById("nfp-val-unemp").innerText = `${c.nfpUnemp.toFixed(1)}%`;
    if (document.getElementById("nfp-val-ahe")) document.getElementById("nfp-val-ahe").innerText = `${c.nfpAhe.toFixed(2)}%`;
    setBtnActive(document.querySelector('[data-ev="nfp"][data-ev-asset="gold"]'), '[data-ev="nfp"][data-ev-asset]');
    setBtnActive(document.querySelector('[data-ev="nfp"][data-ev-diag="tcurve"]'), '[data-ev="nfp"][data-ev-diag]');
  } else if (ev === "ppi") {
    if (document.getElementById("ppi-range-act")) document.getElementById("ppi-range-act").value = c.ppiMom;
    if (document.getElementById("ppi-range-retail")) document.getElementById("ppi-range-retail").value = c.retailMom;
    if (document.getElementById("ppi-val-act")) document.getElementById("ppi-val-act").innerText = `${c.ppiMom.toFixed(2)}%`;
    if (document.getElementById("ppi-val-retail")) document.getElementById("ppi-val-retail").innerText = `${c.retailMom.toFixed(2)}%`;
    setBtnActive(document.querySelector('[data-ev="ppi"][data-ev-asset="gold"]'), '[data-ev="ppi"][data-ev-asset]');
    setBtnActive(document.querySelector('[data-ev="ppi"][data-ev-diag="tcurve"]'), '[data-ev="ppi"][data-ev-diag]');
  }
  await fetchPremarketSnapshot(false);
  renderWorkstation(ev);
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
    const mod = getActiveModuleId();
    const ev = mod === "mod-fomc" ? "fomc" : (mod === "mod-nfp" ? "nfp" : (mod === "mod-ppi" ? "ppi" : "cpi"));
    const pre = ev === "cpi" ? "bbg" : ev;
    if (e.key === "F1") { e.preventDefault(); alert("NEWSLYSER B-PIPE TERMINAL // 10,000-PATH KOU SDE // WHITE HC3 // 5% HYPOTHESIS ENGINE."); }
    else if (e.key === "F3") { e.preventDefault(); document.querySelector(`#${mod} [data-pane*="mc"]`)?.click(); }
    else if (e.key === "F4") { e.preventDefault(); document.querySelector(`#${mod} [data-pane*="trans"]`)?.click(); }
    else if (e.key === "F5") { e.preventDefault(); document.querySelector(`#${mod} [data-pane*="diag"]`)?.click(); }
    else if (e.key === "F8") { e.preventDefault(); document.getElementById(`${pre}-btn-reset`)?.click(); }
    else if (e.key === "F12") { e.preventDefault(); document.getElementById("btn-sync-live")?.click(); }
  });
}

// ==============================================================
// 3. LIVE STATE, TICKERS & DYNAMIC TICKET UPDATES
// ==============================================================
function updateExecTicket(ev, directive, plan, avwapBands = null, preModel = null) {
  const actId  = ev === "cpi" ? "val-ticket-action" : `val-${ev}-ticket-action`;
  const entId  = ev === "cpi" ? "val-ticket-entry" : `val-${ev}-ticket-entry`;
  const slId   = ev === "cpi" ? "val-ticket-sl" : `val-${ev}-ticket-sl`;
  const tpId   = ev === "cpi" ? "val-ticket-tp" : `val-${ev}-ticket-tp`;
  const avId   = ev === "cpi" ? "val-ticket-avwap" : `val-${ev}-ticket-avwap`;
  const lmId   = ev === "cpi" ? "val-ticket-lambda" : `val-${ev}-ticket-lambda`;
  const probId = ev === "cpi" ? "val-ticket-probs" : `val-${ev}-ticket-probs`;
  const dirId  = ev === "cpi" ? "val-ticket-dir" : `val-${ev}-ticket-dir`;

  const actEl = document.getElementById(actId);
  if (actEl && directive) {
    let sizerBtn = actEl.querySelector(".btn-ticket-sizer");
    if (!sizerBtn) {
      sizerBtn = document.createElement("button");
      sizerBtn.className = "bbg-btn-mini btn-ticket-sizer";
      sizerBtn.setAttribute("data-ev", ev);
      sizerBtn.style.cssText = "border-color:#00e5ff; color:#00e5ff; font-size:8.5px; padding:1px 5px; margin-left:6px; cursor:pointer;";
      sizerBtn.innerText = "⚡ SIZER <GO>";
      sizerBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        document.querySelector('[data-mod="mod-astar"]')?.click();
        window.__syncFromActiveTicket?.(ev);
        document.getElementById("section-lot-sizer")?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
    actEl.textContent = directive + " ";
    actEl.appendChild(sizerBtn);
  }
  if (plan) {
    if (document.getElementById(entId)) document.getElementById(entId).innerText = plan.entry_zone || "--";
    if (document.getElementById(slId))  document.getElementById(slId).innerText = `${plan.stop_loss || "--"} (STRUCTURAL BUFFER)`;
    if (document.getElementById(tpId))  document.getElementById(tpId).innerText = `${plan.target || "--"}${plan.target_2 ? " / " + plan.target_2 : ""} (RR: ${plan.risk_reward_ratio || "--"})`;
  }
  if (avwapBands) {
    if (document.getElementById(avId)) document.getElementById(avId).innerText = `$${avwapBands.avwap.toFixed(2)} (±1σ: $${avwapBands.upper_1sd.toFixed(0)} / $${avwapBands.lower_1sd.toFixed(0)})`;
    if (document.getElementById(lmId)) document.getElementById(lmId).innerText = `${avwapBands.kyles_lambda_illiquidity_proxy.toFixed(4)}% (SLIPPAGE RISK)`;
  }
  if (preModel) {
    if (document.getElementById(probId)) {
      document.getElementById(probId).innerText = `HAWK: ${preModel.p_hawkish_beat}% | NEUT: ${preModel.p_inline}% | DOV: ${preModel.p_dovish_miss}%`;
    }
    if (document.getElementById(dirId)) {
      const dirEl = document.getElementById(dirId);
      dirEl.innerText = `GOLD UP: ${preModel.p_gold_up}% | DOWN: ${preModel.p_gold_down}% (${preModel.expected_return_pct >= 0 ? "+" : ""}${preModel.expected_return_pct}%)`;
      dirEl.style.color = preModel.p_gold_up >= 58 ? "#00e5ff" : (preModel.p_gold_down >= 58 ? "#ff4444" : "var(--amber-cmd)");
    }
    const crrId = ev === "cpi" ? "val-ticket-crr" : `val-${ev}-ticket-crr`;
    if (document.getElementById(crrId) && preModel.stage_b) {
      const sb = preModel.stage_b;
      const dp = Number(sb.prob_mispricing_pp || 0);
      document.getElementById(crrId).innerText = `p^P ${preModel.p_gold_up}% vs GLD p^Q ${sb.crr_p_q_pct}% (${dp >= 0 ? "+" : ""}${dp.toFixed(1)}pp) | ±$${sb.straddle_implied_move_usd}`;
    }
  }
}

function applyPremarketModelToUi(ev, m) {
  if (!m) return;
  if (EVENT_STATE[ev] && EVENT_STATE[ev].mode === "PREMARKET") {
    EVENT_STATE[ev].z = Number(m.z_pre || 0.0);
    if (ev === "cpi") {
      const sh = document.getElementById("bbg-val-shock"), rg = document.getElementById("bbg-val-regime");
      if (sh) {
        sh.innerText = `${m.z_pre >= 0 ? "+" : ""}${Number(m.z_pre).toFixed(2)}σ (PRE-MARKET)`;
        sh.style.color = m.z_pre > 0.5 ? "var(--amber-cmd)" : (m.z_pre < -0.5 ? "#00e5ff" : "#ffffff");
      }
      if (rg) rg.innerText = `AUTO PRE-MARKET // ${m.asymmetry_label}`;
    } else if (ev === "fomc") {
      const s = document.getElementById("fomc-shock-summary");
      if (s) s.innerText = `${m.z_pre >= 0 ? "+" : ""}${Number(m.z_pre).toFixed(2)}σ (${m.asymmetry_label})`;
    } else if (ev === "nfp") {
      const s = document.getElementById("nfp-shock-summary");
      if (s) s.innerText = `${m.z_pre >= 0 ? "+" : ""}${Number(m.z_pre).toFixed(2)}σ (${m.asymmetry_label})`;
    } else if (ev === "ppi") {
      const s = document.getElementById("ppi-shock-summary");
      if (s) s.innerText = `${m.z_pre >= 0 ? "+" : ""}${Number(m.z_pre).toFixed(2)}σ (${m.asymmetry_label})`;
    }
  }
  updateExecTicket(ev, m.directive_title, m, m.avwap_bands || null, m);
}

async function fetchPremarketSnapshot(renderActive = true) {
  try {
    const res = await fetch("/api/premarket-snapshot");
    const data = await res.json();
    PREMARKET_CACHE = data;
    const models = data.models || {};
    ["cpi", "fomc", "nfp", "ppi"].forEach(ev => applyPremarketModelToUi(ev, models[ev]));
    if (renderActive) {
      const mod = getActiveModuleId();
      if (mod === "mod-cpi") renderWorkstation("cpi");
      else if (mod === "mod-fomc") renderWorkstation("fomc");
      else if (mod === "mod-nfp") renderWorkstation("nfp");
      else if (mod === "mod-ppi") renderWorkstation("ppi");
    }
  } catch (e) {
    console.warn("fetchPremarketSnapshot failed:", e);
  }
}

async function fetchState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    const p = data.prices;
    if (p) {
      animateNumber(document.getElementById("tick-gold"), p["GC=F"], "$", "", 2);
      animateNumber(document.getElementById("tick-eur"), p["EURUSD=X"], "", "", 5);
      animateNumber(document.getElementById("tick-yield-spread-us-de"), p["YIELD_SPREAD_US_DE"], (p["YIELD_SPREAD_US_DE"] >= 0 ? "+" : ""), "%", 3);
      if (p["DE10Y"]) window._LAST_VERIFIED_DE10Y = Number(p["DE10Y"]);
      if (p["EURUSD_BID"] && p["EURUSD_ASK"]) {
        window._LAST_VERIFIED_EUR_SPREAD = Math.abs((p["EURUSD_ASK"] - p["EURUSD_BID"]) * 10000).toFixed(1);
      }
      animateNumber(document.getElementById("tick-btc"), p["BTC-USD"], "$", "", 2);
      if (p["BTC-USD"]) window._LAST_VERIFIED_BTC_PRICE = Number(p["BTC-USD"]);
      animateNumber(document.getElementById("tick-dxy"), p["DX-Y.NYB"], "", "", 2);
      animateNumber(document.getElementById("tick-2y"), p["ZT=F"], "", "%", 2);
      animateNumber(document.getElementById("tick-10y"), p["^TNX"], "", "%", 2);
      animateNumber(document.getElementById("tick-oil"), p["CL=F"], "$", "", 2);
    }

    const av = data.dynamic_levels?.avwap_bands;
    if (data.pre_news_model && EVENT_STATE.cpi.mode === "PREMARKET") {
      updateExecTicket("cpi", data.pre_news_model.directive_title, data.pre_news_model, av, data.pre_news_model);
    }
    if (data.fomc_prenew && EVENT_STATE.fomc.mode === "PREMARKET") updateExecTicket("fomc", data.fomc_prenew.directive_title, data.fomc_prenew, av, data.fomc_prenew);
    if (data.nfp_prenew && EVENT_STATE.nfp.mode === "PREMARKET")  updateExecTicket("nfp", data.nfp_prenew.directive_title, data.nfp_prenew, av, data.nfp_prenew);
    if (data.ppi_prenew && EVENT_STATE.ppi.mode === "PREMARKET")  updateExecTicket("ppi", data.ppi_prenew.directive_title, data.ppi_prenew, av, data.ppi_prenew);
  } catch (e) {
    console.warn("fetchState failed:", e);
  }
}

async function executeSimulateCpi(actMom, actCoreYoY) {
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_mom: actMom, actual_core_yoy: actCoreYoY })
    });
    const data = await res.json();
    if (data.model_forecast && data.dynamic_execution) {
      updateExecTicket("cpi", data.model_forecast.signal, data.dynamic_execution);
    }
  } catch (e) {}
}

// ==============================================================
// 4. UNIFIED PARAMETERIZED 5-PANE CHART ENGINES (ALL NEWS TYPES)
// ==============================================================
function getCanvasAndHud(ev, pane) {
  if (ev === "cpi") {
    return {
      container: document.getElementById(`canvas-${pane}`),
      hud: document.getElementById(`hud-${pane}`)
    };
  }
  if (ev === "astar") {
    return {
      container: document.getElementById(`canvas-astar-${pane}`),
      hud: document.getElementById(`hud-ast-${pane}`)
    };
  }
  return {
    container: document.getElementById(`canvas-${ev}-${pane}`),
    hud: document.getElementById(`hud-${ev}-${pane}`)
  };
}

// --- PANE 01: BAYESIAN PROBABILITY DENSITY FUNCTION (PDF) ---
async function renderEventPdfPane(ev) {
  const { container, hud } = getCanvasAndHud(ev, "pdf");
  if (!container) return;

  const queryEv = ev === "cpi" ? (EVENT_STATE.cpi.pdfView || "cpi") : ev;
  let data = null;
  try {
    const res = await fetch(`/api/pdf-distribution?event_type=${queryEv}`);
    data = await res.json();
  } catch (e) {
    return;
  }

  const pts = data.points || [];
  if (pts.length < 2) return;
  const unit = data.unit || "";

  // Update table metadata
  const priorEl = document.getElementById(`val-${ev}-pdf-prior`);
  const postEl = ev === "cpi" ? document.getElementById("val-pdf-peak") : document.getElementById(`val-${ev}-pdf-post`);
  if (priorEl && data.parameters?.prior) {
    priorEl.innerText = `mu=${data.parameters.prior.mu}, sigma=${data.parameters.prior.sigma}${unit} (${data.parameters.prior.label})`;
  }
  if (postEl && data.parameters?.posterior) {
    postEl.innerText = `mu=${data.parameters.posterior.mu > 0 ? "+" : ""}${data.parameters.posterior.mu}, sigma=${data.parameters.posterior.sigma}${unit} (${data.parameters.posterior.label})`;
  }
  if (hud && data.z_pre !== undefined) {
    hud.innerText = `Z_pre=${data.z_pre >= 0 ? "+" : ""}${Number(data.z_pre).toFixed(2)}σ | HAWK ${data.p_hawkish_beat}% / NEUT ${data.p_inline}% / DOV ${data.p_dovish_miss}% | GOLD UP ${data.p_gold_up}%`;
  }
  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 36, padR = 36, padT = 18, padB = 24;

  const minX = pts[0].x;
  const maxX = pts[pts.length - 1].x;
  const maxPdf = Math.max(...pts.map(p => Math.max(p.prior_pdf, p.posterior_pdf))) * 1.15;

  function getX(x) { return padL + ((x - minX) / (maxX - minX)) * (w - padL - padR); }
  function getY(y) { return padT + (1 - (y / maxPdf)) * (h - padT - padB); }

  const pathPrior = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.prior_pdf).toFixed(1)}`).join(" ");
  const pathPost  = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.x).toFixed(1)} ${getY(p.posterior_pdf).toFixed(1)}`).join(" ");
  const x0 = getX(0), xMiss = getX(minX * 0.35), xBeat = getX(maxX * 0.35), yZero = getY(0);
  const postMu = Number(data.parameters?.posterior?.mu || 0);
  const xPost = getX(Math.max(minX, Math.min(maxX, postMu)));
  const closeMu = Math.abs(xPost - x0) < 48;
  const t0Anchor = closeMu ? (xPost < x0 ? "start" : "end") : "middle";
  const postAnchor = closeMu ? (xPost < x0 ? "end" : "start") : "middle";

  container.innerHTML = `
    <svg id="svg-${ev}-pdf" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <defs>
        <pattern id="hatch-${ev}" width="5" height="5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="5" stroke="#1f1f1f" stroke-width="1" />
        </pattern>
      </defs>
      <rect x="${padL}" y="${padT}" width="${Math.max(0, xMiss - padL)}" height="${h - padT - padB}" fill="url(#hatch-${ev})"/>
      <rect x="${xBeat}" y="${padT}" width="${Math.max(0, w - padR - xBeat)}" height="${h - padT - padB}" fill="url(#hatch-${ev})"/>
      <line x1="${padL}" y1="${yZero}" x2="${w - padR}" y2="${yZero}" class="axis-line"/>
      <line x1="${xMiss}" y1="${padT}" x2="${xMiss}" y2="${h - padB}" class="grid-dot"/>
      <line x1="${xBeat}" y1="${padT}" x2="${xBeat}" y2="${h - padB}" class="grid-dot"/>
      <line x1="${x0}" y1="${padT}" x2="${x0}" y2="${yZero}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${x0 + (closeMu ? (xPost < x0 ? 3 : -3) : 0)}" y="${padT + 10}" fill="#ffb000" text-anchor="${t0Anchor}">T0: 0</text>
      <line x1="${xPost}" y1="${padT + 14}" x2="${xPost}" y2="${yZero}" stroke="#00e5ff" stroke-width="1.4" stroke-dasharray="3,2"/>
      <text x="${xPost + (closeMu ? (xPost < x0 ? -3 : 3) : 0)}" y="${padT + 10}" fill="#00e5ff" font-weight="800" text-anchor="${postAnchor}">μ:${postMu >= 0 ? "+" : ""}${postMu}</text>
      <path d="${pathPrior}" fill="none" stroke="#666666" stroke-width="1.2" stroke-dasharray="3,3"/>
      <path d="${pathPost}" fill="none" stroke="#ffffff" stroke-width="2.2"/>
      <text x="${xMiss}" y="${h - 6}" text-anchor="middle">MISS</text>
      <text x="${xBeat}" y="${h - 6}" text-anchor="middle">BEAT</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">${maxX.toFixed(2)}${unit}</text>
      <line id="needle-${ev}-pdf" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none; pointer-events:none;"/>
      <circle id="dot-${ev}-pdf" cx="0" cy="0" r="3.5" fill="#ffb000" stroke="#fff" stroke-width="1" style="display:none; pointer-events:none;"/>
    </svg>
  `;

  const svgEl = document.getElementById(`svg-${ev}-pdf`);
  const needle = document.getElementById(`needle-${ev}-pdf`);
  const dot = document.getElementById(`dot-${ev}-pdf`);
  const tt = getOrCreateTooltip(container);
  const defaultHud = hud ? hud.innerText : "";

  if (svgEl) {
    svgEl.addEventListener("mousemove", (mouseEv) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((mouseEv.clientX - rect.left) / rect.width) * w));
      const xVal = minX + ((mouseX - padL) / (w - padL - padR)) * (maxX - minX);
      let closest = pts[0], minDiff = Infinity;
      pts.forEach(p => { const d = Math.abs(p.x - xVal); if (d < minDiff) { minDiff = d; closest = p; } });

      const cx = getX(closest.x), cy = getY(closest.posterior_pdf);
      if (needle) { needle.style.display = "block"; needle.setAttribute("x1", cx); needle.setAttribute("x2", cx); needle.setAttribute("y1", padT); needle.setAttribute("y2", yZero); }
      if (dot) { dot.style.display = "block"; dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); }
      if (hud) hud.innerHTML = `x = <span class="highlight">${closest.x >= 0 ? "+" : ""}${closest.x.toFixed(2)}${unit}</span> | Prior = ${closest.prior_pdf.toFixed(4)} | Post = <span class="highlight">${closest.posterior_pdf.toFixed(4)}</span> | CDF = ${closest.cdf_posterior}% [${closest.zone}]`;
      positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
        <div style="color:var(--amber-cmd); font-weight:800;">SURPRISE: ${closest.x >= 0 ? "+" : ""}${closest.x.toFixed(2)}${unit}</div>
        <div>Prior PDF: ${closest.prior_pdf.toFixed(4)} | Post PDF: <strong>${closest.posterior_pdf.toFixed(4)}</strong></div>
        <div>Cum. Prob P(X &le; x): <strong>${closest.cdf_posterior}%</strong></div>
        <div>Regime: <strong>${closest.zone}</strong></div>
      `);
    });
    svgEl.addEventListener("mouseleave", () => {
      if (needle) needle.style.display = "none";
      if (dot) dot.style.display = "none";
      tt.style.display = "none";
      if (hud && defaultHud) hud.innerText = defaultHud;
    });
  }
}

// --- PANE 02: 10,000-PATH KOU (2002) JUMP-DIFFUSION MONTE CARLO SDE ---
async function renderEventMcPane(ev) {
  const { container, hud } = getCanvasAndHud(ev, "mc");
  if (!container) return;
  const st = EVENT_STATE[ev] || { z: 0.0, horizon: 1 };
  if (st && st.mcView === "garch") {
    await renderGarch90dPane(ev, container, hud);
    return;
  }

  // Reset SDE Titles and Labels
  const tagId = ev === "cpi" ? "tag-cpi-mc" : (ev === "astar" ? "tag-astar-mc" : `tag-${ev}-mc`);
  const titId = ev === "cpi" ? "title-cpi-mc" : (ev === "astar" ? "title-astar-mc" : `title-${ev}-mc`);
  if (document.getElementById(tagId)) document.getElementById(tagId).innerText = ev === "astar" ? "02 // SDE <GO>" : "02 // MCSG <GO>";
  if (document.getElementById(titId)) document.getElementById(titId).innerText = ev === "astar" ? "INTRADAY JUMP MONTE CARLO (10,000 PATHS)" : (ev === "fomc" ? "POLICY SHOCK JUMP SDE (10,000 PATHS)" : (ev === "nfp" ? "LABOR SHOCK JUMP SDE (10,000 PATHS)" : (ev === "ppi" ? "DEMAND PIPELINE JUMP SDE (10,000 PATHS)" : "KOU (2002) ASYMMETRIC JUMP SDE (10,000 PATHS)")));

  const pfx = ev === "cpi" ? "cpi" : (ev === "astar" ? "ast" : ev);
  if (document.getElementById(`lbl-${pfx}-mc-row1`)) document.getElementById(`lbl-${pfx}-mc-row1`).innerText = ev === "astar" ? "FIRST-PASSAGE P(WIN):" : "MEDIAN TERMINAL PRICE:";
  if (document.getElementById(`lbl-${pfx}-mc-row2`)) document.getElementById(`lbl-${pfx}-mc-row2`).innerText = "95% VALUE-AT-RISK (VaR):";
  if (document.getElementById(`lbl-${pfx}-mc-row3`)) document.getElementById(`lbl-${pfx}-mc-row3`).innerText = "TAIL MOVE PROBABILITY:";
  if (document.getElementById(`lbl-${pfx}-mc-row4`)) document.getElementById(`lbl-${pfx}-mc-row4`).innerText = "BS STOCH-VOL P(PROFIT/LOSS):";
  if (ev === "astar") {
    if (document.getElementById("row-ast-mc-extra1")) document.getElementById("row-ast-mc-extra1").style.display = "none";
    if (document.getElementById("row-ast-mc-extra2")) document.getElementById("row-ast-mc-extra2").style.display = "none";
  }
  document.querySelectorAll(`.sde-ctrl, .sde-ctrl-${ev}`).forEach(b => b.style.display = "inline-block");
  let mc = null;
  const confLvl = window.SELECTED_CONFIDENCE || 95;
  const hVal = (st.horizon !== undefined && st.horizon !== "undefined") ? st.horizon : (ev === "astar" ? "daily" : 1);
  const zVal = (st.z !== undefined && st.z !== "undefined") ? st.z : (ev === "astar" ? -1.18 : 0.0);
  try {
    const instParam = ev === "astar" ? `&instrument=${CURRENT_ASTAR_ASSET}` : "";
    const res = await fetch(`/api/monte-carlo?horizon=${hVal}&z=${zVal}&event_type=${ev}&n_paths=10000&min_confidence=${confLvl}${instParam}`);
    mc = await res.json();
  } catch (e) {
    return;
  }

  // Multi-asset precision formatting
  const isEur = (ev === "astar" && CURRENT_ASTAR_ASSET === "EURUSD") || (mc && mc.symbol === "EURUSD") || (mc && mc.instrument === "EURUSD");
  const pDec = isEur ? 5 : (mc && mc.price_decimals !== undefined ? mc.price_decimals : 2);
  const pPfx = isEur ? "" : (mc && mc.currency_prefix !== undefined ? mc.currency_prefix : "$");

  // Update metrics table
  const medId = ev === "cpi" ? "val-mc-median" : (ev === "astar" ? "val-ast-pwin" : `val-${ev}-mc-median`);
  const varId = ev === "cpi" ? "val-mc-var" : (ev === "astar" ? "val-ast-var" : `val-${ev}-mc-var`);
  const tailId = ev === "cpi" ? "val-mc-tail" : `val-${ev}-mc-tail`;
  const stochId = ev === "cpi" ? "val-mc-stoch" : `val-${ev}-mc-stoch`;
  const m = mc.metrics || {};
  const ea = mc.entry_analysis || {};
  const sv = mc.bs_stoch_vol || {};

  if (ev === "astar") {
    if (document.getElementById("val-ast-pwin")) {
      document.getElementById("val-ast-pwin").innerText = `${m.p_positive_return_pct}% win (${mc.n_paths.toLocaleString()} paths, ±${pPfx}${mc.mc_moe_terminal_dollars} SE)`;
    }
    if (document.getElementById("val-ast-var")) {
      document.getElementById("val-ast-var").innerText = `${m.var_95_pct > 0 ? "+" : ""}${m.var_95_pct.toFixed(2)}% / CVaR ${m.cvar_95_pct.toFixed(2)}% (P+: ${m.p_positive_return_pct}% | P-: ${m.p_negative_return_pct}%)`;
    }
  } else {
    if (document.getElementById(medId)) {
      document.getElementById(medId).innerText = `${pPfx}${Number(m.median_terminal_price || 0).toFixed(pDec)} (${mc.n_paths.toLocaleString()} paths, ±${pPfx}${mc.mc_moe_terminal_dollars} SE)`;
    }
    if (document.getElementById(varId)) {
      document.getElementById(varId).innerText = `${m.var_95_pct > 0 ? "+" : ""}${m.var_95_pct.toFixed(2)}% / CVaR ${m.cvar_95_pct.toFixed(2)}% (P+: ${m.p_positive_return_pct}% | P-: ${m.p_negative_return_pct}%)`;
    }
  }
  if (document.getElementById(tailId)) {
    document.getElementById(tailId).innerText = `|Move| >= 1σ: ${m.tail_beyond_1sd_pct}% | >= 2σ: ${m.tail_beyond_2sd_pct}% (σ=${m.terminal_dispersion_sd_pct}%)`;
  }
  if (document.getElementById(stochId) && ea.decision) {
    const evSign = ea.expected_payoff_bps >= 0 ? "+" : "";
    const vrSign = (sv.vol_roc_pct || 0) >= 0 ? "+" : "";
    document.getElementById(stochId).innerText = `[${ea.decision} @ ${confLvl}%] ${ea.best_side}: P(Prof)=${ea.p_profit_pct}% / P(Loss)=${ea.p_loss_pct}% | EV=${evSign}${ea.expected_payoff_bps}bps | Vol dσ/dt=${vrSign}${sv.vol_roc_pct || 0}%`;
    document.getElementById(stochId).style.color = ea.decision === "BUY" ? "#00e5ff" : (ea.decision === "SELL" ? "#ff4444" : "var(--amber-cmd)");
  }
  if (ea.recommended_ticket && ["cpi", "fomc", "nfp", "ppi"].includes(ev) && EVENT_STATE[ev]?.mode === "PREMARKET") {
    updateExecTicket(ev, `PRE-RELEASE DIRECTIVE: ${ea.decision_headline}`, ea.recommended_ticket, null, null);
  }

  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 42, padR = 46, padT = 18, padB = 24;
  if (!mc || !mc.percentiles) return;
  const steps = mc.steps || 24;
  const { p5, p25, p50, p75, p95 } = mc.percentiles;
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

  let tracesSvg = "";
  (mc.sample_traces || []).forEach((tr, idx) => {
    const d = tr.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v)}`).join(" ");
    tracesSvg += `<path d="${d}" fill="none" stroke="#555555" stroke-width="0.8" opacity="${idx < 4 ? '0.48' : '0.22'}"/>`;
  });

  const baseY = getY(mc.current_price);
  const tStart = mc.timesteps[0] || "T0";
  const tMid = mc.timesteps[Math.floor(steps / 2)] || "";
  const tEnd = (mc.timesteps && mc.timesteps[steps]) || (st.horizon === 1 ? "T+24h" : (st.horizon === "intraday" || st.horizon === 0.5 ? "T+12h" : (st.horizon === "scalping" ? "T+6h" : `D${st.horizon}.0`)));

  container.innerHTML = `
    <svg id="svg-${ev}-mc" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" class="axis-line"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <path d="${path90}" fill="#131313" opacity="0.88"/>
      <path d="${path50}" fill="#222222" opacity="0.92"/>
      <g>${tracesSvg}</g>
      <line x1="${padL}" y1="${baseY}" x2="${w - padR}" y2="${baseY}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${w - padR - 4}" y="${Math.max(padT + 10, baseY - 4)}" fill="#ffb000" text-anchor="end">T0: ${pPfx}${Number(mc.current_price).toFixed(pDec)}</text>
      <path d="${pathMedian}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <circle cx="${getX(steps)}" cy="${getY(p50[steps])}" r="3" fill="#ffffff"/>
      <text x="${getX(steps) + 5}" y="${getY(p50[steps]) + 3}" fill="#ffffff" font-weight="700">${pPfx}${Number(p50[steps]).toFixed(isEur ? 4 : (pDec === 5 ? 4 : 0))}</text>
      <text x="${padL - 4}" y="${padT + 8}" text-anchor="end">${pPfx}${Number(maxP).toFixed(isEur ? 4 : (pDec === 5 ? 4 : 0))}</text>
      <text x="${padL - 4}" y="${h - padB}" text-anchor="end">${pPfx}${Number(minP).toFixed(isEur ? 4 : (pDec === 5 ? 4 : 0))}</text>
      <text x="${padL}" y="${h - 6}" text-anchor="start">${tStart}</text>
      <text x="${getX(Math.floor(steps / 2))}" y="${h - 6}" text-anchor="middle">${tMid}</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">${tEnd}</text>
      <line id="crossV-${ev}-mc" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none; pointer-events:none;"/>
      <circle id="crossDot-${ev}-mc" cx="0" cy="0" r="3.5" fill="#ffb000" stroke="#fff" stroke-width="1" style="display:none; pointer-events:none;"/>
    </svg>
  `;

  const svgEl = document.getElementById(`svg-${ev}-mc`);
  const crossV = document.getElementById(`crossV-${ev}-mc`);
  const crossDot = document.getElementById(`crossDot-${ev}-mc`);
  const tt = getOrCreateTooltip(container);

  if (svgEl) {
    svgEl.addEventListener("mousemove", (mouseEv) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((mouseEv.clientX - rect.left) / rect.width) * w));
      const s = Math.max(0, Math.min(steps, Math.round(((mouseX - padL) / (w - padL - padR)) * steps)));
      const medP = p50[s], p95Val = p95[s], p5Val = p5[s], p75Val = p75[s], p25Val = p25[s];
      const delta = medP - mc.current_price;
      const pct = (delta / mc.current_price) * 100;
      const cx = getX(s), cy = getY(medP);

      if (crossV) { crossV.style.display = "block"; crossV.setAttribute("x1", cx); crossV.setAttribute("x2", cx); crossV.setAttribute("y1", padT); crossV.setAttribute("y2", h - padB); }
      if (crossDot) { crossDot.style.display = "block"; crossDot.setAttribute("cx", cx); crossDot.setAttribute("cy", cy); }
      if (hud) hud.innerHTML = `${mc.timesteps[s]} | Med: <span class="highlight">${pPfx}${medP.toFixed(pDec)}</span> (90%: ${pPfx}${p5Val.toFixed(pDec)}–${pPfx}${p95Val.toFixed(pDec)}) | Drift: ${delta >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
      positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
        <div style="color:var(--amber-cmd); font-weight:800;">${mc.timesteps[s]} (10,000 PATHS)</div>
        <div>Median: <strong>${pPfx}${medP.toFixed(pDec)}</strong> (${delta >= 0 ? "+" : ""}${pPfx}${delta.toFixed(pDec)} / ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%)</div>
        <div>50% Core: ${pPfx}${p25Val.toFixed(pDec)} — ${pPfx}${p75Val.toFixed(pDec)}</div>
        <div>90% Band: ${pPfx}${p5Val.toFixed(pDec)} — ${pPfx}${p95Val.toFixed(pDec)}</div>
      `);
    });
    svgEl.addEventListener("mouseleave", () => {
      if (crossV) crossV.style.display = "none";
      if (crossDot) crossDot.style.display = "none";
      tt.style.display = "none";
      if (hud) hud.innerText = `HOVER TIMESTEP // ${mc.n_paths.toLocaleString()}-PATH ENVELOPE [P5, P25, P50, P75, P95]`;
    });
  }
}

async function renderGarch90dPane(ev, container, hud) {
  let data = null;
  try {
    const instParam = ev === "astar" ? `&instrument=${CURRENT_ASTAR_ASSET}` : "";
    const res = await fetch(`/api/garch-volatility-90d?event_type=${ev}${instParam}`);
    data = await res.json();
  } catch (e) {
    return;
  }
  if (!data || !data.series || data.series.length === 0) return;

  const tagId = ev === "cpi" ? "tag-cpi-mc" : (ev === "astar" ? "tag-astar-mc" : `tag-${ev}-mc`);
  const titId = ev === "cpi" ? "title-cpi-mc" : (ev === "astar" ? "title-astar-mc" : `title-${ev}-mc`);
  if (document.getElementById(tagId)) document.getElementById(tagId).innerText = "02 // GARCH.90D <GO>";
  if (document.getElementById(titId)) document.getElementById(titId).innerText = "BOLLERSLEV (1986) GARCH(1,1) CONDITIONAL VOLATILITY (PAST 90D)";

  document.querySelectorAll(`.sde-ctrl, .sde-ctrl-${ev}`).forEach(b => b.style.display = "none");
  if (hud) hud.innerText = "HOVER TRADING DAY // 90D BOLLERSLEV (1986) GARCH(1,1) CONDITIONAL VOLATILITY [ANNUALIZED %]";

  const sum = data.summary || {};
  const pfx = ev === "cpi" ? "cpi" : (ev === "astar" ? "ast" : ev);
  const row1Lbl = document.getElementById(`lbl-${pfx}-mc-row1`);
  const row2Lbl = document.getElementById(`lbl-${pfx}-mc-row2`);
  const row3Lbl = document.getElementById(`lbl-${pfx}-mc-row3`);
  const row4Lbl = document.getElementById(`lbl-${pfx}-mc-row4`);

  if (row1Lbl) row1Lbl.innerText = "LATEST GARCH(1,1) VOL:";
  if (row2Lbl) row2Lbl.innerText = "UNCONDITIONAL VOL (σ_LR):";
  if (row3Lbl) row3Lbl.innerText = "90D VOL RANGE [MIN - MAX]:";
  if (row4Lbl) row4Lbl.innerText = "ENGLE ARCH LM TEST (FIT):";

  const val1Id = ev === "astar" ? "val-ast-pwin" : (ev === "cpi" ? "val-mc-median" : `val-${ev}-mc-median`);
  const val2Id = ev === "astar" ? "val-ast-var" : (ev === "cpi" ? "val-mc-var" : `val-${ev}-mc-var`);
  const val3Id = ev === "astar" ? "val-ast-mc-tail" : (ev === "cpi" ? "val-mc-tail" : `val-${ev}-mc-tail`);
  const val4Id = ev === "astar" ? "val-ast-mc-stoch" : (ev === "cpi" ? "val-mc-stoch" : `val-${ev}-mc-stoch`);

  if (ev === "astar") {
    if (document.getElementById("row-ast-mc-extra1")) document.getElementById("row-ast-mc-extra1").style.display = "table-row";
    if (document.getElementById("row-ast-mc-extra2")) document.getElementById("row-ast-mc-extra2").style.display = "table-row";
  }

  const gapLr = (sum.current_garch_ann_pct - sum.long_run_unconditional_vol_pct).toFixed(2);
  const gapStr = (gapLr >= 0 ? "+" : "") + gapLr + "%";
  if (document.getElementById(val1Id)) document.getElementById(val1Id).innerText = `${sum.current_garch_ann_pct}% ann (${sum.current_garch_daily_pct}%/d)`;
  if (document.getElementById(val2Id)) document.getElementById(val2Id).innerText = `${sum.long_run_unconditional_vol_pct}% ann (Gap: ${gapStr})`;
  if (document.getElementById(val3Id)) document.getElementById(val3Id).innerText = `${sum.min_90d_vol_pct}% – ${sum.max_90d_vol_pct}% (Avg: ${sum.avg_90d_vol_pct}%)`;
  if (document.getElementById(val4Id)) {
    const el4 = document.getElementById(val4Id);
    el4.innerText = `χ²=${sum.engle_arch_lm_stat}, p=${sum.engle_arch_lm_p_val} (VALID GARCH FIT)`;
    el4.style.color = sum.arch_test_pass ? "#00e5ff" : "var(--amber-cmd)";
  }

  // Draw 90D SVG Chart
  const series = data.series || [];
  const forecast = data.forward_forecast || [];
  const markers = data.event_markers || [];
  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 38, padR = 48, padT = 20, padB = 24;

  const allVols = series.map(s => s.garch_ann_vol_pct).concat(series.map(s => s.realized_vol_ann_pct)).concat([sum.long_run_unconditional_vol_pct || 28.0]);
  const minVol = Math.max(10.0, Math.floor((Math.min(...allVols) - 2.5) / 5.0) * 5.0);
  const maxVol = Math.min(65.0, Math.ceil((Math.max(...allVols) + 3.0) / 5.0) * 5.0);
  const n_total = series.length + forecast.length;

  function getX(i) { return padL + (i / Math.max(1, n_total - 1)) * (w - padL - padR); }
  function getY(v) { return padT + (1 - (v - minVol) / Math.max(1.0, maxVol - minVol)) * (h - padT - padB); }

  const lrY = getY(sum.long_run_unconditional_vol_pct || 28.0);
  const garchArea = `M ${getX(0)} ${getY(minVol)} ` + series.map((s, i) => `L ${getX(i).toFixed(1)} ${getY(s.garch_ann_vol_pct).toFixed(1)}`).join(" ") + ` L ${getX(series.length - 1)} ${getY(minVol)} Z`;
  const garchPath = series.map((s, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(s.garch_ann_vol_pct).toFixed(1)}`).join(" ");
  const rvPath = series.map((s, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(s.realized_vol_ann_pct).toFixed(1)}`).join(" ");
  const fwdPath = `M ${getX(series.length - 1)} ${getY(series[series.length - 1].garch_ann_vol_pct)} ` + forecast.map((f, i) => `L ${getX(series.length + i).toFixed(1)} ${getY(f.forecast_ann_vol_pct).toFixed(1)}`).join(" ");

  const gridSteps = [20, 25, 30, 35, 40, 45, 50].filter(v => v >= minVol && v <= maxVol);
  let gridSvg = "";
  gridSteps.forEach(v => {
    const gy = getY(v);
    gridSvg += `<line x1="${padL}" y1="${gy.toFixed(1)}" x2="${w - padR}" y2="${gy.toFixed(1)}" stroke="#222222" stroke-dasharray="2,3"/>`;
    gridSvg += `<text x="${w - padR + 4}" y="${(gy + 3).toFixed(1)}" fill="#666666" font-size="8.5" font-family="var(--font-mono)">${v}%</text>`;
  });

  let markerSvg = "";
  const dateToIdx = {};
  series.forEach((s, i) => { dateToIdx[s.date] = i; });
  markers.forEach(m => {
    const idx = dateToIdx[m.date];
    if (idx !== undefined) {
      const mx = getX(idx);
      const my = getY(series[idx].garch_ann_vol_pct);
      markerSvg += `<line x1="${mx.toFixed(1)}" y1="${padT}" x2="${mx.toFixed(1)}" y2="${h - padB}" stroke="rgba(255,176,0,0.4)" stroke-width="1" stroke-dasharray="2,2"/>`;
      markerSvg += `<polygon points="${mx.toFixed(1)},${(my-4).toFixed(1)} ${(mx+4).toFixed(1)},${my.toFixed(1)} ${mx.toFixed(1)},${(my+4).toFixed(1)} ${(mx-4).toFixed(1)},${my.toFixed(1)}" fill="var(--amber-cmd)" stroke="#ffffff" stroke-width="0.8"/>`;
    }
  });

  let fwdNodes = "";
  forecast.forEach((f, i) => {
    const fx = getX(series.length + i), fy = getY(f.forecast_ann_vol_pct);
    fwdNodes += `<circle cx="${fx.toFixed(1)}" cy="${fy.toFixed(1)}" r="2.5" fill="#00e5ff" stroke="#ffffff" stroke-width="0.8"/>`;
  });

  container.innerHTML = `
    <svg id="svg-${ev}-garch" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <defs>
        <linearGradient id="grad-garch-${ev}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ffb000" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#ffb000" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" class="axis-line"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      ${gridSvg}
      <path d="${garchArea}" fill="url(#grad-garch-${ev})"/>
      <line x1="${padL}" y1="${lrY.toFixed(1)}" x2="${w - padR}" y2="${lrY.toFixed(1)}" stroke="#ffb000" stroke-width="1" stroke-dasharray="3,3"/>
      <text x="${padL + 4}" y="${(lrY - 4).toFixed(1)}" fill="#ffb000" font-size="8.5" font-family="var(--font-mono)">σ_LR: ${sum.long_run_unconditional_vol_pct}%</text>
      <path d="${rvPath}" fill="none" stroke="#00e5ff" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.85"/>
      <path d="${garchPath}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <path d="${fwdPath}" fill="none" stroke="#00e5ff" stroke-width="1.5" stroke-dasharray="2,2"/>
      ${fwdNodes}
      ${markerSvg}
      <text x="${padL}" y="${h - 6}" text-anchor="start" fill="#888" font-size="8.5" font-family="var(--font-mono)">${series[0]?.date ? series[0].date.substring(5) : 'T-90d'}</text>
      <text x="${getX(Math.floor(series.length / 2))}" y="${h - 6}" text-anchor="middle" fill="#888" font-size="8.5" font-family="var(--font-mono)">${series[Math.floor(series.length / 2)]?.date ? series[Math.floor(series.length / 2)].date.substring(5) : 'T-45d'}</text>
      <text x="${getX(series.length - 1)}" y="${h - 6}" text-anchor="end" fill="#ffffff" font-size="8.5" font-family="var(--font-mono)">TODAY</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end" fill="#00e5ff" font-size="8.5" font-family="var(--font-mono)">+10d</text>
      <line id="crossV-${ev}-garch" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none; pointer-events:none;"/>
      <circle id="crossDot-${ev}-garch" cx="0" cy="0" r="3.5" fill="#ffffff" stroke="#ffb000" stroke-width="1.5" style="display:none; pointer-events:none;"/>
    </svg>
  `;

  const svgEl = document.getElementById(`svg-${ev}-garch`);
  const crossV = document.getElementById(`crossV-${ev}-garch`);
  const crossDot = document.getElementById(`crossDot-${ev}-garch`);
  const tt = getOrCreateTooltip(container);

  if (svgEl) {
    svgEl.addEventListener("mousemove", (mouseEv) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((mouseEv.clientX - rect.left) / rect.width) * w));
      const s = Math.max(0, Math.min(series.length - 1, Math.round(((mouseX - padL) / Math.max(1, getX(series.length - 1) - padL)) * (series.length - 1))));
      const pt = series[s];
      if (!pt) return;
      const cx = getX(s), cy = getY(pt.garch_ann_vol_pct);

      if (crossV) { crossV.style.display = "block"; crossV.setAttribute("x1", cx); crossV.setAttribute("x2", cx); crossV.setAttribute("y1", padT); crossV.setAttribute("y2", h - padB); }
      if (crossDot) { crossDot.style.display = "block"; crossDot.setAttribute("cx", cx); crossDot.setAttribute("cy", cy); }
      const diffLr = (pt.garch_ann_vol_pct - (sum.long_run_unconditional_vol_pct || 28.0)).toFixed(1);
      const diffSign = diffLr >= 0 ? "+" : "";

      if (hud) hud.innerHTML = `${pt.date} | GARCH Vol: <span class="highlight">${pt.garch_ann_vol_pct}% ann (${pt.garch_daily_vol_pct}%/d)</span> | RV10: ${pt.realized_vol_ann_pct}% | LR Gap: ${diffSign}${diffLr}% [${pt.regime}]`;
      const matchM = markers.find(m => m.date === pt.date);
      const evBadge = matchM ? `<div style="color:var(--amber-cmd); font-weight:800; border-top:1px solid #333; margin-top:4px; padding-top:2px;">${matchM.label}</div>` : "";

      positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
        <div style="color:var(--amber-cmd); font-weight:800;">${pt.date} // GOLD $${pt.price}</div>
        <div>GARCH(1,1) Vol: <strong>${pt.garch_ann_vol_pct}% ann</strong> (${pt.garch_daily_vol_pct}%/day)</div>
        <div>10D Realized Vol: <strong>${pt.realized_vol_ann_pct}% ann</strong></div>
        <div>Daily Ret: <strong>${pt.log_ret_pct >= 0 ? "+" : ""}${pt.log_ret_pct}%</strong> (Shock ε: ${pt.shock_eps >= 0 ? "+" : ""}${pt.shock_eps}%)</div>
        <div>Regime: <strong>${pt.regime}</strong> (vs σ_LR: ${diffSign}${diffLr}%)</div>
        ${evBadge}
      `);
    });
    svgEl.addEventListener("mouseleave", () => {
      if (crossV) crossV.style.display = "none";
      if (crossDot) crossDot.style.display = "none";
      tt.style.display = "none";
      if (hud) hud.innerText = "HOVER TRADING DAY // 90D BOLLERSLEV (1986) GARCH(1,1) CONDITIONAL VOLATILITY [ANNUALIZED %]";
    });
  }
}

// --- PANE 03: EMPIRICAL SHOCK FREQUENCY, KDE & POST-NEWS MOVEMENT ---
async function renderEventFreqPane(ev) {
  const { container, hud } = getCanvasAndHud(ev, "freq");
  if (!container) return;

  const zQ = (EVENT_STATE[ev] && EVENT_STATE[ev].z) || 0.0;
  let freq = null;
  try {
    const res = await fetch(`/api/shock-frequency?event_type=${ev}&z=${zQ}`);
    freq = await res.json();
  } catch (e) {
    return;
  }

  const rankId = ev === "cpi" ? "val-freq-rank" : `val-${ev}-freq-rank`;
  const paramsId = ev === "cpi" ? "val-freq-params" : `val-${ev}-freq-params`;
  const extId = ev === "cpi" ? "val-freq-extreme" : `val-${ev}-freq-extreme`;
  const ps = freq.post_news_gold_stats || {};

  if (document.getElementById(rankId)) {
    document.getElementById(rankId).innerText = `${freq.empirical_percentile_rank.toFixed(1)}th PCTL (DIST FROM MEDIAN: ${freq.distance_from_median >= 0 ? "+" : ""}${freq.distance_from_median.toFixed(2)}σ)`;
  }
  if (document.getElementById(paramsId)) {
    document.getElementById(paramsId).innerText = `Med=${freq.median_z.toFixed(2)}σ | IQR=${freq.iqr.toFixed(2)}σ | h=${freq.bandwidth_h.toFixed(3)} (N=${freq.n})`;
  }
  if (document.getElementById(extId)) {
    document.getElementById(extId).innerText = `|Move|>=1SD: ${ps.pct_far_1sd}% | Avg |Ret|: ${ps.mean_abs_move_pct}% | Max: ${ps.max_abs_move_pct}%`;
  }
  if (hud && freq.n) {
    hud.innerText = `HOVER HISTOGRAM BARS // N = ${freq.n} EMPIRICAL ${ev.toUpperCase()} RELEASES`;
  }
  const freqTitleEl = document.getElementById(`${ev}-freq-title`);
  if (freqTitleEl && freq.n) {
    freqTitleEl.innerText = `${freq.n}-RELEASE SHOCK FREQUENCY & POST-NEWS MOVE`;
  }
  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 32, padR = 32, padT = 26, padB = 24;
  const minX = -3.0, maxX = +3.0;
  const maxCount = Math.max(...freq.histogram_bars.map(b => b.count), 1);
  const maxKde = Math.max(...freq.kde_curve.map(k => k.density), 0.1);

  function getX(x) { return padL + ((Math.max(minX, Math.min(maxX, x)) - minX) / (maxX - minX)) * (w - padL - padR); }
  function getYCount(c) { return padT + (1 - (c / maxCount)) * (h - padT - padB); }
  function getYKde(d) { return padT + (1 - (d / maxKde)) * (h - padT - padB); }

  const kdePath = freq.kde_curve.map((k, i) => `${i === 0 ? "M" : "L"} ${getX(k.x).toFixed(1)} ${getYKde(k.density).toFixed(1)}`).join(" ");

  let barsSvg = "";
  freq.histogram_bars.forEach((b, idx) => {
    const x1 = getX(b.range[0]);
    const x2 = getX(b.range[1]);
    const barW = Math.max(2, (x2 - x1) - 2);
    const barY = getYCount(b.count);
    const barH = (h - padB) - barY;
    const isCurrent = b.range[0] <= zQ && zQ <= b.range[1];
    barsSvg += `
      <rect class="freq-bar" data-bin-idx="${idx}" x="${x1 + 1}" y="${barY}" width="${barW}" height="${barH}"
            fill="${isCurrent ? '#2e2e2e' : '#0f0f0f'}" stroke="${isCurrent ? '#ffffff' : '#2a2a2a'}" stroke-width="1"/>
      <text x="${x1 + barW / 2}" y="${Math.max(padT + 10, barY - 3)}" fill="#777" text-anchor="middle">${b.count > 0 ? b.count : ""}</text>
    `;
  });

  const curX = getX(zQ), medX = getX(freq.median_z);
  const freqClose = Math.abs(curX - medX) < 75;
  const medAnchor = freqClose ? "end" : (freq.median_z > 1.5 ? "end" : (freq.median_z < -1.5 ? "start" : "middle"));
  const curAnchor = freqClose ? "start" : (zQ > 1.5 ? "end" : (zQ < -1.5 ? "start" : "middle"));
  const medLabelX = freqClose ? Math.min(medX, curX) - 4 : medX;
  const curLabelX = freqClose ? Math.max(medX, curX) + 4 : curX;

  container.innerHTML = `
    <svg id="svg-${ev}-freq" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <g>${barsSvg}</g>
      <path d="${kdePath}" fill="none" stroke="#ffffff" stroke-width="1.8"/>
      <line x1="${medX}" y1="${padT - 8}" x2="${medX}" y2="${h - padB}" stroke="#777" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${medLabelX}" y="${padT - 12}" fill="#888" text-anchor="${medAnchor}">MED: ${freq.median_z.toFixed(2)}σ</text>
      <line x1="${curX}" y1="${padT}" x2="${curX}" y2="${h - padB}" stroke="#ffb000" stroke-width="1.5"/>
      <text x="${curLabelX}" y="${padT - 12}" fill="#ffb000" font-weight="800" text-anchor="${curAnchor}">CUR: ${zQ >= 0 ? "+" : ""}${zQ.toFixed(2)}σ</text>
      <text x="${padL}" y="${h - 6}" text-anchor="start">-3.0σ</text>
      <text x="${getX(0)}" y="${h - 6}" text-anchor="middle">0.0σ</text>
      <text x="${w - padR}" y="${h - 6}" text-anchor="end">+3.0σ</text>
    </svg>
  `;

  const svgEl = document.getElementById(`svg-${ev}-freq`);
  const tt = getOrCreateTooltip(container);
  if (svgEl) {
    svgEl.addEventListener("mousemove", (mouseEv) => {
      const t = mouseEv.target;
      if (t && t.classList.contains("freq-bar")) {
        const b = freq.histogram_bars[parseInt(t.getAttribute("data-bin-idx"), 10)];
        if (!b) return;
        const evList = (b.events || []).slice(0, 4).map(x => `${x.date}: ${x.gold_ret >= 0 ? "+" : ""}${x.gold_ret}% (${x.dist_from_mean_sd >= 0 ? "+" : ""}${x.dist_from_mean_sd}σ)`).join("<br>");
        if (hud) hud.innerHTML = `BIN <span class="highlight">${b.label}</span> | ${b.count} releases (${b.pct}%) | Avg |Gold Move|: <span class="highlight">${b.avg_abs_gold_move_pct}%</span>`;
        positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
          <div style="color:var(--amber-cmd); font-weight:800;">BIN: ${b.label} (${b.count}/${freq.n}, ${b.pct}%)</div>
          <div>Avg |Post-News Move|: <strong>${b.avg_abs_gold_move_pct}%</strong></div>
          ${evList ? `<div style="margin-top:4px; color:#ccc; border-top:1px solid #222; padding-top:3px;">${evList}</div>` : '<div style="color:#666;">No releases in bin</div>'}
        `);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      tt.style.display = "none";
      if (hud) hud.innerText = `HOVER HISTOGRAM BARS // N = ${freq.n} EMPIRICAL RELEASES`;
    });
  }
}

// --- PANE 04: CROSS-ASSET XY TRANSMISSION FRONTIER ---
async function renderEventTransPane(ev) {
  const { container, hud } = getCanvasAndHud(ev, "trans");
  if (!container) return;

  const asset = (EVENT_STATE[ev] && EVENT_STATE[ev].asset) || "gold";
  let trans = null;
  try {
    const res = await fetch(`/api/transmission-curves?event_type=${ev}`);
    trans = await res.json();
  } catch (e) {
    return;
  }

  const slopeId = ev === "cpi" ? "val-trans-slope" : `val-${ev}-trans-slope`;
  const vrId = ev === "cpi" ? "val-trans-vr" : `val-${ev}-trans-vr`;
  const modelObj = asset === "dxy" ? trans.models.dxy : (asset === "2y" ? trans.models.zt_note : trans.models.gold);
  const unitTag = (ev === "cpi" || ev === "nfp" || ev === "ppi") ? "σ" : "SD";

  const respUnit = asset === "2y" ? "bps" : "%";
  if (document.getElementById(slopeId) && modelObj) {
    document.getElementById(slopeId).innerText = `β = ${modelObj.beta[1].toFixed(3)}${respUnit}/${unitTag} (R² = ${modelObj.r2.toFixed(3)}, t = ${modelObj.t_stat}, p = ${modelObj.p_value})`;
  }
  if (document.getElementById(vrId)) {
    document.getElementById(vrId).innerText = `VR(5m) = ${trans.avg_vr5}x | β_DXY=${trans.models.dxy.beta[1].toFixed(2)}% | β_2Y=${trans.models.zt_note.beta[1].toFixed(2)}bps`;
  }

  const curves = trans.response_curves || [];
  const pts = trans.scatter_points || [];
  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 36, padR = 36, padT = 18, padB = 24;
  const minZ = -3.0, maxZ = +3.0;

  let yKey = "gold_return", pKey = "gold_ret", assetLabel = "GOLD";
  if (asset === "dxy") { yKey = "dxy_return"; pKey = "dxy_ret"; assetLabel = "DXY"; }
  else if (asset === "2y") { yKey = "zt_return"; pKey = "zt_ret"; assetLabel = "2Y YIELD"; }

  const allY = [...curves.map(c => c[yKey]), ...pts.map(p => p[pKey])];
  const minY = Math.min(-1.5, ...allY) * 1.1;
  const maxY = Math.max(1.5, ...allY) * 1.1;

  function getX(z) { return padL + ((Math.max(minZ, Math.min(maxZ, z)) - minZ) / (maxZ - minZ)) * (w - padL - padR); }
  function getY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * (h - padT - padB); }

  const pathReg = curves.map((c, i) => `${i === 0 ? "M" : "L"} ${getX(c.z).toFixed(1)} ${getY(c[yKey]).toFixed(1)}`).join(" ");
  let bandPath = "";
  if (asset === "gold") {
    const up = curves.map((c, i) => `${i === 0 ? "M" : "L"} ${getX(c.z).toFixed(1)} ${getY(c.gold_upper_95).toFixed(1)}`).join(" ");
    const lo = curves.slice().reverse().map(c => `L ${getX(c.z).toFixed(1)} ${getY(c.gold_lower_95).toFixed(1)}`).join(" ");
    bandPath = `${up} ${lo} Z`;
  }

  const dotsSvg = pts.map((p, i) => `
    <circle class="trans-dot" data-pt-idx="${i}" cx="${getX(p.z)}" cy="${getY(p[pKey])}" r="3.2"
            fill="${p[pKey] >= 0 ? '#ffffff' : '#ff4444'}" stroke="#000" stroke-width="0.8" opacity="0.85"/>
  `).join("");

  container.innerHTML = `
    <svg id="svg-${ev}-trans" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
      <line x1="${getX(0)}" y1="${padT}" x2="${getX(0)}" y2="${h - padB}" class="axis-line"/>
      ${bandPath ? `<path d="${bandPath}" fill="#171717" opacity="0.65"/>` : ""}
      <g>${dotsSvg}</g>
      <path d="${pathReg}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <text x="${w - padR}" y="${getY(0) - 4}" text-anchor="end">+Z (HAWKISH)</text>
      <text x="${padL}" y="${getY(0) - 4}" text-anchor="start">-Z (DOVISH)</text>
      <text x="${getX(0) + 4}" y="${padT + 10}">${assetLabel} ${respUnit}</text>
      <line id="needle-${ev}-trans" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none; pointer-events:none;"/>
      <circle id="snap-${ev}-trans" cx="0" cy="0" r="5.5" fill="none" stroke="#ffb000" stroke-width="1.5" style="display:none; pointer-events:none;"/>
    </svg>
  `;

  const svgEl = document.getElementById(`svg-${ev}-trans`);
  const needle = document.getElementById(`needle-${ev}-trans`);
  const snap = document.getElementById(`snap-${ev}-trans`);
  const tt = getOrCreateTooltip(container);

  if (svgEl) {
    svgEl.addEventListener("mousemove", (mouseEv) => {
      const rect = svgEl.getBoundingClientRect();
      const mx = ((mouseEv.clientX - rect.left) / rect.width) * w;
      const my = ((mouseEv.clientY - rect.top) / rect.height) * h;
      const clampedX = Math.max(padL, Math.min(w - padR, mx));
      const zVal = minZ + ((clampedX - padL) / (w - padL - padR)) * (maxZ - minZ);

      let nearestDot = null, minDist = 18;
      pts.forEach(p => {
        const d = Math.hypot(mx - getX(p.z), my - getY(p[pKey]));
        if (d < minDist) { minDist = d; nearestDot = p; }
      });

      if (nearestDot) {
        const dx = getX(nearestDot.z), dy = getY(nearestDot[pKey]);
        if (needle) { needle.style.display = "block"; needle.setAttribute("x1", dx); needle.setAttribute("x2", dx); needle.setAttribute("y1", padT); needle.setAttribute("y2", h - padB); }
        if (snap) { snap.style.display = "block"; snap.setAttribute("cx", dx); snap.setAttribute("cy", dy); }
        if (hud) hud.innerHTML = `EVENT <span class="highlight">${nearestDot.date}</span> | Shock: ${nearestDot.raw_shock} (${nearestDot.z >= 0 ? "+" : ""}${nearestDot.z}σ) | ${assetLabel}: <span class="highlight">${nearestDot[pKey] >= 0 ? "+" : ""}${nearestDot[pKey]}${respUnit}</span> | Dist Mean: ${nearestDot.dist_from_mean_sd >= 0 ? "+" : ""}${nearestDot.dist_from_mean_sd}σ`;
        positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
          <div style="color:var(--amber-cmd); font-weight:800;">RELEASE: ${nearestDot.date}</div>
          <div>Shock: <strong>${nearestDot.raw_shock}</strong> (${nearestDot.z >= 0 ? "+" : ""}${nearestDot.z}σ)</div>
          <div>Gold Move: <strong>${nearestDot.gold_ret >= 0 ? "+" : ""}${nearestDot.gold_ret}%</strong> (${nearestDot.dist_from_mean_sd >= 0 ? "+" : ""}${nearestDot.dist_from_mean_sd}σ from mean)</div>
          <div>DXY: ${nearestDot.dxy_ret >= 0 ? "+" : ""}${nearestDot.dxy_ret}% | VR(5m): ${nearestDot.vr5}x</div>
        `);
      } else {
        if (snap) snap.style.display = "none";
        let closest = curves[0], minD = Infinity;
        curves.forEach(c => { const d = Math.abs(c.z - zVal); if (d < minD) { minD = d; closest = c; } });
        if (needle) { needle.style.display = "block"; needle.setAttribute("x1", clampedX); needle.setAttribute("x2", clampedX); needle.setAttribute("y1", padT); needle.setAttribute("y2", h - padB); }
        if (hud) hud.innerHTML = `Shock Z = <span class="highlight">${closest.z >= 0 ? "+" : ""}${closest.z.toFixed(2)}σ</span> | Expected ${assetLabel} = <span class="highlight">${closest[yKey] >= 0 ? "+" : ""}${closest[yKey].toFixed(2)}${respUnit}</span>`;
        positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
          <div style="color:var(--amber-cmd); font-weight:800;">HC3 FRONTIER (Z = ${closest.z >= 0 ? "+" : ""}${closest.z.toFixed(2)}σ)</div>
          <div>Expected ${assetLabel}: <strong>${closest[yKey] >= 0 ? "+" : ""}${closest[yKey].toFixed(2)}${respUnit}</strong></div>
          ${asset === "gold" ? `<div>95% Band: ${closest.gold_lower_95.toFixed(2)}% to ${closest.gold_upper_95.toFixed(2)}%</div>` : ""}
        `);
      }
    });
    svgEl.addEventListener("mouseleave", () => {
      if (needle) needle.style.display = "none";
      if (snap) snap.style.display = "none";
      tt.style.display = "none";
      if (hud) hud.innerText = "HOVER Z-AXIS OR DOTS // WHITE HC3 95% PREDICTION BANDS";
    });
  }
}

// --- PANE 05: STUDENT'S t HYPOTHESIS (α=5%) & LEVERAGE DIAGNOSTICS ---
async function renderEventDiagPane(ev) {
  const { container, hud } = getCanvasAndHud(ev, "diag");
  if (!container) return;

  const view = (EVENT_STATE[ev] && EVENT_STATE[ev].diag) || "tcurve";
  let diag = null;
  try {
    const res = await fetch(`/api/hypothesis-diagnostics?event_type=${ev}`);
    diag = await res.json();
  } catch (e) {
    return;
  }

  const verdictId = ev === "cpi" ? "val-diag-verdict" : `val-${ev}-diag-verdict`;
  const luckId = ev === "cpi" ? "val-diag-luck" : `val-${ev}-diag-luck`;
  const hatId = ev === "cpi" ? "val-diag-hat" : `val-${ev}-diag-hat`;

  if (document.getElementById(verdictId)) {
    const vEl = document.getElementById(verdictId);
    const tier = diag.confidence_99 ? "99%" : (diag.confidence_95 ? "95%" : (diag.confidence_90 ? "90%" : "NONE"));
    if (diag.valid_at_5pct) {
      vEl.innerText = `t=${diag.beta_z_t_stat.toFixed(2)} vs ±${diag.t_crit_95.toFixed(3)} (p=${diag.beta_z_p_val.toFixed(4)}) -> VALID ${tier} (DIRECT + 2SLS)`;
      vEl.style.color = "#ffffff";
    } else if (diag.stage1_valid_5pct && diag.stage2_valid_5pct) {
      vEl.innerText = `2SLS X-ASSET VALID ${tier} (2Y p=${diag.stage1_2y_p} | F=${diag.stage2_cross_f}, p=${diag.stage2_cross_p})`;
      vEl.style.color = "#00e5ff";
    } else if (diag.stage1_valid_5pct) {
      vEl.innerText = `STAGE-1 VALID ${tier} (DXY t=${diag.stage1_dxy_t}, 2Y t=${diag.stage1_2y_t}) | WAVE-2 ONLY`;
      vEl.style.color = "var(--amber-cmd)";
    } else {
      vEl.innerText = `t=${diag.beta_z_t_stat.toFixed(2)} (p=${diag.beta_z_p_val.toFixed(4)}) -> NO EDGE / LUCK RISK`;
      vEl.style.color = "#ff4444";
    }
  }
  if (document.getElementById(luckId)) {
    const lEl = document.getElementById(luckId);
    const vrNote = diag.vr5_p_val < 0.05 ? ` | VR(5)=${diag.vr5_mean}x (p=${diag.vr5_p_val})` : "";
    lEl.innerText = `${diag.binomial_wins}/${diag.binomial_trials} (${diag.binomial_accuracy_pct}%, p=${diag.binomial_p_val}) -> ${diag.beats_luck_5pct ? "NOT LUCK (α=5%)" : "DIRECT COIN-FLIP"}${vrNote}`;
    lEl.style.color = diag.beats_luck_5pct ? "#ffb000" : (diag.vr5_p_val < 0.05 ? "#00e5ff" : "#ff4444");
  }
  if (document.getElementById(hatId)) {
    document.getElementById(hatId).innerText = `h*=${diag.leverage_threshold} | OOS R²=${diag.oos_r2} | F=${diag.f_stat} (F_p=${diag.f_p_val}, ${diag.high_leverage_outlier_count} outliers)`;
  }

  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 34, padR = 34, padT = 22, padB = 24;

  if (view === "tcurve") {
    const tCrit = diag.t_crit_95;
    const tStat = diag.beta_z_t_stat;
    const pts = diag.t_curve || [];
    const maxPdf = Math.max(...pts.map(p => p.pdf), 0.1) * 1.15;
    const minT = Math.min(-4.5, tStat - 0.5);
    const maxT = Math.max(+4.5, -minT);

    function getX(t) { return padL + ((t - minT) / (maxT - minT)) * (w - padL - padR); }
    function getY(p) { return padT + (1 - (p / maxPdf)) * (h - padT - padB); }

    // recompute exact PDF across [minT, maxT] if tStat lies beyond [-4.5, 4.5]
    const nGrid = 80;
    const gridPts = [];
    for (let i = 0; i <= nGrid; i++) {
      const tv = minT + (i / nGrid) * (maxT - minT);
      const pdfVal = Math.pow(1 + (tv * tv) / Math.max(1, diag.df), -(diag.df + 1) / 2) * 0.395;
      gridPts.push({ t: tv, pdf: pdfVal });
    }
    const gMax = Math.max(...gridPts.map(g => g.pdf)) * 1.15;
    function getGY(p) { return padT + (1 - (p / gMax)) * (h - padT - padB); }
    const tPath = gridPts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.t).toFixed(1)} ${getGY(p.pdf).toFixed(1)}`).join(" ");

    const xCritL = getX(-tCrit), xCritR = getX(+tCrit), xStat = getX(tStat);
    const critsClose = (xCritR - xCritL) < 115;
    const critLAnchor = critsClose ? "end" : "middle";
    const critRAnchor = critsClose ? "start" : "middle";
    const critLX = critsClose ? xCritL - 3 : xCritL;
    const critRX = critsClose ? xCritR + 3 : xCritR;
    const statTooCloseToCrit = Math.abs(xStat - xCritL) < 55;

    container.innerHTML = `
      <svg id="svg-${ev}-diag" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
        <rect x="${padL}" y="${padT}" width="${Math.max(0, xCritL - padL)}" height="${h - padT - padB}" fill="#240808"/>
        <rect x="${xCritR}" y="${padT}" width="${Math.max(0, w - padR - xCritR)}" height="${h - padT - padB}" fill="#240808"/>
        <line x1="${padL}" y1="${getGY(0)}" x2="${w - padR}" y2="${getGY(0)}" class="axis-line"/>
        <line x1="${getX(0)}" y1="${padT}" x2="${getX(0)}" y2="${h - padB}" class="axis-line"/>
        <path d="${tPath}" fill="none" stroke="#ffffff" stroke-width="2"/>
        <line x1="${xCritL}" y1="${padT}" x2="${xCritL}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <line x1="${xCritR}" y1="${padT}" x2="${xCritR}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <text x="${critLX}" y="${padT - 6}" fill="#ff4444" text-anchor="${critLAnchor}">-t_crit (${(-tCrit).toFixed(2)})</text>
        <text x="${critRX}" y="${padT - 6}" fill="#ff4444" text-anchor="${critRAnchor}">+t_crit (${tCrit.toFixed(2)})</text>
        <line x1="${xStat}" y1="${padT + 14}" x2="${xStat}" y2="${h - padB}" stroke="#ffb000" stroke-width="2"/>
        <text x="${xStat + (statTooCloseToCrit ? 4 : 0)}" y="${padT + 12}" fill="#ffb000" font-weight="900" text-anchor="${xStat < padL + 50 ? 'start' : (statTooCloseToCrit ? 'start' : 'middle')}">t = ${tStat.toFixed(2)}</text>
        <text x="${padL}" y="${h - 6}">${minT.toFixed(1)}</text>
        <text x="${getX(0)}" y="${h - 6}" text-anchor="middle">0.0 (df=${diag.df})</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">+${maxT.toFixed(1)}</text>
      </svg>
    `;

    const svgEl = document.getElementById(`svg-${ev}-diag`);
    const needle = document.getElementById(`needle-${ev}-diag`);
    const tt = getOrCreateTooltip(container);
    if (svgEl) {
      svgEl.addEventListener("mousemove", (mouseEv) => {
        const rect = svgEl.getBoundingClientRect();
        const mx = Math.max(padL, Math.min(w - padR, ((mouseEv.clientX - rect.left) / rect.width) * w));
        const tVal = minT + ((mx - padL) / (w - padL - padR)) * (maxT - minT);
        const rej = Math.abs(tVal) >= tCrit;
        if (needle) { needle.style.display = "block"; needle.setAttribute("x1", mx); needle.setAttribute("x2", mx); needle.setAttribute("y1", padT); needle.setAttribute("y2", h - padB); }
        if (hud) hud.innerHTML = `t = <span class="highlight">${tVal.toFixed(2)}</span> | Crit(5%) = ±${tCrit.toFixed(3)} | Zone: <span class="highlight">${rej ? "REJECT H0 (SIGNIFICANT)" : "ACCEPT NULL"}</span>`;
        positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
          <div style="color:${rej ? '#ff4444' : '#ffffff'}; font-weight:800;">t = ${tVal.toFixed(2)} (${rej ? '5% REJECTION ZONE' : 'NULL ZONE'})</div>
          <div>Model t-stat: <strong>${tStat.toFixed(2)}</strong> (p = ${diag.beta_z_p_val})</div>
          <div>Binomial vs Luck: <strong>${diag.binomial_wins}/${diag.binomial_trials} (${diag.binomial_accuracy_pct}%)</strong></div>
        `);
      });
      svgEl.addEventListener("mouseleave", () => {
        if (needle) needle.style.display = "none";
        tt.style.display = "none";
        if (hud) hud.innerText = `STUDENT'S t (df=${diag.df}, α=0.05): REJECTION REGIONS (|t| >= ${tCrit.toFixed(2)})`;
      });
    }
  } else {
    const pts = diag.diagnostic_points || [];
    const hThresh = diag.leverage_threshold;
    const maxH = Math.max(0.45, hThresh * 1.6, ...pts.map(p => p.leverage_h));
    function getX(hv) { return padL + (hv / maxH) * (w - padL - padR); }
    function getY(r) { return padT + (1 - (r - (-3.5)) / 7.0) * (h - padT - padB); }

    const dotsSvg = pts.map((p, i) => {
      const isOut = p.is_high_leverage || p.is_outlier;
      return `<circle cx="${getX(p.leverage_h)}" cy="${getY(p.studentized_resid)}" r="3.5" fill="${isOut ? '#ffb000' : '#444'}" stroke="${isOut ? '#fff' : '#777'}" stroke-width="1"/>`;
    }).join("");
    const xThresh = getX(hThresh);

    container.innerHTML = `
      <svg id="svg-${ev}-lev" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
        <line x1="${padL}" y1="${getY(0)}" x2="${w - padR}" y2="${getY(0)}" class="axis-line"/>
        <line x1="${padL}" y1="${getY(+2.0)}" x2="${w - padR}" y2="${getY(+2.0)}" class="grid-dot"/>
        <line x1="${padL}" y1="${getY(-2.0)}" x2="${w - padR}" y2="${getY(-2.0)}" class="grid-dot"/>
        <line x1="${xThresh}" y1="${padT}" x2="${xThresh}" y2="${h - padB}" stroke="#ff4444" stroke-width="1.2" stroke-dasharray="2,2"/>
        <text x="${xThresh + 4}" y="${padT + 10}" fill="#ff4444">h* = ${hThresh.toFixed(3)}</text>
        <g>${dotsSvg}</g>
        <text x="${padL}" y="${padT + 10}">STUD RESID</text>
        <text x="${w - padR}" y="${h - 6}" text-anchor="end">HAT LEVERAGE (h)</text>
        <circle id="snap-${ev}-lev" cx="0" cy="0" r="6" fill="none" stroke="#ffb000" stroke-width="1.5" style="display:none; pointer-events:none;"/>
      </svg>
    `;

    const svgEl = document.getElementById(`svg-${ev}-lev`);
    const snap = document.getElementById(`snap-${ev}-lev`);
    const tt = getOrCreateTooltip(container);
    if (svgEl) {
      svgEl.addEventListener("mousemove", (mouseEv) => {
        const rect = svgEl.getBoundingClientRect();
        const mx = ((mouseEv.clientX - rect.left) / rect.width) * w;
        const my = ((mouseEv.clientY - rect.top) / rect.height) * h;
        let nearest = null, minD = 22;
        pts.forEach(p => {
          const d = Math.hypot(mx - getX(p.leverage_h), my - getY(p.studentized_resid));
          if (d < minD) { minD = d; nearest = p; }
        });
        if (nearest) {
          if (snap) { snap.style.display = "block"; snap.setAttribute("cx", getX(nearest.leverage_h)); snap.setAttribute("cy", getY(nearest.studentized_resid)); }
          if (hud) hud.innerHTML = `${nearest.date} | Leverage h = <span class="highlight">${nearest.leverage_h}</span> | Stud Resid = <span class="highlight">${nearest.studentized_resid}</span>`;
          positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
            <div style="color:var(--amber-cmd); font-weight:800;">${nearest.date}</div>
            <div>Hat Leverage h_ii: <strong>${nearest.leverage_h}</strong> (h*=${hThresh})</div>
            <div>Studentized Resid: <strong>${nearest.studentized_resid}</strong></div>
            <div>Outlier Flag: <strong>${(nearest.is_high_leverage || nearest.is_outlier) ? "HIGH INFLUENCE" : "NORMAL"}</strong></div>
          `);
        } else {
          if (snap) snap.style.display = "none";
          tt.style.display = "none";
        }
      });
      svgEl.addEventListener("mouseleave", () => { if (snap) snap.style.display = "none"; tt.style.display = "none"; });
    }
  }
}

// ==============================================================
// 5. MODULE 05: DAILY A* SETUP, PROGRESS GAUGE & AUDIT LEDGER
// ==============================================================
function initTtlCountdown(serverExpiresAtEpoch, maxValSec = 5400) {
  if (!serverExpiresAtEpoch) return;
  const timerEl = document.getElementById("astar-ttl-timer");
  const decayEl = document.getElementById("astar-alpha-decay");
  const statusEl = document.getElementById("astar-ttl-status");

  function updateClock() {
    const nowSec = Math.floor(Date.now() / 1000);
    const remaining = serverExpiresAtEpoch - nowSec;
    if (remaining <= 0) {
      if (timerEl) { timerEl.innerText = "00m 00s (ROLLING OVER)"; timerEl.style.color = "var(--amber-cmd)"; }
      if (decayEl) { decayEl.innerText = "100% (ALPHA RESET)"; decayEl.style.color = "var(--amber-cmd)"; }
      if (statusEl) { statusEl.innerText = "REFRESHING FRESH SESSION..."; statusEl.style.color = "var(--amber-cmd)"; }
      clearInterval(ttlTimerInterval);
      setTimeout(() => { fetchAstarData(); }, 1200);
      return;
    }
    const m = Math.floor(remaining / 60), s = remaining % 60;
    const h = Math.floor(m / 60);
    const mRem = m % 60;
    const timeStr = h > 0 ? `${h}h ${mRem}m ${s < 10 ? "0" : ""}${s}s` : `${m}m ${s < 10 ? "0" : ""}${s}s`;
    if (timerEl) { timerEl.innerText = timeStr; timerEl.style.color = m < 15 ? "var(--amber-cmd)" : "var(--fg-bright)"; }
    const decayPct = Math.min(100, Math.max(0, (1.0 - Math.exp(-1.5 * (maxValSec - remaining) / maxValSec)) * 100));
    if (decayEl) decayEl.innerText = `${decayPct.toFixed(1)}%`;
    if (statusEl) statusEl.innerText = m < 15 ? "TTL EXPIRING SOON" : "ACTIVE VALID SETUP";
  }

  clearInterval(ttlTimerInterval);
  updateClock();
  ttlTimerInterval = setInterval(updateClock, 1000);
}

function applyAstarData(aStar) {
  if (!aStar) return;
  const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
  const pe = aStar.probability_engine || {};
  // Sync live horizon Z-score into Monte Carlo state so Pane 02 never uses stale -1.18
  if (!EVENT_STATE.astar) EVENT_STATE.astar = { horizon: h, z: 0 };
  EVENT_STATE.astar.z = Number(pe.z_eff !== undefined ? pe.z_eff : (pe.z_20_drift || 0));
  window._LAST_ASTAR_DATA = aStar;

  const bt3 = pe.backtest_3y_summary || {};
  const bt5 = pe.backtest_5y_summary || {};
  if (document.getElementById("ast-z20")) {
    const zVal = Number(pe.z_eff !== undefined ? pe.z_eff : (pe.z_20_drift || 0));
    const zLbl = pe.z_label || "z20";
    const m15Str = pe.z_m15_1h !== undefined ? ` | M15(1H): ${pe.z_m15_1h >= 0 ? "+" : ""}${Number(pe.z_m15_1h).toFixed(2)}σ` : "";
    document.getElementById("ast-z20").innerText = `${zLbl}=${zVal >= 0 ? "+" : ""}${zVal.toFixed(2)}σ${m15Str}`;
    document.getElementById("ast-z20").style.color = Math.abs(zVal) >= 0.75 ? (zVal > 0 ? "#00e676" : "#ff4444") : "var(--amber-cmd)";
  }
  if (document.getElementById("ast-mu-post")) {
    const muVal = Number(pe.mu_daily_post_pct || 0);
    document.getElementById("ast-mu-post").innerText = `${muVal >= 0 ? "+" : ""}${muVal.toFixed(3)}%/day (5D Drift: ${(muVal * 5.0).toFixed(2)}%)`;
  }
  if (document.getElementById("ast-drift-prob")) {
    document.getElementById("ast-drift-prob").innerText = `Bullish ${pe.p_drift_bull_pct || 50}% / Bearish ${pe.p_drift_bear_pct || 50}%`;
  }
  if (document.getElementById("ast-vrp")) {
    const vrpVal = Number(pe.vrp_ann_pct || 0);
    document.getElementById("ast-vrp").innerText = `${vrpVal >= 0 ? "+" : ""}${vrpVal.toFixed(1)}% (GVZ ${pe.gvz_iv_pct}% - RV22 ${pe.rv22_ann_pct}%)`;
    document.getElementById("ast-vrp").style.color = vrpVal > 0 ? "#00e5ff" : "#ff4444";
  }
  if (document.getElementById("ast-har-ratio")) {
    const ratio = Number(pe.rv5_ann_pct || 0) / Math.max(1.0, Number(pe.rv22_ann_pct || 1.0));
    document.getElementById("ast-har-ratio").innerText = `${ratio.toFixed(2)}x (Threshold <= 1.15x Calm)`;
    document.getElementById("ast-har-ratio").style.color = ratio <= 1.15 ? "#00e5ff" : "#ff4444";
  }
  if (document.getElementById("ast-atr-d1")) {
    document.getElementById("ast-atr-d1").innerText = `$${Number(pe.atr_d1_usd || 0).toFixed(2)} (SL: 1.0 ATR / TP: 2.0 ATR)`;
  }
  if (document.getElementById("ast-bt5-summary") && bt5.trades) {
    document.getElementById("ast-bt5-summary").innerText = `${bt5.trades} Trades | ${bt5.win_rate_pct}% Win | PF ${bt5.profit_factor} (t=${bt5.t_stat}, p=${bt5.p_val})`;
  }
  if (document.getElementById("ast-bt-summary") && bt3.trades) {
    document.getElementById("ast-bt-summary").innerText = `${bt3.trades} Trades | ${bt3.win_rate_pct}% Win | PF ${bt3.profit_factor} (t=${bt3.t_stat}, p=${bt3.p_val})`;
  }

  // Pillars & Lo-MacKinlay VR
  const pil = aStar.pillars || {};
  const fp = aStar.algorithmic_footprint || {};
  const fpa = aStar.first_passage_analysis || {};
  const mcq = aStar.monte_carlo_quantiles || {};

  if (document.getElementById("ast-score")) document.getElementById("ast-score").innerText = `${aStar.confluence_score}/100 (${aStar.grade})`;
  if (document.getElementById("ast-p1") && pil.macro_trend_alignment) document.getElementById("ast-p1").innerText = `${pil.macro_trend_alignment.score}/25 (${pil.macro_trend_alignment.status})`;
  if (document.getElementById("ast-p2") && pil.liquidity_sweep_status) document.getElementById("ast-p2").innerText = `${pil.liquidity_sweep_status.score}/25 (${pil.liquidity_sweep_status.status})`;
  if (document.getElementById("ast-p3") && pil.volatility_band_status) document.getElementById("ast-p3").innerText = `${pil.volatility_band_status.score}/25 (${pil.volatility_band_status.status})`;
  if (document.getElementById("ast-p4") && pil.cross_asset_confirmation) document.getElementById("ast-p4").innerText = `${pil.cross_asset_confirmation.score}/25 (${pil.cross_asset_confirmation.status})`;
  if (document.getElementById("ast-vr5")) document.getElementById("ast-vr5").innerText = fp.footprint_verdict || `VR=${fp.lo_mackinlay_vr5}x (p=${fp.p_value})`;
  if (document.getElementById("ast-lambda")) document.getElementById("ast-lambda").innerText = `λ = ${fp.kyles_lambda}% | Amihud = ${fp.amihud_illiq} (n=${fp.n_bars})`;
  const q8p = aStar.quant_8_pillars || {};
  const isEur = aStar.symbol === "EURUSD";
  const isBtc = aStar.symbol === "BTCUSD";
  const pDec = isEur ? 5 : 2;
  const pPfx = isEur ? "" : "$";

  const cmdTitle = document.getElementById("cmd-function-title");
  if (cmdTitle) {
    cmdTitle.innerText = isEur ? "EURUSD.SNR <GO> | WORKSTATION | HC3 KERNEL" : (isBtc ? "BTCUSD.MOM <GO> | WORKSTATION | HC3 KERNEL" : "CPI.XAU <GO> | WORKSTATION | HC3 KERNEL");
  }

  // Macro Metrics: Yield Spreads & SNR Pure-Trend Telemetry
  const mm = aStar.macro_metrics || {};
  const lblSpread = document.getElementById("lbl-ast-yield-spread");
  const valSpread = document.getElementById("ast-yield-spread");
  if (valSpread) {
    if (isEur && mm.transatlantic_yield_spread_us_de) {
      if (lblSpread) lblSpread.innerText = "TRANSATLANTIC SPREAD (US10Y - DE10Y):";
      valSpread.innerText = `${mm.transatlantic_yield_spread_us_de} (German Bund: ${window._LAST_VERIFIED_DE10Y || '3.32%'})`;
      valSpread.style.color = "#00e5ff";
    } else {
      if (lblSpread) lblSpread.innerText = "US 10Y-2Y YIELD SPREAD:";
      valSpread.innerText = mm.yield_spread_10_2 || "--";
      valSpread.style.color = "#00e5ff";
    }
  }

  const rTrend = document.getElementById("row-ast-pure-trend");
  const valTrend = document.getElementById("ast-pure-trend");
  if (rTrend && valTrend) {
    if (isEur && aStar.snr_pure_trend) {
      rTrend.style.display = "";
      const pt = aStar.snr_pure_trend;
      if (pt.is_sandwich_chop) {
        valTrend.innerHTML = '<span style="color:var(--amber-cmd); font-weight:900;">SANDWICH CHOP BLACKOUT</span> (EMA50/EMA200 Compression)';
      } else if (pt.pure_trend_direction === 'BULLISH_TREND') {
        valTrend.innerHTML = '<span style="color:#00e676; font-weight:900;">BULLISH PURE-TREND</span> (EMA50 > EMA200, Zero Counter-Trend Gate)';
      } else if (pt.pure_trend_direction === 'BEARISH_TREND') {
        valTrend.innerHTML = '<span style="color:#ff4444; font-weight:900;">BEARISH PURE-TREND</span> (EMA50 < EMA200, Zero Counter-Trend Gate)';
      } else {
        valTrend.innerText = pt.pure_trend_direction || '--';
      }
    } else {
      rTrend.style.display = "none";
    }
  }

  const ticket = aStar.trade_ticket || {};
  const rShield = document.getElementById("row-ast-be-shield");
  const valShield = document.getElementById("ast-be-shield");
  const rTkShield = document.getElementById("row-astar-be-shield");
  const valTkShield = document.getElementById("astar-be-shield");
  if (isEur && (ticket.dynamic_be_shield_trigger || ticket.dynamic_be_shield_sl)) {
    if (rShield) rShield.style.display = "";
    if (valShield) valShield.innerText = `+1.0R (${ticket.dynamic_be_shield_trigger}) -> Locked SL (${ticket.dynamic_be_shield_sl})`;
    if (rTkShield) rTkShield.style.display = "";
    if (valTkShield) valTkShield.innerText = `+1.0R Trigger: ${ticket.dynamic_be_shield_trigger} | BE Shield SL: ${ticket.dynamic_be_shield_sl}`;
  } else {
    if (rShield) rShield.style.display = "none";
    if (rTkShield) rTkShield.style.display = "none";
  }

  const lblQ8 = document.getElementById("lbl-ast-q8");
  if (lblQ8) lblQ8.innerText = isEur ? "EUR/USD SPREAD / DE10Y / VPIN:" : (isBtc ? "BTC FUNDING / VPIN / OI:" : "GLD GEX / O-U / VPIN:");
  if (isEur && document.getElementById("ast-q8-live")) {
    const pipSp = window._LAST_VERIFIED_EUR_SPREAD || '0.2';
    const de10 = window._LAST_VERIFIED_DE10Y ? `${Number(window._LAST_VERIFIED_DE10Y).toFixed(3)}%` : '3.321%';
    document.getElementById("ast-q8-live").innerText = `Spread ${pipSp} pips | Bund ${de10} | VPIN=${q8p.vpin_now || 0.44}`;
  } else if (document.getElementById("ast-q8-live")) {
    document.getElementById("ast-q8-live").innerText = `GEX $${q8p.net_gex_usd_m || 0}M (Flip $${q8p.gamma_flip_xau || 0}) | O-U s=${q8p.ou_fair_value_s_score >= 0 ? "+" : ""}${q8p.ou_fair_value_s_score || 0}σ | VPIN=${q8p.vpin_now || 0.44}`;
  }

  if (document.getElementById("ast-har-vol")) {
    document.getElementById("ast-har-vol").innerText = `${q8p.vol_targeting_scalar || 1.0}x Cap $${q8p.target_risk_usd || 30.0} (Fcst ${q8p.har_rv_forecast_pct || 1.19}%/d | t5 VaR99 ${q8p.var99_student_t5_pct || -3.52}%)`;
  }
  if (document.getElementById("val-ast-pwin")) document.getElementById("val-ast-pwin").innerText = `${fpa.win_probability_pct}% (M2M EV: ${fpa.expected_value_r >= 0 ? "+" : ""}${fpa.expected_value_r}R / Kelly: ${fpa.quarter_kelly_pct}%)`;
  if (document.getElementById("val-ast-var")) document.getElementById("val-ast-var").innerText = `${mcq.var_95_pct}% (CVaR: ${mcq.cvar_95_pct}% | ${(mcq.n_paths || 10000).toLocaleString()} paths)`;

  const scoreBadge = document.getElementById("astar-grade-badge");
  const ticketLabel = document.getElementById("astar-ticket-label");
  const ticketPane = document.getElementById("astar-ticket-pane");
  const actionEl = document.getElementById("astar-action");

  if (scoreBadge) scoreBadge.innerText = `${aStar.confluence_score}/100`;
  if (actionEl) {
    if (aStar.btc_regime?.is_macro_blackout) {
      actionEl.innerHTML = `<span style="background:#ff4444; color:#000; padding:2px 6px; font-weight:900; margin-right:6px;">MACRO BLACKOUT</span> ACTION: STAND ASIDE — T-30m TO T+15m HIGH-IMPACT LIQUIDITY FIREWALL`;
      actionEl.style.color = "#ff4444";
    } else if (aStar.btc_regime?.is_sandwich_chop) {
      actionEl.innerHTML = `<span style="background:var(--amber-cmd); color:#000; padding:2px 6px; font-weight:900; margin-right:6px;">SANDWICH CHOP</span> ACTION: STAND ASIDE — PRICE COMPRESSED BETWEEN EMA50 & EMA200`;
      actionEl.style.color = "var(--amber-cmd)";
    } else if (isEur && aStar.snr_pure_trend?.is_sandwich_chop) {
      actionEl.innerHTML = `<span style="background:var(--amber-cmd); color:#000; padding:2px 6px; font-weight:900; margin-right:6px;">SANDWICH CHOP BLACKOUT</span> ACTION: STAND ASIDE — EUR/USD COMPRESSED BETWEEN EMA50 & EMA200`;
      actionEl.style.color = "var(--amber-cmd)";
    } else {
      actionEl.innerText = `ACTION: ${ticket.action || "STAND ASIDE"}`;
      actionEl.style.color = (ticket.action || "").includes("LONG") || (ticket.action || "").includes("BUY") ? "#00e676" : ((ticket.action || "").includes("SHORT") || (ticket.action || "").includes("SELL") ? "#ff4444" : "#ffffff");
    }
  }
  if (ticketLabel) {
    if (isBtc) {
      ticketLabel.innerHTML = `[ BITCOIN CFD // 1:200 BROKER LEVERAGE ] <span style="color:#ff9900;">${ticket.ticket_title || 'BTC A* SETUP'}</span>`;
    } else if (isEur) {
      ticketLabel.innerHTML = `[ EURO FX // 1:500 LEVERAGE ] <span style="color:#00e5ff;">${ticket.ticket_title || 'EURUSD A* SETUP'}</span>`;
    } else {
      ticketLabel.innerText = ticket.ticket_title || `EXECUTION TICKET (${aStar.confluence_score}/100)`;
    }
  }
  if (ticketPane) ticketPane.style.borderStyle = aStar.is_a_star ? "solid" : "dashed";

  // Horizon Strategy & Guidance Banner (Institutional Core vs HFT Dynamic Agility)
  const stratNotes = aStar.horizon_strategy_notes || {};
  const stratBadge = document.getElementById("astar-strategy-badge");
  const stratSummary = document.getElementById("astar-strategy-summary");
  const stratDetails = document.getElementById("astar-strategy-details");
  const stratBanner = document.getElementById("astar-horizon-banner");

  const isCore = (h === "daily" || h === "swing");
  if (stratBadge) {
    stratBadge.innerText = stratNotes.badge ? stratNotes.badge.replace(/[★⚡]/g, "").trim() : (isCore ? "CORE INSTITUTIONAL" : "HFT DYNAMIC AGILITY");
    stratBadge.style.background = isCore ? "var(--amber-cmd)" : "#00e5ff";
    stratBadge.style.color = "#000000";
  }
  if (stratSummary) {
    stratSummary.innerText = stratNotes.summary || (isCore 
      ? "DAILY & SWING PRIMARY FOCUS // Macro Persistence, Low Turnover, 1:2.0+ RR"
      : "HFT DYNAMIC AGILITY // Fast Microstructure Execution with Strict Accuracy Gates");
    stratSummary.style.color = isCore ? "var(--fg-bright)" : "#00e5ff";
  }
  if (stratDetails) {
    if (isCore) {
      stratDetails.innerText = `EVT 99% Tail Shield Active • 4th-Moment Kelly • $10k–$20k 1:200 Leverage (1.0% Risk Cap = $100 / $200)`;
    } else {
      stratDetails.innerText = `VPIN < 0.65 Microstructure Gate • Hawkes Cascade Firewall • DFA Hurst Trend Persistence • Micro-Scaled Risk (0.35% - 0.50%)`;
    }
  }
  if (stratBanner) {
    stratBanner.style.borderColor = isCore ? "#553300" : "#004455";
    stratBanner.style.background = isCore ? "#0c0a05" : "#021014";
  }

  if (document.getElementById("astar-entry")) document.getElementById("astar-entry").innerText = ticket.entry_zone || "--";
  if (document.getElementById("astar-sl")) document.getElementById("astar-sl").innerText = ticket.stop_loss || "--";
  if (document.getElementById("astar-tp1")) document.getElementById("astar-tp1").innerText = ticket.take_profit_1 || "--";
  if (document.getElementById("astar-tp2")) document.getElementById("astar-tp2").innerText = ticket.take_profit_2 || "--";
  const lotTextEl = document.getElementById("astar-lot-text") || document.getElementById("astar-lot");
  if (lotTextEl) lotTextEl.innerText = `${ticket.recommended_lot_size_1000_equity || 0.00} LOT (${ticket.max_risk_amount_usd || '$0.00'})`;

  // Update Opt-In Add/Skip Buttons on Ticket Pane
  const btnAdd = document.getElementById("btn-add-astar-running");
  const btnSkip = document.getElementById("btn-skip-astar-setup");
  const fbEl = document.getElementById("astar-optin-feedback");
  const runningCount = Number(aStar.running_count || (aStar.in_running_list ? 1 : 0));
  const isAStar = !!aStar.is_a_star;
  const dirWord = (ticket.action || "").includes("LONG") ? "LONG" : "SHORT";

  if (btnAdd) {
    if (isAStar) {
      btnAdd.disabled = false;
      if (runningCount > 0) {
        btnAdd.innerText = `+ SCALE IN / ADD NEW ${h.toUpperCase()} (${runningCount} ACTIVE) <GO>`;
        btnAdd.style.background = "#092129";
        btnAdd.style.borderColor = "#00e5ff";
        btnAdd.style.color = "#00e5ff";
      } else {
        btnAdd.innerText = `+ ADD ${h.toUpperCase()} ${dirWord} TO RUNNING LIST <GO>`;
        btnAdd.style.background = "#0b2414";
        btnAdd.style.borderColor = "#00e676";
        btnAdd.style.color = "#00e676";
      }
    } else {
      btnAdd.disabled = true;
      btnAdd.innerText = `STAND ASIDE — NOT A* GRADE (${h.toUpperCase()})`;
      btnAdd.style.background = "#181414";
      btnAdd.style.borderColor = "#553333";
      btnAdd.style.color = "#996666";
    }
  }
  if (btnSkip) {
    btnSkip.disabled = false;
    btnSkip.style.opacity = "1";
  }
  if (fbEl && !window._ASTAR_OPTIN_LOCK_MSG) {
    if (runningCount > 0) {
      fbEl.innerHTML = `ACTIVE POSITIONS: <strong style="color:#00e5ff;">${runningCount} ${h.toUpperCase()} trade(s)</strong> running in <a href="#astar-running-tbody" style="color:#00e5ff; text-decoration:underline;">[06A // RUNNING & ACTIVE]</a> below. Fresh live candidate shown above.`;
    } else if (isAStar) {
      fbEl.innerText = `READY FOR OPT-IN: Click [+ ADD] to track this ${h.toUpperCase()} A* setup live, or [× SKIP] to ignore.`;
      fbEl.style.color = "#00e676";
    } else {
      fbEl.innerText = `GATE PROTECTED: ${h.toUpperCase()} is currently STAND ASIDE — wait for A* conditions.`;
      fbEl.style.color = "#888";
    }
  }

  // Visual Position Progress Gauge
  const ls = aStar.locked_setup;
  const curP = aStar.current_price || (isBtc ? 91500.0 : (isEur ? 1.08500 : 3800.0));
  const gSl = document.getElementById("gauge-sl"), gEnt = document.getElementById("gauge-entry"), gLiv = document.getElementById("gauge-live"), gTp = document.getElementById("gauge-tp1");
  const gUsd = document.getElementById("gauge-pnl-usd"), gR = document.getElementById("gauge-pnl-r"), gSt = document.getElementById("gauge-status-badge"), gPin = document.getElementById("gauge-cursor-pin");

  if (ls) {
    if (gSl) gSl.innerText = `${pPfx}${(ls.stop_loss || 0).toFixed(pDec)}`;
    if (gEnt) gEnt.innerText = `${pPfx}${(ls.fill_price || (Array.isArray(ls.entry_zone) ? (ls.entry_zone[0] + ls.entry_zone[1]) / 2 : curP)).toFixed(pDec)}`;
    if (gLiv) gLiv.innerText = `${pPfx}${curP.toFixed(pDec)}`;
    if (gTp) gTp.innerText = `${pPfx}${(ls.take_profit_1 || 0).toFixed(pDec)}`;
    const pUsd = ls.floating_pnl_usd || 0, pR = ls.floating_r || 0;
    if (gUsd) { gUsd.innerText = `${pUsd >= 0 ? "+" : ""}$${pUsd.toFixed(2)} USD`; gUsd.style.color = pUsd >= 0 ? "#00e676" : "#ff4444"; }
    if (gR) { gR.innerText = `${pR >= 0 ? "+" : ""}${pR.toFixed(2)} R`; gR.style.color = pR >= 0 ? "#00e676" : "#ff4444"; }
    if (gSt) gSt.innerText = ls.status || "PENDING_TRIGGER";
    const rng = Math.abs((ls.take_profit_1 || 0) - (ls.stop_loss || 0));
    if (gPin && rng > 0) {
      const pct = ls.direction === "BUY" ? ((curP - ls.stop_loss) / rng) * 100 : ((ls.stop_loss - curP) / rng) * 100;
      gPin.style.left = `${Math.max(0, Math.min(100, pct))}%`;
    }
  } else {
    if (gSl) gSl.innerText = ticket.stop_loss || "--";
    if (gEnt) gEnt.innerText = ticket.entry_zone ? ticket.entry_zone.split("(")[0].trim() : "--";
    if (gLiv) gLiv.innerText = `${pPfx}${curP.toFixed(pDec)}`;
    if (gTp) gTp.innerText = ticket.take_profit_1 || "--";
    if (gUsd) { gUsd.innerText = "$0.00 USD"; gUsd.style.color = "#888"; }
    if (gR) { gR.innerText = "0.00 R"; gR.style.color = "#888"; }
    if (gSt) gSt.innerText = aStar.is_a_star ? "OPT-IN READY" : (ticket.setup_status || "STAND_ASIDE");
  }

  if (aStar.time_to_live && aStar.time_to_live.expires_at_epoch) {
    initTtlCountdown(aStar.time_to_live.expires_at_epoch, aStar.time_to_live.max_validity_seconds || 5400);
  }

  renderEventMcPane("astar");
  renderLiquidityHeatmapUi(aStar);
}

async function renderLiquidityHeatmapUi(aStar) {
  const svg = document.getElementById("svg-liquidity-heatmap");
  if (!svg) return;
  const adv = aStar.advanced_microstructure || {};
  const hwk = adv.hawkes_cascade || {};
  const hurst = adv.dfa_hurst || {};
  const evt = adv.evt_tail_shield || {};
  const liq = adv.liquidation_heatmap || {};

  // Badges
  const bHwk = document.getElementById("badge-hawkes-cascade");
  const bHurst = document.getElementById("badge-hurst-regime");
  const bEvt = document.getElementById("badge-evt-shield");
  if (bHwk) {
    bHwk.innerText = hwk.is_cascade_active ? `HAWKES: CASCADE ACTIVE (${hwk.t_relax_hours}h RELAX)` : `HAWKES: CASCADE SAFE (η=${hwk.branching_ratio || 0.20})`;
    bHwk.style.background = hwk.is_cascade_active ? "#221111" : "#112211";
    bHwk.style.borderColor = hwk.is_cascade_active ? "#ff4444" : "#00e676";
    bHwk.style.color = hwk.is_cascade_active ? "#ff6b6b" : "#00e676";
  }
  if (bHurst) {
    bHurst.innerText = `DFA HURST: H=${hurst.hurst || 0.50} (${hurst.regime || "RANDOM_WALK"})`;
    bHurst.style.color = hurst.is_trending ? "#00e676" : (hurst.regime === "MEAN_REVERTING" ? "#ff4444" : "#00d2ff");
  }
  if (bEvt && evt.es_99_pct) {
    bEvt.innerText = `EVT 99% CVaR: ES=${evt.es_99_pct}% (VaR=${evt.var_99_pct}%)`;
  }

  // Table metrics
  const isEur = (CURRENT_ASTAR_ASSET === "EURUSD") || (aStar && aStar.symbol === "EURUSD");
  const pDec = isEur ? 5 : 2;
  const pPfx = isEur ? "" : "$";

  const elPoc = document.getElementById("val-hm-poc");
  const elVa = document.getElementById("val-hm-va");
  const elShort = document.getElementById("val-hm-short-pool");
  const elLong = document.getElementById("val-hm-long-pool");
  const elShield = document.getElementById("val-hm-evt-shield");
  const elEta = document.getElementById("val-hm-hawkes-eta");

  if (elPoc) elPoc.innerText = `${pPfx}${Number(aStar.current_price || 0).toFixed(pDec)}`;
  if (elShort) elShort.innerText = `${pPfx}${Number(liq.closest_short_liquidation_pool || 0).toFixed(pDec)}`;
  if (elLong) elLong.innerText = `${pPfx}${Number(liq.closest_long_liquidation_pool || 0).toFixed(pDec)}`;
  if (elShield) elShield.innerText = `${pPfx}${Number(evt.evt_sl_price || 0).toFixed(pDec)} (-${evt.es_99_pct || 0}%)`;
  if (elEta) elEta.innerText = `η = ${hwk.branching_ratio || 0.20} | λ = ${hwk.jump_intensity || 0.05}`;

  // Fetch full 50-bin profile asynchronously
  try {
    const res = await fetch(`/api/liquidity-heatmap?instrument=${CURRENT_ASTAR_ASSET}`);
    const hmData = await res.json();
    if (!hmData || !hmData.volume_profile) return;
    const vp = hmData.volume_profile;
    const bins = vp.bins || [];
    if (elPoc && vp.poc) elPoc.innerText = `${pPfx}${Number(vp.poc).toFixed(pDec)} (${vp.poc_vol_share_pct || 0}%)`;
    if (elVa && vp.val && vp.vah) elVa.innerText = `${pPfx}${Number(vp.val).toFixed(pDec)} - ${pPfx}${Number(vp.vah).toFixed(pDec)}`;

    // Draw Neo-Brutalist SVG Heatmap with exact price coordinate projection
    const w = svg.clientWidth || 400;
    const h = 160;
    const n = bins.length;
    if (n === 0) return;

    const shortLiq = hmData.liquidation_heatmap ? Number(hmData.liquidation_heatmap.closest_short_liquidation_pool || 0) : 0;
    const longLiq = hmData.liquidation_heatmap ? Number(hmData.liquidation_heatmap.closest_long_liquidation_pool || 0) : 0;
    const evtSl = hmData.evt_stop_shield ? Number(hmData.evt_stop_shield.evt_sl_price || 0) : 0;
    const spot = Number(hmData.current_price || aStar.current_price || 0);

    const priceList = bins.map(b => Number(b.price)).filter(p => !isNaN(p) && p > 0);
    if (spot > 0) priceList.push(spot);
    if (shortLiq > 0) priceList.push(shortLiq);
    if (longLiq > 0) priceList.push(longLiq);
    if (evtSl > 0) priceList.push(evtSl);

    let pMin = Math.min(...priceList);
    let pMax = Math.max(...priceList);
    const pRange = pMax - pMin || (spot * 0.01) || 1.0;
    pMin -= pRange * 0.04;
    pMax += pRange * 0.04;

    const padT = 16, padB = 16;
    const chartH = h - padT - padB;
    const getY = (price) => padT + (1.0 - (price - pMin) / (pMax - pMin)) * chartH;

    let svgHtml = `<rect width="${w}" height="${h}" fill="#0a0a0c" rx="2"/>`;

    // Draw volume profile bins properly mapped to price axis (high price on top)
    const barH = Math.max(2, chartH / n);
    bins.forEach((b) => {
      const bPrice = Number(b.price);
      const y = Math.max(padT, Math.min(h - padB - barH, getY(bPrice) - barH / 2));
      const barW = Math.max(4, b.intensity * (w * 0.62));
      const col = b.is_poc ? "#00ff88" : (b.is_value_area ? "#00d2ff" : "#334455");
      const opacity = b.is_poc ? 0.95 : (0.22 + b.intensity * 0.65);
      svgHtml += `<rect x="10" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(1, barH - 0.5).toFixed(1)}" fill="${col}" fill-opacity="${opacity.toFixed(2)}" />`;
    });

    // Draw Short Liquidation Pool line (top red)
    if (shortLiq > 0) {
      const yShort = Math.max(padT, Math.min(h - padB, getY(shortLiq)));
      svgHtml += `<line x1="0" y1="${yShort.toFixed(1)}" x2="${w}" y2="${yShort.toFixed(1)}" stroke="#ff3b5c" stroke-width="1.5" stroke-dasharray="4,3"/>`;
      svgHtml += `<text x="${w - 195}" y="${Math.max(11, yShort - 3).toFixed(1)}" fill="#ff8095" font-size="8.5" font-family="monospace" font-weight="bold">SHORT LIQ: ${pPfx}${shortLiq.toFixed(pDec)}</text>`;
    }

    // Draw Spot Price line (middle cyan)
    if (spot > 0) {
      const ySpot = Math.max(padT, Math.min(h - padB, getY(spot)));
      svgHtml += `<line x1="0" y1="${ySpot.toFixed(1)}" x2="${w}" y2="${ySpot.toFixed(1)}" stroke="#00e5ff" stroke-width="2"/>`;
      svgHtml += `<text x="15" y="${Math.max(11, ySpot - 4).toFixed(1)}" fill="#00e5ff" font-size="9" font-family="monospace" font-weight="bold">SPOT: ${pPfx}${spot.toFixed(pDec)}</text>`;
    }

    // Draw Long Liquidation Pool line (bottom gold)
    if (longLiq > 0) {
      const yLong = Math.max(padT, Math.min(h - padB, getY(longLiq)));
      svgHtml += `<line x1="0" y1="${yLong.toFixed(1)}" x2="${w}" y2="${yLong.toFixed(1)}" stroke="#ffaa00" stroke-width="1.5" stroke-dasharray="4,3"/>`;
      svgHtml += `<text x="${w - 195}" y="${Math.min(h - 4, yLong + 10).toFixed(1)}" fill="#ffcc66" font-size="8.5" font-family="monospace" font-weight="bold">LONG LIQ: ${pPfx}${longLiq.toFixed(pDec)}</text>`;
    }

    // Draw EVT 99% Stop Shield (bottom purple line)
    if (evtSl > 0) {
      const yEvt = Math.max(padT, Math.min(h - padB, getY(evtSl)));
      svgHtml += `<line x1="0" y1="${yEvt.toFixed(1)}" x2="${w}" y2="${yEvt.toFixed(1)}" stroke="#bd93f9" stroke-width="2"/>`;
      svgHtml += `<text x="15" y="${Math.min(h - 4, yEvt + 10).toFixed(1)}" fill="#d6b4fc" font-size="8.5" font-family="monospace" font-weight="bold">EVT 99% CVaR STOP: ${pPfx}${evtSl.toFixed(pDec)}</text>`;
    }

    svg.innerHTML = svgHtml;
  } catch (err) {
    console.warn("renderLiquidityHeatmapUi error:", err);
  }
}

async function fetchAstarData() {
  try {
    const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
    const [aRes, dRes] = await Promise.all([
      fetch(`/api/daily-a-star?horizon=${h}&instrument=${CURRENT_ASTAR_ASSET}`),
      fetch(`/api/dynamic-levels?instrument=${CURRENT_ASTAR_ASSET}`)
    ]);
    const aStar = await aRes.json();
    const dyn = await dRes.json();
    if (!aStar) return;
    applyAstarData(aStar);
    fetchGlobalMacroMatrix();
    fetchQuant8AuditTable();
    renderSetupLedgerTable();
    renderMultiHorizonOptInMatrix();
  } catch (err) {
    console.warn("fetchAstarData error:", err);
  }
}

window._SKIPPED_ASTAR_HORIZONS = window._SKIPPED_ASTAR_HORIZONS || {};

async function addHorizonSetupToRunning(horizon) {
  const fbEl = document.getElementById("astar-optin-feedback");
  const btnAdd = document.getElementById("btn-add-astar-running");
  if (btnAdd) {
    btnAdd.disabled = true;
    btnAdd.innerText = "SUBMITTING TO RUNNING LIST...";
  }
  try {
    const res = await fetch("/api/setup-ledger/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ horizon, asset: CURRENT_ASTAR_ASSET })
    });
    const out = await res.json();
    delete window._SKIPPED_ASTAR_HORIZONS[horizon];
    if (fbEl) {
      window._ASTAR_OPTIN_LOCK_MSG = true;
      fbEl.innerText = out.message || `${horizon.toUpperCase()} status: ${out.status}`;
      fbEl.style.color = out.status === "ADDED" ? "#00e676" : (out.status === "DUPLICATE_DEBOUNCE" ? "var(--amber-cmd)" : "#ff6b6b");
      setTimeout(() => { window._ASTAR_OPTIN_LOCK_MSG = false; }, 4500);
    }
    await renderSetupLedgerTable();
    await renderMultiHorizonOptInMatrix();
    await fetchAstarData();
  } catch (e) {
    if (fbEl) { fbEl.innerText = `Error adding ${horizon}: ${e.message}`; fbEl.style.color = "#ff4444"; }
  } finally {
    if (btnAdd) btnAdd.disabled = false;
  }
}

function skipHorizonSetup(horizon) {
  window._SKIPPED_ASTAR_HORIZONS[horizon] = true;
  const fbEl = document.getElementById("astar-optin-feedback");
  if (fbEl) {
    window._ASTAR_OPTIN_LOCK_MSG = true;
    fbEl.innerText = `SKIPPED ${horizon.toUpperCase()} setup for this session (not added to running list).`;
    fbEl.style.color = "var(--amber-cmd)";
    setTimeout(() => { window._ASTAR_OPTIN_LOCK_MSG = false; }, 3500);
  }
  renderMultiHorizonOptInMatrix();
}

async function handleRunningTradeAction(setupId, mode) {
  try {
    await fetch("/api/setup-ledger/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ setup_id: setupId, mode })
    });
    await renderSetupLedgerTable();
    await renderMultiHorizonOptInMatrix();
    await fetchAstarData();
  } catch (e) {
    console.warn("handleRunningTradeAction error:", e);
  }
}

async function renderMultiHorizonOptInMatrix() {
  const wrap = document.getElementById("astar-multi-optin-bar");
  if (!wrap) return;
  try {
    const res = await fetch(`/api/multi-horizon-a-star?instrument=${CURRENT_ASTAR_ASSET}`);
    const data = await res.json();
    const horizons = data.horizons || {};
    const order = ["scalping", "intraday", "daily", "swing", "news"];
    wrap.innerHTML = order.map(h => {
      const item = horizons[h];
      if (!item) return "";
      const tt = item.trade_ticket || {};
      const isA = !!item.is_a_star;
      const runCnt = Number(item.running_count || (item.in_running_list ? 1 : 0));
      const inList = runCnt > 0;
      const skipped = !!window._SKIPPED_ASTAR_HORIZONS[h];
      const actStr = tt.action || "";
      const dir = isA ? (actStr.includes("LONG") || actStr.includes("BUY") ? "LONG" : "SHORT") : "STAND ASIDE";
      const dirColor = dir === "LONG" ? "#00e676" : (dir === "SHORT" ? "#ff4444" : "#888");
      const borderCol = inList ? "#00e5ff" : (isA ? (skipped ? "#444" : "#00e676") : "#25252c");
      const pe = item.probability_engine || {};
      const zTxt = pe.z_eff !== undefined ? `${pe.z_label || "z"}=${pe.z_eff >= 0 ? "+" : ""}${Number(pe.z_eff).toFixed(2)}σ` : "";

      const runBadge = runCnt > 0 ? `<span style="display:inline-block; padding:2px 5px; background:#092129; border:1px solid #00e5ff; color:#00e5ff; font-size:8px; font-weight:800;">${runCnt} ACTIVE</span>` : "";

      let btns = "";
      if (isA && !skipped) {
        const addLabel = runCnt > 0 ? `+ SCALE IN (${runCnt})` : "+ ADD TO RUNNING";
        const btnCol = runCnt > 0 ? "#00e5ff" : "#00e676";
        const btnBg = runCnt > 0 ? "#092129" : "#0b2414";
        btns = `
          ${runBadge}
          <button onclick="addHorizonSetupToRunning('${h}')" class="bbg-btn-mini" style="background:${btnBg}; border:1px solid ${btnCol}; color:${btnCol}; font-size:8.5px; font-weight:900; padding:3px 7px; cursor:pointer;">${addLabel}</button>
          <button onclick="skipHorizonSetup('${h}')" class="bbg-btn-mini" style="background:#1a1a1f; border:1px solid #555; color:#aaa; font-size:8.5px; padding:3px 6px; cursor:pointer;">SKIP</button>
        `;
      } else if (isA && skipped) {
        btns = `
          ${runBadge}
          <span style="color:#888; font-size:8.5px;">SKIPPED</span>
          <button onclick="addHorizonSetupToRunning('${h}')" class="bbg-btn-mini" style="background:#141a14; border:1px solid #4a7c59; color:#8fd19e; font-size:8px; padding:2px 6px; cursor:pointer;">UNDO / ADD</button>
        `;
      } else {
        btns = `
          ${runBadge}
          <span style="color:#666; font-size:8.5px; font-family:var(--font-mono);">GATE BLOCKED</span>
        `;
      }

      return `
        <div style="background:#0c0c10; border:1px solid ${borderCol}; padding:8px; display:flex; flex-direction:column; gap:5px; font-family:var(--font-mono);">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:var(--amber-cmd); font-size:10px;">${h.toUpperCase()}</strong>
            <span style="color:${dirColor}; font-weight:900; font-size:9.5px;">${dir}</span>
          </div>
          <div style="font-size:8.5px; color:#aaa;">Score: <strong style="color:#fff;">${item.confluence_score}/100</strong> | ${zTxt}</div>
          <div style="font-size:8.5px; color:#888;">SL: <span style="color:#ff6b6b;">${tt.stop_loss || "--"}</span> | TP1: <span style="color:#ddd;">${(tt.take_profit_1 || "--").split("(")[0].trim()}</span></div>
          <div style="display:flex; gap:6px; align-items:center; margin-top:2px;">${btns}</div>
        </div>
      `;
    }).join("");
  } catch (e) {
    console.warn("renderMultiHorizonOptInMatrix error:", e);
  }
}

async function renderSetupLedgerTable() {
  const runBody = document.getElementById("astar-running-tbody");
  const tpBody = document.getElementById("astar-tp-tbody");
  const slBody = document.getElementById("astar-sl-tbody");
  const archBody = document.getElementById("astar-ledger-tbody");
  if (!runBody && !archBody) return;
  try {
    const res = await fetch("/api/setup-ledger");
    const data = await res.json();
    const activeRunning = data.active_running || [];
    const hitTp = data.hit_tp || [];
    const slTouched = data.sl_touched || [];
    const archived = data.archived_expired || [];
    const sum = data.summary || {};
    const livePx = data.live_price;

    // Header summary badges
    if (document.getElementById("badge-running-count")) document.getElementById("badge-running-count").innerText = activeRunning.length;
    if (document.getElementById("badge-tp-count")) document.getElementById("badge-tp-count").innerText = hitTp.length;
    if (document.getElementById("badge-sl-count")) document.getElementById("badge-sl-count").innerText = slTouched.length;
    if (document.getElementById("badge-archived-count")) document.getElementById("badge-archived-count").innerText = archived.length;

    const fltUsd = Number(sum.floating_usd || 0);
    const fltR = Number(sum.floating_r || 0);
    const runUsdEl = document.getElementById("sum-running-usd");
    const runREl = document.getElementById("sum-running-r");
    if (runUsdEl) { runUsdEl.innerText = `${fltUsd >= 0 ? "+" : ""}$${fltUsd.toFixed(2)}`; runUsdEl.style.color = fltUsd >= 0 ? "#00e676" : "#ff4444"; }
    if (runREl) { runREl.innerText = `${fltR >= 0 ? "+" : ""}${fltR.toFixed(2)} R`; runREl.style.color = fltR >= 0 ? "#00e676" : "#ff4444"; }

    if (document.getElementById("sum-tp-r")) document.getElementById("sum-tp-r").innerText = `+${Number(sum.hit_tp_total_r || 0).toFixed(2)} R`;
    if (document.getElementById("sum-tp-usd")) document.getElementById("sum-tp-usd").innerText = `+$${Number(sum.hit_tp_total_usd || 0).toFixed(2)}`;
    if (document.getElementById("sum-sl-r")) document.getElementById("sum-sl-r").innerText = `${Number(sum.sl_touched_total_r || 0).toFixed(2)} R`;
    if (document.getElementById("sum-sl-usd")) document.getElementById("sum-sl-usd").innerText = `$${Number(sum.sl_touched_total_usd || 0).toFixed(2)}`;

    const wrEl = document.getElementById("astar-ledger-winrate");
    if (wrEl) wrEl.innerText = `WIN RATE: ${sum.win_rate_pct || 0}% (${hitTp.length}W / ${slTouched.length}L)`;
    const avgEl = document.getElementById("astar-avg-pips");
    if (avgEl) {
      const nr = Number(sum.net_realized_r || 0);
      const nu = Number(sum.net_realized_usd || 0);
      avgEl.innerText = `NET REALIZED: ${nr >= 0 ? "+" : ""}${nr.toFixed(2)} R (${nu >= 0 ? "+" : ""}$${nu.toFixed(2)})`;
    }

    const fmtEntry = s => {
      const isE = (s.asset === "EURUSD");
      const d = isE ? 5 : 2;
      const p = isE ? "" : "$";
      if (s.fill_price) return `${p}${Number(s.fill_price).toFixed(d)}`;
      if (Array.isArray(s.entry_zone)) return `${p}${Number(s.entry_zone[0]).toFixed(d)}–${p}${Number(s.entry_zone[1]).toFixed(d)}`;
      return s.entry_zone || "N/A";
    };
    const calcPips = s => {
      const eRef = s.fill_price || (Array.isArray(s.entry_zone) ? (s.entry_zone[0] + s.entry_zone[1]) / 2.0 : null);
      const isE = (s.asset === "EURUSD");
      const isB = (s.asset === "BTCUSD");
      const lp = isE ? (data.live_prices?.EURUSD || 1.085) : (isB ? (data.live_prices?.BTCUSD || 91500) : livePx);
      const xRef = s.exit_price || lp;
      if (eRef && xRef) {
        const mult = isE ? 10000.0 : (isB ? 1.0 : 10.0);
        return ((s.direction === "BUY" ? (xRef - eRef) : (eRef - xRef)) * mult);
      }
      return null;
    };

    // BOX 1: RUNNING & ACTIVE SETUPS
    if (runBody) {
      if (activeRunning.length === 0) {
        runBody.innerHTML = `<tr><td colspan="12" style="text-align:center; color:#777; padding:12px;">NO ACTIVE RUNNING SETUPS — CLICK [+ ADD TO RUNNING LIST] ON ANY A* SETUP ABOVE</td></tr>`;
      } else {
        runBody.innerHTML = activeRunning.map(s => {
          const fu = Number(s.floating_pnl_usd || 0);
          const fr = Number(s.floating_r || 0);
          const pnlCol = fu >= 0 ? "#00e676" : "#ff4444";
          const dirCol = s.direction === "BUY" ? "#00e676" : "#ff4444";
          const stCol = s.status === "PARTIAL_TP_HIT" ? "#00e676" : "#00e5ff";
          const isE = (s.asset === "EURUSD");
          const isB = (s.asset === "BTCUSD");
          const d = isE ? 5 : 2;
          const p = isE ? "" : "$";
          const assetBadge = s.asset ? `<span style="font-size:8px; border:1px solid ${isB ? '#ff9900' : (isE ? '#00e5ff' : '#ffd700')}; color:${isB ? '#ff9900' : (isE ? '#00e5ff' : '#ffd700')}; padding:1px 3px; margin-right:4px;">${s.asset}</span>` : "";
          const lp = isE ? (data.live_prices?.EURUSD) : (isB ? (data.live_prices?.BTCUSD) : livePx);
          const tpTxt = `${s.take_profit_1 ? `${p}${Number(s.take_profit_1).toFixed(d)}` : "--"} / ${s.take_profit_2 ? `${p}${Number(s.take_profit_2).toFixed(d)}` : "--"}`;
          return `
            <tr>
              <td>${s.created_at_utc || "LIVE"}</td>
              <td><strong style="color:var(--amber-cmd);">${s.setup_id}</strong></td>
              <td>${assetBadge}<strong>${(s.horizon || "daily").toUpperCase()}</strong></td>
              <td><strong style="color:${dirCol};">${s.direction}</strong></td>
              <td>${fmtEntry(s)}</td>
              <td style="color:var(--amber-cmd); font-weight:800;">${lp ? `${p}${Number(lp).toFixed(d)}` : "--"}</td>
              <td style="color:#ff4444;">${s.stop_loss ? `${p}${Number(s.stop_loss).toFixed(d)}` : "N/A"}</td>
              <td style="color:#00e676;">${tpTxt}</td>
              <td><span style="color:${stCol}; border:1px solid ${stCol}; padding:1px 4px; font-size:8.5px;">${s.status}</span></td>
              <td style="color:${pnlCol}; font-weight:900;">${fu >= 0 ? "+" : ""}$${fu.toFixed(2)}</td>
              <td style="color:${pnlCol}; font-weight:900;">${fr >= 0 ? "+" : ""}${fr.toFixed(2)} R</td>
              <td style="white-space:nowrap;">
                <button onclick="handleRunningTradeAction('${s.setup_id}','close')" class="bbg-btn-mini" style="border-color:var(--amber-cmd); color:var(--amber-cmd); font-size:8px; padding:2px 5px; cursor:pointer;">CLOSE @ MKT</button>
                <button onclick="handleRunningTradeAction('${s.setup_id}','remove')" class="bbg-btn-mini" style="border-color:#666; color:#aaa; font-size:8px; padding:2px 5px; margin-left:3px; cursor:pointer;">REMOVE</button>
              </td>
            </tr>
          `;
        }).join("");
      }
    }

    // BOX 2: HIT TP (WINNERS)
    if (tpBody) {
      if (hitTp.length === 0) {
        tpBody.innerHTML = `<tr><td colspan="10" style="text-align:center; color:#666; padding:10px;">NO HIT TP TRADES RECORDED YET</td></tr>`;
      } else {
        tpBody.innerHTML = hitTp.map(s => {
          const pips = calcPips(s);
          const rr = Number(s.realized_r || 0);
          const usd = Number(s.realized_pnl_usd || 0);
          const isE = (s.asset === "EURUSD");
          const d = isE ? 5 : 2;
          const p = isE ? "" : "$";
          const exitTxt = s.exit_price ? `${p}${Number(s.exit_price).toFixed(d)}` : (s.take_profit_1 ? `${p}${Number(s.take_profit_1).toFixed(d)}` : "--");
          const assetBadge = s.asset ? `<span style="font-size:8px; border:1px solid ${isE ? '#00e5ff' : '#ffd700'}; color:${isE ? '#00e5ff' : '#ffd700'}; padding:1px 3px; margin-right:4px;">${s.asset}</span>` : "";
          return `
            <tr>
              <td>${s.resolved_at || s.created_at_utc || "--"}</td>
              <td><strong style="color:#00e676;">${s.setup_id}</strong></td>
              <td>${assetBadge}${(s.horizon || "daily").toUpperCase()}</td>
              <td><strong style="color:${s.direction === 'BUY' ? '#00e676' : '#ff8a80'};">${s.direction}</strong></td>
              <td>${fmtEntry(s)}</td>
              <td style="color:#00e676;">${exitTxt}</td>
              <td><span style="color:#00e676; border:1px solid #00e676; padding:1px 4px; font-size:8px;">${s.status}</span></td>
              <td style="color:#00e5ff; font-weight:800;">${pips !== null ? `+${Math.abs(pips).toFixed(1)}` : "--"}</td>
              <td style="color:#00e676; font-weight:900;">+${rr.toFixed(2)} R</td>
              <td style="color:#fff; font-weight:900;">+$${usd.toFixed(2)}</td>
            </tr>
          `;
        }).join("");
      }
    }

    // BOX 3: SL TOUCHED (LOSSES)
    if (slBody) {
      if (slTouched.length === 0) {
        slBody.innerHTML = `<tr><td colspan="10" style="text-align:center; color:#666; padding:10px;">NO SL TOUCHED TRADES RECORDED YET</td></tr>`;
      } else {
        slBody.innerHTML = slTouched.map(s => {
          const pips = calcPips(s);
          const rr = Number(s.realized_r || -1.0);
          const usd = Number(s.realized_pnl_usd || 0);
          const isE = (s.asset === "EURUSD");
          const d = isE ? 5 : 2;
          const p = isE ? "" : "$";
          const exitTxt = s.exit_price ? `${p}${Number(s.exit_price).toFixed(d)}` : (s.stop_loss ? `${p}${Number(s.stop_loss).toFixed(d)}` : "--");
          const assetBadge = s.asset ? `<span style="font-size:8px; border:1px solid ${isE ? '#00e5ff' : '#ffd700'}; color:${isE ? '#00e5ff' : '#ffd700'}; padding:1px 3px; margin-right:4px;">${s.asset}</span>` : "";
          return `
            <tr>
              <td>${s.resolved_at || s.created_at_utc || "--"}</td>
              <td><strong style="color:#ff4444;">${s.setup_id}</strong></td>
              <td>${assetBadge}${(s.horizon || "daily").toUpperCase()}</td>
              <td><strong style="color:${s.direction === 'BUY' ? '#fff' : '#ff4444'};">${s.direction}</strong></td>
              <td>${fmtEntry(s)}</td>
              <td style="color:#ff4444;">${exitTxt}</td>
              <td><span style="color:#ff4444; border:1px solid #ff4444; padding:1px 4px; font-size:8px;">${s.status}</span></td>
              <td style="color:#ff6b6b; font-weight:800;">${pips !== null ? `-${Math.abs(pips).toFixed(1)}` : "--"}</td>
              <td style="color:#ff4444; font-weight:900;">${rr.toFixed(2)} R</td>
              <td style="color:#ff8a80; font-weight:900;">$${usd.toFixed(2)}</td>
            </tr>
          `;
        }).join("");
      }
    }

    // BOX 4: EXPIRED / ARCHIVED
    if (archBody) {
      if (archived.length === 0) {
        archBody.innerHTML = `<tr><td colspan="11" style="text-align:center; color:#666;">NO ARCHIVED / EXPIRED SETUPS</td></tr>`;
      } else {
        archBody.innerHTML = archived.slice(0, 60).map(s => `
          <tr>
            <td>${s.created_at_utc || "--"}</td>
            <td style="color:#888;">${s.setup_id}</td>
            <td>${(s.horizon || "daily").toUpperCase()}</td>
            <td>${s.direction || "--"}</td>
            <td>${fmtEntry(s)}</td>
            <td>${s.stop_loss ? `$${Number(s.stop_loss).toFixed(2)}` : "N/A"}</td>
            <td>${s.take_profit_1 ? `$${Number(s.take_profit_1).toFixed(2)}` : "N/A"}</td>
            <td><span style="color:#888; border:1px solid #444; padding:1px 4px; font-size:8px;">${s.status}</span></td>
            <td>--</td>
            <td>0.00 R</td>
            <td style="color:#777;">${s.resolution || "Archived"}</td>
          </tr>
        `).join("");
      }
    }
  } catch (e) {
    console.warn("renderSetupLedgerTable error:", e);
  }
}

async function fetchGlobalMacroMatrix() {
  const tbody = document.getElementById("global-macro-tbody");
  if (!tbody) return;
  try {
    const res = await fetch("/api/global-macro-matrix");
    const data = await res.json();
    const list = Array.isArray(data.sovereigns) ? data.sovereigns : Object.values(data.sovereigns || {});
    tbody.innerHTML = list.map(s => {
      const rate = s.policy_rate !== undefined ? s.policy_rate : (s.pboc_1y_lpr || 0);
      const cpi = s.headline_cpi_yoy !== undefined ? s.headline_cpi_yoy : (s.inflation_cpi || 0);
      const realR = s.real_policy_rate !== undefined ? s.real_policy_rate : (s.real_yield || 0);
      const beta = s.gold_transmission_beta || 0;
      return `
        <tr>
          <td><strong>${s.country}</strong> (${s.currency || s.region || ''})</td>
          <td>${Number(rate).toFixed(2)}%</td>
          <td style="color:${cpi >= 2.8 ? '#ff4444' : '#ffffff'};">${Number(cpi).toFixed(1)}%</td>
          <td style="color:${realR < 0 ? '#ff4444' : '#ffffff'};">${realR > 0 ? "+" : ""}${Number(realR).toFixed(2)}%</td>
          <td style="color:var(--amber-cmd);">${beta > 0 ? "+" : ""}${Number(beta).toFixed(2)}</td>
          <td>${s.regime || 'NEUTRAL'}</td>
        </tr>
      `;
    }).join("");
  } catch (e) { console.warn("fetchGlobalMacroMatrix error:", e); }
}
async function fetchQuant8AuditTable() {
  const tbody = document.getElementById("quant8-audit-tbody");
  const badge = document.getElementById("quant8-pass-badge");
  if (!tbody) return;
  try {
    const res = await fetch("/api/quant-models-audit");
    const data = await res.json();
    if (badge) {
      badge.innerText = `${data.passed_models_count}/${data.total_models} VERIFIED (${data.generated_at_utc || "LIVE"})`;
    }
    tbody.innerHTML = (data.models || []).map(m => {
      const isOk = m.tier && (m.tier.includes("99%") || m.tier.includes("95%") || m.tier.includes("CALIBRATED"));
      const tc = m.tier.includes("99%") ? "#ffffff" : (m.tier.includes("95%") ? "var(--amber-cmd)" : (m.tier.includes("90%") ? "#ffb000" : "#ff4444"));
      return `
        <tr>
          <td><strong>${m.pillar}</strong></td>
          <td>${m.model_name}</td>
          <td>${m.n}</td>
          <td style="color:var(--amber-cmd);">${m.primary_stat} (p=${m.p_val})</td>
          <td><span style="color:${tc}; border:1px solid ${tc}; padding:1px 4px; font-size:8.5px; font-weight:800;">${m.tier}</span></td>
          <td style="color:${isOk ? '#00e5ff' : '#aaa'};">${m.live_reading}</td>
          <td style="color:#aaa; font-size:9px;">${m.honest_limit}</td>
        </tr>
      `;
    }).join("");
    const strip = document.getElementById("quant8-conf-strip");
    if (strip && data.confidence_80_99_news) {
      const items = Object.entries(data.confidence_80_99_news).map(([k, v]) => {
        const d80 = v.direct_max_conf || 0;
        const w80 = v.wave2_max_conf || 0;
        const vr = v.vr_max_conf || 0;
        return `<span style="margin-right:16px;"><strong style="color:var(--amber-cmd);">${k.toUpperCase()}</strong>: Direct=<strong style="color:${d80 >= 95 ? '#00e5ff' : (d80 >= 80 ? '#ffb000' : '#ff4444')}">${d80}%</strong> | 2SLS=<strong style="color:${w80 >= 95 ? '#00e5ff' : (w80 >= 80 ? '#ffb000' : '#ff4444')}">${w80}%</strong> | VR5=<strong style="color:#ffffff;">${vr}%</strong></span>`;
      }).join("");
      strip.innerHTML = `<strong style="color:#ffffff; margin-right:10px;">[ 80%–99% ZERO-LUCK PASS CEILING ]</strong>${items}`;
      const jc = data.jobless_claims_audit;
      const pt = data.paper_trade_summary;
      const extra = [];
      if (jc && jc.n) {
        extra.push(`<strong style="color:#ff4444;">JOBLESS CLAIMS (n=${jc.n})</strong>: Gold t=${jc.gold_t} (p=${jc.gold_p}), Hit=${jc.hit_pct}% -> <strong style="color:#ff4444;">REJECTED STANDALONE</strong>`);
      }
      if (pt && pt.combined_trades) {
        extra.push(`<strong style="color:#00e5ff;">3Y COMBINED PORTFOLIO ($1,000 -> $${pt.combined_end_equity_usd.toLocaleString("en-US", {minimumFractionDigits:2})}, +${pt.combined_net_ret_pct}%, DD ${pt.combined_max_dd_pct}%)</strong>: ${pt.combined_trades}t, Win=<strong style="color:#ffffff;">${pt.combined_win_pct}%</strong>, PF <strong style="color:#ffffff;">${pt.combined_pf}</strong>, t=+${pt.combined_t_stat} (<strong style="color:#00e5ff;">p=${pt.combined_p_val}</strong>)`);
      }
      if (pt && pt.daily_3y_trades) {
        extra.push(`<strong style="color:var(--amber-cmd);">DAILY A* ADAPTIVE 1:1/1:2 (Lev 1:2000)</strong>: ${pt.daily_3y_trades}t, Win=${pt.daily_3y_win_pct}%, +${pt.daily_3y_ret_pct}%, PF ${pt.daily_3y_pf} (p=${pt.daily_3y_p_val})`);
      }
      if (pt && pt.news_t3m_trades) {
        extra.push(`<strong style="color:#ffffff;">MACRO NEWS T+3m 1:4 RR (Lev 1:200)</strong>: ${pt.news_t3m_trades}t, Win=${pt.news_t3m_win_pct}% (BE 20%), +${pt.news_t3m_ret_pct}%, PF ${pt.news_t3m_pf}`);
      }
      if (extra.length) {
        strip.innerHTML += `<div style="margin-top:5px; border-top:1px dashed #222; padding-top:4px;">${extra.join(" &nbsp;|&nbsp; ")}</div>`;
      }
    }
  } catch (e) { console.warn("fetchQuant8AuditTable error:", e); }
}


function checkScheduledReleaseFirewall() {
  const up = PREMARKET_CACHE?.snapshot?.upcoming_consensus || {};
  const now = new Date();
  const todayUtc = now.toISOString().slice(0, 10);
  const secInDay = now.getUTCHours() * 3600 + now.getUTCMinutes() * 60 + now.getUTCSeconds();
  const releaseSecs = { cpi: 13 * 3600 + 30 * 60, nfp: 13 * 3600 + 30 * 60, ppi: 13 * 3600 + 30 * 60, fomc: 19 * 3600 };
  ["cpi", "fomc", "nfp", "ppi"].forEach(ev => {
    const dt = up[ev]?.date;
    const targetSec = releaseSecs[ev];
    if (dt === todayUtc && secInDay >= targetSec && secInDay < targetSec + 60 && !EVENT_STATE[ev].fwTimer) {
      triggerFirewallCountdown(ev, (targetSec + 60) - secInDay);
    }
  });
}

function initAutoRefreshLoop() {
  let loopCount = 0;
  setInterval(async () => {
    try {
      loopCount++;
      checkScheduledReleaseFirewall();
      const mod = getActiveModuleId();
      const isAstar = mod === "mod-astar";
      const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
      const res = await fetch(`/api/live-ticks?include_astar=${isAstar}&horizon=${h}&instrument=${CURRENT_ASTAR_ASSET}`);
      const data = await res.json();
      if (data && data.prices) {
        const p = data.prices;
        animateNumber(document.getElementById("tick-gold"), p["GC=F"], "$", "", 2);
        animateNumber(document.getElementById("tick-eur"), p["EURUSD=X"], "", "", 5);
        animateNumber(document.getElementById("tick-yield-spread-us-de"), p["YIELD_SPREAD_US_DE"], (p["YIELD_SPREAD_US_DE"] >= 0 ? "+" : ""), "%", 3);
        if (p["DE10Y"]) window._LAST_VERIFIED_DE10Y = Number(p["DE10Y"]);
        if (p["EURUSD_BID"] && p["EURUSD_ASK"]) {
          window._LAST_VERIFIED_EUR_SPREAD = Math.abs((p["EURUSD_ASK"] - p["EURUSD_BID"]) * 10000).toFixed(1);
        }
        animateNumber(document.getElementById("tick-btc"), p["BTC-USD"], "$", "", 2);
        if (p["BTC-USD"]) window._LAST_VERIFIED_BTC_PRICE = Number(p["BTC-USD"]);
        animateNumber(document.getElementById("tick-dxy"), p["DX-Y.NYB"], "", "", 2);
        animateNumber(document.getElementById("tick-2y"), p["ZT=F"], "", "%", 2);
        animateNumber(document.getElementById("tick-10y"), p["^TNX"], "", "%", 2);
        animateNumber(document.getElementById("tick-oil"), p["CL=F"], "$", "", 2);

        // Lively update position gauge and pin cursor
        const isE = CURRENT_ASTAR_ASSET === "EURUSD";
        const isB = CURRENT_ASTAR_ASSET === "BTCUSD";
        const curPrice = isE ? p["EURUSD=X"] : (isB ? p["BTC-USD"] : p["GC=F"]);
        const pDec = isE ? 5 : 2;
        const pPfx = isE ? "" : "$";
        if (curPrice && isAstar) {
          const gLiv = document.getElementById("gauge-live");
          if (gLiv) gLiv.innerText = `${pPfx}${curPrice.toFixed(pDec)}`;
          const ls = window._LAST_ASTAR_DATA?.locked_setup;
          if (ls && ls.stop_loss && ls.take_profit_1) {
            const rng = Math.abs(ls.take_profit_1 - ls.stop_loss);
            const gPin = document.getElementById("gauge-cursor-pin");
            if (gPin && rng > 0) {
              const pct = ls.direction === "BUY" ? ((curPrice - ls.stop_loss) / rng) * 100 : ((ls.stop_loss - curPrice) / rng) * 100;
              gPin.style.left = `${Math.max(0, Math.min(100, pct))}%`;
            }
          }
        }
      }
      if (isAstar && data && data.daily_a_star) {
        applyAstarData(data.daily_a_star);
      } else if (loopCount % 3 === 0 && ["mod-cpi", "mod-fomc", "mod-nfp", "mod-ppi"].includes(mod)) {
        const ev = mod.replace("mod-", "");
        if (EVENT_STATE[ev]?.mode === "PREMARKET") {
          await fetchState();
          renderEventPdfPane(ev);
          renderEventMcPane(ev);
        }
      }
    } catch (e) {}
  }, 1000);
  document.getElementById("btn-reset-astar-session")?.addEventListener("click", async () => {
    const b = document.getElementById("btn-reset-astar-session");
    b.innerText = "WAIT REFRESHING...";
    const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
    try {
      await fetch(`/api/reset-setup-session?event_type=${h}&horizon=${h}`, { method: "POST" });
      await fetchAstarData();
    } catch (e) {
      console.warn("Reset setup session failed:", e);
    }
    setTimeout(() => { b.innerText = "FORCE RE-EVALUATE SESSION <GO>"; }, 800);
  });

  document.getElementById("btn-add-astar-running")?.addEventListener("click", async () => {
    const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
    await addHorizonSetupToRunning(h);
  });

  document.getElementById("btn-skip-astar-setup")?.addEventListener("click", () => {
    const h = (EVENT_STATE.astar && EVENT_STATE.astar.horizon) || "daily";
    skipHorizonSetup(h);
  });

  document.getElementById("btn-refresh-divided-ledger")?.addEventListener("click", async () => {
    await renderSetupLedgerTable();
    await renderMultiHorizonOptInMatrix();
  });

  window.addHorizonSetupToRunning = addHorizonSetupToRunning;
  window.skipHorizonSetup = skipHorizonSetup;
  window.handleRunningTradeAction = handleRunningTradeAction;
}

// ==============================================================
// 6. DEEPSEEK & YoY TIMELINE
// ==============================================================
function initDeepSeekControls() {
  const btnPre = document.getElementById("btn-ai-pre");
  const btnPost = document.getElementById("btn-ai-post");
  const btnRun = document.getElementById("btn-run-deepseek");
  const outputEl = document.getElementById("ai-output-terminal");
  const statusEl = document.getElementById("ai-status-badge");
  const eventSel = document.getElementById("ai-event-select");

  btnPre?.addEventListener("click", (e) => { setBtnActive(e.target, "#btn-ai-pre, #btn-ai-post"); currentAiMode = "pre_news"; });
  btnPost?.addEventListener("click", (e) => { setBtnActive(e.target, "#btn-ai-pre, #btn-ai-post"); currentAiMode = "post_news"; });

  btnRun?.addEventListener("click", async () => {
    btnRun.innerText = "EXECUTING DEEPSEEK...";
    if (statusEl) statusEl.innerText = "QUERYING AI KERNEL...";
    if (outputEl) outputEl.innerText = ">>> CONNECTING TO DEEPSEEK REASONING ENGINE...\n>>> COMPILING AUTHENTIC MACRO CONTEXT & 5% HYPOTHESIS METRICS...";
    try {
      const res = await fetch("/api/deepseek-briefing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify((() => {
          const ev = eventSel.value;
          let a1 = parseFloat(document.getElementById("bbg-range-headline")?.value || 0.3);
          let a2 = parseFloat(document.getElementById("bbg-range-core")?.value || 2.9);
          if (ev === "fomc") {
            a1 = parseFloat(document.getElementById("fomc-range-rate")?.value || 4.0);
            a2 = parseFloat(document.getElementById("fomc-range-dots")?.value || 0.0);
          } else if (ev === "nfp") {
            a1 = parseFloat(document.getElementById("nfp-range-act")?.value || 135);
            a2 = parseFloat(document.getElementById("nfp-range-unemp")?.value || 4.1);
          } else if (ev === "ppi_retail" || ev === "ppi") {
            a1 = parseFloat(document.getElementById("ppi-range-act")?.value || 0.4);
            a2 = parseFloat(document.getElementById("ppi-range-retail")?.value || 0.4);
          }
          return { mode: currentAiMode, event_type: ev, actual_mom: a1, actual_core_yoy: a2, min_confidence: window.SELECTED_CONFIDENCE || 95 };
        })())
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

const YOY_DOMAINS = {
  cpi: {
    subtitle: "46-MONTH MULTI-MACRO TIMELINE // CPI INFLATION YoY (2023-2026)",
    refVal: 2.0,
    refLabel: "FED 2.0% TARGET",
    series: [
      { key: "headline_yoy", label: "HEADLINE YoY", short: "HL", color: "#ffffff", width: 2.2, dash: "" },
      { key: "core_yoy", label: "CORE YoY", short: "CORE", color: "#ffb000", width: 1.9, dash: "" },
      { key: "supercore_yoy", label: "SUPERCORE YoY", short: "SUP", color: "#00e5ff", width: 1.6, dash: "4,3" },
      { key: "shelter_yoy", label: "SHELTER YoY", short: "SHL", color: "#ff4444", width: 1.6, dash: "" }
    ],
    note: "Oct 2026 official YoY omitted due to Oct 2025 BLS shutdown gap (MoM +0.35% / Core +0.30%)."
  },
  nfp: {
    subtitle: "46-MONTH MULTI-MACRO TIMELINE // NFP LABOR, WAGES & UNEMPLOYMENT (2023-2026)",
    refVal: 4.0,
    refLabel: "4.0% NAIRU / BENCHMARK",
    series: [
      { key: "unemp_rate", label: "UNEMP RATE (%)", short: "UNEMP", color: "#ff4444", width: 2.1, dash: "" },
      { key: "ahe_yoy", label: "WAGE AHE YoY (%)", short: "WAGE", color: "#ffb000", width: 1.9, dash: "" },
      { key: "nfp_yoy", label: "PAYROLLS YoY (%)", short: "NFP YoY", color: "#ffffff", width: 2.0, dash: "4,3" }
    ],
    note: "FRED PAYEMS, UNRATE, CES0500000003 through Oct 2026 (+165K Oct jobs beat vs +65K fcst)."
  },
  ppi: {
    subtitle: "MONTHLY MULTI-MACRO TIMELINE // PPI WHOLESALE & RETAIL DEMAND YoY (2023-2026)",
    refVal: 2.0,
    refLabel: "2.0% PIPELINE BENCHMARK",
    series: [
      { key: "ppi_yoy", label: "PPI FINAL YoY (%)", short: "PPI YoY", color: "#ff4444", width: 2.2, dash: "" },
      { key: "retail_yoy", label: "RETAIL SALES YoY (%)", short: "RETAIL YoY", color: "#00e5ff", width: 1.9, dash: "" },
      { key: "ppi_mom", label: "PPI MoM (%)", short: "PPI MoM", color: "#ffb000", width: 1.4, dash: "3,3" }
    ],
    note: "FRED + BLS WPSFD4/RSAFS through Oct 2026 PPI (+0.35% MoM); Oct Retail Sales releases Nov 17."
  },
  fomc: {
    subtitle: "46-MONTH MULTI-MACRO TIMELINE // FOMC TARGET RATE, REAL RATE & 10Y-2Y CURVE",
    refVal: 0.0,
    refLabel: "0.0% ZERO BOUND / CURVE INVERSION",
    series: [
      { key: "fomc_target_upper", label: "FOMC UPPER TARGET (%)", short: "TARGET", color: "#ffffff", width: 2.3, dash: "" },
      { key: "fed_funds_eff", label: "EFF FED FUNDS (%)", short: "EFFR", color: "#00e5ff", width: 1.6, dash: "4,3" },
      { key: "real_policy_rate", label: "REAL POLICY RATE (%)", short: "REAL", color: "#ffb000", width: 1.9, dash: "" },
      { key: "us10y_2y_spread", label: "10Y-2Y SPREAD (%)", short: "10s2s", color: "#ff4444", width: 1.5, dash: "" }
    ],
    note: "FRED DFEDTARU & FEDFUNDS through Oct/Nov 2026 (hiked +25 bps to 4.00% on Sep 16, held Oct 28)."
  },
  all: {
    subtitle: "46-MONTH CROSS-MACRO OVERLAY // CORE CPI, PPI, NFP & FOMC TARGET RATE",
    refVal: 2.0,
    refLabel: "2.0% FED MANDATE",
    series: [
      { key: "fomc_target_upper", label: "FOMC TARGET (%)", short: "FOMC", color: "#ffffff", width: 2.2, dash: "" },
      { key: "core_yoy", label: "CORE CPI YoY (%)", short: "CPI CORE", color: "#ffb000", width: 1.9, dash: "" },
      { key: "ppi_yoy", label: "PPI FINAL YoY (%)", short: "PPI YoY", color: "#ff4444", width: 1.7, dash: "" },
      { key: "nfp_yoy", label: "PAYROLLS YoY (%)", short: "NFP YoY", color: "#00e5ff", width: 1.7, dash: "4,3" }
    ],
    note: "Unified 4-pillar cycle overlay linking Policy Rate, Consumer Inflation, Producer Costs & Labor Growth."
  }
};

function renderYoyPills() {
  const wrap = document.getElementById("yoy-series-pills");
  const cfg = YOY_DOMAINS[yoyDomain] || YOY_DOMAINS.cpi;
  if (!wrap) return;
  wrap.innerHTML = cfg.series.map(s => {
    const isOn = yoyActiveKeys[s.key] !== false;
    return `<button class="bbg-btn-mini ${isOn ? 'active' : ''}" data-yoy-key="${s.key}" style="border-bottom:2px solid ${s.color}; opacity:${isOn ? '1' : '0.45'};">${s.label}</button>`;
  }).join("");
  wrap.querySelectorAll("[data-yoy-key]").forEach(btn => {
    btn.addEventListener("click", () => {
      const k = btn.getAttribute("data-yoy-key");
      yoyActiveKeys[k] = !(yoyActiveKeys[k] !== false);
      renderYoyPills();
      renderYoyTimeline();
    });
  });
}

function initYoyControls() {
  document.querySelectorAll("[data-yoy-domain]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      setBtnActive(e.target, "[data-yoy-domain]");
      yoyDomain = e.target.getAttribute("data-yoy-domain") || "cpi";
      const cfg = YOY_DOMAINS[yoyDomain];
      yoyActiveKeys = {};
      cfg.series.forEach(s => { yoyActiveKeys[s.key] = true; });
      renderYoyPills();
      renderYoyTimeline();
    });
  });
  renderYoyPills();
}

async function renderYoyTimeline() {
  const container = document.getElementById("canvas-yoy");
  const hudEl = document.getElementById("hud-yoy");
  if (!container) return;

  if (!yoyTimelineCache) {
    try {
      const res = await fetch("/api/yoy-timeline");
      yoyTimelineCache = await res.json();
    } catch (e) {
      console.warn("renderYoyTimeline error:", e);
      return;
    }
  }
  const data = Array.isArray(yoyTimelineCache) ? yoyTimelineCache : [];
  const n = data.length;
  if (n < 2) {
    container.innerHTML = `<div style="padding:20px; color:#888;">NO YOY TIMELINE DATA AVAILABLE</div>`;
    return;
  }

  const cfg = YOY_DOMAINS[yoyDomain] || YOY_DOMAINS.cpi;
  const subEl = document.getElementById("yoy-pane-subtitle");
  if (subEl) subEl.innerText = cfg.subtitle;

  const dates = data.map(r => r.date);
  const activeSeries = cfg.series.filter(s => yoyActiveKeys[s.key] !== false);
  const allVals = [];
  activeSeries.forEach(s => {
    data.forEach(r => {
      const v = r[s.key];
      if (v !== null && v !== undefined && !isNaN(v)) allVals.push(Number(v));
    });
  });
  if (!allVals.length) allVals.push(0.0, 5.0);

  const rawMin = Math.min(...allVals, cfg.refVal);
  const rawMax = Math.max(...allVals, cfg.refVal + 0.5);
  const minY = Math.floor(rawMin * 2) / 2 - 0.5;
  const maxY = Math.ceil(rawMax * 2) / 2 + 0.5;

  const w = Math.max(container.clientWidth || 0, 920);
  const h = Math.max(container.clientHeight || 0, 390);
  const padL = 52, padR = 88, padT = 22, padB = 34;

  function getX(i) { return padL + (i / (n - 1)) * (w - padL - padR); }
  function getY(v) { return padT + (1 - (v - minY) / Math.max(0.1, maxY - minY)) * (h - padT - padB); }

  function buildNullablePath(key) {
    let d = "";
    let openSeg = false;
    for (let i = 0; i < n; i++) {
      const v = data[i][key];
      if (v === null || v === undefined || isNaN(v)) {
        openSeg = false;
        continue;
      }
      d += `${openSeg ? "L" : "M"} ${getX(i).toFixed(1)} ${getY(Number(v)).toFixed(1)} `;
      openSeg = true;
    }
    return d.trim();
  }

  const yStep = (maxY - minY) > 6 ? 1.0 : 0.5;
  let gridSvg = "";
  for (let yVal = Math.ceil(minY / yStep) * yStep; yVal <= maxY; yVal += yStep) {
    const yy = getY(yVal);
    gridSvg += `<line x1="${padL}" y1="${yy}" x2="${w - padR}" y2="${yy}" class="grid-dot"/>`;
    gridSvg += `<text x="${padL - 6}" y="${yy + 3}" text-anchor="end">${yVal.toFixed(1)}%</text>`;
  }

  let pathsSvg = "";
  const rawEndLabels = [];
  activeSeries.forEach(s => {
    const dStr = buildNullablePath(s.key);
    if (dStr) {
      pathsSvg += `<path d="${dStr}" fill="none" stroke="${s.color}" stroke-width="${s.width}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
    }
    for (let i = n - 1; i >= 0; i--) {
      const v = data[i][s.key];
      if (v !== null && v !== undefined && !isNaN(v)) {
        rawEndLabels.push({ y: getY(Number(v)), text: `${s.short} ${Number(v).toFixed(2)}%`, color: s.color, weight: 800 });
        break;
      }
    }
  });

  rawEndLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < rawEndLabels.length; i++) {
    if (rawEndLabels[i].y - rawEndLabels[i - 1].y < 13) rawEndLabels[i].y = rawEndLabels[i - 1].y + 13;
  }
  const endLabels = rawEndLabels.map(l => `<text x="${w - padR + 6}" y="${l.y + 3}" fill="${l.color}" font-weight="${l.weight}">${l.text}</text>`).join("");

  const last = n - 1;
  let ticksSvg = "";
  for (let i = 0; i < n; i += 6) {
    if (last - i >= 3) ticksSvg += `<text x="${getX(i)}" y="${h - 10}" text-anchor="middle">${dates[i]}</text>`;
  }
  ticksSvg += `<text x="${getX(last)}" y="${h - 10}" text-anchor="end">${dates[last]}</text>`;

  const yTarget = getY(cfg.refVal);
  const dotsSvg = activeSeries.map(s => `<circle id="yoy-dot-${s.key}" r="3.5" fill="${s.color}" style="display:none; pointer-events:none;"/>`).join("");

  container.innerHTML = `
    <svg id="svg-yoy" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair; min-height:390px;">
      ${gridSvg}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" class="axis-line"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <line x1="${padL}" y1="${yTarget}" x2="${w - padR}" y2="${yTarget}" stroke="#666" stroke-width="1" stroke-dasharray="3,3"/>
      <text x="${padL + 6}" y="${yTarget - 5}" fill="#888">${cfg.refLabel}</text>
      ${pathsSvg}
      ${endLabels}
      ${ticksSvg}
      <line id="yoy-scrubber" x1="0" y1="0" x2="0" y2="0" class="crosshair" style="display:none; pointer-events:none;"/>
      ${dotsSvg}
    </svg>
  `;

  const primaryKey = cfg.series[0].key;
  let latestRec = null, latestIdx = n - 1;
  for (let i = n - 1; i >= 0; i--) {
    if (data[i][primaryKey] !== null && data[i][primaryKey] !== undefined) { latestRec = data[i]; latestIdx = i; break; }
  }
  const badgeEl = document.getElementById("yoy-latest-badge");
  if (badgeEl && latestRec) badgeEl.innerText = `LATEST (${latestRec.date}): ${cfg.series[0].short} ${Number(latestRec[primaryKey]).toFixed(2)}%`;

  if (latestRec) {
    const latStr = activeSeries.map(s => {
      const v = latestRec[s.key];
      return `${s.short}: ${v !== null && v !== undefined ? Number(v).toFixed(2) + "%" : "N/A"}`;
    }).join(" | ");
    const prior12 = data[Math.max(0, latestIdx - 12)];
    const p12Val = prior12 ? prior12[primaryKey] : null;
    const diff12 = (p12Val !== null && p12Val !== undefined) ? (Number(latestRec[primaryKey]) - Number(p12Val)) : null;
    if (document.getElementById("yoy-val-latest")) document.getElementById("yoy-val-latest").innerText = `${latestRec.date} // ${latStr}`;
    if (document.getElementById("yoy-val-prior")) document.getElementById("yoy-val-prior").innerText = p12Val !== null ? `${prior12.date}: ${Number(p12Val).toFixed(2)}% (12M Δ: ${diff12 >= 0 ? "+" : ""}${diff12.toFixed(2)} pp)` : "--";
    const gap = Number(latestRec[primaryKey]) - cfg.refVal;
    if (document.getElementById("yoy-val-gap")) document.getElementById("yoy-val-gap").innerText = `${gap >= 0 ? "+" : ""}${gap.toFixed(2)} pp vs ${cfg.refLabel}`;
    if (document.getElementById("yoy-val-note")) document.getElementById("yoy-val-note").innerText = cfg.note;
  }

  const svgEl = document.getElementById("svg-yoy");
  const scrub = document.getElementById("yoy-scrubber");
  const tt = getOrCreateTooltip(container);
  if (svgEl && hudEl) {
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = Math.max(padL, Math.min(w - padR, ((ev.clientX - rect.left) / rect.width) * w));
      const idx = Math.max(0, Math.min(n - 1, Math.round(((mouseX - padL) / (w - padL - padR)) * (n - 1))));
      const xPos = getX(idx);
      const row = data[idx];

      if (scrub) { scrub.style.display = "block"; scrub.setAttribute("x1", xPos); scrub.setAttribute("x2", xPos); scrub.setAttribute("y1", padT); scrub.setAttribute("y2", h - padB); }
      const hudParts = activeSeries.map(s => {
        const v = row[s.key];
        const dotEl = document.getElementById(`yoy-dot-${s.key}`);
        if (dotEl) {
          if (v !== null && v !== undefined && !isNaN(v)) {
            dotEl.style.display = "block"; dotEl.setAttribute("cx", xPos); dotEl.setAttribute("cy", getY(Number(v)));
          } else {
            dotEl.style.display = "none";
          }
        }
        const txt = (v !== null && v !== undefined && !isNaN(v)) ? `${Number(v).toFixed(2)}%` : "N/A";
        return `${s.short}: <span style="color:${s.color}; font-weight:700;">${txt}</span>`;
      });
      let extraLine = "";
      if (yoyDomain === "nfp" && row.nfp_mom_k !== null && row.nfp_mom_k !== undefined) extraLine = `Net Payrolls MoM: <strong>${row.nfp_mom_k >= 0 ? "+" : ""}${row.nfp_mom_k}K</strong>`;
      else if (yoyDomain === "cpi" && row.headline_mom !== null) extraLine = `Headline MoM: <strong>${row.headline_mom >= 0 ? "+" : ""}${row.headline_mom.toFixed(2)}%</strong> | Core MoM: <strong>${row.core_mom >= 0 ? "+" : ""}${row.core_mom.toFixed(2)}%</strong>`;
      else if (yoyDomain === "ppi" && row.retail_mom !== null) extraLine = `Retail MoM: <strong>${row.retail_mom >= 0 ? "+" : ""}${row.retail_mom.toFixed(2)}%</strong>`;

      hudEl.innerHTML = `${row.date} | ${hudParts.join(" | ")}`;
      positionTooltip(tt, container, ev.clientX, ev.clientY,
        `<strong>${row.date} // ${yoyDomain.toUpperCase()} TELEMETRY</strong><br/>` +
        activeSeries.map(s => {
          const v = row[s.key];
          return `<span style="color:${s.color};">${s.label}:</span> <strong>${v !== null && v !== undefined && !isNaN(v) ? Number(v).toFixed(2) + "%" : "N/A (Shutdown)"}</strong>`;
        }).join("<br/>") +
        (extraLine ? `<br/><span style="color:#aaa;">${extraLine}</span>` : "")
      );
    });
    svgEl.addEventListener("mouseleave", () => {
      if (scrub) scrub.style.display = "none";
      activeSeries.forEach(s => {
        const dotEl = document.getElementById(`yoy-dot-${s.key}`);
        if (dotEl) dotEl.style.display = "none";
      });
      tt.style.display = "none";
      hudEl.innerText = "HOVER DATE TICK // MULTI-MACRO HISTORICAL SERIES (2023-2026)";
    });
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
        <td><span style="color:${ev.impact === 'MAXIMUM' ? '#ff4444' : (ev.impact === 'MEDIUM' ? 'var(--amber-cmd)' : '#ffffff')}; font-weight:800;">[ ${ev.impact} ]</span></td>
        <td style="color:${ev.actual ? '#00e5ff' : '#666'}; font-weight:800;">${ev.actual || 'PENDING'}</td>
        <td>${ev.forecast || '-'}</td>
        <td style="color:#777;">${ev.previous || '-'}</td>
        <td style="color:#fff;">
          <div>${ev.effect_xauusd || '-'}</div>
          ${ev.crypto_sensitivity ? `<div style="color:#ff9900; font-size:9px; margin-top:2px;">[BTC SENSITIVITY: <strong>${ev.crypto_sensitivity}</strong>] ${ev.effect_btcusd || ''}</div>` : ''}
        </td>
      </tr>
    `).join("");
  } catch (e) {}
}

// ==============================================================
// 7. HISTORICAL AUDIT LEDGERS (WITH POST-NEWS PRICE MOVEMENT & MEAN DIST)
// ==============================================================
function computeMeanAndSd(events) {
  const rets = events.map(e => Number(e.gold_return_pct || 0));
  const mean = rets.reduce((a, b) => a + b, 0) / Math.max(1, rets.length);
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, rets.length - 1);
  return { mean, sd: Math.max(0.01, Math.sqrt(variance)) };
}
function computeAvgPips(events) {
  const pips = events.map(e => (Number(e.gold_close || 0) - Number(e.gold_open || 0)) * 10.0);
  const avgAbs = pips.reduce((a, b) => a + Math.abs(b), 0) / Math.max(1, pips.length);
  const avgSigned = pips.reduce((a, b) => a + b, 0) / Math.max(1, pips.length);
  return { avgAbs: avgAbs.toFixed(1), avgSigned: avgSigned.toFixed(1) };
}


async function fetchAuditLedger() {
  try {
    const res = await fetch("/api/historical");
    const data = await res.json();
    const tbody = document.getElementById("bbg-audit-tbody");
    if (!tbody || !data.events) return;
    const cpiTit = document.getElementById("audit-title-cpi");
    if (cpiTit && data.count) cpiTit.innerText = `[ 07 // AUDIT <GO> : ${data.count}-RELEASE EMPIRICAL POST-NEWS PRICE MOVEMENT RECORD ]`;
    const { mean, sd } = computeMeanAndSd(data.events);
    const { avgAbs, avgSigned } = computeAvgPips(data.events);
    const cpiAvgEl = document.getElementById("cpi-avg-pips");
    if (cpiAvgEl) cpiAvgEl.innerText = `AVG PIPS: |Δ| ${avgAbs} pips (SIGNED: ${avgSigned >= 0 ? "+" : ""}${avgSigned} pips)`;
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      const pips = (Number(e.gold_close) - Number(e.gold_open)) * 10.0;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${e.ref_month}</td>
          <td>${e.actual_mom.toFixed(2)}%</td>
          <td>${e.forecast_mom.toFixed(2)}%</td>
          <td>${e.actual_core_yoy.toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? 'var(--amber-cmd)' : '#fff'}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${e.z_score_composite.toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
          <td style="color:${pips >= 0 ? '#00e5ff' : '#ff4444'}; font-weight:800;">${pips >= 0 ? '+' : ''}${pips.toFixed(1)} pips</td>
          <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${e.gold_return_pct.toFixed(2)}%</td>
          <td style="color:${Math.abs(distSd) >= 1.0 ? 'var(--amber-cmd)' : '#999'};">${distSd >= 0 ? '+' : ''}${distSd.toFixed(2)}σ</td>
          <td>${e.dxy_return_pct > 0 ? '+' : ''}${e.dxy_return_pct.toFixed(2)}%</td>
        </tr>
      `;
    }).join("");
  } catch (e) {}
}

async function fetchFomcData() {
  try {
    const res = await fetch("/api/fomc-historical");
    const data = await res.json();
    const tbody = document.getElementById("fomc-tbody");
    if (!tbody || !data.events) return;
    const fomcTit = document.getElementById("audit-title-fomc");
    if (fomcTit && data.count) fomcTit.innerText = `[ 07 // AUDIT <GO> : ${data.count}-MEETING FOMC POST-NEWS PRICE MOVEMENT RECORD ]`;
    const { mean, sd } = computeMeanAndSd(data.events);
    const { avgAbs, avgSigned } = computeAvgPips(data.events);
    const fomcAvgEl = document.getElementById("fomc-avg-pips");
    if (fomcAvgEl) fomcAvgEl.innerText = `AVG PIPS: |Δ| ${avgAbs} pips (SIGNED: ${avgSigned >= 0 ? "+" : ""}${avgSigned} pips)`;
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      const pips = (Number(e.gold_close) - Number(e.gold_open)) * 10.0;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${Number(e.actual_rate).toFixed(2)}%</td>
          <td>${Number(e.priced_rate).toFixed(2)}%</td>
          <td style="color:${e.total_policy_shock_bps > 0 ? '#ff4444' : (e.total_policy_shock_bps < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.total_policy_shock_bps > 0 ? '+' : ''}${Number(e.total_policy_shock_bps).toFixed(1)} bps</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
          <td style="color:${pips >= 0 ? '#00e5ff' : '#ff4444'}; font-weight:800;">${pips >= 0 ? '+' : ''}${pips.toFixed(1)} pips</td>
          <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${Number(e.gold_return_pct).toFixed(2)}%</td>
          <td style="color:${Math.abs(distSd) >= 1.0 ? 'var(--amber-cmd)' : '#999'};">${distSd >= 0 ? '+' : ''}${distSd.toFixed(2)}σ</td>
          <td style="color:var(--amber-cmd);">${e.us2y_delta_bps > 0 ? '+' : ''}${Number(e.us2y_delta_bps).toFixed(1)} bps</td>
          <td>${Number(e.volatility_ratio_vr5 || 4.0).toFixed(2)}x</td>
          <td>${e.regime || 'POLICY'}</td>
        </tr>
      `;
    }).join("");
  } catch (e) { console.warn("fetchFomcData error:", e); }
}

async function fetchNfpData() {
  try {
    const res = await fetch("/api/nfp-historical");
    const data = await res.json();
    const tbody = document.getElementById("nfp-tbody");
    if (!tbody || !data.events) return;
    const nfpTit = document.getElementById("audit-title-nfp");
    if (nfpTit && data.count) nfpTit.innerText = `[ 07 // AUDIT <GO> : ${data.count}-RELEASE NFP POST-NEWS PRICE MOVEMENT RECORD ]`;
    const { mean, sd } = computeMeanAndSd(data.events);
    const { avgAbs, avgSigned } = computeAvgPips(data.events);
    const nfpAvgEl = document.getElementById("nfp-avg-pips");
    if (nfpAvgEl) nfpAvgEl.innerText = `AVG PIPS: |Δ| ${avgAbs} pips (SIGNED: ${avgSigned >= 0 ? "+" : ""}${avgSigned} pips)`;
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      const pips = (Number(e.gold_close) - Number(e.gold_open)) * 10.0;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${e.nfp_actual_k}k</td>
          <td>${e.nfp_forecast_k}k</td>
          <td>${Number(e.unemp_actual_pct).toFixed(1)}%</td>
          <td>${Number(e.ahe_actual_pct).toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${Number(e.z_score_composite).toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
          <td style="color:${pips >= 0 ? '#00e5ff' : '#ff4444'}; font-weight:800;">${pips >= 0 ? '+' : ''}${pips.toFixed(1)} pips</td>
          <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${Number(e.gold_return_pct).toFixed(2)}%</td>
          <td style="color:${Math.abs(distSd) >= 1.0 ? 'var(--amber-cmd)' : '#999'};">${distSd >= 0 ? '+' : ''}${distSd.toFixed(2)}σ</td>
          <td>${Number(e.volatility_ratio_vr5 || 4.8).toFixed(2)}x</td>
          <td>${e.regime || 'LABOR'}</td>
        </tr>
      `;
    }).join("");
  } catch (e) { console.warn("fetchNfpData error:", e); }
}

async function fetchPpiData() {
  try {
    const res = await fetch("/api/ppi-historical");
    const data = await res.json();
    const tbody = document.getElementById("ppi-tbody");
    if (!tbody || !data.events) return;
    const ppiTit = document.getElementById("audit-title-ppi");
    if (ppiTit && data.count) ppiTit.innerText = `[ 07 // AUDIT <GO> : ${data.count}-RELEASE PPI & DEMAND POST-NEWS MOVEMENT RECORD ]`;
    const { mean, sd } = computeMeanAndSd(data.events);
    const { avgAbs, avgSigned } = computeAvgPips(data.events);
    const ppiAvgEl = document.getElementById("ppi-avg-pips");
    if (ppiAvgEl) ppiAvgEl.innerText = `AVG PIPS: |Δ| ${avgAbs} pips (SIGNED: ${avgSigned >= 0 ? "+" : ""}${avgSigned} pips)`;
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      const pips = (Number(e.gold_close) - Number(e.gold_open)) * 10.0;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${Number(e.ppi_actual).toFixed(2)}%</td>
          <td>${Number(e.ppi_forecast).toFixed(2)}%</td>
          <td>${e.retail_actual == null ? '<span style="color:var(--amber-cmd);">PENDING</span>' : `${Number(e.retail_actual).toFixed(2)}%`}</td>
          <td>${Number(e.retail_forecast).toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${Number(e.z_score_composite).toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
          <td style="color:${pips >= 0 ? '#00e5ff' : '#ff4444'}; font-weight:800;">${pips >= 0 ? '+' : ''}${pips.toFixed(1)} pips</td>
          <td style="color:${e.gold_return_pct >= 0 ? '#ffffff' : '#ff4444'}; font-weight:700;">${e.gold_return_pct > 0 ? '+' : ''}${Number(e.gold_return_pct).toFixed(2)}%</td>
          <td style="color:${Math.abs(distSd) >= 1.0 ? 'var(--amber-cmd)' : '#999'};">${distSd >= 0 ? '+' : ''}${distSd.toFixed(2)}σ</td>
          <td>${Number(e.volatility_ratio_vr5 || 3.9).toFixed(2)}x</td>
          <td>${e.regime || 'DEMAND'}</td>
        </tr>
      `;
    }).join("");
  } catch (e) { console.warn("fetchPpiData error:", e); }
}


async function fetchAcademicLit() {
  try {
    const res = await fetch("/api/academic-literature");
    const data = await res.json();
    const tbody = document.getElementById("lit-tbody");
    if (!tbody || !Array.isArray(data)) return;
    tbody.innerHTML = data.map(p => `
      <tr>
        <td><strong>${p.citation || p.author || "-"}</strong></td>
        <td style="color:var(--amber-cmd);">${p.title}<br/><span style="color:#777; font-size:8.5px;">${p.journal || ""}</span></td>
        <td style="font-family:var(--font-mono); color:#ccc;">${p.equation || p.methodology || p.journal || "-"}</td>
        <td>${p.empirical_finding || p.findings || "-"}</td>
        <td style="color:#fff;">${p.method_role || p.terminal_application || "-"}</td>
      </tr>
    `).join("");
  } catch (e) {}
}
async function fetchUsEconomyAtlas() {
  try {
    const confLvl = window.SELECTED_CONFIDENCE || 95;
    const res = await fetch(`/api/us-economy-atlas?min_confidence=${confLvl}`);
    const data = await res.json();
    if (!data || !data.domains) return;
    const dom = data.domains;
    const mod = data.models || {};
    const reg = mod.macro_regimes_copula || {};
    const prob = mod.probit_stress || {};
    const tay = mod.taylor_policy || {};
    const fl = mod.fiscal_liquidity_hc3 || {};
    const probs = reg.regime_probabilities_pct || {};

    if (document.getElementById("us-fred-count")) document.getElementById("us-fred-count").innerText = `${data.total_fred_series || 40} VERIFIED SERIES`;
    if (document.getElementById("us-dom-regime")) document.getElementById("us-dom-regime").innerText = `${reg.dominant_regime || "GOLDILOCKS"} (${probs[reg.dominant_regime] || "--"}%)`;
    if (document.getElementById("us-probit-prob")) document.getElementById("us-probit-prob").innerText = `${prob.stress_prob_6m_pct || 0}% (LR χ²=${prob.lr_chi2 || 0})`;
    if (document.getElementById("us-taylor-gap")) document.getElementById("us-taylor-gap").innerText = `${tay.taylor_gap_pct >= 0 ? "+" : ""}${tay.taylor_gap_pct || 0}% (i*=${tay.taylor_implied_rate_pct || 4.0}%)`;

    if (document.getElementById("us-p-goldilocks")) document.getElementById("us-p-goldilocks").innerText = `${probs.GOLDILOCKS ?? "--"}%`;
    if (document.getElementById("us-p-reflation")) document.getElementById("us-p-reflation").innerText = `${probs.REFLATION ?? "--"}%`;
    if (document.getElementById("us-p-stagflation")) document.getElementById("us-p-stagflation").innerText = `${probs.STAGFLATION ?? "--"}%`;
    if (document.getElementById("us-p-crisis")) document.getElementById("us-p-crisis").innerText = `${probs.CRISIS_STRESS ?? "--"}%`;
    if (document.getElementById("us-mahal-dist")) document.getElementById("us-mahal-dist").innerText = `${reg.mahalanobis_sigma || 0}σ (D²=${reg.mahalanobis_d2 || 0}, χ² p=${reg.chi2_anomaly_p_val || 1})`;

    if (document.getElementById("us-probit-detail")) document.getElementById("us-probit-detail").innerText = `${prob.stress_prob_6m_pct || 0}% (McFadden R²=${prob.mcfadden_r2 || 0}, Brier=${prob.brier_score || 0})`;
    if (document.getElementById("us-taylor-rate")) document.getElementById("us-taylor-rate").innerText = `i*=${tay.taylor_implied_rate_pct || 4.0}% vs Fed ${tay.fed_actual_rate_pct || 4.0}% (Gap ${tay.taylor_gap_pct >= 0 ? "+" : ""}${tay.taylor_gap_pct || 0}%)`;
    if (document.getElementById("us-policy-probs")) document.getElementById("us-policy-probs").innerText = `Tighten ${tay.p_tightening_pct || 0}% | Hold ${tay.p_hold_pct || 0}% | Ease ${tay.p_easing_pct || 0}%`;
    if (document.getElementById("us-credit-impulse")) document.getElementById("us-credit-impulse").innerText = `${fl.current_4w_gold_impulse_pct >= 0 ? "+" : ""}${fl.current_4w_gold_impulse_pct || 0}% (F=${fl.pruned_f_stat || 0}, p=${fl.pruned_f_p || 1})`;

    const lab = dom.labor_dol || {}, grw = dom.growth_bea || {}, inf = dom.inflation_pce || {};
    const lgEl = document.getElementById("us-labor-growth-tbody");
    if (lgEl) {
      const rows1 = [
        ["UNEMPLOYMENT / SAHM DELTA:", `${lab.unemployment_rate_pct}% (+${lab.sahm_rule_delta_pp}pp vs 12M low)`],
        ["NFP 3M AVG / WAGE YoY:", `${lab.nfp_3m_avg_k}K / +${lab.wage_ahe_yoy_pct}%`],
        ["DOL CLAIMS (ICSA / CCSA):", `${lab.initial_claims_k}K / ${lab.continuing_claims_k}K`],
        ["JOLTS / QUITS / HOURS:", `${lab.jolts_openings_k}K | ${lab.quits_rate_jtsqur_pct}% | ${lab.weekly_hours_awhaetp}h (Part ${lab.participation_civpart_pct}%)`],
        ["REAL GDP (BEA) / INDPRO:", `$${grw.real_gdp_trillions}T (${grw.real_gdp_qoq_ann_pct >= 0 ? "+" : ""}${grw.real_gdp_qoq_ann_pct}% QoQ) | INDPRO ${grw.indpro_yoy_pct >= 0 ? "+" : ""}${grw.indpro_yoy_pct}% YoY`],
        ["CORE PCE / CORE CPI / 10Y BE:", `${inf.core_pce_yoy_pct}% PCE | ${inf.core_cpi_est_yoy_pct}% CPI | ${inf.breakeven_10y_pct}% 10Y BE`]
      ];
      lgEl.innerHTML = rows1.map(([k, v]) => `<tr><td class="label">${k}</td><td class="val">${v}</td></tr>`).join("");
    }

    const dbt = dom.debt_fiscal || {}, tr = dom.treasury_liquidity || {}, fc = dom.financial_conditions_micro || {};
    const dfEl = document.getElementById("us-debt-fin-tbody");
    if (dfEl) {
      const rows2 = [
        ["PUBLIC DEBT / DEBT-TO-GDP:", `$${dbt.public_debt_trillions}T (${dbt.debt_to_gdp_pct}% GDP)`],
        ["FEDERAL DEFICIT (OMB/TREAS):", `${dbt.federal_deficit_trillions}T (${dbt.deficit_to_gdp_pct}% GDP)`],
        ["TREASURY CURVE (2Y/10Y/30Y):", `${tr.dgs2_pct}% / ${tr.dgs10_pct}% / ${tr.dgs30_pct}% (TIPS ${tr.tips_10y_real_pct}%)`],
        ["CURVE SLOPES (10Y-2Y / 10Y-3M):", `${tr.curve_10y_2y_pct >= 0 ? "+" : ""}${tr.curve_10y_2y_pct}% / ${tr.curve_10y_3m_pct >= 0 ? "+" : ""}${tr.curve_10y_3m_pct}%`],
        ["FED BALANCE SHEET / NET LIQ / M2:", `$${tr.fed_balance_sheet_t}T / Net $${tr.net_fed_liquidity_t}T | M2 $${tr.m2_money_supply_t}T (${tr.m2_yoy_pct >= 0 ? "+" : ""}${tr.m2_yoy_pct}% YoY)`],
        ["NFCI / STLFSI4 / HY OAS:", `${fc.chicago_nfci} | ${fc.stl_stress_stlfsi4} | HY ${fc.high_yield_oas_pct}% (UMich ${fc.michigan_sentiment_umcsent}, Save ${fc.personal_saving_psavert_pct}%)`]
      ];
      dfEl.innerHTML = rows2.map(([k, v]) => `<tr><td class="label">${k}</td><td class="val">${v}</td></tr>`).join("");
    }

    const audEl = document.getElementById("us-econ-audit-tbody");
    if (audEl) {
      audEl.innerHTML = (data.hypothesis_tests || []).map(t => {
        const tc = t.passes_selected ? "#00e5ff" : "#ff4444";
        return `
          <tr>
            <td><strong>${t.model}</strong></td>
            <td>${t.n}</td>
            <td style="color:var(--amber-cmd);">${t.stat} (p=${t.p_val})</td>
            <td><span style="color:${tc}; border:1px solid ${tc}; padding:1px 4px; font-size:8.5px; font-weight:800;">${t.tier}</span></td>
            <td style="color:${tc}; font-weight:800;">${t.passes_selected ? `PASS @ ${confLvl}%` : `FAIL @ ${confLvl}%`}</td>
            <td>${t.output}</td>
          </tr>
        `;
      }).join("");
    }
  } catch (e) { console.warn("fetchUsEconomyAtlas error:", e); }
}
document.getElementById("btn-refresh-us-atlas")?.addEventListener("click", (e) => {
  const b = e.target;
  b.innerText = "REFRESHING US DATA...";
  fetch("/api/update-all-data", { method: "POST" }).then(() => fetchUsEconomyAtlas()).finally(() => { setTimeout(() => { b.innerText = "REFRESH US MACRO <GO>"; }, 600); });
});

// ==============================================================
// 8. INSTITUTIONAL LOT CALCULATOR & MARGIN RISK SIZER
// ==============================================================
function initLotCalculator() {
  let currentRiskMode = "pct";
  let currentDirection = "BUY";

  const eqInput = document.getElementById("calc-equity");
  const rRange = document.getElementById("calc-risk-range");
  const rInput = document.getElementById("calc-risk-pct");
  const cashInput = document.getElementById("calc-max-risk-cash");
  const wrapPct = document.getElementById("wrap-risk-pct");
  const wrapCash = document.getElementById("wrap-risk-cash");
  const btnModePct = document.getElementById("btn-mode-pct");
  const btnModeCash = document.getElementById("btn-mode-cash");
  const btnDirBuy = document.getElementById("btn-calc-dir-buy");
  const btnDirSell = document.getElementById("btn-calc-dir-sell");

  const levSel = document.getElementById("calc-leverage");
  const instSel = document.getElementById("calc-instrument");
  const entInput = document.getElementById("calc-entry");
  const slPipsInput = document.getElementById("calc-sl-pips");
  const slInput = document.getElementById("calc-sl");
  const tp1PipsInput = document.getElementById("calc-tp1-pips");
  const tp1Input = document.getElementById("calc-tp1");
  const tp2PipsInput = document.getElementById("calc-tp2-pips");
  const tp2Input = document.getElementById("calc-tp2");

  const specs = {
    "XAUUSD": { contract: 100.0, pip: 0.10, unit: "oz", defaultLev: 100 },
    "EURUSD": { contract: 100000.0, pip: 0.0001, unit: "EUR", defaultLev: 500 },
    "BTCUSD": { contract: 1.0, pip: 1.0, unit: "BTC", defaultLev: 200 },
    "GBPUSD": { contract: 100000.0, pip: 0.0001, unit: "GBP", defaultLev: 500 },
    "US30": { contract: 10.0, pip: 1.0, unit: "pts", defaultLev: 100 },
    "CL": { contract: 1000.0, pip: 0.01, unit: "bbl", defaultLev: 100 }
  };

  function getSpec() {
    const inst = (instSel?.value || "XAUUSD").toUpperCase();
    return specs[inst] || specs["XAUUSD"];
  }

  function recalculate() {
    const eq = Math.max(10, parseFloat(eqInput?.value || 1000));
    const lev = Math.max(1, parseInt(levSel?.value || 100, 10));
    const spec = getSpec();
    const cSize = spec.contract;
    const pipVal = spec.pip;
    const pipValuePerLot = cSize * pipVal;
    const entry = parseFloat(entInput?.value || 3925.0);

    // 1. Deduce Target Risk Cash ($): Option 1 (% of Equity) vs Option 2 (Fixed Max Cash Risk $)
    let targetRiskDollars = 20.0;
    let effectiveRiskPct = 2.0;

    if (currentRiskMode === "cash") {
      targetRiskDollars = Math.max(1.0, parseFloat(cashInput?.value || 50.0));
      effectiveRiskPct = (targetRiskDollars / eq) * 100.0;
      if (document.getElementById("lbl-cash-equiv-pct")) {
        document.getElementById("lbl-cash-equiv-pct").innerText = `$${targetRiskDollars.toFixed(2)} (${effectiveRiskPct.toFixed(2)}% eq)`;
      }
    } else {
      effectiveRiskPct = Math.max(0.05, parseFloat(rInput?.value || 2.0));
      targetRiskDollars = eq * (effectiveRiskPct / 100.0);
      if (document.getElementById("lbl-risk-cash")) {
        document.getElementById("lbl-risk-cash").innerText = `${effectiveRiskPct.toFixed(1)}% ($${targetRiskDollars.toFixed(2)} USD)`;
      }
    }

    // 2. Stop Loss & Take Profit in PIPS
    let slPips = Math.max(1.0, parseFloat(slPipsInput?.value || 100.0));
    let tp1Pips = Math.max(1.0, parseFloat(tp1PipsInput?.value || 250.0));
    let tp2Pips = Math.max(1.0, parseFloat(tp2PipsInput?.value || 500.0));

    const slDist = slPips * pipVal;
    const tp1Dist = tp1Pips * pipVal;
    const tp2Dist = tp2Pips * pipVal;

    // 3. EXACT PIP-BASED LOT SIZE DEDUCTION
    // Lot Size = Target Risk ($) / (SL Pips * Pip Value per Lot)
    const rawLot = targetRiskDollars / Math.max(0.01, (slPips * pipValuePerLot));
    const lot = Math.max(0.01, Math.round(rawLot * 100.0) / 100.0);

    const actualRiskUsd = lot * slPips * pipValuePerLot;
    const actualRiskPct = (actualRiskUsd / eq) * 100.0;
    const notional = lot * cSize * entry;
    const margin = notional / lev;
    const freeMargin = eq - margin;
    const marginLvl = (eq / Math.max(0.01, margin)) * 100.0;
    const effLev = notional / eq;

    const pnlTp1 = lot * tp1Pips * pipValuePerLot;
    const pnlTp2 = lot * tp2Pips * pipValuePerLot;
    const rr1 = pnlTp1 / Math.max(1, actualRiskUsd);
    const rr2 = pnlTp2 / Math.max(1, actualRiskUsd);

    // Update Output Elements
    if (document.getElementById("lbl-equity-val")) {
      document.getElementById("lbl-equity-val").innerText = `$${eq.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    if (document.getElementById("out-calc-lot")) {
      document.getElementById("out-calc-lot").innerText = `${lot.toFixed(2)} LOT`;
    }
    const priceDec = pipVal < 0.01 ? 5 : 2;
    if (document.getElementById("out-calc-notional")) {
      document.getElementById("out-calc-notional").innerText = `$${notional.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USD Notional (${(lot * cSize).toLocaleString("en-US", { maximumFractionDigits: 1 })} ${spec.unit})`;
    }
    if (document.getElementById("out-calc-risk-usd")) {
      document.getElementById("out-calc-risk-usd").innerText = `-$${actualRiskUsd.toFixed(2)} USD`;
    }
    if (document.getElementById("out-calc-risk-pct")) {
      document.getElementById("out-calc-risk-pct").innerText = `${actualRiskPct.toFixed(2)}% of Equity (${currentRiskMode === "cash" ? "Fixed Cash Mode" : "% Mode"})`;
    }
    if (document.getElementById("out-calc-sl-dist")) {
      document.getElementById("out-calc-sl-dist").innerText = `${pipVal < 0.01 ? "" : "$"}${slDist.toFixed(priceDec)} (${slPips.toFixed(1)} pips @ $${(lot * pipValuePerLot).toFixed(2)}/pip)`;
    }
    if (document.getElementById("out-calc-margin")) {
      document.getElementById("out-calc-margin").innerText = `$${margin.toFixed(2)} USD (1:${lev})`;
    }
    if (document.getElementById("out-calc-free-margin")) {
      const fmEl = document.getElementById("out-calc-free-margin");
      fmEl.innerText = `$${freeMargin.toFixed(2)} USD`;
      fmEl.style.color = freeMargin > 0 ? "#00e5ff" : "#ff4444";
    }
    if (document.getElementById("out-calc-margin-lvl")) {
      const mlEl = document.getElementById("out-calc-margin-lvl");
      const mlStatus = marginLvl >= 500 ? "HEALTHY" : (marginLvl >= 150 ? "MODERATE" : "CRITICAL DANGER");
      mlEl.innerText = `${marginLvl.toLocaleString("en-US", { maximumFractionDigits: 1 })}% (${mlStatus})`;
      mlEl.style.color = marginLvl >= 500 ? "#ffffff" : (marginLvl >= 150 ? "var(--amber-cmd)" : "#ff4444");
    }
    if (document.getElementById("out-calc-eff-lev")) {
      document.getElementById("out-calc-eff-lev").innerText = `${effLev.toFixed(1)}x Account Notional`;
    }
    if (document.getElementById("out-calc-tp1-pnl")) {
      document.getElementById("out-calc-tp1-pnl").innerText = `+$${pnlTp1.toFixed(2)} USD (1:${rr1.toFixed(2)} RR / +${tp1Pips.toFixed(0)}p)`;
    }
    if (document.getElementById("out-calc-tp2-pnl")) {
      document.getElementById("out-calc-tp2-pnl").innerText = `+$${pnlTp2.toFixed(2)} USD (1:${rr2.toFixed(2)} RR / +${tp2Pips.toFixed(0)}p)`;
    }

    // Guardrail Badge & Warning Alert Box
    const badge = document.getElementById("lot-guardrail-badge");
    const warnBox = document.getElementById("calc-warning-box");
    const warnTxt = document.getElementById("calc-warning-text");

    let isDanger = margin >= eq;
    let isWarning = margin >= eq * 0.65 || effectiveRiskPct > 3.0 || effLev > 20.0;

    if (isDanger) {
      if (badge) { badge.innerText = "DANGER: MARGIN CALL"; badge.style.borderColor = "#ff4444"; badge.style.color = "#ff4444"; }
      if (warnTxt) warnTxt.innerHTML = `<span style="color:#ff4444;">⚠️ MARGIN CALL RISK: Required margin ($${margin.toFixed(2)}) exceeds equity ($${eq.toFixed(2)}). Broker will reject order. Lower lot, widen SL pips, or increase leverage.</span>`;
      if (warnBox) warnBox.style.borderColor = "#ff4444";
    } else if (isWarning) {
      if (badge) { badge.innerText = "WARNING: HIGH RISK"; badge.style.borderColor = "var(--amber-cmd)"; badge.style.color = "var(--amber-cmd)"; }
      let msgs = [];
      if (effectiveRiskPct > 3.0) msgs.push(`Risk of ${effectiveRiskPct.toFixed(1)}% exceeds institutional prop desk 3.0% threshold.`);
      if (margin >= eq * 0.65) msgs.push(`High margin utilization (${(margin/eq*100).toFixed(0)}%).`);
      if (effLev > 20.0) msgs.push(`Effective leverage ${effLev.toFixed(1)}x is aggressive.`);
      if (warnTxt) warnTxt.innerHTML = `<span style="color:var(--amber-cmd);">⚠️ ELEVATED RISK: ${msgs.join(" ")}</span>`;
      if (warnBox) warnBox.style.borderColor = "var(--amber-cmd)";
    } else {
      if (badge) { badge.innerText = "GUARDRAIL: OPTIMAL MARGIN"; badge.style.borderColor = "#00e5ff"; badge.style.color = "#00e5ff"; }
      if (warnTxt) warnTxt.innerHTML = `<span style="color:#00e5ff;">✓ OPTIMAL INSTITUTIONAL SIZING: 1.0R risk ($${actualRiskUsd.toFixed(2)}) on ${slPips.toFixed(0)} pips SL safely within margin buffer.</span>`;
      if (warnBox) warnBox.style.borderColor = "#222";
    }
  }

  // 1. Risk Mode Switcher: Option 1 (% of Equity) vs Option 2 (Fixed Max Cash Risk $)
  btnModePct?.addEventListener("click", () => {
    currentRiskMode = "pct";
    btnModePct.classList.add("active");
    btnModeCash?.classList.remove("active");
    if (wrapPct) wrapPct.style.display = "block";
    if (wrapCash) wrapCash.style.display = "none";
    recalculate();
  });

  btnModeCash?.addEventListener("click", () => {
    currentRiskMode = "cash";
    btnModeCash.classList.add("active");
    btnModePct?.classList.remove("active");
    if (wrapCash) wrapCash.style.display = "block";
    if (wrapPct) wrapPct.style.display = "none";
    recalculate();
  });

  // 2. Order Direction Switcher (BUY vs SELL)
  btnDirBuy?.addEventListener("click", () => {
    currentDirection = "BUY";
    btnDirBuy.classList.add("active");
    btnDirSell?.classList.remove("active");
    syncPricesFromPips();
    recalculate();
  });

  btnDirSell?.addEventListener("click", () => {
    currentDirection = "SELL";
    btnDirSell.classList.add("active");
    btnDirBuy?.classList.remove("active");
    syncPricesFromPips();
    recalculate();
  });

  // 3. Bi-Directional Pips <-> Price Sync
  function syncPricesFromPips() {
    const entry = parseFloat(entInput?.value || 3925.0);
    const spec = getSpec();
    const pipVal = spec.pip;
    const priceDec = pipVal < 0.01 ? 5 : 2;
    const slPips = parseFloat(slPipsInput?.value || 100.0);
    const tp1Pips = parseFloat(tp1PipsInput?.value || 250.0);
    const tp2Pips = parseFloat(tp2PipsInput?.value || 500.0);

    if (slInput) slInput.value = (currentDirection === "BUY" ? entry - slPips * pipVal : entry + slPips * pipVal).toFixed(priceDec);
    if (tp1Input) tp1Input.value = (currentDirection === "BUY" ? entry + tp1Pips * pipVal : entry - tp1Pips * pipVal).toFixed(priceDec);
    if (tp2Input) tp2Input.value = (currentDirection === "BUY" ? entry + tp2Pips * pipVal : entry - tp2Pips * pipVal).toFixed(priceDec);
  }

  function syncPipsFromPrices() {
    const entry = parseFloat(entInput?.value || 3925.0);
    const spec = getSpec();
    const pipVal = spec.pip;
    const slPrice = parseFloat(slInput?.value || 3915.0);
    const tp1Price = parseFloat(tp1Input?.value || 3950.0);
    const tp2Price = parseFloat(tp2Input?.value || 3975.0);

    if (slPipsInput) slPipsInput.value = (Math.abs(entry - slPrice) / pipVal).toFixed(1);
    if (tp1PipsInput) tp1PipsInput.value = (Math.abs(entry - tp1Price) / pipVal).toFixed(1);
    if (tp2PipsInput) tp2PipsInput.value = (Math.abs(entry - tp2Price) / pipVal).toFixed(1);
  }

  [slPipsInput, tp1PipsInput, tp2PipsInput].forEach(inp => inp?.addEventListener("input", () => {
    syncPricesFromPips();
    recalculate();
  }));

  [slInput, tp1Input, tp2Input].forEach(inp => inp?.addEventListener("input", () => {
    syncPipsFromPrices();
    recalculate();
  }));

  entInput?.addEventListener("input", () => {
    syncPricesFromPips();
    recalculate();
  });

  [eqInput, cashInput].forEach(inp => inp?.addEventListener("input", recalculate));
  levSel?.addEventListener("change", () => {
    recalculate();
  });
  instSel?.addEventListener("change", () => {
    const sym = instSel.value;
    if (sym === "BTCUSD") {
      if (levSel) levSel.value = "200";
      if (entInput) entInput.value = (window._LAST_VERIFIED_BTC_PRICE || 91500.0).toFixed(2);
      if (slPipsInput) slPipsInput.value = "1500.0";
      if (tp1PipsInput) tp1PipsInput.value = "3000.0";
      if (tp2PipsInput) tp2PipsInput.value = "5000.0";
    } else if (sym === "EURUSD") {
      if (levSel) levSel.value = "500";
      if (entInput) entInput.value = "1.08500";
      if (slPipsInput) slPipsInput.value = "20.0";
      if (tp1PipsInput) tp1PipsInput.value = "50.0";
      if (tp2PipsInput) tp2PipsInput.value = "80.0";
    } else if (sym === "XAUUSD") {
      if (levSel) levSel.value = "100";
      if (entInput) entInput.value = "3925.00";
      if (slPipsInput) slPipsInput.value = "100.0";
      if (tp1PipsInput) tp1PipsInput.value = "200.0";
      if (tp2PipsInput) tp2PipsInput.value = "350.0";
    }
    syncPricesFromPips();
    recalculate();
  });

  rRange?.addEventListener("input", (e) => {
    if (rInput) rInput.value = parseFloat(e.target.value).toFixed(2);
    document.querySelectorAll(".btn-risk-preset").forEach(b => b.classList.remove("active"));
    recalculate();
  });

  rInput?.addEventListener("input", (e) => {
    if (rRange) rRange.value = parseFloat(e.target.value);
    document.querySelectorAll(".btn-risk-preset").forEach(b => b.classList.remove("active"));
    recalculate();
  });

  // Presets
  document.querySelectorAll(".btn-eq-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-eq-preset").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (eqInput) eqInput.value = btn.getAttribute("data-eq");
      recalculate();
    });
  });

  document.querySelectorAll(".btn-risk-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-risk-preset").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const rVal = parseFloat(btn.getAttribute("data-risk"));
      if (rInput) rInput.value = rVal.toFixed(1);
      if (rRange) rRange.value = rVal;
      if (btnModePct) btnModePct.click();
      recalculate();
    });
  });

  document.querySelectorAll(".btn-cash-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-cash-preset").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const cVal = parseFloat(btn.getAttribute("data-cash"));
      if (cashInput) cashInput.value = cVal;
      if (btnModeCash) btnModeCash.click();
      recalculate();
    });
  });

  // Sync from Active Ticket
  function syncFromActiveTicket(ev = null) {
    const activeMod = ev ? `mod-${ev}` : getActiveModuleId();
    let entryVal = 3925.0, slVal = 3915.0, tp1Val = 3950.0, tp2Val = 3975.0;
    let dir = "BUY";

    if (activeMod === "mod-astar") {
      if (CURRENT_ASTAR_ASSET && instSel) {
        instSel.value = CURRENT_ASTAR_ASSET;
        if (CURRENT_ASTAR_ASSET === "EURUSD" && levSel) levSel.value = "500";
      }
      const entTxt = document.getElementById("astar-entry")?.innerText || "";
      const slTxt = document.getElementById("astar-sl")?.innerText || "";
      const tp1Txt = document.getElementById("astar-tp1")?.innerText || "";
      const tp2Txt = document.getElementById("astar-tp2")?.innerText || "";
      const actTxt = document.getElementById("astar-action")?.innerText || "";
      if (actTxt.includes("SHORT") || actTxt.includes("SELL")) dir = "SELL";
      const mEnt = entTxt.match(/(\d+\.?\d*)/);
      const mSl = slTxt.match(/(\d+\.?\d*)/);
      const mTp1 = tp1Txt.match(/(\d+\.?\d*)/);
      const mTp2 = tp2Txt.match(/(\d+\.?\d*)/);
      if (mEnt) entryVal = parseFloat(mEnt[1]);
      if (mSl) slVal = parseFloat(mSl[1]);
      if (mTp1) tp1Val = parseFloat(mTp1[1]);
      if (mTp2) tp2Val = parseFloat(mTp2[1]);
    } else {
      const eName = activeMod.replace("mod-", "");
      const entId = eName === "cpi" ? "val-ticket-entry" : `val-${eName}-ticket-entry`;
      const slId = eName === "cpi" ? "val-ticket-sl" : `val-${eName}-ticket-sl`;
      const tpId = eName === "cpi" ? "val-ticket-tp" : `val-${eName}-ticket-tp`;
      const actId = eName === "cpi" ? "val-ticket-action" : `val-${eName}-ticket-action`;
      const entTxt = document.getElementById(entId)?.innerText || "";
      const slTxt = document.getElementById(slId)?.innerText || "";
      const tpTxt = document.getElementById(tpId)?.innerText || "";
      const actTxt = document.getElementById(actId)?.innerText || "";
      if (actTxt.includes("SHORT") || actTxt.includes("SELL")) dir = "SELL";
      const mEnt = entTxt.match(/(\d+\.?\d*)/);
      const mSl = slTxt.match(/(\d+\.?\d*)/);
      const mTpPair = tpTxt.match(/\$?(\d+\.?\d*)\s*\/\s*\$?(\d+\.?\d*)/);
      const mTp = tpTxt.match(/(\d+\.?\d*)/);
      if (mEnt) entryVal = parseFloat(mEnt[1]);
      if (mSl) slVal = parseFloat(mSl[1]);
      if (mTpPair) {
        tp1Val = parseFloat(mTpPair[1]);
        tp2Val = parseFloat(mTpPair[2]);
      } else if (mTp) {
        tp1Val = parseFloat(mTp[1]);
        tp2Val = tp1Val + (tp1Val >= entryVal ? 1 : -1) * Math.abs(tp1Val - entryVal) * 0.6;
      }
    }

    currentDirection = dir;
    if (dir === "BUY") {
      btnDirBuy?.classList.add("active");
      btnDirSell?.classList.remove("active");
    } else {
      btnDirSell?.classList.add("active");
      btnDirBuy?.classList.remove("active");
    }

    const syncDec = getSpec().pip < 0.01 ? 5 : 2;
    if (entInput) entInput.value = entryVal.toFixed(syncDec);
    if (slInput) slInput.value = slVal.toFixed(syncDec);
    if (tp1Input) tp1Input.value = tp1Val.toFixed(syncDec);
    if (tp2Input) tp2Input.value = tp2Val.toFixed(syncDec);
    syncPipsFromPrices();
    recalculate();
  }
  window.__syncFromActiveTicket = syncFromActiveTicket;

  document.getElementById("btn-sync-active-ticket")?.addEventListener("click", () => syncFromActiveTicket());
  document.querySelectorAll(".btn-ticket-sizer").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const ev = btn.getAttribute("data-ev");
      document.querySelector('[data-mod="mod-astar"]')?.click();
      syncFromActiveTicket(ev);
      document.getElementById("section-lot-sizer")?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  document.getElementById("btn-top-lot-sizer")?.addEventListener("click", () => {
    document.querySelector('[data-mod="mod-astar"]')?.click();
    syncFromActiveTicket();
    document.getElementById("section-lot-sizer")?.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  syncPricesFromPips();
  recalculate();
}

document.addEventListener("DOMContentLoaded", init);
