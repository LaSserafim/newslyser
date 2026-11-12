/**
 * NEWSLYSER // BLOOMBERG PROFESSIONAL TERMINAL WORKSTATION
 * Authentic Monochrome White-on-Black Architecture.
 * 10,000-Path Kou (2002) Jump-Diffusion MC, White (1980) HC3 Regressions,
 * Student's t + Binomial 5% Significance Hypothesis Testing & Live TradingView B-PIPE.
 */

const EVENT_STATE = {
  cpi:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", pdfView: "cpi", mode: "PREMARKET", fwTimer: null },
  fomc: { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mode: "PREMARKET", fwTimer: null },
  nfp:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mode: "PREMARKET", fwTimer: null },
  ppi:  { z: 0.00, horizon: 1, asset: "gold", diag: "tcurve", mode: "PREMARKET", fwTimer: null }
};

let PREMARKET_CACHE = null;
let currentAiMode = "pre_news";
let maximizedPaneId = null;
let yoyDomain = "cpi";
let yoyTimelineCache = null;
let yoyActiveKeys = { headline_yoy: true, core_yoy: true, supercore_yoy: true, shelter_yoy: true };
let ttlTimerInterval = null;

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

  // Load frozen Pre-Market Snapshot FIRST so Pane 01, 02, 03, 06 match identically on first paint
  await Promise.all([fetchPremarketSnapshot(false), fetchState()]);
  renderWorkstation("cpi");
  fetchAuditLedger();
  fetchFomcData();
  fetchNfpData();
  fetchPpiData();
  fetchCalendarData();
  fetchAcademicLit();
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
function initCpiSliders() {
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

    const zHl = (hl - 0.3) / 0.075;
    const zCore = (core - 2.9) / 0.075;
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
    const act = parseFloat(rAct.value);
    const un = parseFloat(rUnemp.value);
    const ah = parseFloat(rAhe.value);

    vAct.innerText = `${act}k`;
    vUnemp.innerText = `${un.toFixed(1)}%`;
    vAhe.innerText = `${ah.toFixed(2)}%`;

    const zNfp = (act - 135) / 45.0;
    const zUn = (un - 4.1) / 0.15;
    const zAh = (ah - 0.3) / 0.10;
    const zLabor = 0.50 * zNfp - 0.30 * zUn + 0.20 * zAh;
    EVENT_STATE.nfp.z = zLabor;
    EVENT_STATE.nfp.mode = "WHATIF";

    sSum.innerText = `${zLabor > 0 ? "+" : ""}${zLabor.toFixed(2)}σ (WHAT-IF)`;
    renderEventMcPane("nfp");
    renderEventFreqPane("nfp");
    try {
      const res = await fetch(`/api/nfp-model?nfp_actual=${act}&nfp_forecast=135&unemp_actual=${un}&unemp_forecast=4.1&ahe_actual=${ah}&ahe_forecast=0.3`);
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
    const act = parseFloat(rAct.value);
    const ret = parseFloat(rRetail.value);

    vAct.innerText = `${act.toFixed(2)}%`;
    vRetail.innerText = `${ret.toFixed(2)}%`;

    const zPpi = (act - 0.4) / 0.20;
    const zRet = (ret - 0.4) / 0.35;
    const zDemand = 0.55 * zPpi + 0.45 * zRet;
    EVENT_STATE.ppi.z = zDemand;
    EVENT_STATE.ppi.mode = "WHATIF";

    sSum.innerText = `${zDemand > 0 ? "+" : ""}${zDemand.toFixed(2)}σ (WHAT-IF)`;
    renderEventMcPane("ppi");
    renderEventFreqPane("ppi");
    try {
      const res = await fetch(`/api/ppi-retail-model?ppi_actual=${act}&ppi_forecast=0.4&retail_actual=${ret}&retail_forecast=0.4`);
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

  if (ev === "cpi") {
    EVENT_STATE.cpi.pdfView = "cpi";
    const rH = document.getElementById("bbg-range-headline"), rC = document.getElementById("bbg-range-core");
    if (rH) rH.value = 0.3;
    if (rC) rC.value = 2.9;
    if (document.getElementById("bbg-val-headline")) document.getElementById("bbg-val-headline").innerText = "0.30%";
    if (document.getElementById("bbg-val-core")) document.getElementById("bbg-val-core").innerText = "2.90%";
    document.querySelectorAll("[data-z]").forEach(b => b.classList.remove("active"));
    setBtnActive(document.querySelector("[data-asset='gold']"), "[data-asset]");
    setBtnActive(document.querySelector("[data-diag='tcurve']"), "[data-diag]");
  } else if (ev === "fomc") {
    if (document.getElementById("fomc-range-rate")) document.getElementById("fomc-range-rate").value = 4.0;
    if (document.getElementById("fomc-range-priced")) document.getElementById("fomc-range-priced").value = 4.0;
    if (document.getElementById("fomc-range-dots")) document.getElementById("fomc-range-dots").value = 0.0;
    if (document.getElementById("fomc-val-rate")) document.getElementById("fomc-val-rate").innerText = "4.00%";
    if (document.getElementById("fomc-val-priced")) document.getElementById("fomc-val-priced").innerText = "4.00%";
    if (document.getElementById("fomc-val-dots")) document.getElementById("fomc-val-dots").innerText = "0.0 bps";
    setBtnActive(document.querySelector('[data-ev="fomc"][data-ev-asset="gold"]'), '[data-ev="fomc"][data-ev-asset]');
    setBtnActive(document.querySelector('[data-ev="fomc"][data-ev-diag="tcurve"]'), '[data-ev="fomc"][data-ev-diag]');
  } else if (ev === "nfp") {
    if (document.getElementById("nfp-range-act")) document.getElementById("nfp-range-act").value = 135;
    if (document.getElementById("nfp-range-unemp")) document.getElementById("nfp-range-unemp").value = 4.1;
    if (document.getElementById("nfp-range-ahe")) document.getElementById("nfp-range-ahe").value = 0.3;
    if (document.getElementById("nfp-val-act")) document.getElementById("nfp-val-act").innerText = "135k";
    if (document.getElementById("nfp-val-unemp")) document.getElementById("nfp-val-unemp").innerText = "4.1%";
    if (document.getElementById("nfp-val-ahe")) document.getElementById("nfp-val-ahe").innerText = "0.30%";
    setBtnActive(document.querySelector('[data-ev="nfp"][data-ev-asset="gold"]'), '[data-ev="nfp"][data-ev-asset]');
    setBtnActive(document.querySelector('[data-ev="nfp"][data-ev-diag="tcurve"]'), '[data-ev="nfp"][data-ev-diag]');
  } else if (ev === "ppi") {
    if (document.getElementById("ppi-range-act")) document.getElementById("ppi-range-act").value = 0.4;
    if (document.getElementById("ppi-range-retail")) document.getElementById("ppi-range-retail").value = 0.4;
    if (document.getElementById("ppi-val-act")) document.getElementById("ppi-val-act").innerText = "0.40%";
    if (document.getElementById("ppi-val-retail")) document.getElementById("ppi-val-retail").innerText = "0.40%";
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

  if (document.getElementById(actId) && directive) document.getElementById(actId).innerText = directive;
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
  let mc = null;
  try {
    const res = await fetch(`/api/monte-carlo?horizon=${st.horizon}&z=${st.z}&event_type=${ev}&n_paths=10000`);
    mc = await res.json();
  } catch (e) {
    return;
  }

  // Update metrics table
  const medId = ev === "cpi" ? "val-mc-median" : `val-${ev}-mc-median`;
  const varId = ev === "cpi" ? "val-mc-var" : `val-${ev}-mc-var`;
  const tailId = ev === "cpi" ? "val-mc-tail" : `val-${ev}-mc-tail`;
  const m = mc.metrics || {};

  if (document.getElementById(medId)) {
    document.getElementById(medId).innerText = `$${m.median_terminal_price.toFixed(2)} (${mc.n_paths.toLocaleString()} paths, ±$${mc.mc_moe_terminal_dollars} SE)`;
  }
  if (document.getElementById(varId)) {
    document.getElementById(varId).innerText = `${m.var_95_pct > 0 ? "+" : ""}${m.var_95_pct.toFixed(2)}% / CVaR ${m.cvar_95_pct.toFixed(2)}% (P+: ${m.p_positive_return_pct}% | P-: ${m.p_negative_return_pct}%)`;
  }
  if (document.getElementById(tailId)) {
    document.getElementById(tailId).innerText = `|Move| >= 1σ: ${m.tail_beyond_1sd_pct}% | >= 2σ: ${m.tail_beyond_2sd_pct}% (σ=${m.terminal_dispersion_sd_pct}%)`;
  }

  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 42, padR = 46, padT = 18, padB = 24;
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
  const tEnd = mc.timesteps[steps] || (st.horizon === 1 ? "T+24h" : "D5.0");

  container.innerHTML = `
    <svg id="svg-${ev}-mc" viewBox="0 0 ${w} ${h}" class="bbg-svg" style="cursor:crosshair;">
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" class="axis-line"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" class="axis-line"/>
      <path d="${path90}" fill="#131313" opacity="0.88"/>
      <path d="${path50}" fill="#222222" opacity="0.92"/>
      <g>${tracesSvg}</g>
      <line x1="${padL}" y1="${baseY}" x2="${w - padR}" y2="${baseY}" stroke="#ffb000" stroke-width="1" stroke-dasharray="2,2"/>
      <text x="${w - padR - 4}" y="${Math.max(padT + 10, baseY - 4)}" fill="#ffb000" text-anchor="end">T0: $${mc.current_price.toFixed(1)}</text>
      <path d="${pathMedian}" fill="none" stroke="#ffffff" stroke-width="2"/>
      <circle cx="${getX(steps)}" cy="${getY(p50[steps])}" r="3" fill="#ffffff"/>
      <text x="${getX(steps) + 5}" y="${getY(p50[steps]) + 3}" fill="#ffffff" font-weight="700">$${p50[steps].toFixed(0)}</text>
      <text x="${padL - 4}" y="${padT + 8}" text-anchor="end">$${maxP.toFixed(0)}</text>
      <text x="${padL - 4}" y="${h - padB}" text-anchor="end">$${minP.toFixed(0)}</text>
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
      if (hud) hud.innerHTML = `${mc.timesteps[s]} | Med: <span class="highlight">$${medP.toFixed(1)}</span> (90%: $${p5Val.toFixed(0)}–$${p95Val.toFixed(0)}) | Drift: ${delta >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
      positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
        <div style="color:var(--amber-cmd); font-weight:800;">${mc.timesteps[s]} (10,000 PATHS)</div>
        <div>Median: <strong>$${medP.toFixed(2)}</strong> (${delta >= 0 ? "+" : ""}$${delta.toFixed(2)} / ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%)</div>
        <div>50% Core: $${p25Val.toFixed(1)} — $${p75Val.toFixed(1)}</div>
        <div>90% Band: $${p5Val.toFixed(1)} — $${p95Val.toFixed(1)}</div>
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

  if (document.getElementById(slopeId) && modelObj) {
    document.getElementById(slopeId).innerText = `β = ${modelObj.beta[1].toFixed(3)}%/${unitTag} (R² = ${modelObj.r2.toFixed(3)}, t = ${modelObj.t_stat}, p = ${modelObj.p_value})`;
  }
  if (document.getElementById(vrId)) {
    document.getElementById(vrId).innerText = `VR(5m) = ${trans.avg_vr5}x | β_DXY=${trans.models.dxy.beta[1].toFixed(2)}% | β_2Y=${trans.models.zt_note.beta[1].toFixed(2)}`;
  }

  const curves = trans.response_curves || [];
  const pts = trans.scatter_points || [];
  const w = container.clientWidth || 380;
  const h = container.clientHeight || 215;
  const padL = 36, padR = 36, padT = 18, padB = 24;
  const minZ = -3.0, maxZ = +3.0;

  let yKey = "gold_return", pKey = "gold_ret", assetLabel = "GOLD";
  if (asset === "dxy") { yKey = "dxy_return"; pKey = "dxy_ret"; assetLabel = "DXY"; }
  else if (asset === "2y") { yKey = "zt_return"; pKey = "zt_ret"; assetLabel = "2Y NOTE"; }

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
      <text x="${getX(0) + 4}" y="${padT + 10}">${assetLabel} %</text>
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
        if (hud) hud.innerHTML = `EVENT <span class="highlight">${nearestDot.date}</span> | Shock: ${nearestDot.raw_shock} (${nearestDot.z >= 0 ? "+" : ""}${nearestDot.z}σ) | ${assetLabel}: <span class="highlight">${nearestDot[pKey] >= 0 ? "+" : ""}${nearestDot[pKey]}%</span> | Dist Mean: ${nearestDot.dist_from_mean_sd >= 0 ? "+" : ""}${nearestDot.dist_from_mean_sd}σ`;
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
        if (hud) hud.innerHTML = `Shock Z = <span class="highlight">${closest.z >= 0 ? "+" : ""}${closest.z.toFixed(2)}σ</span> | Expected ${assetLabel} = <span class="highlight">${closest[yKey] >= 0 ? "+" : ""}${closest[yKey].toFixed(2)}%</span>`;
        positionTooltip(tt, container, mouseEv.clientX, mouseEv.clientY, `
          <div style="color:var(--amber-cmd); font-weight:800;">HC3 FRONTIER (Z = ${closest.z >= 0 ? "+" : ""}${closest.z.toFixed(2)}σ)</div>
          <div>Expected ${assetLabel}: <strong>${closest[yKey] >= 0 ? "+" : ""}${closest[yKey].toFixed(2)}%</strong></div>
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
    const tagTxt = diag.slope_valid_5pct
      ? `VALID (${tier} / REJECT H0)`
      : (diag.multi_valid_5pct ? `UNI FAIL | X-ASSET F VALID (p=${diag.f_p_val})` : "NO EDGE / LUCK RISK");
    vEl.innerText = `t=${diag.beta_z_t_stat.toFixed(2)} vs ±${diag.t_crit_95.toFixed(3)} (p=${diag.beta_z_p_val.toFixed(4)}) -> ${tagTxt}`;
    vEl.style.color = diag.slope_valid_5pct ? "#ffffff" : (diag.multi_valid_5pct ? "var(--amber-cmd)" : "#ff4444");
  }
  if (document.getElementById(luckId)) {
    const lEl = document.getElementById(luckId);
    lEl.innerText = `${diag.binomial_wins}/${diag.binomial_trials} (${diag.binomial_accuracy_pct}%, p=${diag.binomial_p_val}) -> ${diag.beats_luck_5pct ? "NOT LUCK (α=5%)" : "COIN-FLIP / LUCK RISK"}`;
    lEl.style.color = diag.beats_luck_5pct ? "#ffb000" : "#ff4444";
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
    const m = Math.floor(remaining / 60), s = remaining % 60;
    if (timerEl) { timerEl.innerText = `${m}m ${s < 10 ? "0" : ""}${s}s`; timerEl.style.color = m < 15 ? "var(--amber-cmd)" : "var(--fg-bright)"; }
    const maxValSec = 5400;
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
    const [aRes, dRes] = await Promise.all([fetch("/api/daily-a-star"), fetch("/api/dynamic-levels")]);
    const aStar = await aRes.json();
    const dyn = await dRes.json();
    if (!aStar) return;

    if (dyn) {
      const res = dyn.resistance || {}, sup = dyn.support || {};
      if (document.getElementById("ast-r3")) document.getElementById("ast-r3").innerText = `$${(res.r3 || 0).toFixed(2)}`;
      if (document.getElementById("ast-r2")) document.getElementById("ast-r2").innerText = `$${(res.r2 || 0).toFixed(2)}`;
      if (document.getElementById("ast-r1")) document.getElementById("ast-r1").innerText = `$${(res.r1 || 0).toFixed(2)}`;
      if (document.getElementById("ast-pp")) document.getElementById("ast-pp").innerText = `$${(dyn.pivot_point || 0).toFixed(2)}`;
      if (document.getElementById("ast-s1")) document.getElementById("ast-s1").innerText = `$${(sup.s1 || 0).toFixed(2)}`;
      if (document.getElementById("ast-s2")) document.getElementById("ast-s2").innerText = `$${(sup.s2 || 0).toFixed(2)}`;
      if (document.getElementById("ast-s3")) document.getElementById("ast-s3").innerText = `$${(sup.s3 || 0).toFixed(2)}`;
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
    if (document.getElementById("val-ast-pwin")) document.getElementById("val-ast-pwin").innerText = `${fpa.win_probability_pct}% (P_up: ${mcq.p_upward_pct}% / P_down: ${mcq.p_downward_pct}%)`;
    if (document.getElementById("val-ast-var")) document.getElementById("val-ast-var").innerText = `${mcq.var_95_pct}% (CVaR: ${mcq.cvar_95_pct}% | ${(mcq.n_paths || 10000).toLocaleString()} paths)`;

    const ticket = aStar.trade_ticket || {};
    const scoreBadge = document.getElementById("astar-grade-badge");
    const ticketLabel = document.getElementById("astar-ticket-label");
    const ticketPane = document.getElementById("astar-ticket-pane");
    const actionEl = document.getElementById("astar-action");

    if (scoreBadge) scoreBadge.innerText = `${aStar.confluence_score}/100`;
    if (actionEl) actionEl.innerText = `ACTION: ${ticket.action || "STAND ASIDE"}`;
    if (ticketLabel) ticketLabel.innerText = ticket.ticket_title || `EXECUTION TICKET (${aStar.confluence_score}/100)`;
    if (ticketPane) ticketPane.style.borderStyle = aStar.is_a_star ? "solid" : "dashed";

    if (document.getElementById("astar-entry")) document.getElementById("astar-entry").innerText = ticket.entry_zone || "--";
    if (document.getElementById("astar-sl")) document.getElementById("astar-sl").innerText = ticket.stop_loss || "--";
    if (document.getElementById("astar-tp1")) document.getElementById("astar-tp1").innerText = ticket.take_profit_1 || "--";
    if (document.getElementById("astar-tp2")) document.getElementById("astar-tp2").innerText = ticket.take_profit_2 || "--";
    if (document.getElementById("astar-lot")) document.getElementById("astar-lot").innerText = `${ticket.recommended_lot_size_1000_equity || 0.00} LOT (${ticket.max_risk_amount_usd || '$0.00'})`;

    // Visual Position Progress Gauge
    const ls = aStar.locked_setup;
    const curP = aStar.current_price || 3970.0;
    const gSl = document.getElementById("gauge-sl"), gEnt = document.getElementById("gauge-entry"), gLiv = document.getElementById("gauge-live"), gTp = document.getElementById("gauge-tp1");
    const gUsd = document.getElementById("gauge-pnl-usd"), gR = document.getElementById("gauge-pnl-r"), gSt = document.getElementById("gauge-status-badge"), gPin = document.getElementById("gauge-cursor-pin");

    if (ls) {
      if (gSl) gSl.innerText = `$${(ls.stop_loss || 0).toFixed(2)}`;
      if (gEnt) gEnt.innerText = `$${(ls.fill_price || (Array.isArray(ls.entry_zone) ? (ls.entry_zone[0] + ls.entry_zone[1]) / 2 : curP)).toFixed(2)}`;
      if (gLiv) gLiv.innerText = `$${curP.toFixed(2)}`;
      if (gTp) gTp.innerText = `$${(ls.take_profit_1 || 0).toFixed(2)}`;
      const pUsd = ls.floating_pnl_usd || 0, pR = ls.floating_r || 0;
      if (gUsd) { gUsd.innerText = `${pUsd >= 0 ? "+" : ""}$${pUsd.toFixed(2)} USD`; gUsd.style.color = pUsd >= 0 ? "#fff" : "#ff4444"; }
      if (gR) { gR.innerText = `${pR >= 0 ? "+" : ""}${pR.toFixed(2)} R`; gR.style.color = pR >= 0 ? "#fff" : "#ff4444"; }
      if (gSt) gSt.innerText = ls.status || "PENDING_TRIGGER";
      const rng = Math.abs((ls.take_profit_1 || 0) - (ls.stop_loss || 0));
      if (gPin && rng > 0) {
        const pct = ls.direction === "BUY" ? ((curP - ls.stop_loss) / rng) * 100 : ((ls.stop_loss - curP) / rng) * 100;
        gPin.style.left = `${Math.max(0, Math.min(100, pct))}%`;
      }
    } else {
      if (gSl) gSl.innerText = ticket.stop_loss || "--";
      if (gEnt) gEnt.innerText = ticket.entry_zone ? ticket.entry_zone.split("(")[0].trim() : "--";
      if (gLiv) gLiv.innerText = `$${curP.toFixed(2)}`;
      if (gTp) gTp.innerText = ticket.take_profit_1 || "--";
      if (gSt) gSt.innerText = ticket.setup_status || "STAND_ASIDE";
    }

    if (aStar.time_to_live && aStar.time_to_live.expires_at_epoch) {
      initTtlCountdown(aStar.time_to_live.expires_at_epoch);
    }

    renderEventMcPane("astar");
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
      const c = isWin ? "#ffffff" : (isLoss ? "#ff4444" : "var(--amber-cmd)");
      const rStr = s.realized_r !== null && s.realized_r !== undefined ? `${s.realized_r >= 0 ? "+" : ""}${s.realized_r.toFixed(2)} R` : "0.00 R";
      const eStr = Array.isArray(s.entry_zone) ? `$${s.entry_zone[0].toFixed(2)} - $${s.entry_zone[1].toFixed(2)}` : (s.entry_zone || "N/A");
      return `
        <tr>
          <td>${s.created_at_utc || "TODAY"}</td>
          <td><strong style="color:var(--amber-cmd);">${s.setup_id}</strong></td>
          <td>${s.event_type || "DAILY_A*"} (${s.asset || "XAUUSD"})</td>
          <td><strong style="color:${s.direction === 'BUY' ? '#ffffff' : '#ff4444'}">${s.direction}</strong></td>
          <td>${eStr}</td>
          <td style="color:#ff4444;">${s.stop_loss ? `$${s.stop_loss.toFixed(2)}` : "N/A"}</td>
          <td>${s.take_profit_1 ? `$${s.take_profit_1.toFixed(2)}` : "N/A"}</td>
          <td><span style="color:${c}; border:1px solid ${c}; padding:1px 4px; font-size:8.5px;">${s.status}</span></td>
          <td style="color:${c}; font-weight:900;">${rStr}</td>
          <td style="color:#aaa;">${s.resolution || "Active"}</td>
        </tr>
      `;
    }).join("");
  } catch (e) {}
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

function initAutoRefreshLoop() {
  setInterval(async () => {
    try {
      const isAstar = getActiveModuleId() === "mod-astar";
      const res = await fetch(`/api/live-ticks?include_astar=${isAstar}`);
      const data = await res.json();
      if (data && data.prices) {
        const p = data.prices;
        animateNumber(document.getElementById("tick-gold"), p["GC=F"], "$", "", 2);
        animateNumber(document.getElementById("tick-dxy"), p["DX-Y.NYB"], "", "", 2);
        animateNumber(document.getElementById("tick-2y"), p["ZT=F"], "", "%", 2);
        animateNumber(document.getElementById("tick-10y"), p["^TNX"], "", "%", 2);
        animateNumber(document.getElementById("tick-oil"), p["CL=F"], "$", "", 2);
      }
      if (isAstar && data && data.daily_a_star) {
        fetchAstarData();
      }
    } catch (e) {}
  }, 6000);

  document.getElementById("btn-reset-astar-session")?.addEventListener("click", async () => {
    const b = document.getElementById("btn-reset-astar-session");
    b.innerText = "WAIT REFRESHING...";
    try { await fetch("/api/reset-setup-session", { method: "POST" }); await fetchAstarData(); } catch (e) {}
    setTimeout(() => { b.innerText = "FORCE RE-EVALUATE SESSION <GO>"; }, 800);
  });
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
    subtitle: "45-MONTH MULTI-MACRO TIMELINE // PPI WHOLESALE & RETAIL DEMAND YoY (2023-2026)",
    refVal: 2.0,
    refLabel: "2.0% PIPELINE BENCHMARK",
    series: [
      { key: "ppi_yoy", label: "PPI FINAL YoY (%)", short: "PPI YoY", color: "#ff4444", width: 2.2, dash: "" },
      { key: "retail_yoy", label: "RETAIL SALES YoY (%)", short: "RETAIL YoY", color: "#00e5ff", width: 1.9, dash: "" },
      { key: "ppi_mom", label: "PPI MoM (%)", short: "PPI MoM", color: "#ffb000", width: 1.4, dash: "3,3" }
    ],
    note: "FRED PPIFIS & RSAFS through Sep 2026 (+1.34% PPI jump); Oct 2026 PPI releases Nov 13."
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
        <td><span style="color:${ev.impact === 'MAXIMUM' ? '#ff4444' : '#ffffff'}; font-weight:800;">[ ${ev.impact} ]</span></td>
        <td style="color:${ev.actual ? '#00e5ff' : '#666'}; font-weight:800;">${ev.actual || 'PENDING'}</td>
        <td>${ev.forecast || '-'}</td>
        <td style="color:#777;">${ev.previous || '-'}</td>
        <td style="color:#fff;">${ev.effect_xauusd}</td>
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

async function fetchAuditLedger() {
  try {
    const res = await fetch("/api/historical");
    const data = await res.json();
    const tbody = document.getElementById("bbg-audit-tbody");
    if (!tbody || !data.events) return;
    const cpiTit = document.getElementById("audit-title-cpi");
    if (cpiTit && data.count) cpiTit.innerText = `[ 07 // AUDIT <GO> : ${data.count}-RELEASE EMPIRICAL POST-NEWS PRICE MOVEMENT RECORD ]`;
    const { mean, sd } = computeMeanAndSd(data.events);
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${e.ref_month}</td>
          <td>${e.actual_mom.toFixed(2)}%</td>
          <td>${e.forecast_mom.toFixed(2)}%</td>
          <td>${e.actual_core_yoy.toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? 'var(--amber-cmd)' : '#fff'}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${e.z_score_composite.toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
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
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${Number(e.actual_rate).toFixed(2)}%</td>
          <td>${Number(e.priced_rate).toFixed(2)}%</td>
          <td style="color:${e.total_policy_shock_bps > 0 ? '#ff4444' : (e.total_policy_shock_bps < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.total_policy_shock_bps > 0 ? '+' : ''}${Number(e.total_policy_shock_bps).toFixed(1)} bps</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
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
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${e.nfp_actual_k}k</td>
          <td>${e.nfp_forecast_k}k</td>
          <td>${Number(e.unemp_actual_pct).toFixed(1)}%</td>
          <td>${Number(e.ahe_actual_pct).toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${Number(e.z_score_composite).toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
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
    tbody.innerHTML = data.events.map(e => {
      const distSd = (e.gold_return_pct - mean) / sd;
      return `
        <tr>
          <td><strong>${e.date}</strong></td>
          <td>${Number(e.ppi_actual).toFixed(2)}%</td>
          <td>${Number(e.ppi_forecast).toFixed(2)}%</td>
          <td>${Number(e.retail_actual).toFixed(2)}%</td>
          <td>${Number(e.retail_forecast).toFixed(2)}%</td>
          <td style="color:${e.z_score_composite > 0 ? '#ff4444' : (e.z_score_composite < 0 ? '#ffffff' : '#888')}; font-weight:700;">${e.z_score_composite > 0 ? '+' : ''}${Number(e.z_score_composite).toFixed(2)}σ</td>
          <td>$${Number(e.gold_open).toFixed(1)} → $${Number(e.gold_close).toFixed(1)}</td>
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

document.addEventListener("DOMContentLoaded", init);
