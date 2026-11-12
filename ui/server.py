"""
NEWSLYSER // BLOOMBERG PROFESSIONAL TERMINAL WORKSTATION (STANDBY MIRROR)
Runs on dedicated port 8889.
Delivers:
1. Bayesian Probability Density Function (Prior vs. Posterior)
2. Kou (2002) Jump-Diffusion Monte Carlo SDE Fan
3. Shock Frequency Distribution & Empirical Distance from Median (Histogram + KDE)
4. Cross-Asset Expected Return Transmission Curves
5. Student's t Hypothesis Testing & White HC3 Outlier Leverage Diagnostics
6. Microstructure 3-Wave Execution Firewall & Anchored VWAP Geometry
"""

import os
import sys
import json
import math
import subprocess
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
import quant_model
import data_pipeline

DATA_DIR = os.path.join(PARENT_DIR, "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="NEWSLYSER // Bloomberg Professional Terminal", version="6.0.0")

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def load_data():
    with open(os.path.join(DATA_DIR, "cpi_historical.json")) as f:
        hist = json.load(f)
    with open(os.path.join(DATA_DIR, "tomorrow_preview.json")) as f:
        tomorrow = json.load(f)
    with open(os.path.join(DATA_DIR, "macro_summary.json")) as f:
        macro = json.load(f)
    return hist, tomorrow, macro

def fetch_live_quotes():
    tickers = ["FX_IDC:XAUUSD", "OANDA:XAUUSD", "FX:XAUUSD", "TVC:DXY", "TVC:US02Y", "TVC:US10Y", "NYMEX:CL1!", "TVC:VIX"]
    cmd = [
        "curl", "-s", "-m", "3", "-X", "POST", "https://scanner.tradingview.com/global/scan",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"symbols": {"tickers": tickers}, "columns": ["close", "bid", "ask", "change", "high", "low", "open", "ATR|30"]})
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            data = json.loads(res.stdout).get("data", [])
            mapping = {}
            for row in data:
                s = row.get("s", "")
                d = row.get("d", [])
                if len(d) >= 1 and d[0] is not None:
                    if s == "FX_IDC:XAUUSD":
                        live_px = round(float(d[0]), 2)
                        mapping["GC=F"] = live_px
                        mapping["XAUUSD_SPOT"] = live_px
                        if len(d) > 2 and d[1] and d[2] and abs(((float(d[1]) + float(d[2])) / 2.0) - live_px) <= 1.50:
                            mapping["XAUUSD_BID"] = round(float(d[1]), 2)
                            mapping["XAUUSD_ASK"] = round(float(d[2]), 2)
                        else:
                            mapping["XAUUSD_BID"] = live_px
                            mapping["XAUUSD_ASK"] = live_px
                        if len(d) > 5 and d[4] is not None and d[5] is not None:
                            mapping["SESSION_HIGH"] = round(float(d[4]), 2)
                            mapping["SESSION_LOW"] = round(float(d[5]), 2)
                            mapping["SESSION_CLOSE"] = live_px
                        if len(d) > 7 and d[7] is not None and float(d[7]) > 0:
                            mapping["ATR_M30"] = round(float(d[7]), 2)
                    elif s == "TVC:DXY":
                        mapping["DX-Y.NYB"] = round(float(d[0]), 3)
                    elif s == "TVC:US02Y":
                        mapping["ZT=F"] = round(float(d[0]), 3)
                    elif s == "TVC:US10Y":
                        mapping["^TNX"] = round(float(d[0]), 3)
                    elif s == "NYMEX:CL1!":
                        mapping["CL=F"] = round(float(d[0]), 2)
                    elif s == "TVC:VIX":
                        mapping["VIX"] = round(float(d[0]), 2)
            if "GC=F" not in mapping:
                for row in data:
                    s = row.get("s", "")
                    d = row.get("d", [])
                    if s in ["FX:XAUUSD", "OANDA:XAUUSD"] and len(d) >= 1 and d[0] is not None:
                        mapping["GC=F"] = round(float(d[0]), 2)
                        mapping["XAUUSD_SPOT"] = mapping["GC=F"]
                        break
            if mapping:
                return mapping
    except Exception:
        pass
    return {"GC=F": 3975.0, "DX-Y.NYB": 102.28, "ZT=F": 4.75, "^TNX": 5.00, "CL=F": 86.64}

@app.get("/")
def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/state")
def get_state():
    hist, tomorrow, macro = load_data()
    prices = fetch_live_quotes()
    gold_price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)

    dyn = quant_model.calculate_dynamic_levels(current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    astar = quant_model.calculate_daily_a_star_setup(current_price=gold_price, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y, wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"), n_paths=10000, session_h=sh, session_l=sl, session_c=sc)
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=gold_price, atr_m30=atr_val)
    fomc_pre = quant_model.calculate_fomc_prenew_model(current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    nfp_pre = quant_model.calculate_nfp_prenew_model(current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    ppi_pre = quant_model.calculate_ppi_prenew_model(current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)

    return {
        "prices": prices,
        "tomorrow": tomorrow,
        "pre_news_model": pre_news,
        "fomc_prenew": fomc_pre,
        "nfp_prenew": nfp_pre,
        "ppi_prenew": ppi_pre,
        "dynamic_levels": dyn,
        "daily_a_star": astar,
        "historical_count": len(hist)
    }

@app.get("/api/pdf-distribution")
def get_pdf_distribution(event_type: str = "cpi"):
    return quant_model.calculate_probability_density_function(event_type=event_type)

@app.get("/api/monte-carlo")
def get_monte_carlo(horizon: int = 1, z: float = 0.0, price: float = None, event_type: str = "cpi", n_paths: int = 10000):
    if price is None:
        prices = fetch_live_quotes()
        price = prices.get("GC=F", 3975.0)
    return quant_model.run_monte_carlo(current_price=price, z_shock=z, horizon_days=horizon, n_paths=n_paths, event_type=event_type)

@app.get("/api/shock-frequency")
def get_shock_frequency(z: float = 0.0, event_type: str = "cpi"):
    return quant_model.get_event_shock_frequency(event_type=event_type, z=z)

@app.get("/api/transmission-curves")
def get_transmission_curves(event_type: str = "cpi"):
    return quant_model.get_event_transmission_curves(event_type=event_type)

@app.get("/api/hypothesis-diagnostics")
def get_hypothesis_diagnostics(event_type: str = "cpi"):
    return quant_model.get_event_hypothesis_diagnostics(event_type=event_type)

@app.get("/api/hypothesis-audit-all")
def get_hypothesis_audit_all():
    return quant_model.get_all_methods_hypothesis_audit()
@app.get("/api/quant-models-audit")
def get_quant_models_audit():
    return quant_model.get_eight_model_quant_audit()


class SimReq(BaseModel):
    actual_mom: float
    actual_core_yoy: float
    price: float = None

@app.post("/api/simulate")
def simulate(req: SimReq):
    _, tomorrow, _ = load_data()
    price = req.price or fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow, current_price=price)

@app.get("/api/cpi-historical")
def get_historical():
    hist, _, _ = load_data()
    eval_results = quant_model.evaluate_models(hist)
    return {"count": len(hist), "events": hist, "models": eval_results}

@app.get("/api/historical")
def get_historical_alias():
    return get_historical()

@app.get("/api/fomc-historical")
def get_fomc_historical():
    with open(os.path.join(DATA_DIR, "fomc_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_fomc_models(events)
    return {"count": len(events), "events": events, "models": eval_results}

@app.get("/api/nfp-historical")
def get_nfp_historical():
    with open(os.path.join(DATA_DIR, "nfp_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_nfp_models(events)
    return {"count": len(events), "events": events, "models": eval_results}

@app.get("/api/ppi-historical")
def get_ppi_historical():
    with open(os.path.join(DATA_DIR, "ppi_retail_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_ppi_models(events)
    return {"count": len(events), "events": events, "models": eval_results}

@app.get("/api/tomorrow")
def get_tomorrow():
    _, tomorrow, macro = load_data()
    return {"tomorrow_meta": tomorrow, "macro_summary": macro}

@app.get("/api/premarket-snapshot")
def get_premarket_snapshot():
    _, tomorrow, _ = load_data()
    prices = fetch_live_quotes()
    price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
    snap_path = os.path.join(DATA_DIR, "premarket_snapshot.json")
    snapshot_meta = {}
    if os.path.exists(snap_path):
        with open(snap_path) as f:
            snapshot_meta = json.load(f)
    return {
        "snapshot": snapshot_meta,
        "models": {
            "cpi": quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc),
            "fomc": quant_model.calculate_fomc_prenew_model(current_price=price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc),
            "nfp": quant_model.calculate_nfp_prenew_model(current_price=price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc),
            "ppi": quant_model.calculate_ppi_prenew_model(current_price=price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc),
        },
        "quant_8": quant_model.get_eight_model_quant_audit()
    }

@app.get("/api/live-ticks")
def get_live_ticks(include_astar: bool = False):
    prices = fetch_live_quotes()
    res = {"status": "SUCCESS", "prices": prices}
    if include_astar:
        current_gold = prices.get("GC=F", 3975.0)
        atr_val = prices.get("ATR_M30", 10.50)
        sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
        dxy = prices.get("DX-Y.NYB", 102.28)
        us10y = prices.get("^TNX", 5.00)
        us2y = prices.get("ZT=F", 4.75)
        res["daily_a_star"] = quant_model.calculate_daily_a_star_setup(current_price=current_gold, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y, wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"), n_paths=10000, session_h=sh, session_l=sl, session_c=sc)
    return res

@app.get("/api/pre-news-model")
def get_pre_news_model(price: float = None, atr: float = None):
    _, tomorrow, _ = load_data()
    prices = fetch_live_quotes()
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=price, atr_m30=atr_val, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))

@app.get("/api/fomc-prenew-model")
def get_fomc_prenew_model(price: float = None, atr: float = None):
    prices = fetch_live_quotes()
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_fomc_prenew_model(current_price=price, atr_m30=atr_val, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))

@app.get("/api/nfp-prenew-model")
def get_nfp_prenew_model(price: float = None, atr: float = None):
    prices = fetch_live_quotes()
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_nfp_prenew_model(current_price=price, atr_m30=atr_val, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))

@app.get("/api/ppi-prenew-model")
def get_ppi_prenew_model(price: float = None, atr: float = None):
    prices = fetch_live_quotes()
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_ppi_prenew_model(current_price=price, atr_m30=atr_val, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))
@app.get("/api/fomc-model")
def get_fomc_model(actual_rate: float = 4.00, priced_rate: float = 4.00, dots_shift_bps: float = 0.0, price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_fomc_model(actual_rate, priced_rate, dots_shift_bps, current_price=price)

@app.get("/api/nfp-model")
def get_nfp_model(nfp_actual: float = 135.0, nfp_forecast: float = 135.0, unemp_actual: float = 4.1, unemp_forecast: float = 4.1, ahe_actual: float = 0.3, ahe_forecast: float = 0.3, price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_nfp_model(nfp_actual, nfp_forecast, unemp_actual, unemp_forecast, ahe_actual, ahe_forecast, current_price=price)

@app.get("/api/ppi-retail-model")
def get_ppi_retail_model(ppi_actual: float = 0.4, ppi_forecast: float = 0.4, retail_actual: float = 0.4, retail_forecast: float = 0.4, price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_ppi_retail_model(ppi_actual, ppi_forecast, retail_actual, retail_forecast, current_price=price)

@app.get("/api/high-impact-calendar")
def get_high_impact_calendar():
    with open(os.path.join(DATA_DIR, "high_impact_calendar.json")) as f:
        return json.load(f)

@app.get("/api/academic-literature")
def get_academic_literature():
    with open(os.path.join(DATA_DIR, "academic_literature.json")) as f:
        return json.load(f)

@app.get("/api/yoy-timeline")
def get_yoy_timeline():
    with open(os.path.join(DATA_DIR, "cpi_yoy_timeline.json")) as f:
        return json.load(f)

@app.get("/api/setup-ledger")
def get_setup_ledger():
    if hasattr(quant_model, "SetupLifecycleManager"):
        mgr = quant_model.SetupLifecycleManager()
        return mgr.get_ledger()
    return {"active_setups": [], "completed_setups": []}

@app.get("/api/global-macro-matrix")
def get_global_macro_matrix():
    matrix_path = os.path.join(DATA_DIR, "global_macro_matrix.json")
    if os.path.exists(matrix_path):
        with open(matrix_path) as f:
            return json.load(f)
    return {}

@app.get("/api/daily-a-star")
def get_daily_a_star(price: float = None):
    prices = fetch_live_quotes()
    gold = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    return quant_model.calculate_daily_a_star_setup(current_price=gold, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y, wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"), n_paths=10000, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))

@app.get("/api/dynamic-levels")
def get_dynamic_levels(price: float = None):
    prices = fetch_live_quotes()
    if price is None:
        price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    return quant_model.calculate_dynamic_levels(current_price=price, atr_m30=atr_val, session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE"))

@app.post("/api/sync-releases")
def sync_releases(force: bool = True):
    return data_pipeline.sync_live_releases_if_stale(force=force)

@app.post("/api/refresh-all")
def refresh_all():
    prices = fetch_live_quotes()
    _, tomorrow, _ = load_data()
    current_gold = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)

    dyn_levels = quant_model.calculate_dynamic_levels(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    mgr = quant_model.SetupLifecycleManager()
    mgr.evaluate_live_tick(current_gold, live_high=sh, live_low=sl)
    a_star = quant_model.calculate_daily_a_star_setup(current_price=current_gold, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y, wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"), n_paths=10000, session_h=sh, session_l=sl, session_c=sc)
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    fomc_pre = quant_model.calculate_fomc_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    nfp_pre = quant_model.calculate_nfp_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    ppi_pre = quant_model.calculate_ppi_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    fomc = quant_model.calculate_fomc_model(4.00, 4.00, 0.0, current_price=current_gold, atr_m30=atr_val)
    nfp = quant_model.calculate_nfp_model(135.0, 135.0, 4.1, 4.1, 0.3, 0.3, current_price=current_gold, atr_m30=atr_val)
    ppi = quant_model.calculate_ppi_retail_model(0.4, 0.4, 0.4, 0.4, current_price=current_gold, atr_m30=atr_val)
    return {
        "status": "SUCCESS",
        "prices": prices,
        "pre_news_model": pre_news,
        "fomc_prenew": fomc_pre,
        "nfp_prenew": nfp_pre,
        "ppi_prenew": ppi_pre,
        "dynamic_levels": dyn_levels,
        "daily_a_star": a_star,
        "fomc_model": fomc,
        "nfp_model": nfp,
        "ppi_model": ppi
    }

def get_deepseek_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        env_path = os.path.join(PARENT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("DEEPSEEK_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        break
    return key or ""

class DeepSeekReq(BaseModel):
    mode: str = "pre_news"
    event_type: str = "cpi"
    actual_mom: float = 0.3
    actual_core_yoy: float = 2.8

@app.post("/api/deepseek-briefing")
def deepseek_briefing(req: DeepSeekReq):
    api_key = get_deepseek_key()
    if not api_key:
        return {
            "status": "FALLBACK_OFFLINE",
            "memo": f"""[ EXECUTIVE MACRO BRIEFING // {req.event_type.upper()} DUAL-KERNEL TELEMETRY ]
- EVENT HORIZON: U.S. {req.event_type.upper()} MACROECONOMIC ANNOUNCEMENT
- MODEL: TWO-STAGE LEAST SQUARES (2SLS) WITH WHITE (1980) HC3 COVARIANCE
- REGIME VERDICT: PURIFIED COMPOSITE SHOCK AT +1.33σ (94.1TH PERCENTILE HISTORICAL EXTENSION)
- CROSS-ASSET VECTOR: 2Y TREASURY EXPANSION (+4.5 bps/σ) EXERTS DOWNWARD IMPULSE ON GOLD ($XAUUSD)
- MICROSTRUCTURE RULE: STRICT 60-SECOND WAVE 1 FIREWALL LOCKOUT. NO MARKET ORDERS. FADE SPIKES AT S2/R2 BOUNDARIES."""
        }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        _, tomorrow, _ = load_data()
        prompt = f"""You are the Chief Quantitative Macroeconomist at a Tier-1 macro proprietary trading desk.
Deliver an executive institutional execution memo for {req.event_type.upper()} ahead of release.
Write in stark uppercase industrial format [ SECTION ] and hyphen bullets (- ). Every statement must be a quantitative fact or defined risk."""
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        return {"status": "SUCCESS", "memo": res.choices[0].message.content}
    except Exception as e:
        return {"status": "FALLBACK_OFFLINE", "memo": f"DeepSeek API error: {str(e)}"}

@app.get("/api/deepseek-test")
def deepseek_test():
    api_key = get_deepseek_key()
    return {"has_key": bool(api_key), "status": "READY" if api_key else "OFFLINE_FALLBACK"}
@app.get("/api/hypothesis-test")
def get_hypothesis_test(event_type: str = "cpi"):
    return quant_model.get_event_hypothesis_test(event_type=event_type)

@app.post("/api/reset-setup-session")
def reset_setup_session(event_type: str = "daily", validity_seconds: int = 5400):
    import time
    now_ts = int(time.time())
    expires_at = now_ts + validity_seconds
    data = {"created_at_epoch": now_ts, "expires_at_epoch": expires_at, "max_validity_minutes": validity_seconds // 60}
    epoch_file = os.path.join(DATA_DIR, "setup_epochs.json")
    try:
        epochs = {}
        if os.path.exists(epoch_file):
            with open(epoch_file) as f:
                epochs = json.load(f)
        epochs[event_type] = data
        with open(epoch_file, "w") as f:
            json.dump(epochs, f, indent=2)
    except Exception as e:
        print(f"Error resetting epoch: {e}")
    return {"status": "SUCCESS", "message": f"Fresh session window initialized for {event_type.upper()}", **data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8889, reload=False)
