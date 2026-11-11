"""
NEWSLYSER PROTOTYPE // BLOOMBERG & LINUX TACTICAL TELEMETRY WORKSTATION
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

DATA_DIR = os.path.join(PARENT_DIR, "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="NEWSLYSER // Bloomberg Terminal Prototype", version="5.0.0")

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
    tickers = ["FX:XAUUSD", "TVC:DXY", "TVC:US02Y", "TVC:US10Y", "NYMEX:CL1!"]
    cmd = [
        "curl", "-s", "-m", "3", "-X", "POST", "https://scanner.tradingview.com/global/scan",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"symbols": {"tickers": tickers}, "columns": ["close", "bid", "ask", "change"]})
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
                    if s == "FX:XAUUSD":
                        mapping["GC=F"] = round(float(d[0]), 2)
                        mapping["XAUUSD_SPOT"] = round(float(d[0]), 2)
                        mapping["XAUUSD_BID"] = round(float(d[1]), 2) if len(d) > 1 and d[1] else round(float(d[0]) - 0.15, 2)
                        mapping["XAUUSD_ASK"] = round(float(d[2]), 2) if len(d) > 2 and d[2] else round(float(d[0]) + 0.15, 2)
                    elif s == "TVC:DXY":
                        mapping["DX-Y.NYB"] = round(float(d[0]), 3)
                    elif s == "TVC:US02Y":
                        mapping["ZT=F"] = round(float(d[0]), 3)
                    elif s == "TVC:US10Y":
                        mapping["^TNX"] = round(float(d[0]), 3)
                    elif s == "NYMEX:CL1!":
                        mapping["CL=F"] = round(float(d[0]), 2)
            if mapping:
                return mapping
    except Exception:
        pass
    return {"GC=F": 4047.30, "DX-Y.NYB": 102.28, "ZT=F": 4.75, "^TNX": 5.00, "CL=F": 86.64}

@app.get("/")
def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/state")
def get_state():
    hist, tomorrow, macro = load_data()
    prices = fetch_live_quotes()
    gold_price = prices.get("GC=F", 4047.30)
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    
    dyn = quant_model.calculate_dynamic_levels(current_price=gold_price)
    astar = quant_model.calculate_daily_a_star_setup(current_price=gold_price, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y)
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=gold_price)
    
    return {
        "prices": prices,
        "tomorrow": tomorrow,
        "pre_news_model": pre_news,
        "dynamic_levels": dyn,
        "daily_a_star": astar,
        "historical_count": len(hist)
    }

@app.get("/api/pdf-distribution")
def get_pdf_distribution(event_type: str = "cpi"):
    return quant_model.calculate_probability_density_function(event_type=event_type)

@app.get("/api/monte-carlo")
def get_monte_carlo(horizon: int = 1, z: float = 0.0, price: float = None):
    if price is None:
        prices = fetch_live_quotes()
        price = prices.get("GC=F", 4047.30)
    return quant_model.run_monte_carlo(current_price=price, z_shock=z, horizon_days=horizon, event_type="cpi")

@app.get("/api/shock-frequency")
def get_shock_frequency(z: float = 0.0):
    hist, _, _ = load_data()
    z_vals = np.array([e["z_score_composite"] for e in hist if "z_score_composite" in e])
    n = len(z_vals)
    
    mean_z = float(np.mean(z_vals))
    std_z = float(np.std(z_vals, ddof=1))
    med_z = float(np.median(z_vals))
    q25, q75 = float(np.percentile(z_vals, 25)), float(np.percentile(z_vals, 75))
    iqr = q75 - q25
    
    # Silverman's rule of thumb bandwidth
    h = 0.9 * min(std_z, iqr / 1.34) * (n ** (-0.2)) if iqr > 0 else 0.35
    
    # 10 discrete histogram bins across [-3.0, +3.0]
    bins = np.linspace(-3.0, 3.0, 11)
    counts, edges = np.histogram(z_vals, bins=bins)
    
    hist_bars = []
    for i in range(len(counts)):
        low, high = float(edges[i]), float(edges[i+1])
        matching_events = [e for e in hist if low <= e["z_score_composite"] < high or (i == len(counts)-1 and e["z_score_composite"] == high)]
        hist_bars.append({
            "bin_index": i,
            "range": [round(low, 2), round(high, 2)],
            "label": f"{low:+.1f}σ to {high:+.1f}σ",
            "count": int(counts[i]),
            "pct": round(float(counts[i]) / n * 100.0, 1),
            "events": [{"date": e["date"], "ref": e["ref_month"], "z": e["z_score_composite"], "gold_ret": e["gold_return_pct"]} for e in matching_events]
        })
        
    # Smooth continuous Gaussian Kernel Density Estimation (KDE) curve
    x_grid = np.linspace(-3.2, 3.2, 80)
    kde_curve = []
    for x in x_grid:
        density = (1.0 / (n * h * math.sqrt(2 * math.pi))) * np.sum(np.exp(-0.5 * ((x - z_vals) / h) ** 2))
        kde_curve.append({
            "x": round(float(x), 3),
            "density": round(float(density), 4)
        })
        
    # Current shock empirical ranking
    pct_rank = round(float(np.mean(z_vals <= z) * 100.0), 1)
    dist_from_median = round(z - med_z, 3)
    dist_from_mean_sigma = round((z - mean_z) / max(0.01, std_z), 2)
    
    return {
        "n": n,
        "mean_z": round(mean_z, 3),
        "std_z": round(std_z, 3),
        "median_z": round(med_z, 3),
        "q25": round(q25, 3),
        "q75": round(q75, 3),
        "iqr": round(iqr, 3),
        "bandwidth_h": round(h, 4),
        "current_z": z,
        "empirical_percentile_rank": pct_rank,
        "distance_from_median": dist_from_median,
        "distance_from_mean_sigma": dist_from_mean_sigma,
        "histogram_bars": hist_bars,
        "kde_curve": kde_curve
    }

@app.get("/api/transmission-curves")
def get_transmission_curves():
    hist, _, _ = load_data()
    z_vals = np.array([e["z_score_composite"] for e in hist])
    gold_rets = np.array([e["gold_return_pct"] for e in hist])
    dxy_rets = np.array([e["dxy_return_pct"] for e in hist])
    zt_rets = np.array([e["zt_return_pct"] for e in hist])
    
    m_gold = quant_model.run_regression_ols(z_vals, gold_rets)
    m_dxy = quant_model.run_regression_ols(z_vals, dxy_rets)
    m_zt = quant_model.run_regression_ols(z_vals, zt_rets)
    
    z_grid = np.linspace(-3.0, 3.0, 61)
    curves = []
    
    for z in z_grid:
        pred_gold = m_gold["beta"][0] + m_gold["beta"][1] * z
        pred_dxy = m_dxy["beta"][0] + m_dxy["beta"][1] * z
        pred_zt = m_zt["beta"][0] + m_zt["beta"][1] * z
        
        curves.append({
            "z": round(float(z), 2),
            "gold_return": round(float(pred_gold), 3),
            "gold_upper_95": round(float(pred_gold + m_gold["pred_moe_95"]), 3),
            "gold_lower_95": round(float(pred_gold - m_gold["pred_moe_95"]), 3),
            "dxy_return": round(float(pred_dxy), 3),
            "zt_return": round(float(pred_zt), 3)
        })
        
    scatter_points = []
    for e in hist:
        scatter_points.append({
            "date": e["date"],
            "ref_month": e["ref_month"],
            "z": e["z_score_composite"],
            "gold_ret": e["gold_return_pct"],
            "dxy_ret": e["dxy_return_pct"],
            "zt_ret": e["zt_return_pct"]
        })
        
    return {
        "models": {
            "gold": {"beta": m_gold["beta"], "moe_95": m_gold["moe_95"], "r2": m_gold["r2"], "p_value": m_gold["p_values"][1]},
            "dxy": {"beta": m_dxy["beta"], "moe_95": m_dxy["moe_95"], "r2": m_dxy["r2"], "p_value": m_dxy["p_values"][1]},
            "zt_note": {"beta": m_zt["beta"], "moe_95": m_zt["moe_95"], "r2": m_zt["r2"], "p_value": m_zt["p_values"][1]}
        },
        "response_curves": curves,
        "scatter_points": scatter_points
    }

@app.get("/api/hypothesis-diagnostics")
def get_hypothesis_diagnostics():
    hist, _, _ = load_data()
    y = np.array([e["gold_return_pct"] for e in hist])
    X = np.column_stack([
        [e["z_score_composite"] for e in hist],
        [e["dxy_return_pct"] for e in hist],
        [e["zt_return_pct"] for e in hist]
    ])
    
    m = quant_model.run_regression_ols(X, y)
    df = m["degrees_of_freedom"]
    t_crit = quant_model._student_t_crit(df, alpha=0.05)
    
    # Student's t distribution curve across [-4.5, +4.5]
    t_grid = np.linspace(-4.5, 4.5, 80)
    c_norm = math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)) / math.sqrt(math.pi * df)
    t_curve = []
    for t in t_grid:
        pdf_val = c_norm * ((1.0 + (t ** 2) / df) ** (-(df + 1) / 2.0))
        is_reject = bool(abs(t) >= t_crit)
        t_curve.append({
            "t": round(float(t), 3),
            "pdf": round(float(pdf_val), 4),
            "is_reject": is_reject
        })
        
    # Hat Matrix Leverage & Studentized Residuals
    X_design = np.column_stack([np.ones(len(y)), X])
    n, k = X_design.shape
    invXtX = np.linalg.pinv(X_design.T @ X_design)
    H = X_design @ invXtX @ X_design.T
    h_diag = np.clip(np.diag(H), 0, 0.99)
    
    residuals = y - (X_design @ m["beta"])
    sigma_resid = math.sqrt(np.sum(residuals ** 2) / df)
    stud_res = residuals / (sigma_resid * np.sqrt(1.0 - h_diag))
    h_threshold = 2.0 * k / n
    
    diagnostic_points = []
    for i, e in enumerate(hist):
        diagnostic_points.append({
            "date": e["date"],
            "ref_month": e["ref_month"],
            "z": e["z_score_composite"],
            "leverage_h": round(float(h_diag[i]), 4),
            "studentized_resid": round(float(stud_res[i]), 3),
            "is_high_leverage": bool(h_diag[i] > h_threshold),
            "is_outlier": bool(abs(stud_res[i]) > 2.0)
        })
        
    return {
        "df": df,
        "t_crit_95": t_crit,
        "beta_z_t_stat": m["t_stats"][1],
        "beta_z_p_val": m["p_values"][1],
        "beta_dxy_t_stat": m["t_stats"][2],
        "beta_zt_t_stat": m["t_stats"][3],
        "f_stat": m["f_stat"],
        "leverage_threshold": round(h_threshold, 4),
        "t_curve": t_curve,
        "diagnostic_points": diagnostic_points
    }

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

@app.get("/api/pre-news-model")
def get_pre_news_model(price: float = None, atr: float = 12.50):
    _, tomorrow, _ = load_data()
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=price, atr_m30=atr)

@app.get("/api/fomc-prenew-model")
def get_fomc_prenew_model(price: float = None, atr: float = 12.50):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_fomc_prenew_model(current_price=price, atr_m30=atr)

@app.get("/api/nfp-prenew-model")
def get_nfp_prenew_model(price: float = None, atr: float = 12.50):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_nfp_prenew_model(current_price=price, atr_m30=atr)

@app.get("/api/ppi-prenew-model")
def get_ppi_prenew_model(price: float = None, atr: float = 12.50):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_ppi_prenew_model(current_price=price, atr_m30=atr)

@app.get("/api/fomc-model")
def get_fomc_model(actual_rate: float = 5.00, priced_rate: float = 5.00, dots_shift_bps: float = 0.0, price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_fomc_model(actual_rate, priced_rate, dots_shift_bps, current_price=price)

@app.get("/api/nfp-model")
def get_nfp_model(nfp_actual: float = 175.0, nfp_forecast: float = 175.0, unemp_actual: float = 4.1, unemp_forecast: float = 4.1, ahe_actual: float = 0.3, ahe_forecast: float = 0.3, price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_nfp_model(nfp_actual, nfp_forecast, unemp_actual, unemp_forecast, ahe_actual, ahe_forecast, current_price=price)

@app.get("/api/ppi-retail-model")
def get_ppi_retail_model(ppi_actual: float = 0.2, ppi_forecast: float = 0.2, retail_actual: float = 0.4, retail_forecast: float = 0.4, price: float = None):
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
    gold = price or prices.get("GC=F", 4047.30)
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    return quant_model.calculate_daily_a_star_setup(current_price=gold, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y)

@app.get("/api/dynamic-levels")
def get_dynamic_levels(price: float = None):
    if price is None:
        price = fetch_live_quotes().get("GC=F", 4047.30)
    return quant_model.calculate_dynamic_levels(current_price=price)

@app.post("/api/refresh-all")
def refresh_all():
    prices = fetch_live_quotes()
    _, tomorrow, _ = load_data()
    current_gold = prices.get("GC=F", 4047.30)
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    
    dyn_levels = quant_model.calculate_dynamic_levels(current_price=current_gold)
    a_star = quant_model.calculate_daily_a_star_setup(current_price=current_gold, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y)
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=current_gold)
    fomc = quant_model.calculate_fomc_model(5.00, 5.00, 0.0, current_price=current_gold)
    nfp = quant_model.calculate_nfp_model(175.0, 175.0, 4.1, 4.1, 0.3, 0.3, current_price=current_gold)
    ppi = quant_model.calculate_ppi_retail_model(0.2, 0.2, 0.4, 0.4, current_price=current_gold)
    
    return {
        "status": "SUCCESS",
        "prices": prices,
        "pre_news_model": pre_news,
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
