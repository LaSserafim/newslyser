"""
NEWSLYSER - Institutional CPI Quantitative Terminal & Macro Engine
FastAPI application delivering real econometric analysis, historical telemetry, and DeepSeek AI macro briefing.
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

import quant_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

def get_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("DEEPSEEK_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        break
    return key or ""

DEEPSEEK_API_KEY = get_api_key()
app = FastAPI(title="NEWSLYSER // CPI Quantitative Terminal", version="2.4.0")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def load_data():
    with open(os.path.join(DATA_DIR, "cpi_historical.json")) as f:
        hist = json.load(f)
    with open(os.path.join(DATA_DIR, "tomorrow_preview.json")) as f:
        tomorrow = json.load(f)
    with open(os.path.join(DATA_DIR, "macro_summary.json")) as f:
        macro = json.load(f)
    return hist, tomorrow, macro
def fetch_live_macro_data():
    """
    Fetches verified live data from ForexFactory (CPI release check)
    and Yahoo Finance (real-time market prices). Zero simulated data.
    """
    import datetime, urllib.request, re
    import yfinance as yf
    
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 1. Fetch real-time market quotes
    prices = {}
    try:
        tickers = ["GC=F", "DX-Y.NYB", "ZT=F", "^TNX"]
        data = yf.download(tickers, period="1d", interval="1m", progress=False)
        if not data.empty and "Close" in data:
            for t in tickers:
                s = data["Close"][t].dropna()
                if not s.empty:
                    prices[t] = round(float(s.iloc[-1]), 3)
    except Exception as e:
        print(f"[LIVE SYNC] Yahoo Finance warning: {e}")
        
    # Fallback to daily close if minute data unavailable
    if "GC=F" not in prices:
        try:
            daily = yf.download(["GC=F", "DX-Y.NYB", "ZT=F", "^TNX"], period="5d", progress=False)["Close"]
            for t in ["GC=F", "DX-Y.NYB", "ZT=F", "^TNX"]:
                s = daily[t].dropna()
                if not s.empty:
                    prices[t] = round(float(s.iloc[-1]), 3)
        except Exception as e:
            print(f"[LIVE SYNC] Fallback price fetch error: {e}")

    # 2. Check ForexFactory official calendar for actual release
    headline_actual = None
    core_actual = None
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            events = json.loads(resp.read().decode("utf-8"))
        
        for item in events:
            if item.get("country") == "USD" and "2026-11-10" in item.get("date", ""):
                title = item.get("title", "")
                act_str = item.get("actual", "")
                if act_str:
                    m = re.search(r"([\-\+]?\d+\.?\d*)", act_str)
                    if m:
                        val = float(m.group(1))
                        if "Core" in title and "m/m" in title:
                            core_actual = val
                        elif "CPI m/m" in title and "Core" not in title:
                            headline_actual = val
    except Exception as e:
        print(f"[LIVE SYNC] ForexFactory check error: {e}")
        
    return {
        "timestamp": now_utc,
        "prices": prices,
        "headline_actual": headline_actual,
        "core_actual": core_actual
    }

@app.post("/api/sync-live")
def sync_live():
    _, tomorrow, _ = load_data()
    live_data = fetch_live_macro_data()
    
    hl_act = live_data["headline_actual"]
    core_act = live_data["core_actual"]
    
    if hl_act is not None:
        est_core_yoy = tomorrow["forecast_core_yoy"] + (core_act - tomorrow["forecast_core_mom"]) if core_act else tomorrow["forecast_core_yoy"]
        est_core_yoy = round(est_core_yoy, 2)
        sim = quant_model.simulate_tomorrow_scenario(hl_act, est_core_yoy, tomorrow)
        
        return {
            "status": "RELEASED",
            "message": "Official CPI Release detected and parsed.",
            "timestamp": live_data["timestamp"],
            "prices": live_data["prices"],
            "actual_mom": hl_act,
            "actual_core_yoy": est_core_yoy,
            "simulation": sim
        }
        
    return {
        "status": "AWAITING_RELEASE",
        "message": "Live market baseline synced. Official October 2026 CPI release not yet published (Scheduled Nov 10 @ 08:30 AM EST).",
        "timestamp": live_data["timestamp"],
        "prices": live_data["prices"],
        "forecast_mom": tomorrow["forecast_mom"],
        "forecast_core_yoy": tomorrow["forecast_core_yoy"],
        "target_date": tomorrow["date"],
        "target_time": tomorrow["time"]
    }

@app.get("/")
def get_index():
    res = FileResponse(os.path.join(STATIC_DIR, "index.html"))
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    res.headers["Pragma"] = "no-cache"
    res.headers["Expires"] = "0"
    return res

@app.get("/api/historical")
def get_historical():
    hist, _, _ = load_data()
    eval_results = quant_model.evaluate_models(hist)
    return {
        "count": len(hist),
        "events": hist,
        "models": eval_results
    }

@app.get("/api/tomorrow")
def get_tomorrow():
    _, tomorrow, macro = load_data()
    return {
        "event": tomorrow,
        "macro_snapshots": macro
    }
@app.get("/api/pre-news-model")
def get_pre_news_model():
    _, tomorrow, _ = load_data()
    return quant_model.calculate_pre_news_bayesian_model(tomorrow)

@app.get("/api/yoy-timeline")
def get_yoy_timeline():
    with open(os.path.join(DATA_DIR, "cpi_yoy_timeline.json")) as f:
        data = json.load(f)
    return data

@app.get("/api/academic-literature")
def get_academic_literature():
    with open(os.path.join(DATA_DIR, "academic_literature.json")) as f:
        data = json.load(f)
    return data
@app.get("/api/monte-carlo")
def get_monte_carlo(horizon: int = 5, z: float = 0.0):
    return quant_model.run_monte_carlo(horizon_days=horizon, z_shock=z)

@app.get("/api/pdf-distribution")
def get_pdf_distribution():
    return quant_model.calculate_probability_density_function()



class SimRequest(BaseModel):
    actual_mom: float
    actual_core_yoy: float

@app.post("/api/simulate")
def simulate(req: SimRequest):
    _, tomorrow, _ = load_data()
    sim_result = quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow)
    return sim_result

class DeepSeekRequest(BaseModel):
    mode: str = "pre_news" # "pre_news" or "post_news"
    actual_mom: float = 0.3
    actual_core_yoy: float = 2.8
    market_context: str = "Gold trading at $4,136.60, DXY at 101.82, US 2Y yield at 4.46%, 10Y yield at 4.84%, 10Y TIPS real yield at 2.60%."

@app.post("/api/deepseek-briefing")
def deepseek_briefing(req: DeepSeekRequest):
    import re
    _, tomorrow, _ = load_data()
    
    if req.mode == "pre_news":
        bayesian = quant_model.calculate_pre_news_bayesian_model(tomorrow)
        prompt = f"""You are the Chief Quantitative Macroeconomist and Head of Fixed Income & Commodities Trading at a Tier-1 macro hedge fund.
Deliver an executive PRE-NEWS predictive macro memo ahead of tomorrow's U.S. October 2026 CPI release (Nov 10 @ 08:30 AM EST).

AUTHENTIC EMPIRICAL CONTEXT:
- Consensus Forecast: Headline MoM 0.3%, Core YoY 2.8%, Core MoM 0.2%.
- Cleveland Fed Inflation Nowcast: Headline 0.28% MoM, Core 0.22% MoM.
- Geopolitical / Energy Factor: October WTI Crude Oil surged +12.11% ($92.87 -> $104.12) due to Middle East supply tension.
- Energy Contribution to Headline MoM: approx +0.08 percentage points upside pressure.
- 10Y Breakeven Inflation (FRED T10YIE): 2.23%.
- Historical 34-Release Consensus Distribution: In-line 53.8%, Beat 32.4%, Miss 13.8%.
- Pre-Release Bayesian Expected Value on Gold: -0.196% (Downside Asymmetry).
- Live Baseline: {req.market_context}

STRICT INSTRUCTIONS:
1. DO NOT USE ANY MARKDOWN ASTERISKS (** or *), NO HASHTAG HEADINGS (###), AND NO CONVERSATIONAL FILLER.
2. Write in stark, uppercase industrial engineering format using brackets [ SECTION NAME ] and hyphen bullets (- ).
3. Deliver:
   [ 1. LEADING MACRO & GEOPOLITICAL ENERGY VECTOR ]
   [ 2. BAYESIAN PROBABILITY DECOMPOSITION ]
   [ 3. ASYMMETRIC EXPECTED PAYOFF ON GOLD (XAUUSD) ]
   [ 4. PRE-NEWS POSITIONING DIRECTIVE (ENTRY, STOP, TARGET, RR) ]
   [ 5. DUAL-STRATEGY COMPARISON: PRE-NEWS EXECUTION VS WAITING FOR POST-NEWS CONFIRMATION ]
4. Every statement must be a quantitative fact, decision, or defined risk. No marketing."""
    else:
        sim = quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow)
        prompt = f"""You are the Chief Quantitative Macroeconomist and Head of Trading at an institutional proprietary trading desk.
Analyze the official U.S. CPI release numbers through empirical econometric principles and cross-asset transmission.

SCENARIO DATA:
- Actual Headline MoM: {req.actual_mom:.2f}% (Forecast: {tomorrow['forecast_mom']:.2f}%)
- Actual Core YoY: {req.actual_core_yoy:.2f}% (Forecast: {tomorrow['forecast_core_yoy']:.2f}%)
- Model Composite Z-Score: {sim['surprises']['z_score_composite']:+.2f} sigma
- Model Regime: {sim['model_forecast']['regime']}
- Model Signal: {sim['model_forecast']['signal']}
- Expected Gold Move: {sim['model_forecast']['expected_gold_return_pct']:+.2f}%
- Expected DXY Move: {sim['model_forecast']['expected_dxy_return_pct']:+.2f}%
- Live Baseline: {req.market_context}

STRICT INSTRUCTIONS:
1. DO NOT USE ANY MARKDOWN ASTERISKS (** or *), NO HASHTAG HEADINGS (###), AND NO CONVERSATIONAL FILLER.
2. Write in stark uppercase industrial format with brackets [ SECTION NAME ] and hyphen bullets (- ).
3. Deliver:
   [ 1. COMPONENT REALITY (SUPERCORE VS SHELTER LAG) ]
   [ 2. CROSS-ASSET TRANSMISSION (2Y YIELDS & 10Y TIPS) ]
   [ 3. 3-WAVE EXECUTION PROTOCOL (WAVE 1 FIREWALL, WAVE 2 FADE, WAVE 3 DRIFT) ]
   [ 4. TACTICAL VERDICT (BIAS, ENTRY, INVALIDATION) ]"""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an elite quantitative macro analyst. Output clean, raw plain text without asterisks or markdown styling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=950
        )
        content = response.choices[0].message.content
        
        # Clean out any stray asterisks or markdown headers
        clean_text = content.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        
        return {
            "status": "success",
            "mode": req.mode,
            "briefing": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8888, reload=False)
