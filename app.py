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

@app.get("/")
def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

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

class SimRequest(BaseModel):
    actual_mom: float
    actual_core_yoy: float

@app.post("/api/simulate")
def simulate(req: SimRequest):
    _, tomorrow, _ = load_data()
    sim_result = quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow)
    return sim_result

class DeepSeekRequest(BaseModel):
    actual_mom: float
    actual_core_yoy: float
    supercore_mom_est: float = 0.2
    market_context: str = "Gold trading at $4,128, DXY at 101.87, US 2Y yield at 4.46%, 10Y TIPS at 2.60%."

@app.post("/api/deepseek-briefing")
def deepseek_briefing(req: DeepSeekRequest):
    _, tomorrow, _ = load_data()
    sim = quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow)
    
    prompt = f"""You are the Chief Quantitative Macroeconomist and Head of Trading at an institutional fixed income & commodities proprietary trading desk.
Analyze tomorrow's U.S. CPI release scenario strictly through empirical econometric principles, cross-asset transmission mechanisms, and market microstructure.

SCENARIO PARAMETERS:
- Release: U.S. October 2026 CPI (BLS Release: Nov 10, 2026, 08:30 AM EST)
- Actual Headline CPI MoM: {req.actual_mom:.2f}% (Consensus Forecast: {tomorrow['forecast_mom']:.2f}%, Prev: {tomorrow['prev_mom']:.2f}%)
- Actual Core CPI YoY: {req.actual_core_yoy:.2f}% (Consensus Forecast: {tomorrow['forecast_core_yoy']:.2f}%, Prev: {tomorrow['prev_core_yoy']:.2f}%)
- Supercore (Services ex-Shelter) MoM Estimate: {req.supercore_mom_est:.2f}%
- Quantitative Model Composite Z-Score: {sim['surprises']['z_score_composite']:+.2f} sigma
- Model Regime: {sim['model_forecast']['regime']}
- Model Signal: {sim['model_forecast']['signal']}
- Baseline Expected Gold Move: {sim['model_forecast']['expected_gold_return_pct']:+.2f}%
- Baseline Expected DXY Move: {sim['model_forecast']['expected_dxy_return_pct']:+.2f}%
- Market Baseline: {req.market_context}

INSTRUCTIONS:
1. Provide an executive institutional desk briefing formatted in stark, professional industrial bullet points.
2. Break down the Transmission Mechanism: CPI shock -> Fed terminal rate pricing (SOFR futures) -> US 2Y Yields -> 10Y TIPS Real Yields -> Gold (XAUUSD) & DXY.
3. Detail the 3-Wave Execution Protocol:
   - Wave 1 (T0 to T+60s): Microstructure liquidity gap, spread blowout warning ($2-$5 spread on Gold), and why market orders are prohibited.
   - Wave 2 (T+2m to T+15m): Component digestion (Core vs Supercore vs Shelter lag) and filtering false knee-jerk moves.
   - Wave 3 (T+15m to T+60m): Post-Announcement Drift (PEAD) entry trigger and invalidation levels.
4. Conclude with an unambiguous, terse Tactical Verdict (Bias, Execution Trigger, Invalidation). Do NOT use conversational fluff or generic AI disclaimers."""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an elite quantitative macro research system. Provide terse, rigorous, mathematical and institutional analysis. Avoid generic AI slop."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        return {
            "status": "success",
            "briefing": content,
            "simulation": sim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8888, reload=False)
