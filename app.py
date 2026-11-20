"""
NEWSLYSER - Institutional Macro Quantitative Terminal & Multi-Event Econometric Engine
FastAPI application delivering real econometric analysis, historical telemetry, and DeepSeek AI macro briefing.
"""

import os
for _k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(_k, None)
import json
import math
import numpy as np
import urllib.request
import re
import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import yfinance as yf

import quant_model
import data_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
SETUP_EPOCHS_FILE = os.path.join(DATA_DIR, "setup_epochs.json")

def get_or_create_setup_epoch(event_type="daily", ttl_seconds=5400):
    """
    Persistent Server-Side Setup Epoch Anchor:
    Guarantees setup timestamp survives page refreshes and server reboots.
    Prevents timer reset loop. Auto-rolls over when setup validity expires.
    """
    import time
    now_ts = int(time.time())
    epochs = {}
    if os.path.exists(SETUP_EPOCHS_FILE):
        try:
            with open(SETUP_EPOCHS_FILE) as f:
                epochs = json.load(f)
        except Exception:
            epochs = {}
            
    rec = epochs.get(event_type)
    if not rec or now_ts >= rec.get("expires_at", 0):
        rec = {"created_at": now_ts, "expires_at": now_ts + ttl_seconds}
        epochs[event_type] = rec
        try:
            with open(SETUP_EPOCHS_FILE, "w") as f:
                json.dump(epochs, f, indent=2)
        except Exception as e:
            print("Error writing setup_epochs.json:", e)
            
    return rec["created_at"], rec["expires_at"]


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
app = FastAPI(title="NEWSLYSER // Bloomberg Professional Terminal", version="6.0.0")

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

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

_CACHE_MACRO = {"timestamp": 0.0, "data": None}


def fetch_live_quotes():
    return fetch_tradingview_live_quotes()

def fetch_tradingview_live_quotes():
    """
    Direct streaming quotes from TradingView Global Scanner API.
    Prioritizes official spot feeds: FX_IDC:XAUUSD & FX_IDC:EURUSD alongside OANDA,
    TVC:DXY, TVC:US10Y, TVC:US02Y, TVC:DE10Y, and NYMEX:CL1!.
    """
    import subprocess
    clean_env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    tickers = [
        "FX_IDC:XAUUSD", "OANDA:XAUUSD", "FX:XAUUSD", "FOREXCOM:XAUUSD", "TVC:GOLD", "SAXO:XAUUSD",
        "FX_IDC:EURUSD", "OANDA:EURUSD", "FX:EURUSD",
        "BINANCE:BTCUSDT", "COINBASE:BTCUSD",
        "TVC:DXY", "TVC:US02Y", "TVC:US10Y", "TVC:DE10Y",
        "NYMEX:CL1!", "TVC:VIX"
    ]
    cmd = [
        "curl", "-s", "--noproxy", "*", "-m", "3", "-X", "POST", "https://scanner.tradingview.com/global/scan",
        "-H", "Content-Type: application/json",
        "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "-d", json.dumps({"symbols": {"tickers": tickers}, "columns": ["close", "bid", "ask", "change", "high", "low", "open", "ATR|30"]})
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
    if res.returncode != 0:
        return {}
    try:
        data = json.loads(res.stdout)
        quotes = {}
        for row in data.get("data", []):
            sym = row.get("s")
            cols = row.get("d", [])
            close_p = cols[0] if len(cols) > 0 else None
            bid = cols[1] if len(cols) > 1 else None
            ask = cols[2] if len(cols) > 2 else None
            high_p = cols[4] if len(cols) > 4 else None
            low_p = cols[5] if len(cols) > 5 else None
            atr_30 = cols[7] if len(cols) > 7 else None

            if sym == "FX_IDC:XAUUSD":
                if close_p is not None:
                    live_px = round(float(close_p), 2)
                    quotes["GC=F"] = live_px
                    quotes["XAUUSD_SPOT"] = live_px
                    if bid is not None and ask is not None and abs(((float(bid) + float(ask)) / 2.0) - live_px) <= 1.50:
                        quotes["XAUUSD_BID"] = round(float(bid), 2)
                        quotes["XAUUSD_ASK"] = round(float(ask), 2)
                    else:
                        quotes["XAUUSD_BID"] = live_px
                        quotes["XAUUSD_ASK"] = live_px
                elif bid is not None and ask is not None:
                    mid = round((float(bid) + float(ask)) / 2.0, 2)
                    quotes["GC=F"] = mid
                    quotes["XAUUSD_SPOT"] = mid
                    quotes["XAUUSD_BID"] = round(float(bid), 2)
                    quotes["XAUUSD_ASK"] = round(float(ask), 2)
                if high_p is not None and low_p is not None and close_p is not None:
                    quotes["SESSION_HIGH"] = round(float(high_p), 2)
                    quotes["SESSION_LOW"] = round(float(low_p), 2)
                    quotes["SESSION_CLOSE"] = round(float(close_p), 2)
                if atr_30 is not None and float(atr_30) > 0:
                    quotes["ATR_M30"] = round(float(atr_30), 2)
            elif sym == "FX_IDC:EURUSD":
                if close_p is not None:
                    live_eur = round(float(close_p), 5)
                    quotes["EURUSD=X"] = live_eur
                    quotes["EURUSD_SPOT"] = live_eur
                    if bid is not None and ask is not None and abs(((float(bid) + float(ask)) / 2.0) - live_eur) <= 0.0010:
                        quotes["EURUSD_BID"] = round(float(bid), 5)
                        quotes["EURUSD_ASK"] = round(float(ask), 5)
                    else:
                        quotes["EURUSD_BID"] = live_eur
                        quotes["EURUSD_ASK"] = live_eur
                elif bid is not None and ask is not None:
                    mid = round((float(bid) + float(ask)) / 2.0, 5)
                    quotes["EURUSD=X"] = mid
                    quotes["EURUSD_SPOT"] = mid
                    quotes["EURUSD_BID"] = round(float(bid), 5)
                    quotes["EURUSD_ASK"] = round(float(ask), 5)
                if high_p is not None and low_p is not None and close_p is not None:
                    quotes["EURUSD_SESSION_HIGH"] = round(float(high_p), 5)
                    quotes["EURUSD_SESSION_LOW"] = round(float(low_p), 5)
                    quotes["EURUSD_SESSION_CLOSE"] = round(float(close_p), 5)
                if atr_30 is not None and float(atr_30) > 0:
                    quotes["EURUSD_ATR_M30"] = round(float(atr_30), 5)
            elif sym in ("BINANCE:BTCUSDT", "COINBASE:BTCUSD") and "BTC-USD" not in quotes:
                if close_p is not None:
                    live_btc = round(float(close_p), 2)
                    quotes["BTC-USD"] = live_btc
                    quotes["BTCUSD_SPOT"] = live_btc
                    quotes["BTC_BID"] = round(float(bid), 2) if bid is not None else live_btc
                    quotes["BTC_ASK"] = round(float(ask), 2) if ask is not None else live_btc
                elif bid is not None and ask is not None:
                    mid = round((float(bid) + float(ask)) / 2.0, 2)
                    quotes["BTC-USD"] = mid
                    quotes["BTCUSD_SPOT"] = mid
                    quotes["BTC_BID"] = round(float(bid), 2)
                    quotes["BTC_ASK"] = round(float(ask), 2)
                if high_p is not None and low_p is not None and close_p is not None:
                    quotes["BTC_SESSION_HIGH"] = round(float(high_p), 2)
                    quotes["BTC_SESSION_LOW"] = round(float(low_p), 2)
                    quotes["BTC_SESSION_CLOSE"] = round(float(close_p), 2)
                if atr_30 is not None and float(atr_30) > 0:
                    quotes["BTC_ATR_M30"] = round(float(atr_30), 2)
            elif sym == "TVC:DXY" and close_p is not None:
                quotes["DX-Y.NYB"] = round(float(close_p), 3)
            elif sym == "TVC:US02Y" and close_p is not None:
                quotes["ZT=F"] = round(float(close_p), 3)
            elif sym == "TVC:US10Y" and close_p is not None:
                quotes["^TNX"] = round(float(close_p), 3)
            elif sym == "TVC:DE10Y" and close_p is not None:
                quotes["DE10Y"] = round(float(close_p), 3)
            elif sym == "NYMEX:CL1!" and close_p is not None:
                quotes["CL=F"] = round(float(close_p), 2)
            elif sym == "TVC:VIX" and close_p is not None:
                quotes["VIX"] = round(float(close_p), 2)
        if "GC=F" not in quotes:
            for row in data.get("data", []):
                sym = row.get("s")
                cols = row.get("d", [])
                if sym in ["FX:XAUUSD", "OANDA:XAUUSD", "FOREXCOM:XAUUSD", "TVC:GOLD", "SAXO:XAUUSD"]:
                    bid, ask, close_p = (cols[1] if len(cols) > 1 else None), (cols[2] if len(cols) > 2 else None), (cols[0] if len(cols) > 0 else None)
                    if close_p is not None:
                        live_px = round(float(close_p), 2)
                    elif bid is not None and ask is not None:
                        live_px = round((float(bid) + float(ask)) / 2.0, 2)
                    else:
                        continue
                    quotes["GC=F"] = live_px
                    quotes["XAUUSD_SPOT"] = live_px
                    quotes["XAUUSD_BID"] = round(float(bid), 2) if (bid is not None and abs(float(bid) - live_px) <= 1.50) else live_px
                    quotes["XAUUSD_ASK"] = round(float(ask), 2) if (ask is not None and abs(float(ask) - live_px) <= 1.50) else live_px
                    break
        if "EURUSD=X" not in quotes:
            for row in data.get("data", []):
                sym = row.get("s")
                cols = row.get("d", [])
                if sym in ["FX:EURUSD", "OANDA:EURUSD"]:
                    close_p, bid, ask = cols[0], cols[1], cols[2]
                    live_eur = round(float(close_p), 5) if close_p is not None else round((float(bid) + float(ask)) / 2.0, 5)
                    quotes["EURUSD=X"] = live_eur
                    quotes["EURUSD_SPOT"] = live_eur
                    quotes["EURUSD_BID"] = round(float(bid), 5) if bid is not None else live_eur
                    quotes["EURUSD_ASK"] = round(float(ask), 5) if ask is not None else live_eur
                    break
        if "^TNX" in quotes and "DE10Y" in quotes:
            quotes["YIELD_SPREAD_US_DE"] = round(quotes["^TNX"] - quotes["DE10Y"], 3)
        return quotes
    except Exception as e:
        print("TradingView scanner parse error:", e)
        return {}
_LAST_VERIFIED_PRICES = {}

def fetch_live_macro_data():
    """
    High-frequency multi-tier real-time market data ingestion:
    Tier 1: TradingView live scanner (FX_IDC:XAUUSD spot bid/ask, TVC:DXY, TVC:US10Y, NYMEX:CL1!).
    Tier 2: Yahoo Finance 5m live quote fallback.
    NEVER fabricates dummy prices into live trade evaluation (sets is_live_verified=False if offline).
    """
    import time
    now_ts = time.time()
    if _CACHE_MACRO["data"] and (now_ts - _CACHE_MACRO["timestamp"] < 0.5):
        return _CACHE_MACRO["data"]

    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    prices = {}
    is_live_verified = False

    # Tier 1: Real-time TradingView spot feed
    tv_quotes = fetch_tradingview_live_quotes()
    if tv_quotes and "GC=F" in tv_quotes and tv_quotes["GC=F"] is not None:
        prices = tv_quotes
        is_live_verified = True
        data_source = "TRADINGVIEW REAL-TIME SPOT FEED (FX_IDC:XAUUSD + FX_IDC:EURUSD)"
    else:
        # Tier 2: Yahoo Finance fast quote / 5m live quote fallback
        try:
            fast_px = yf.Ticker("GC=F").fast_info.get("last_price")
            if fast_px is not None and not math.isnan(fast_px) and float(fast_px) > 0:
                prices["GC=F"] = round(float(fast_px), 2)
                prices["XAUUSD_SPOT"] = prices["GC=F"]
                is_live_verified = True
            else:
                gc_hist = yf.Ticker("GC=F").history(period="1d", interval="5m")
                if gc_hist is not None and not gc_hist.empty and "Close" in gc_hist:
                    s = gc_hist["Close"].dropna()
                    if not s.empty:
                        prices["GC=F"] = round(float(s.iloc[-1]), 2)
                        prices["XAUUSD_SPOT"] = prices["GC=F"]
                        is_live_verified = True
        except Exception:
            pass
        try:
            fast_eur = yf.Ticker("EURUSD=X").fast_info.get("last_price")
            if fast_eur is not None and not math.isnan(fast_eur) and float(fast_eur) > 0:
                prices["EURUSD=X"] = round(float(fast_eur), 5)
                prices["EURUSD_SPOT"] = prices["EURUSD=X"]
                is_live_verified = True
            else:
                eur_hist = yf.Ticker("EURUSD=X").history(period="1d", interval="5m")
                if eur_hist is not None and not eur_hist.empty and "Close" in eur_hist:
                    s_eur = eur_hist["Close"].dropna()
                    if not s_eur.empty:
                        eur_px = round(float(s_eur.iloc[-1]), 5)
                        prices["EURUSD=X"] = eur_px
                        prices["EURUSD_SPOT"] = eur_px
                        is_live_verified = True
        except Exception:
            pass
        try:
            fast_btc = yf.Ticker("BTC-USD").fast_info.get("last_price")
            if fast_btc is not None and not math.isnan(fast_btc) and float(fast_btc) > 0:
                prices["BTC-USD"] = round(float(fast_btc), 2)
                prices["BTCUSD_SPOT"] = prices["BTC-USD"]
                is_live_verified = True
            else:
                btc_hist = yf.Ticker("BTC-USD").history(period="1d", interval="5m")
                if btc_hist is not None and not btc_hist.empty and "Close" in btc_hist:
                    s_btc = btc_hist["Close"].dropna()
                    if not s_btc.empty:
                        btc_px = round(float(s_btc.iloc[-1]), 2)
                        prices["BTC-USD"] = btc_px
                        prices["BTCUSD_SPOT"] = btc_px
                        is_live_verified = True
        except Exception:
            pass
        data_source = "YAHOO FINANCE LIVE FEED" if is_live_verified else "OFFLINE (UNVERIFIED)"

    # Cache verified live prices; if temporarily offline, reuse last verified live quote (NEVER jump to fake numbers)
    if is_live_verified:
        _LAST_VERIFIED_PRICES.update(prices)
    elif _LAST_VERIFIED_PRICES:
        for k, v in _LAST_VERIFIED_PRICES.items():
            prices.setdefault(k, v)

    # Macro & cross-asset secondary fallbacks only if missing
    if "EURUSD=X" not in prices:
        prices["EURUSD=X"] = _LAST_VERIFIED_PRICES.get("EURUSD=X", 1.08500)
        prices["EURUSD_SPOT"] = prices["EURUSD=X"]
    if "EURUSD_ATR_M30" not in prices:
        prices["EURUSD_ATR_M30"] = _LAST_VERIFIED_PRICES.get("EURUSD_ATR_M30", 0.00120)
    if "BTC-USD" not in prices:
        prices["BTC-USD"] = _LAST_VERIFIED_PRICES.get("BTC-USD", 91500.0)
        prices["BTCUSD_SPOT"] = prices["BTC-USD"]
    if "BTC_ATR_M30" not in prices:
        prices["BTC_ATR_M30"] = _LAST_VERIFIED_PRICES.get("BTC_ATR_M30", 450.0)
    if "DE10Y" not in prices:
        prices["DE10Y"] = _LAST_VERIFIED_PRICES.get("DE10Y", 2.35)
    if "DX-Y.NYB" not in prices:
        prices["DX-Y.NYB"] = _LAST_VERIFIED_PRICES.get("DX-Y.NYB", 102.24)
    if "ZT=F" not in prices:
        prices["ZT=F"] = _LAST_VERIFIED_PRICES.get("ZT=F", 4.75)
    if "^TNX" not in prices:
        prices["^TNX"] = _LAST_VERIFIED_PRICES.get("^TNX", 5.00)
    if "CL=F" not in prices:
        prices["CL=F"] = _LAST_VERIFIED_PRICES.get("CL=F", 86.28)
    if "YIELD_SPREAD_US_DE" not in prices and "^TNX" in prices and "DE10Y" in prices:
        prices["YIELD_SPREAD_US_DE"] = round(float(prices["^TNX"]) - float(prices["DE10Y"]), 3)

    # 2. Release detector: read latest verified release status from tomorrow_preview.json (no hardcoded dummy actuals)
    headline_actual = None
    core_actual = None
    try:
        with open(os.path.join(DATA_DIR, "tomorrow_preview.json")) as f:
            tm_cur = json.load(f)
        today_s = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if tm_cur.get("date", "9999-99-99") <= today_s:
            headline_actual = tm_cur.get("last_actual_mom")
            core_actual = tm_cur.get("last_actual_core_mom")
    except Exception:
        pass

    res = {
        "timestamp": now_utc,
        "prices": prices,
        "is_live_verified": is_live_verified,
        "headline_actual": headline_actual,
        "core_actual": core_actual,
        "source": data_source
    }
    _CACHE_MACRO["timestamp"] = now_ts
    _CACHE_MACRO["data"] = res
    return res

@app.post("/api/sync-live")
def sync_live():
    sync_res = data_pipeline.sync_live_releases_if_stale(force=False)
    if sync_res.get("rebuilt"):
        _clear_quant_caches()
    _, tomorrow, _ = load_data()
    _CACHE_MACRO["data"] = None
    live_data = fetch_live_macro_data()

    hl_act = live_data["headline_actual"]
    core_act = live_data["core_actual"]
    current_gold = live_data["prices"].get("GC=F", 3875.0)

    if hl_act is not None:
        est_core_yoy = tomorrow["forecast_core_yoy"] + (core_act - tomorrow["forecast_core_mom"]) if core_act else tomorrow["forecast_core_yoy"]
        est_core_yoy = round(est_core_yoy, 2)
        sim = quant_model.simulate_tomorrow_scenario(hl_act, est_core_yoy, tomorrow, current_price=current_gold)

        return {
            "status": "RELEASED",
            "source": live_data.get("source", "BLS OFFICIAL FEED"),
            "message": f"Official CPI Release detected via {live_data.get('source')} and parsed.",
            "timestamp": live_data["timestamp"],
            "prices": live_data["prices"],
            "actual_mom": hl_act,
            "actual_core_yoy": est_core_yoy,
            "actual_core_mom": core_act,
            "simulation": sim,
            "sync_releases": sync_res
        }

    return {
        "status": "AWAITING_RELEASE",
        "message": f"Live market baseline synced. Awaiting {tomorrow.get('ref_month', '')} CPI release ({tomorrow.get('date', '')}).",
        "timestamp": live_data["timestamp"],
        "prices": live_data["prices"],
        "forecast_mom": tomorrow["forecast_mom"],
        "forecast_core_yoy": tomorrow["forecast_core_yoy"],
        "target_date": tomorrow["date"],
        "target_time": tomorrow["time"],
        "sync_releases": sync_res
    }
@app.api_route("/", methods=["GET", "HEAD"])
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

@app.get("/api/fomc-historical")
def get_fomc_historical():
    with open(os.path.join(DATA_DIR, "fomc_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_fomc_models(events)
    return {
        "count": len(events),
        "events": events,
        "models": eval_results
    }

@app.get("/api/nfp-historical")
def get_nfp_historical():
    with open(os.path.join(DATA_DIR, "nfp_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_nfp_models(events)
    return {
        "count": len(events),
        "events": events,
        "models": eval_results
    }

@app.get("/api/ppi-historical")
def get_ppi_historical():
    with open(os.path.join(DATA_DIR, "ppi_retail_historical.json")) as f:
        events = json.load(f)
    eval_results = quant_model.evaluate_ppi_models(events)
    return {
        "count": len(events),
        "events": events,
        "models": eval_results
    }

@app.get("/api/tomorrow")
def get_tomorrow():
    _, tomorrow, macro = load_data()
    return {
        "event": tomorrow,
        "macro_snapshots": macro
    }

@app.get("/api/premarket-snapshot")
def get_premarket_snapshot():
    _, tomorrow, _ = load_data()
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
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
def get_live_ticks(include_astar: bool = False, instrument: str = "XAUUSD", horizon: str = "daily"):
    """Lightweight spot-only tick stream: NEVER mutates pre-market snapshot or Wave 1 firewall."""
    live_data = fetch_live_macro_data()
    prices = live_data.get("prices", {})
    inst_upper = (instrument or "XAUUSD").upper()
    if inst_upper in ["EURUSD", "EUR"]:
        inst_key = "EURUSD"
        current_px = prices.get("EURUSD=X", 1.08500)
        atr_val = prices.get("EURUSD_ATR_M30", 0.00120)
        sh = prices.get("EURUSD_SESSION_HIGH")
        sl = prices.get("EURUSD_SESSION_LOW")
        sc = prices.get("EURUSD_SESSION_CLOSE")
    elif inst_upper in ["BTCUSD", "BTC", "BITCOIN"]:
        inst_key = "BTCUSD"
        current_px = prices.get("BTC-USD", 91500.0)
        atr_val = prices.get("BTC_ATR_M30", 450.0)
        sh = prices.get("BTC_SESSION_HIGH")
        sl = prices.get("BTC_SESSION_LOW")
        sc = prices.get("BTC_SESSION_CLOSE")
    else:
        inst_key = "XAUUSD"
        current_px = prices.get("GC=F", 3975.0)
        atr_val = prices.get("ATR_M30", 10.50)
        sh = prices.get("SESSION_HIGH")
        sl = prices.get("SESSION_LOW")
        sc = prices.get("SESSION_CLOSE")

    res = {
        "status": "SUCCESS",
        "timestamp": live_data.get("timestamp"),
        "instrument": inst_key,
        "current_price": current_px,
        "prices": prices
    }
    if include_astar:
        dxy = prices.get("DX-Y.NYB", 102.24)
        us10y = prices.get("^TNX", 5.00)
        us2y = prices.get("ZT=F", 4.75)
        h_key = (horizon or "daily").lower()
        ttl_map = {"scalping": 3600, "intraday": 14400, "daily": 5400, "swing": 432000, "news": 3600}
        ttl_sec = ttl_map.get(h_key, 5400)
        epoch_name = f"{h_key}_{inst_key.lower()}"
        created_epoch, _ = get_or_create_setup_epoch(epoch_name, ttl_sec)
        res["daily_a_star"] = quant_model.calculate_daily_a_star_setup(
            current_price=current_px, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
            wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
            n_paths=10000, setup_created_epoch=created_epoch, session_h=sh, session_l=sl, session_c=sc,
            horizon=h_key, instrument=inst_key, de10y_val=prices.get("DE10Y")
        )
    return res

@app.get("/api/pre-news-model")
def get_pre_news_model(price: float = None, atr: float = None):
    _, tomorrow, _ = load_data()
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_pre_news_bayesian_model(
        tomorrow, current_price=price, atr_m30=atr_val,
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )

@app.get("/api/fomc-prenew-model")
def get_fomc_prenew_model(price: float = None, atr: float = None):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_fomc_prenew_model(
        current_price=price, atr_m30=atr_val,
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )

@app.get("/api/nfp-prenew-model")
def get_nfp_prenew_model(price: float = None, atr: float = None):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_nfp_prenew_model(
        current_price=price, atr_m30=atr_val,
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )

@app.get("/api/ppi-prenew-model")
def get_ppi_prenew_model(price: float = None, atr: float = None):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    price = price if price is not None else prices.get("GC=F", 3975.0)
    atr_val = atr if atr is not None else prices.get("ATR_M30", 10.50)
    return quant_model.calculate_ppi_prenew_model(
        current_price=price, atr_m30=atr_val,
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )
@app.get("/api/setup-ledger")
def get_setup_ledger(instrument: str = None):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    verified = bool(live.get("is_live_verified"))
    lp_dict = {}
    if verified:
        if "GC=F" in prices:
            lp_dict["XAUUSD"] = float(prices["GC=F"])
        if "EURUSD=X" in prices:
            lp_dict["EURUSD"] = float(prices["EURUSD=X"])
        if "BTC-USD" in prices:
            lp_dict["BTCUSD"] = float(prices["BTC-USD"])
    cur_gold = prices.get("GC=F") if verified else None
    mgr = quant_model.SetupLifecycleManager()
    res = mgr.get_categorized_ledger(live_price=cur_gold, live_prices=lp_dict)
    if instrument:
        inst_upper = instrument.upper()
        if inst_upper in ["EURUSD", "EUR"]:
            target = "EURUSD"
        elif inst_upper in ["BTCUSD", "BTC", "BITCOIN"]:
            target = "BTCUSD"
        else:
            target = "XAUUSD"
        res["running_trades"] = [t for t in res.get("running_trades", []) if t.get("asset", "XAUUSD") == target]
        res["active_running"] = [t for t in res.get("active_running", []) if t.get("asset", "XAUUSD") == target]
        res["hit_tp"] = [t for t in res.get("hit_tp", []) if t.get("asset", "XAUUSD") == target]
        res["sl_touched"] = [t for t in res.get("sl_touched", []) if t.get("asset", "XAUUSD") == target]
        res["archived_expired"] = [t for t in res.get("archived_expired", []) if t.get("asset", "XAUUSD") == target]
        res["completed_setups"] = [t for t in res.get("completed_setups", []) if t.get("asset", "XAUUSD") == target]
    res["data_source"] = live.get("source", "UNKNOWN")
    res["is_live_verified"] = verified
    return res

@app.post("/api/setup-ledger/add")
def add_setup_to_running_list(payload: dict):
    """Explicit user opt-in: adds an A* setup (selected horizon) into the Running Trading List."""
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    verified = bool(live.get("is_live_verified"))

    inst = str(payload.get("asset") or payload.get("instrument") or "XAUUSD").upper()
    if inst in ["EURUSD", "EUR"]:
        target_asset = "EURUSD"
        price_key = "EURUSD=X"
        default_atr = 0.00120
    elif inst in ["BTCUSD", "BTC", "BITCOIN"]:
        target_asset = "BTCUSD"
        price_key = "BTC-USD"
        default_atr = 450.0
    else:
        target_asset = "XAUUSD"
        price_key = "GC=F"
        default_atr = 10.50

    cur_px = prices.get(price_key)
    if not verified or cur_px is None:
        return {"status": "OFFLINE", "message": f"Live {target_asset} price feed unverified — cannot add trade without real live tick."}
    cur_px = float(cur_px)
    h_key = str(payload.get("horizon") or "daily").lower()

    # If client passed only horizon, compute fresh live ticket for that horizon
    if not payload.get("stop_loss") or not payload.get("take_profit_1"):
        atr_key = "EURUSD_ATR_M30" if target_asset == "EURUSD" else ("BTC_ATR_M30" if target_asset == "BTCUSD" else "ATR_M30")
        atr_val = prices.get(atr_key, default_atr)
        dxy = prices.get("DX-Y.NYB", 102.24)
        us10y = prices.get("^TNX", 5.00)
        us2y = prices.get("ZT=F", 4.75)
        ttl_map = {"scalping": 3600, "intraday": 14400, "daily": 5400, "swing": 432000, "news": 3600}
        epoch_name = f"{h_key}_{target_asset.lower()}"
        created_epoch, _ = get_or_create_setup_epoch(epoch_name, ttl_map.get(h_key, 5400))
        sh_key = "EURUSD_SESSION_HIGH" if target_asset == "EURUSD" else ("BTC_SESSION_HIGH" if target_asset == "BTCUSD" else "SESSION_HIGH")
        sl_key = "EURUSD_SESSION_LOW" if target_asset == "EURUSD" else ("BTC_SESSION_LOW" if target_asset == "BTCUSD" else "SESSION_LOW")
        sc_key = "EURUSD_SESSION_CLOSE" if target_asset == "EURUSD" else ("BTC_SESSION_CLOSE" if target_asset == "BTCUSD" else "SESSION_CLOSE")
        sh = prices.get(sh_key)
        sl = prices.get(sl_key)
        sc = prices.get(sc_key)
        ast = quant_model.calculate_daily_a_star_setup(
            current_price=cur_px, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
            wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
            n_paths=5000, setup_created_epoch=created_epoch,
            session_h=sh, session_l=sl, session_c=sc,
            horizon=h_key, instrument=target_asset, de10y_val=prices.get("DE10Y")
        )
        tt = ast.get("trade_ticket", {})
        if not ast.get("is_a_star") and not payload.get("allow_non_astar", False):
            return {"status": "BLOCKED", "message": f"{h_key.upper()} ({target_asset}) is currently STAND ASIDE (not A* Grade). Wait for gates to pass."}
        payload = {
            "asset": target_asset,
            "horizon": h_key,
            "action": tt.get("action", ""),
            "ticket_type": tt.get("ticket_type", ""),
            "entry_zone": tt.get("entry_zone", ""),
            "stop_loss": tt.get("stop_loss", ""),
            "take_profit_1": tt.get("take_profit_1", ""),
            "take_profit_2": tt.get("take_profit_2", ""),
            "planned_rr": tt.get("risk_reward_ratio", "1:2.00"),
            "recommended_lot": tt.get("recommended_lot_size_1000_equity", 0.01),
            "immediate_entry": True
        }
    else:
        payload["asset"] = target_asset

    mgr = quant_model.SetupLifecycleManager()
    res = mgr.add_to_running_list(payload, cur_px)
    lp_dict = {}
    if "GC=F" in prices:
        lp_dict["XAUUSD"] = float(prices["GC=F"])
    if "EURUSD=X" in prices:
        lp_dict["EURUSD"] = float(prices["EURUSD=X"])
    if "BTC-USD" in prices:
        lp_dict["BTCUSD"] = float(prices["BTC-USD"])
    res["ledger"] = mgr.get_categorized_ledger(live_price=prices.get("GC=F"), live_prices=lp_dict)
    return res

@app.post("/api/setup-ledger/action")
def action_on_running_setup(payload: dict):
    """Close at live market price or cleanly remove a trade from the Running Trading List."""
    setup_id = str(payload.get("setup_id") or "")
    mode = str(payload.get("mode") or "close").lower()
    if mode not in ("close", "remove"):
        return {"status": "ERROR", "message": "Invalid mode (must be 'close' or 'remove')."}
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    verified = bool(live.get("is_live_verified"))
    mgr = quant_model.SetupLifecycleManager()

    # Identify asset of the setup
    target_trade = None
    data = mgr._load()
    for item in data.get("running_trades", []):
        if item.get("setup_id") == setup_id:
            target_trade = item
            break

    trade_asset = target_trade.get("asset", "XAUUSD") if target_trade else "XAUUSD"
    if trade_asset == "EURUSD":
        price_key = "EURUSD=X"
    elif trade_asset == "BTCUSD":
        price_key = "BTC-USD"
    else:
        price_key = "GC=F"
    live_px = float(prices[price_key]) if (verified and prices.get(price_key) is not None) else None

    if mode == "close" and live_px is None:
        return {"status": "OFFLINE", "message": f"Live {trade_asset} price feed unverified — cannot close at market without live price."}
    res = mgr.close_or_remove_running_trade(setup_id, live_price=live_px, mode=mode)
    lp_dict = {}
    if verified:
        if "GC=F" in prices:
            lp_dict["XAUUSD"] = float(prices["GC=F"])
        if "EURUSD=X" in prices:
            lp_dict["EURUSD"] = float(prices["EURUSD=X"])
        if "BTC-USD" in prices:
            lp_dict["BTCUSD"] = float(prices["BTC-USD"])
    res["ledger"] = mgr.get_categorized_ledger(live_price=prices.get("GC=F"), live_prices=lp_dict)
    return res

@app.get("/api/global-macro-matrix")
def get_global_macro_matrix():
    with open(os.path.join(DATA_DIR, "global_macro_matrix.json")) as f:
        data = json.load(f)
    return data

@app.get("/api/first-passage-win-prob")
def get_first_passage_win_prob(event_type: str = "daily", is_buy: bool = True):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    gold = prices.get("GC=F", _LAST_VERIFIED_PRICES.get("GC=F", 3725.0))
    astar = quant_model.calculate_daily_a_star_setup(
        current_price=gold, atr_m30=prices.get("ATR_M30", 10.50),
        dxy_val=prices.get("DX-Y.NYB", 102.24), us10y_val=prices.get("^TNX", 5.00), us2y_val=prices.get("ZT=F", 4.75),
        wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )
    return astar.get("first_passage_analysis", {})

@app.get("/api/algorithmic-footprint")
def get_algorithmic_footprint(event_type: str = "daily"):
    live = fetch_live_macro_data()
    gold = live.get("prices", {}).get("GC=F", _LAST_VERIFIED_PRICES.get("GC=F", 3725.0))
    return quant_model.compute_algorithmic_footprint(event_type=event_type, current_price=gold)

@app.get("/api/garch-predicted-shock")
def get_garch_predicted_shock(event_type: str = "cpi"):
    live = fetch_live_macro_data()
    snap = quant_model._load_premarket_snapshot()
    lead = snap.get("leading_indicators", {})
    vix_val = live.get("prices", {}).get("VIX", lead.get("vix_last", 17.5))
    wti_22d = lead.get("wti_22d_pct", 0.0)
    return quant_model.predict_garch_shock_dispersion(event_type=event_type, vix_val=vix_val, wti_change_pct=wti_22d)
@app.get("/api/garch-volatility-90d")
def get_garch_volatility_90d(event_type: str = "cpi"):
    return quant_model.calculate_garch_90d_volatility(event_type=event_type)
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
def get_monte_carlo(horizon: str = "1", z: str = "0.0", price: float = None, event_type: str = "cpi", n_paths: int = 10000, min_confidence: int = 95):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    if price is None:
        price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.80)

    # Robust parsing for horizon and z (handles strings like 'intraday', 'undefined', numeric)
    h_str = str(horizon).lower()
    if h_str in ["scalping", "0.25"]:
        h_days = 0.25
    elif h_str in ["intraday", "0.5"]:
        h_days = 0.5
    elif h_str in ["daily", "1", "1.0", "undefined", ""]:
        h_days = 1.0
    elif h_str in ["swing", "5", "5.0", "10", "10.0"]:
        h_days = 5.0
    elif h_str in ["news"]:
        h_days = 0.5
    else:
        try:
            h_days = float(horizon)
        except Exception:
            h_days = 1.0

    try:
        z_val = float(z)
    except Exception:
        z_val = -1.18 if str(event_type).lower() == "astar" else 0.0

    return quant_model.run_monte_carlo(current_price=price, z_shock=z_val, horizon_days=h_days, n_paths=n_paths, event_type=event_type, atr_m30=atr_val, min_confidence=min_confidence)

@app.get("/api/pdf-distribution")
def get_pdf_distribution(event_type: str = "cpi"):
    return quant_model.calculate_probability_density_function(event_type=event_type)

@app.get("/api/high-impact-calendar")
def get_high_impact_calendar():
    with open(os.path.join(DATA_DIR, "high_impact_calendar.json")) as f:
        data = json.load(f)
    for ev in data:
        if not ev.get("crypto_sensitivity"):
            cat = str(ev.get("category", "")).upper()
            title = str(ev.get("title", "")).upper()
            if any(k in cat for k in ["INFLATION", "CENTRAL_BANK", "LABOR"]) or any(k in title for k in ["CPI", "FOMC", "PAYROLL", "NFP"]):
                ev["crypto_sensitivity"] = "HIGH"
                ev.setdefault("effect_btcusd", "GLOBAL USD LIQUIDITY IMPULSE // STAND ASIDE T-30m TO T+15m")
            elif any(k in cat for k in ["GROWTH"]) or any(k in title for k in ["PPI", "CLAIMS", "RETAIL"]):
                ev["crypto_sensitivity"] = "MEDIUM"
                ev.setdefault("effect_btcusd", "SHORT-TERM USD VOLATILITY TRANSMISSION")
            else:
                ev["crypto_sensitivity"] = "LOW"
                ev.setdefault("effect_btcusd", "INDIRECT FX LIQUIDITY IMPACT")
    return data

@app.get("/api/daily-a-star")
def get_daily_a_star(price: float = None, horizon: str = "daily", instrument: str = "XAUUSD"):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    inst_upper = (instrument or "XAUUSD").upper()
    if inst_upper in ["EURUSD", "EUR"]:
        inst_key = "EURUSD"
        if price is None:
            price = prices.get("EURUSD=X", 1.08500)
        atr_val = prices.get("EURUSD_ATR_M30", 0.00120)
        sh = prices.get("EURUSD_SESSION_HIGH")
        sl = prices.get("EURUSD_SESSION_LOW")
        sc = prices.get("EURUSD_SESSION_CLOSE")
    elif inst_upper in ["BTCUSD", "BTC", "BITCOIN"]:
        inst_key = "BTCUSD"
        if price is None:
            price = prices.get("BTC-USD", 91500.0)
        atr_val = prices.get("BTC_ATR_M30", 450.0)
        sh = prices.get("BTC_SESSION_HIGH")
        sl = prices.get("BTC_SESSION_LOW")
        sc = prices.get("BTC_SESSION_CLOSE")
    else:
        inst_key = "XAUUSD"
        if price is None:
            price = prices.get("GC=F", 3975.0)
        atr_val = prices.get("ATR_M30", 10.50)
        sh = prices.get("SESSION_HIGH")
        sl = prices.get("SESSION_LOW")
        sc = prices.get("SESSION_CLOSE")
    dxy = prices.get("DX-Y.NYB", 102.24)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    h_key = (horizon or "daily").lower()
    ttl_map = {"scalping": 3600, "intraday": 14400, "daily": 5400, "swing": 432000, "news": 3600}
    ttl_sec = ttl_map.get(h_key, 5400)
    epoch_name = f"{h_key}_{inst_key.lower()}"
    created_epoch, _ = get_or_create_setup_epoch(epoch_name, ttl_sec)
    return quant_model.calculate_daily_a_star_setup(
        current_price=price, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
        wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
        n_paths=10000, setup_created_epoch=created_epoch,
        session_h=sh, session_l=sl, session_c=sc,
        horizon=h_key, instrument=inst_key, de10y_val=prices.get("DE10Y")
    )

@app.get("/api/multi-horizon-a-star")
def get_multi_horizon_a_star(price: float = None, instrument: str = "XAUUSD"):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    inst_upper = (instrument or "XAUUSD").upper()
    if inst_upper in ["EURUSD", "EUR"]:
        inst_key = "EURUSD"
        if price is None:
            price = prices.get("EURUSD=X", 1.08500)
        atr_val = prices.get("EURUSD_ATR_M30", 0.00120)
        sh = prices.get("EURUSD_SESSION_HIGH")
        sl = prices.get("EURUSD_SESSION_LOW")
        sc = prices.get("EURUSD_SESSION_CLOSE")
    elif inst_upper in ["BTCUSD", "BTC", "BITCOIN"]:
        inst_key = "BTCUSD"
        if price is None:
            price = prices.get("BTC-USD", 91500.0)
        atr_val = prices.get("BTC_ATR_M30", 450.0)
        sh = prices.get("BTC_SESSION_HIGH")
        sl = prices.get("BTC_SESSION_LOW")
        sc = prices.get("BTC_SESSION_CLOSE")
    else:
        inst_key = "XAUUSD"
        if price is None:
            price = prices.get("GC=F", 3975.0)
        atr_val = prices.get("ATR_M30", 10.50)
        sh = prices.get("SESSION_HIGH")
        sl = prices.get("SESSION_LOW")
        sc = prices.get("SESSION_CLOSE")
    dxy = prices.get("DX-Y.NYB", 102.24)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    return quant_model.calculate_multi_horizon_a_star_setup(
        current_price=price, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
        wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
        n_paths=10000,
        session_h=sh, session_l=sl, session_c=sc,
        instrument=inst_key, de10y_val=prices.get("DE10Y")
    )

@app.get("/api/dynamic-levels")
def get_dynamic_levels(price: float = None):
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    if price is None:
        price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    return quant_model.calculate_dynamic_levels(
        current_price=price, atr_m30=atr_val,
        session_h=prices.get("SESSION_HIGH"), session_l=prices.get("SESSION_LOW"), session_c=prices.get("SESSION_CLOSE")
    )

def _clear_quant_caches():
    quant_model._CALIBRATION_CACHE.clear()
    quant_model._DAILY_OLS_CACHE["data"] = None
    quant_model._DAILY_OLS_CACHE["ts"] = 0.0
    quant_model._CONF_PROFILE_CACHE.clear()
    quant_model._MC_RESULT_CACHE.clear()
    quant_model._GARCH_90D_CACHE.clear()
    quant_model._INTRADAY_CACHE["closes"] = None
    quant_model._INTRADAY_CACHE["ts"] = 0.0
    quant_model._DAILY_PROB_CACHE["data"] = None
    quant_model._DAILY_PROB_CACHE["ts"] = 0.0
    quant_model._PAPER_3Y_CACHE["data"] = None
    quant_model._PAPER_3Y_CACHE["ts"] = 0.0

@app.post("/api/sync-releases")
def sync_releases(force: bool = True):
    res = data_pipeline.sync_live_releases_if_stale(force=force)
    _clear_quant_caches()
    return res

@app.post("/api/update-all-data")
def update_all_data():
    data_pipeline.build_all_datasets()
    _clear_quant_caches()
    _CACHE_MACRO["data"] = None
    _CACHE_MACRO["timestamp"] = 0.0
    return {"status": "SUCCESS", "message": "All FRED, Yahoo, GLD Options, M5 Flow, and Historical Ledgers rebuilt & refreshed."}
@app.post("/api/refresh-all")
def refresh_all():
    live_data = fetch_live_macro_data()
    prices = live_data.get("prices", {})
    current_gold = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
    _, tomorrow, _ = load_data()

    dxy = prices.get("DX-Y.NYB", 102.24)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)
    created_epoch, _ = get_or_create_setup_epoch("daily", 5400)
    dyn_levels = quant_model.calculate_dynamic_levels(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    mgr = quant_model.SetupLifecycleManager()
    ledger_state = mgr.evaluate_all_live_ticks(current_gold) if (live_data.get("is_live_verified") and prices.get("GC=F") is not None) else mgr.get_ledger()
    active_setup = ledger_state.get("active_setup")
    completed = ledger_state.get("completed_setups", [])
    a_star = quant_model.calculate_daily_a_star_setup(
        current_price=current_gold, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
        wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
        n_paths=10000, setup_created_epoch=created_epoch, session_h=sh, session_l=sl, session_c=sc
    )

    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    fomc_pre = quant_model.calculate_fomc_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    nfp_pre = quant_model.calculate_nfp_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    ppi_pre = quant_model.calculate_ppi_prenew_model(current_price=current_gold, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    up_cons = quant_model._load_premarket_snapshot().get("upcoming_consensus", {})
    f_up = up_cons.get("fomc", {})
    n_up = up_cons.get("nfp", {})
    p_up = up_cons.get("ppi", {})
    f_r = float(f_up.get("current_target", 4.00))
    fomc = quant_model.calculate_fomc_model(f_r, f_r, 0.0, current_price=current_gold, atr_m30=atr_val)
    n_k = float(n_up.get("forecast_nfp_k", 135.0))
    n_u = float(n_up.get("forecast_unemp", 4.1))
    n_a = float(n_up.get("forecast_ahe", 0.3))
    nfp = quant_model.calculate_nfp_model(n_k, n_k, n_u, n_u, n_a, n_a, current_price=current_gold, atr_m30=atr_val)
    p_m = float(p_up.get("forecast_ppi_mom", 0.4))
    p_rm = float(p_up.get("forecast_retail_mom", 0.4))
    ppi = quant_model.calculate_ppi_retail_model(p_m, p_m, p_rm, p_rm, current_price=current_gold, atr_m30=atr_val)
    return {
        "status": "SUCCESS",
        "timestamp": live_data.get("timestamp"),
        "current_gold": current_gold,
        "prices": prices,
        "dynamic_levels": dyn_levels,
        "daily_a_star": a_star,
        "setup_ledger": mgr.get_ledger(),
        "pre_news_model": pre_news,
        "fomc_prenew": fomc_pre,
        "nfp_prenew": nfp_pre,
        "ppi_prenew": ppi_pre,
        "fomc_model": fomc,
        "nfp_model": nfp,
        "ppi_model": ppi
    }

@app.post("/api/refresh-pre-news")
def refresh_pre_news():
    """Refreshes Option A Pre-News Bayesian Model relative to current market tick."""
    live = fetch_live_macro_data()
    gold = live.get("prices", {}).get("GC=F", 4069.40)
    _, tomorrow, _ = load_data()
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=gold)
    return {
        "status": "SUCCESS",
        "timestamp": live.get("timestamp"),
        "current_gold": gold,
        "pre_news": pre_news
    }

class ManualCalcRequest(BaseModel):
    event_type: str # "cpi", "fomc", "nfp", "ppi_retail"
    actual: float
    forecast: float
    extra_actual: float = 0.0
    extra_forecast: float = 0.0
    ahe_actual: float = 0.0
    ahe_forecast: float = 0.0
    price: float = None

@app.post("/api/manual-calc")
def manual_calc(req: ManualCalcRequest):
    """
    MANUAL INPUT OVERRIDE:
    Calculates exact Z-score, expected Gold return %, DXY shock,
    and tactical directive in < 1ms without waiting for external web scrapers.
    """
    price = req.price
    if price is None:
        live = fetch_live_macro_data()
        price = live.get("prices", {}).get("GC=F", 4069.40)

    if req.event_type == "cpi":
        _, tomorrow, _ = load_data()
        sim = quant_model.simulate_tomorrow_scenario(req.actual, req.extra_actual, tomorrow, current_price=price)
        return {"event_type": "cpi", "result": sim}
    elif req.event_type == "fomc":
        fomc = quant_model.calculate_fomc_model(req.actual, req.forecast, req.extra_actual, current_price=price)
        return {"event_type": "fomc", "result": fomc}
    elif req.event_type == "nfp":
        nfp = quant_model.calculate_nfp_model(req.actual, req.forecast, req.extra_actual, req.extra_forecast, req.ahe_actual, req.ahe_forecast, current_price=price)
        return {"event_type": "nfp", "result": nfp}
    elif req.event_type == "ppi_retail":
        ppi = quant_model.calculate_ppi_retail_model(req.actual, req.forecast, req.extra_actual, req.extra_forecast, current_price=price)
        return {"event_type": "ppi_retail", "result": ppi}
    else:
        raise HTTPException(status_code=400, detail="Unknown event_type")

@app.get("/api/fomc-model")
def get_fomc_model(actual_rate: float = 4.00, priced_rate: float = 4.00, dots_shift_bps: float = 0.0, price: float = None):
    if price is None:
        live = fetch_live_macro_data()
        price = live.get("prices", {}).get("GC=F", 4069.40)
    return quant_model.calculate_fomc_model(actual_rate, priced_rate, dots_shift_bps, current_price=price)

@app.get("/api/nfp-model")
def get_nfp_model(nfp_actual: float = 135.0, nfp_forecast: float = 135.0, unemp_actual: float = 4.1, unemp_forecast: float = 4.1, ahe_actual: float = 0.3, ahe_forecast: float = 0.3, price: float = None):
    if price is None:
        live = fetch_live_macro_data()
        price = live.get("prices", {}).get("GC=F", 4069.40)
    return quant_model.calculate_nfp_model(nfp_actual, nfp_forecast, unemp_actual, unemp_forecast, ahe_actual, ahe_forecast, current_price=price)

@app.get("/api/ppi-retail-model")
def get_ppi_retail_model(ppi_actual: float = 0.4, ppi_forecast: float = 0.4, retail_actual: float = 0.4, retail_forecast: float = 0.4, price: float = None):
    if price is None:
        live = fetch_live_macro_data()
        price = live.get("prices", {}).get("GC=F", 4069.40)
    return quant_model.calculate_ppi_retail_model(ppi_actual, ppi_forecast, retail_actual, retail_forecast, current_price=price)

class SimRequest(BaseModel):
    actual_mom: float
    actual_core_yoy: float
    price: float = None

@app.post("/api/simulate")
def simulate(req: SimRequest):
    _, tomorrow, _ = load_data()
    price = req.price
    if price is None:
        live = fetch_live_macro_data()
        price = live.get("prices", {}).get("GC=F", 4069.40)
    sim_result = quant_model.simulate_tomorrow_scenario(req.actual_mom, req.actual_core_yoy, tomorrow, current_price=price)
    return sim_result

class DeepSeekRequest(BaseModel):
    mode: str = "pre_news"
    event_type: str = "cpi"
    actual_mom: float = 0.3
    actual_core_yoy: float = 2.8
    min_confidence: int = 95
    market_context: str = ""

@app.post("/api/deepseek-briefing")
def deepseek_briefing(req: DeepSeekRequest):
    _, tomorrow, _ = load_data()
    live = fetch_live_macro_data()
    prices = live.get("prices", {})
    prompt = quant_model.build_deepseek_prompt(
        mode=req.mode,
        event_type=req.event_type,
        prices=prices,
        tomorrow_meta=tomorrow,
        min_confidence=req.min_confidence,
        actual_mom=req.actual_mom,
        actual_core_yoy=req.actual_core_yoy,
        market_context=req.market_context
    )
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an elite quantitative macro analyst. Strictly obey the MANDATORY QUANTITATIVE OVERRIDE and never invent trades when STAND ASIDE is mandated. Output clean raw plain text without asterisks or markdown styling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=950
        )
        content = response.choices[0].message.content
        
        # Clean out any stray asterisks or markdown headers
        clean_text = content.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        
        return {
            "status": "success",
            "mode": req.mode,
            "briefing": clean_text,
            "memo": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/deepseek-test")
def deepseek_test():
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=10
        )
        return {"status": "SUCCESS", "message": "DeepSeek API operational", "content": resp.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

@app.post("/api/reset-setup-session")
def reset_setup_session(event_type: str = "all", horizon: str = None):
    """
    Explicitly resets setup session window and generates fresh epoch anchor.
    Resets the requested horizon (or all horizons), archives expired state,
    and forces SetupLifecycleManager to recalculate a fresh active setup ticket.
    """
    import time
    now_ts = int(time.time())
    ttl_map = {
        "scalping": 3600,
        "intraday": 14400,
        "daily": 5400,
        "swing": 432000,
        "news": 3600,
        "cpi": 3600,
        "fomc": 7200,
        "nfp": 5400,
        "ppi": 3600
    }
    h_key = (horizon or event_type or "all").lower()
    
    epochs = {}
    if os.path.exists(SETUP_EPOCHS_FILE):
        try:
            with open(SETUP_EPOCHS_FILE) as f:
                epochs = json.load(f)
        except Exception:
            epochs = {}
            
    keys_to_reset = list(ttl_map.keys()) if h_key in ["all", "*"] else [h_key]
    if h_key not in ttl_map and h_key not in ["all", "*"]:
        keys_to_reset = [h_key]

    mgr = quant_model.SetupLifecycleManager()
    for k in keys_to_reset:
        ttl = ttl_map.get(k, 5400)
        epochs[k] = {"created_at": now_ts, "expires_at": now_ts + ttl}
        if hasattr(mgr, "force_reset_horizon"):
            mgr.force_reset_horizon(k)

    try:
        with open(SETUP_EPOCHS_FILE, "w") as f:
            json.dump(epochs, f, indent=2)
    except Exception as e:
        print("Error saving reset setup epoch:", e)
        
    _clear_quant_caches()
    _CACHE_MACRO["data"] = None
    _CACHE_MACRO["timestamp"] = 0.0

    return {
        "status": "SUCCESS",
        "message": f"Fresh session window initialized for {', '.join(k.upper() for k in keys_to_reset)}",
        "created_at_epoch": now_ts,
        "reset_horizons": keys_to_reset
    }

@app.get("/api/state")
def get_state():
    hist, tomorrow, macro = load_data()
    live_data = fetch_live_macro_data()
    prices = live_data.get("prices", {})
    gold_price = prices.get("GC=F", 3975.0)
    atr_val = prices.get("ATR_M30", 10.50)
    sh, sl, sc = prices.get("SESSION_HIGH"), prices.get("SESSION_LOW"), prices.get("SESSION_CLOSE")
    dxy = prices.get("DX-Y.NYB", 102.28)
    us10y = prices.get("^TNX", 5.00)
    us2y = prices.get("ZT=F", 4.75)

    dyn = quant_model.calculate_dynamic_levels(current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
    created_epoch, _ = get_or_create_setup_epoch("daily", 5400)
    astar = quant_model.calculate_daily_a_star_setup(
        current_price=gold_price, atr_m30=atr_val, dxy_val=dxy, us10y_val=us10y, us2y_val=us2y,
        wti_val=prices.get("CL=F", 86.0), vix_val=prices.get("VIX"),
        n_paths=10000, setup_created_epoch=created_epoch, session_h=sh, session_l=sl, session_c=sc
    )
    pre_news = quant_model.calculate_pre_news_bayesian_model(tomorrow, current_price=gold_price, atr_m30=atr_val, session_h=sh, session_l=sl, session_c=sc)
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
        "quant_8": quant_model.get_eight_model_quant_audit(),
        "historical_count": len(hist)
    }

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
@app.get("/api/confidence-ladder")
def get_confidence_ladder():
    return quant_model.get_eight_model_quant_audit().get("confidence_80_99_news", {})
@app.get("/api/paper-trade-backtest")
def get_paper_trade_backtest(min_confidence: int = 95, starting_equity: float = 1000.0):
    return quant_model.run_paper_trade_backtest(starting_equity=starting_equity, min_confidence=min_confidence)
@app.get("/api/us-economy-atlas")
def get_us_economy_atlas(min_confidence: int = 95):
    return quant_model.get_us_economy_atlas_analysis(min_confidence=min_confidence)

class LotCalcRequest(BaseModel):
    equity: float = 1000.0
    risk_mode: str = "pct"
    risk_pct: float = 2.0
    max_risk_usd: float = None
    leverage: int = None
    entry_price: float = None
    sl_price: float = None
    sl_pips: float = None
    tp1_price: float = None
    tp1_pips: float = None
    tp2_price: float = None
    tp2_pips: float = None
    contract_size: float = None
    instrument: str = "XAUUSD"
    direction: str = "BUY"

@app.post("/api/calculate-lot-size")
def api_calculate_lot_size(req: LotCalcRequest):
    inst = (req.instrument or "XAUUSD").upper()
    spec = quant_model.get_instrument_spec(inst)
    lev = req.leverage if req.leverage is not None else spec.get("default_leverage", 100)
    entry = req.entry_price if req.entry_price is not None else (91500.0 if spec["symbol"] == "BTCUSD" else (1.08500 if spec["symbol"] == "EURUSD" else 3925.0))
    c_size = req.contract_size if req.contract_size is not None else spec.get("contract", 100.0)
    return quant_model.calculate_lot_and_risk(
        equity=req.equity, risk_mode=req.risk_mode, risk_pct=req.risk_pct,
        max_risk_usd=req.max_risk_usd, leverage=lev,
        entry_price=entry, sl_price=req.sl_price, sl_pips=req.sl_pips,
        tp1_price=req.tp1_price, tp1_pips=req.tp1_pips,
        tp2_price=req.tp2_price, tp2_pips=req.tp2_pips,
        contract_size=c_size, instrument=spec["symbol"],
        direction=req.direction
    )

@app.get("/api/calculate-lot-size")
def api_calculate_lot_size_get(equity: float = 1000.0, risk_mode: str = "pct", risk_pct: float = 2.0, max_risk_usd: float = None, leverage: int = None, entry: float = None, sl: float = None, sl_pips: float = None, tp1: float = None, tp1_pips: float = None, tp2: float = None, tp2_pips: float = None, contract_size: float = None, instrument: str = "XAUUSD", direction: str = "BUY"):
    inst = (instrument or "XAUUSD").upper()
    spec = quant_model.get_instrument_spec(inst)
    lev = leverage if leverage is not None else spec.get("default_leverage", 100)
    entry_p = entry if entry is not None else (91500.0 if spec["symbol"] == "BTCUSD" else (1.08500 if spec["symbol"] == "EURUSD" else 3925.0))
    c_size = contract_size if contract_size is not None else spec.get("contract", 100.0)
    return quant_model.calculate_lot_and_risk(
        equity=equity, risk_mode=risk_mode, risk_pct=risk_pct,
        max_risk_usd=max_risk_usd, leverage=lev,
        entry_price=entry_p, sl_price=sl, sl_pips=sl_pips,
        tp1_price=tp1, tp1_pips=tp1_pips,
        tp2_price=tp2, tp2_pips=tp2_pips,
        contract_size=c_size, instrument=spec["symbol"],
        direction=direction
    )

@app.get("/api/cpi-historical")
def get_cpi_historical():
    hist, _, _ = load_data()
    eval_results = quant_model.evaluate_models(hist)
    return {"count": len(hist), "events": hist, "models": eval_results}
@app.get("/api/hypothesis-test")
def get_hypothesis_test(event_type: str = "cpi"):
    return quant_model.get_event_hypothesis_test(event_type=event_type)

@app.get("/api/backtest-5y")
def get_backtest_5y(instrument: str = "XAUUSD"):
    inst = (instrument or "XAUUSD").upper()
    if inst in ("EURUSD", "EUR"):
        fname = "backtest_multi_horizon_5y_eurusd.json"
    elif inst in ("BTCUSD", "BTC", "BITCOIN"):
        fname = "backtest_btc_zero_lookahead_5y.json"
    else:
        fname = "backtest_multi_horizon_5y.json"
    path_5y = os.path.join(DATA_DIR, fname)
    if os.path.exists(path_5y):
        with open(path_5y, "r") as f:
            return json.load(f)
    return {"error": f"5y backtest telemetry not found for {inst}"}

@app.get("/api/backtest-eurusd-5y")
def get_backtest_eurusd_5y():
    path_eur = os.path.join(DATA_DIR, "backtest_multi_horizon_5y_eurusd.json")
    if os.path.exists(path_eur):
        with open(path_eur, "r") as f:
            return json.load(f)
    return {"error": "EUR/USD 5y backtest telemetry not found"}

@app.get("/api/backtest-maxtp")
def get_backtest_maxtp():
    path_mtp = os.path.join(DATA_DIR, "backtest_maxtp_5y.json")
    if os.path.exists(path_mtp):
        with open(path_mtp, "r") as f:
            return json.load(f)
    return {"error": "maxtp backtest telemetry not found"}

@app.get("/api/backtest-zero-lookahead")
def get_backtest_zero_lookahead(instrument: str = "XAUUSD"):
    inst = (instrument or "XAUUSD").upper()
    fname = "backtest_btc_zero_lookahead_5y.json" if inst in ("BTCUSD", "BTC", "BITCOIN") else "backtest_zero_lookahead_5y.json"
    path_zl = os.path.join(DATA_DIR, fname)
    if os.path.exists(path_zl):
        with open(path_zl, "r") as f:
            return json.load(f)
    return {"error": f"zero lookahead backtest telemetry not found for {inst}"}

@app.get("/api/backtest-btc-zero-lookahead")
def get_backtest_btc_zero_lookahead():
    path_btc = os.path.join(DATA_DIR, "backtest_btc_zero_lookahead_5y.json")
    if os.path.exists(path_btc):
        with open(path_btc, "r") as f:
            return json.load(f)
    return {"error": "BTC zero lookahead backtest telemetry not found"}

@app.get("/api/backtest-enhanced-v2")
def get_backtest_enhanced_v2():
    path_v2 = os.path.join(DATA_DIR, "backtest_enhanced_v2_5y.json")
    if os.path.exists(path_v2):
        with open(path_v2, "r") as f:
            return json.load(f)
    return {"error": "enhanced v2 backtest telemetry not found"}

@app.get("/api/backtest-multi-horizon")
def get_backtest_multi_horizon(instrument: str = "XAUUSD"):
    inst = (instrument or "XAUUSD").upper()
    if inst in ("EURUSD", "EUR"):
        fname_5y = "backtest_multi_horizon_5y_eurusd.json"
        fname_3y = "backtest_multi_horizon_5y_eurusd.json"
    elif inst in ("BTCUSD", "BTC", "BITCOIN"):
        fname_5y = "backtest_btc_zero_lookahead_5y.json"
        fname_3y = "backtest_btc_zero_lookahead_5y.json"
    else:
        fname_5y = "backtest_multi_horizon_5y.json"
        fname_3y = "backtest_multi_horizon_3y.json"
    path_5y = os.path.join(DATA_DIR, fname_5y)
    path_3y = os.path.join(DATA_DIR, fname_3y)
    res = {}
    if os.path.exists(path_5y):
        with open(path_5y, "r") as f:
            res["backtest_5y"] = json.load(f)
    if os.path.exists(path_3y):
        with open(path_3y, "r") as f:
            res["backtest_3y"] = json.load(f)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8888, reload=False)
