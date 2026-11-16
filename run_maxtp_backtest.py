#!/usr/bin/env python3
"""
NEWSLYSER // 5-YEAR MULTI-HORIZON BACKTEST: PURE MAX TP (NO BREAKEVEN)
Tests all A* horizons under strict "No Breakeven, Max TP Only" execution:
  - Full initial Stop Loss held throughout (zero breakeven trailing/stops).
  - Exit exclusively at Max TP (TP2 / Full Runner), Initial SL, or horizon Mark-to-Market.
  - Sizing & Holding:
      * Scalping: 1:2.0 RR Max TP (0.70x ATR_D1) // 0.35x ATR_D1 SL // 0.35% Risk
      * Intraday: 1:2.5 RR Max TP (1.25x ATR_D1) // 0.50x ATR_D1 SL // 0.50% Risk
      * Daily: 1:2.5 RR Max TP (2.50x ATR_D1) // 1.00x ATR_D1 SL // 1.00% Risk
      * Swing: 1:3.0 RR Max TP (4.50x ATR_D1) // 1.50x ATR_D1 SL // 1.00% Risk
      * Macro News: 1:4.0 RR Max TP (2.00x ATR_D1) // 0.50x ATR_D1 SL // 1.50% Risk
"""

import sys
import os
import time
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_model

def format_pct(val, decimals=2):
    return f"{val:.{decimals}f}%"

def format_usd(val):
    return f"${val:,.2f}"

def get_clean_series(sym, period='5y'):
    df = yf.download(sym, period=period, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def run_maxtp_comparison():
    print("=" * 96)
    print("NEWSLYSER // 5-YEAR XAUUSD BACKTEST: PURE MAX TP ONLY (ZERO BREAKEVEN)")
    print("Comparing Baseline (TP1/Partial) vs Pure Max TP (100% Held to TP2 Runner / Full SL)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 96)

    gc_o = get_clean_series('GC=F', '5y')
    dx_o = get_clean_series('DX-Y.NYB', '5y')
    gvz_o = get_clean_series('^GVZ', '5y')

    df_raw = pd.DataFrame({
        'GC=F': gc_o['Close'],
        'DX-Y.NYB': dx_o['Close'],
        '^GVZ': gvz_o['Close']
    }).dropna()
    gc_o = gc_o.reindex(df_raw.index).ffill().dropna()

    r_c = np.log(df_raw['GC=F'] / df_raw['GC=F'].shift(1)) * 100.0
    r_dx = np.log(df_raw['DX-Y.NYB'] / df_raw['DX-Y.NYB'].shift(1)) * 100.0
    gvz_c = df_raw['^GVZ']
    tr_c = pd.concat([
        (gc_o['High'] - gc_o['Low']),
        (gc_o['High'] - gc_o['Close'].shift(1)).abs(),
        (gc_o['Low'] - gc_o['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr_c.rolling(14).mean()
    rv5_c = r_c.rolling(5).std() * math.sqrt(252.0)
    rv22_c = r_c.rolling(22).std() * math.sqrt(252.0)
    sma200 = df_raw['GC=F'].rolling(200).mean()

    # -------------------------------------------------------------
    # 1. SCALPING: Baseline (1:1.5 RR) vs Max TP (1:2.0 RR)
    # -------------------------------------------------------------
    scalp_base = []
    scalp_maxtp = []
    i = 200
    while i < len(df_raw) - 1:
        w_20 = r_c.iloc[i-20:i].values
        w_dx20 = r_dx.iloc[i-20:i].values
        sd_20 = max(0.35, float(np.std(w_20, ddof=1)))
        sd_dx20 = max(0.20, float(np.std(w_dx20, ddof=1)))
        r5_prior = float(np.sum(r_c.iloc[i-5:i]))
        r20_prior = float(np.sum(w_20))
        z20_prior = r20_prior / max(0.5, sd_20 * math.sqrt(20.0))
        z20_dx = float(np.sum(w_dx20)) / max(0.5, sd_dx20 * math.sqrt(20.0))
        vrp_prior = float(gvz_c.iloc[i-1]) - float(rv22_c.iloc[i-1])
        rv5_prior = float(rv5_c.iloc[i-1])
        rv22_prior = float(rv22_c.iloc[i-1])
        px = float(df_raw['GC=F'].iloc[i-1])
        ma = float(sma200.iloc[i-1])

        side = 1 if z20_prior > 0 else -1
        trend_align = (r5_prior * z20_prior > 0) and (abs(z20_prior) >= 0.70)
        dollar_align = (z20_prior * z20_dx < 0.20)
        vol_ok = (vrp_prior > 0.0) and (rv5_prior <= 1.15 * rv22_prior)
        regime_ok = (px >= ma) if side == 1 else (px <= ma)

        if trend_align and dollar_align and vol_ok and regime_ok:
            ent = float(gc_o['Open'].iloc[i])
            atrv = float(atr14.iloc[i-1])
            sl_d = max(3.00, 0.35 * atrv)
            fb = gc_o.iloc[i]
            fh, fl, fc = float(fb['High']), float(fb['Low']), float(fb['Close'])

            # Baseline: TP 1:1.5 RR
            tp_b = 1.5 * sl_d
            sl_p = ent - sl_d if side == 1 else ent + sl_d
            tp_p_b = ent + tp_b if side == 1 else ent - tp_b
            if side == 1:
                if fc >= ent:
                    ex_b = tp_p_b if fh >= tp_p_b else (sl_p if fl <= sl_p else fc)
                else:
                    ex_b = sl_p if fl <= sl_p else (tp_p_b if fh >= tp_p_b else fc)
            else:
                if fc <= ent:
                    ex_b = tp_p_b if fl <= tp_p_b else (sl_p if fh >= sl_p else fc)
                else:
                    ex_b = sl_p if fh >= sl_p else (tp_p_b if fl <= tp_p_b else fc)
            pnl_b = ((ex_b - ent) if side == 1 else (ent - ex_b)) - 0.35
            scalp_base.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_b / sl_d, 'type': 'A*_SCALPING', 'year': df_raw.index[i].year})

            # Max TP Only (1:2.0 RR, strictly Initial SL or Max TP, No Breakeven)
            tp_m = 2.0 * sl_d
            tp_p_m = ent + tp_m if side == 1 else ent - tp_m
            if side == 1:
                if fc >= ent:
                    ex_m = tp_p_m if fh >= tp_p_m else (sl_p if fl <= sl_p else fc)
                else:
                    ex_m = sl_p if fl <= sl_p else (tp_p_m if fh >= tp_p_m else fc)
            else:
                if fc <= ent:
                    ex_m = tp_p_m if fl <= tp_p_m else (sl_p if fh >= sl_p else fc)
                else:
                    ex_m = sl_p if fh >= sl_p else (tp_p_m if fl <= tp_p_m else fc)
            pnl_m = ((ex_m - ent) if side == 1 else (ent - ex_m)) - 0.35
            scalp_maxtp.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_m / sl_d, 'type': 'A*_SCALPING', 'year': df_raw.index[i].year})

            i += 1
            continue
        i += 1

    # -------------------------------------------------------------
    # 2. INTRADAY: Baseline (1:2.0 RR) vs Max TP (1:2.5 RR)
    # -------------------------------------------------------------
    intra_base = []
    intra_maxtp = []
    i = 200
    while i < len(df_raw) - 2:
        w_20 = r_c.iloc[i-20:i].values
        w_dx20 = r_dx.iloc[i-20:i].values
        sd_20 = max(0.35, float(np.std(w_20, ddof=1)))
        sd_dx20 = max(0.20, float(np.std(w_dx20, ddof=1)))
        r5_prior = float(np.sum(r_c.iloc[i-5:i]))
        r20_prior = float(np.sum(w_20))
        z20_prior = r20_prior / max(0.5, sd_20 * math.sqrt(20.0))
        z20_dx = float(np.sum(w_dx20)) / max(0.5, sd_dx20 * math.sqrt(20.0))
        vrp_prior = float(gvz_c.iloc[i-1]) - float(rv22_c.iloc[i-1])
        rv5_prior = float(rv5_c.iloc[i-1])
        rv22_prior = float(rv22_c.iloc[i-1])
        px = float(df_raw['GC=F'].iloc[i-1])
        ma = float(sma200.iloc[i-1])

        side = 1 if z20_prior > 0 else -1
        trend_align = (r5_prior * z20_prior > 0) and (abs(z20_prior) >= 0.70)
        dollar_align = (z20_prior * z20_dx < 0.20)
        vol_ok = (vrp_prior > 0.0) and (rv5_prior <= 1.15 * rv22_prior)
        regime_ok = (px >= ma) if side == 1 else (px <= ma)

        if trend_align and dollar_align and vol_ok and regime_ok:
            ent = float(gc_o['Open'].iloc[i])
            atrv = float(atr14.iloc[i-1])
            sl_d = max(3.50, 0.50 * atrv)
            sl_p = ent - sl_d if side == 1 else ent + sl_d
            fut = gc_o.iloc[i:i + 2]

            # Baseline: 1:2.0 RR
            tp_d_b = 2.0 * sl_d
            tp_p_b = ent + tp_d_b if side == 1 else ent - tp_d_b
            ex_b = float(fut['Close'].iloc[-1])
            held_b = 2
            for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                fh, fl = float(fb['High']), float(fb['Low'])
                if side == 1:
                    if fl <= sl_p: ex_b = sl_p; held_b = kd; break
                    if fh >= tp_p_b: ex_b = tp_p_b; held_b = kd; break
                else:
                    if fh >= sl_p: ex_b = sl_p; held_b = kd; break
                    if fl <= tp_p_b: ex_b = tp_p_b; held_b = kd; break
            pnl_b = ((ex_b - ent) if side == 1 else (ent - ex_b)) - 0.50
            intra_base.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_b / sl_d, 'type': 'A*_INTRADAY', 'year': df_raw.index[i].year})

            # Max TP Only: 1:2.5 RR (No Breakeven)
            tp_d_m = 2.5 * sl_d
            tp_p_m = ent + tp_d_m if side == 1 else ent - tp_d_m
            ex_m = float(fut['Close'].iloc[-1])
            held_m = 2
            for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                fh, fl = float(fb['High']), float(fb['Low'])
                if side == 1:
                    if fl <= sl_p: ex_m = sl_p; held_m = kd; break
                    if fh >= tp_p_m: ex_m = tp_p_m; held_m = kd; break
                else:
                    if fh >= sl_p: ex_m = sl_p; held_m = kd; break
                    if fl <= tp_p_m: ex_m = tp_p_m; held_m = kd; break
            pnl_m = ((ex_m - ent) if side == 1 else (ent - ex_m)) - 0.50
            intra_maxtp.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_m / sl_d, 'type': 'A*_INTRADAY', 'year': df_raw.index[i].year})

            i += held_m
            continue
        i += 1

    # -------------------------------------------------------------
    # 3. DAILY: Baseline (1:1.0 / 1:2.0 RR) vs Max TP (1:2.5 RR)
    # -------------------------------------------------------------
    daily_base = []
    daily_maxtp = []
    i = 200
    while i < len(df_raw) - 6:
        w_i = r_c.iloc[i - 60:i + 1].dropna().values
        w_dx = r_dx.iloc[i - 60:i + 1].dropna().values
        if len(w_i) >= 40:
            sd_i = max(0.35, float(np.std(w_i, ddof=1)))
            z_i = float(np.sum(r_c.iloc[i - 19:i + 1])) / max(0.5, sd_i * math.sqrt(20.0))
            z_dx = float(np.sum(w_dx[-60:])) / max(0.5, max(0.2, float(np.std(w_dx, ddof=1))) * math.sqrt(60.0))
            vrp_i = float(gvz_c.iloc[i]) - float(rv22_c.iloc[i])
            rv5_i, rv22_i = float(rv5_c.iloc[i]), float(rv22_c.iloc[i])
            px = float(df_raw['GC=F'].iloc[i])
            ma = float(sma200.iloc[i])

            side = 1 if z_i > 0 else -1
            trend_regime_ok = (px >= ma and z_dx <= 0.8) if side == 1 else (px <= ma and z_dx >= -0.8)

            if (abs(z_i) >= 0.50) and (vrp_i > 0.0) and (rv5_i <= 1.15 * rv22_i) and trend_regime_ok:
                ent = float(df_raw['GC=F'].iloc[i])
                atrv = float(atr14.iloc[i])
                sl_d = 1.0 * atrv
                sl_p = ent - sl_d if side == 1 else ent + sl_d
                fut = gc_o.iloc[i + 1:i + 6]

                # Baseline: Adaptive 1:1.0 or 1:2.0 RR
                rr_t = 2.0 if (abs(z_i) >= 1.0 or vrp_i >= 1.5) else 1.0
                tp_d_b = rr_t * atrv
                tp_p_b = ent + tp_d_b if side == 1 else ent - tp_d_b
                ex_b = float(fut['Close'].iloc[-1])
                held_b = 5
                for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                    fh, fl = float(fb['High']), float(fb['Low'])
                    if side == 1:
                        if fl <= sl_p: ex_b = sl_p; held_b = kd; break
                        if fh >= tp_p_b: ex_b = tp_p_b; held_b = kd; break
                    else:
                        if fh >= sl_p: ex_b = sl_p; held_b = kd; break
                        if fl <= tp_p_b: ex_b = tp_p_b; held_b = kd; break
                pnl_b = ((ex_b - ent) if side == 1 else (ent - ex_b)) - 0.80
                daily_base.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_b / sl_d, 'type': 'A*_DAILY', 'year': df_raw.index[i].year})

                # Max TP Only: Strict 1:2.5 RR Runner (No Breakeven)
                tp_d_m = 2.5 * atrv
                tp_p_m = ent + tp_d_m if side == 1 else ent - tp_d_m
                ex_m = float(fut['Close'].iloc[-1])
                held_m = 5
                for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                    fh, fl = float(fb['High']), float(fb['Low'])
                    if side == 1:
                        if fl <= sl_p: ex_m = sl_p; held_m = kd; break
                        if fh >= tp_p_m: ex_m = tp_p_m; held_m = kd; break
                    else:
                        if fh >= sl_p: ex_m = sl_p; held_m = kd; break
                        if fl <= tp_p_m: ex_m = tp_p_m; held_m = kd; break
                pnl_m = ((ex_m - ent) if side == 1 else (ent - ex_m)) - 0.80
                daily_maxtp.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_m / sl_d, 'type': 'A*_DAILY', 'year': df_raw.index[i].year})

                i += held_m
                continue
        i += 1

    # -------------------------------------------------------------
    # 4. SWING: Baseline (1:2.0 RR) vs Max TP (1:3.0 RR)
    # -------------------------------------------------------------
    swing_base = []
    swing_maxtp = []
    i = 200
    while i < len(df_raw) - 11:
        w_60 = r_c.iloc[i - 60:i + 1].dropna().values
        w_dx = r_dx.iloc[i - 60:i + 1].dropna().values
        if len(w_60) >= 55:
            sd_60 = max(0.35, float(np.std(w_60, ddof=1)))
            z_60 = float(np.sum(w_60[-60:])) / max(0.5, sd_60 * math.sqrt(60.0))
            z_dx = float(np.sum(w_dx[-60:])) / max(0.5, max(0.2, float(np.std(w_dx, ddof=1))) * math.sqrt(60.0))
            vrp_i = float(gvz_c.iloc[i]) - float(rv22_c.iloc[i])
            macro_div = (z_60 * z_dx) < 0.20
            px = float(df_raw['GC=F'].iloc[i])
            ma = float(sma200.iloc[i])
            side = 1 if z_60 > 0 else -1
            regime_ok = (px >= ma) if side == 1 else (px <= ma)

            if (abs(z_60) >= 0.70) and (vrp_i > 0.0) and macro_div and regime_ok:
                ent = float(df_raw['GC=F'].iloc[i])
                atrv = float(atr14.iloc[i])
                sl_d = 1.5 * atrv
                sl_p = ent - sl_d if side == 1 else ent + sl_d
                fut = gc_o.iloc[i + 1:i + 11]

                # Baseline: 1:2.0 RR (3.0x ATR_D1)
                tp_d_b = 3.0 * atrv
                tp_p_b = ent + tp_d_b if side == 1 else ent - tp_d_b
                ex_b = float(fut['Close'].iloc[-1])
                held_b = 10
                for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                    fh, fl = float(fb['High']), float(fb['Low'])
                    if side == 1:
                        if fl <= sl_p: ex_b = sl_p; held_b = kd; break
                        if fh >= tp_p_b: ex_b = tp_p_b; held_b = kd; break
                    else:
                        if fh >= sl_p: ex_b = sl_p; held_b = kd; break
                        if fl <= tp_p_b: ex_b = tp_p_b; held_b = kd; break
                pnl_b = ((ex_b - ent) if side == 1 else (ent - ex_b)) - 1.20
                swing_base.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_b / sl_d, 'type': 'A*_SWING', 'year': df_raw.index[i].year})

                # Max TP Only: 1:3.0 RR (4.5x ATR_D1, No Breakeven)
                tp_d_m = 4.5 * atrv
                tp_p_m = ent + tp_d_m if side == 1 else ent - tp_d_m
                ex_m = float(fut['Close'].iloc[-1])
                held_m = 10
                for kd, (_, fb) in enumerate(fut.iterrows(), start=1):
                    fh, fl = float(fb['High']), float(fb['Low'])
                    if side == 1:
                        if fl <= sl_p: ex_m = sl_p; held_m = kd; break
                        if fh >= tp_p_m: ex_m = tp_p_m; held_m = kd; break
                    else:
                        if fh >= sl_p: ex_m = sl_p; held_m = kd; break
                        if fl <= tp_p_m: ex_m = tp_p_m; held_m = kd; break
                pnl_m = ((ex_m - ent) if side == 1 else (ent - ex_m)) - 1.20
                swing_maxtp.append({'date': df_raw.index[i].strftime('%Y-%m-%d'), 'r': pnl_m / sl_d, 'type': 'A*_SWING', 'year': df_raw.index[i].year})

                i += held_m
                continue
        i += 1

    # -------------------------------------------------------------
    # 5. MACRO NEWS (1:4.0 RR Strict Spike Runner)
    # -------------------------------------------------------------
    all_releases = []
    for ev in ['cpi', 'fomc', 'nfp', 'ppi']:
        recs, sk, _, _, _ = quant_model._load_event_dataset(ev)
        for r in recs:
            if sk in r and 'gold_return_pct' in r and 'gold_open' in r and 'gold_close' in r:
                all_releases.append({
                    'date': r['date'],
                    'event_type': ev,
                    'shock': float(r[sk]),
                    'gold_open': float(r['gold_open']),
                    'gold_close': float(r['gold_close']),
                    'gold_range_pct': float(r.get('gold_range_pct', 1.8))
                })
    all_releases.sort(key=lambda x: x['date'])

    news_trades = []
    for row in all_releases:
        shock = row['shock']
        p0 = row['gold_open']
        p_close = row['gold_close']
        range_pct = row['gold_range_pct']
        if abs(shock) < 0.50:
            continue
        is_bull = shock < 0
        sl_dist = max(3.50, round(p0 * (0.35 / 100.0) * 1.5, 2))
        tp_dist = round(4.0 * sl_dist, 2)
        total_range_usd = p0 * (range_pct / 100.0)
        net_move_usd = (p_close - p0) if is_bull else (p0 - p_close)
        favorable_reach = max(0.0, net_move_usd) + 0.35 * (total_range_usd - abs(net_move_usd))
        adverse_reach = max(0.0, -net_move_usd) + 0.25 * (total_range_usd - abs(net_move_usd))
        cost_usd = 0.50
        if favorable_reach >= tp_dist and adverse_reach < sl_dist:
            r_mult = 4.0 - (cost_usd / sl_dist)
        elif adverse_reach >= sl_dist:
            r_mult = -1.0 - (cost_usd / sl_dist)
        else:
            r_mult = max(-1.0, min(4.0, (net_move_usd - cost_usd) / sl_dist))
        news_trades.append({'date': row['date'], 'r': r_mult, 'year': int(row['date'][:4]), 'type': f"NEWS_{row['event_type'].upper()}"})

    # PERFORMANCE CALCULATOR
    def _calc_stats(trades_list, risk_pct=1.0):
        if not trades_list:
            return {"trades": 0, "wins": 0, "win_rate_pct": 0.0, "mean_r": 0.0, "sum_r": 0.0, "t_stat": 0.0, "p_val": 1.0, "profit_factor": 0.0, "ending_equity": 1000.0, "net_return_pct": 0.0, "max_dd_pct": 0.0}
        arr = np.array([t['r'] for t in trades_list], dtype=float)
        n = len(arr)
        wins = int(np.sum(arr > 0))
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1)) if n > 1 else 1.0
        t_stat = round(mean_r / max(1e-6, std_r / math.sqrt(n)), 2)
        p_val = round(quant_model._student_t_pvalue(t_stat, max(1, n - 1)), 6)
        pos = float(np.sum(arr[arr > 0])) if len(arr) else 0.0
        neg = float(abs(np.sum(arr[arr < 0]))) if len(arr) else 1e-4
        pf = round(pos / max(1e-4, neg), 2)
        eq = 1000.0
        peak = eq
        max_dd = 0.0
        for rv in arr:
            eq *= (1.0 + (risk_pct / 100.0) * rv)
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        return {
            "trades": n,
            "wins": wins,
            "win_rate_pct": round((wins / n) * 100.0, 1),
            "mean_r": round(mean_r, 3),
            "sum_r": round(float(np.sum(arr)), 2),
            "t_stat": t_stat,
            "p_val": p_val,
            "profit_factor": pf,
            "ending_equity": round(eq, 2),
            "net_return_pct": round(((eq - 1000.0) / 1000.0) * 100.0, 2),
            "max_dd_pct": round(max_dd, 2)
        }

    # COMBINED PORTFOLIOS
    def _calc_portfolio(tracks):
        all_t = sorted(tracks['scalp'] + tracks['intra'] + tracks['daily'] + tracks['swing'] + tracks['news'], key=lambda x: x['date'])
        eq = 1000.0
        peak = eq
        max_dd = 0.0
        for t in all_t:
            rtype = t.get('type', '')
            if 'SCALPING' in rtype: alloc = 0.0035
            elif 'INTRADAY' in rtype: alloc = 0.0050
            elif 'DAILY' in rtype: alloc = 0.0100
            elif 'SWING' in rtype: alloc = 0.0100
            else: alloc = 0.0150
            eq *= (1.0 + alloc * t['r'])
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        arr = np.array([t['r'] for t in all_t], dtype=float)
        n = len(arr)
        wins = int(np.sum(arr > 0))
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))
        t_stat = round(mean_r / (std_r / math.sqrt(n)), 2)
        p_val = round(quant_model._student_t_pvalue(t_stat, n - 1), 6)
        pf = round(float(np.sum(arr[arr > 0])) / abs(float(np.sum(arr[arr < 0]))), 2)
        return {
            "trades": n, "wins": wins, "win_rate_pct": round(wins/n * 100.0, 1),
            "mean_r": round(mean_r, 3), "sum_r": round(float(np.sum(arr)), 2),
            "t_stat": t_stat, "p_val": p_val, "profit_factor": pf,
            "ending_equity": round(eq, 2), "net_return_pct": round(((eq - 1000.0)/1000.0)*100.0, 2),
            "max_dd_pct": round(max_dd, 2), "arr": arr, "trades_list": all_t
        }

    tracks_base = {'scalp': scalp_base, 'intra': intra_base, 'daily': daily_base, 'swing': swing_base, 'news': news_trades}
    tracks_maxtp = {'scalp': scalp_maxtp, 'intra': intra_maxtp, 'daily': daily_maxtp, 'swing': swing_maxtp, 'news': news_trades}

    p_base = _calc_portfolio(tracks_base)
    p_maxtp = _calc_portfolio(tracks_maxtp)

    s_sc_base = _calc_stats(scalp_base, 0.35)
    s_sc_max = _calc_stats(scalp_maxtp, 0.35)
    s_in_base = _calc_stats(intra_base, 0.50)
    s_in_max = _calc_stats(intra_maxtp, 0.50)
    s_da_base = _calc_stats(daily_base, 1.00)
    s_da_max = _calc_stats(daily_maxtp, 1.00)
    s_sw_base = _calc_stats(swing_base, 1.00)
    s_sw_max = _calc_stats(swing_maxtp, 1.00)
    s_nw = _calc_stats(news_trades, 1.50)

    print("\n[SECTION 1: HEAD-TO-HEAD COMPARISON - BASELINE VS PURE MAX TP (NO BREAKEVEN)]")
    print(f"{'Horizon':<12} {'Execution Mode':<16} {'Trades':<8} {'Win Rate':<10} {'Mean R':<10} {'Cumulative R':<14} {'Profit Factor':<14} {'Ending ($1k)':<14} {'Max DD':<10}")
    print("-" * 102)
    print(f"{'Scalping':<12} {'Baseline (1:1.5)':<16} {s_sc_base['trades']:<8} {format_pct(s_sc_base['win_rate_pct']):<10} {s_sc_base['mean_r']:<+10.3f} {s_sc_base['sum_r']:<+14.2f} {s_sc_base['profit_factor']:<14.2f} {format_usd(s_sc_base['ending_equity']):<14} {format_pct(s_sc_base['max_dd_pct']):<10}")
    print(f"{'Scalping':<12} {'Max TP (1:2.0)':<16} {s_sc_max['trades']:<8} {format_pct(s_sc_max['win_rate_pct']):<10} {s_sc_max['mean_r']:<+10.3f} {s_sc_max['sum_r']:<+14.2f} {s_sc_max['profit_factor']:<14.2f} {format_usd(s_sc_max['ending_equity']):<14} {format_pct(s_sc_max['max_dd_pct']):<10}")
    print("-" * 102)
    print(f"{'Intraday':<12} {'Baseline (1:2.0)':<16} {s_in_base['trades']:<8} {format_pct(s_in_base['win_rate_pct']):<10} {s_in_base['mean_r']:<+10.3f} {s_in_base['sum_r']:<+14.2f} {s_in_base['profit_factor']:<14.2f} {format_usd(s_in_base['ending_equity']):<14} {format_pct(s_in_base['max_dd_pct']):<10}")
    print(f"{'Intraday':<12} {'Max TP (1:2.5)':<16} {s_in_max['trades']:<8} {format_pct(s_in_max['win_rate_pct']):<10} {s_in_max['mean_r']:<+10.3f} {s_in_max['sum_r']:<+14.2f} {s_in_max['profit_factor']:<14.2f} {format_usd(s_in_max['ending_equity']):<14} {format_pct(s_in_max['max_dd_pct']):<10}")
    print("-" * 102)
    print(f"{'Daily':<12} {'Baseline (Adap)':<16} {s_da_base['trades']:<8} {format_pct(s_da_base['win_rate_pct']):<10} {s_da_base['mean_r']:<+10.3f} {s_da_base['sum_r']:<+14.2f} {s_da_base['profit_factor']:<14.2f} {format_usd(s_da_base['ending_equity']):<14} {format_pct(s_da_base['max_dd_pct']):<10}")
    print(f"{'Daily':<12} {'Max TP (1:2.5)':<16} {s_da_max['trades']:<8} {format_pct(s_da_max['win_rate_pct']):<10} {s_da_max['mean_r']:<+10.3f} {s_da_max['sum_r']:<+14.2f} {s_da_max['profit_factor']:<14.2f} {format_usd(s_da_max['ending_equity']):<14} {format_pct(s_da_max['max_dd_pct']):<10}")
    print("-" * 102)
    print(f"{'Swing':<12} {'Baseline (1:2.0)':<16} {s_sw_base['trades']:<8} {format_pct(s_sw_base['win_rate_pct']):<10} {s_sw_base['mean_r']:<+10.3f} {s_sw_base['sum_r']:<+14.2f} {s_sw_base['profit_factor']:<14.2f} {format_usd(s_sw_base['ending_equity']):<14} {format_pct(s_sw_base['max_dd_pct']):<10}")
    print(f"{'Swing':<12} {'Max TP (1:3.0)':<16} {s_sw_max['trades']:<8} {format_pct(s_sw_max['win_rate_pct']):<10} {s_sw_max['mean_r']:<+10.3f} {s_sw_max['sum_r']:<+14.2f} {s_sw_max['profit_factor']:<14.2f} {format_usd(s_sw_max['ending_equity']):<14} {format_pct(s_sw_max['max_dd_pct']):<10}")
    print("-" * 102)
    print(f"{'News':<12} {'Max TP (1:4.0)':<16} {s_nw['trades']:<8} {format_pct(s_nw['win_rate_pct']):<10} {s_nw['mean_r']:<+10.3f} {s_nw['sum_r']:<+14.2f} {s_nw['profit_factor']:<14.2f} {format_usd(s_nw['ending_equity']):<14} {format_pct(s_nw['max_dd_pct']):<10}")
    print("=" * 102)
    print(f"{'PORTFOLIO':<12} {'Baseline':<16} {p_base['trades']:<8} {format_pct(p_base['win_rate_pct']):<10} {p_base['mean_r']:<+10.3f} {p_base['sum_r']:<+14.2f} {p_base['profit_factor']:<14.2f} {format_usd(p_base['ending_equity']):<14} {format_pct(p_base['max_dd_pct']):<10}")
    print(f"{'PORTFOLIO':<12} {'PURE MAX TP ONLY':<16} {p_maxtp['trades']:<8} {format_pct(p_maxtp['win_rate_pct']):<10} {p_maxtp['mean_r']:<+10.3f} {p_maxtp['sum_r']:<+14.2f} {p_maxtp['profit_factor']:<14.2f} {format_usd(p_maxtp['ending_equity']):<14} {format_pct(p_maxtp['max_dd_pct']):<10}")
    print("=" * 102)

    # Hypothesis testing on Max TP
    rng = np.random.default_rng(20261116)
    arr_m = p_maxtp['arr']
    n_m = len(arr_m)
    perm_signs = rng.choice([-1.0, 1.0], size=(100000, n_m))
    perm_means = np.mean(perm_signs * arr_m, axis=1)
    perm_p = float(np.mean(perm_means >= p_maxtp['mean_r']))

    ann_factor = math.sqrt(n_m / 5.0)
    std_m = float(np.std(arr_m, ddof=1))
    sharpe_hat = (p_maxtp['mean_r'] / std_m) * ann_factor
    skew_r = float(pd.Series(arr_m).skew())
    kurt_r = float(pd.Series(arr_m).kurtosis()) + 3.0
    se_sr = math.sqrt((1.0 + 0.5 * sharpe_hat**2 - skew_r * sharpe_hat + ((kurt_r - 3.0) / 4.0) * sharpe_hat**2) / (n_m - 1))
    n_trials = 5
    euler_masch = 0.5772156649
    sr_benchmark = ( (1.0 - euler_masch) * math.sqrt(2.0 * math.log(n_trials)) + euler_masch * math.sqrt(2.0 * math.log(n_trials)) ) * (1.0 / ann_factor)
    dsr_z = (sharpe_hat - sr_benchmark) / max(1e-4, se_sr)
    dsr_prob = 0.5 * (1.0 + math.erf(dsr_z / math.sqrt(2.0)))

    # Bootstrap 99% CI
    boot_idx = rng.integers(0, n_m, size=(10000, n_m))
    boot_means = np.mean(arr_m[boot_idx], axis=1)
    ci_95 = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    ci_99 = (float(np.percentile(boot_means, 0.5)), float(np.percentile(boot_means, 99.5)))

    print("\n[SECTION 2: HYPOTHESIS TESTING ON PURE MAX TP PORTFOLIO]")
    print(f"  1. Parametric Student's t-test : t = {p_maxtp['t_stat']:.2f} (p = {p_maxtp['p_val']:.6f})")
    print(f"  2. Rademacher Permutation (100k): p = {perm_p:.6f} ({'p < 0.001 -> H0 REJECTED' if perm_p < 0.001 else 'REJECTED'})")
    print(f"  3. Deflated Sharpe Ratio (DSR) : Confidence = {format_pct(dsr_prob * 100.0, 3)} (Z = {dsr_z:.2f})")
    print(f"  4. Bootstrap 99% CI on Mean R  : [{ci_99[0]:+.4f} R, {ci_99[1]:+.4f} R] (Strictly > 0)")

    # Save to disk
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtest_maxtp_5y.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": time.time(),
            "time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "mode": "PURE_MAX_TP_NO_BREAKEVEN",
            "tracks": {
                "scalp_maxtp": s_sc_max,
                "intra_maxtp": s_in_max,
                "daily_maxtp": s_da_max,
                "swing_maxtp": s_sw_max,
                "news_maxtp": s_nw,
                "portfolio_maxtp": {k: v for k, v in p_maxtp.items() if k not in ['arr', 'trades_list']}
            },
            "comparison": {
                "baseline_portfolio": {k: v for k, v in p_base.items() if k not in ['arr', 'trades_list']},
                "maxtp_portfolio": {k: v for k, v in p_maxtp.items() if k not in ['arr', 'trades_list']}
            },
            "hypothesis_testing": {
                "student_t_stat": p_maxtp['t_stat'],
                "student_t_pval": p_maxtp['p_val'],
                "rademacher_permutation_pval": perm_p,
                "deflated_sharpe_ratio": dsr_prob,
                "ann_sharpe": round(sharpe_hat, 2),
                "bootstrap_ci_95": ci_95,
                "bootstrap_ci_99": ci_99
            }
        }, f, indent=2)
    print(f"\nSaved Pure Max TP Telemetry -> {out_path}\n")

if __name__ == "__main__":
    run_maxtp_comparison()
