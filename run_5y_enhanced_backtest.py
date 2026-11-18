#!/usr/bin/env python3
"""
NEWSLYSER // 5-YEAR ENHANCED ZERO-LOOKAHEAD ABLATION BACKTEST (V2)
Tests the 3 daily-verifiable quantitative enhancements against the V1 Zero-Lookahead Baseline:
  1. Macro Real Yield Circuit Breaker (TIP ETF + DXY 200 SMA regime lockout on Longs)
  2. VRP Tightening Filter (VRP = ^GVZ - RV22 >= +1.50% for momentum tracks)
  3. Dynamic Sharpe/Kelly Per-Track Risk Allocation (News 1.75%, Daily 1.25%, Swing 0.75%, Intra/Scalp 0.25%)
     + Scaled 4% Risk Budget Kelly Allocation (News 4.0%, Daily 2.5%, Swing 1.5%, Intra/Scalp 0.5%)

Note: Enhancement 4 (Session Liquidity Window 07:00-10:00 & 12:30-16:30 UTC) requires intraday
timestamps and is enforced in live M5/H1 execution in quant_model.py, not fabricated on D1 bars.

Strict Zero-Lookahead Rules (identical to V1):
  - Signals calculated at Day i Close using data <= Day i.
  - Execution strictly on Day i+1 Open.
  - Worst-case same-bar collision: SL hit first (-1.0 R).
  - Overnight gap past SL fills at Open (slippage loss).
  - Parameter freeze across In-Sample (2021-2023) and Out-of-Sample (2024-2026).
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


def format_usd(val):
    return f"${val:,.2f}"


def run_enhanced_ablation():
    print("=" * 104)
    print("NEWSLYSER // 5-YEAR ENHANCED ZERO-LOOKAHEAD ABLATION BACKTEST (V1 BASELINE vs V2 ENHANCEMENTS)")
    print(f"Audit Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 104)

    # 1. Ingest 5Y real market data (GC=F, ^GVZ, DX-Y.NYB, TIP, ^TNX)
    raw = yf.download(
        ['GC=F', '^GVZ', 'DX-Y.NYB', 'TIP', '^TNX'],
        period='5y',
        progress=False,
        auto_adjust=False,
        threads=False
    )

    gc_c = raw['Close']['GC=F'].ffill().dropna()
    df = pd.DataFrame({
        'GC_Open': raw['Open']['GC=F'].reindex(gc_c.index).ffill(),
        'GC_High': raw['High']['GC=F'].reindex(gc_c.index).ffill(),
        'GC_Low': raw['Low']['GC=F'].reindex(gc_c.index).ffill(),
        'GC_Close': gc_c,
        'DXY': raw['Close']['DX-Y.NYB'].reindex(gc_c.index).ffill().bfill(),
        'GVZ': raw['Close']['^GVZ'].reindex(gc_c.index).ffill().bfill(),
        'TIP': raw['Close']['TIP'].reindex(gc_c.index).ffill().bfill(),
        'TNX': raw['Close']['^TNX'].reindex(gc_c.index).ffill().bfill(),
    }).dropna()

    r_gc = np.log(df['GC_Close'] / df['GC_Close'].shift(1)) * 100.0
    r_dx = np.log(df['DXY'] / df['DXY'].shift(1)) * 100.0
    r_tip = np.log(df['TIP'] / df['TIP'].shift(1)) * 100.0

    tr = pd.concat([
        (df['GC_High'] - df['GC_Low']),
        (df['GC_High'] - df['GC_Close'].shift(1)).abs(),
        (df['GC_Low'] - df['GC_Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    rv5 = r_gc.rolling(5).std() * math.sqrt(252.0)
    rv22 = r_gc.rolling(22).std() * math.sqrt(252.0)
    sma200 = df['GC_Close'].rolling(200).mean()
    dxy_sma200 = df['DXY'].rolling(200).mean()
    tip_sma50 = df['TIP'].rolling(50).mean()

    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"  Data Range: {start_date} -> {end_date} ({len(df)} bars) | Assets: GC=F, ^GVZ, DX-Y.NYB, TIP, ^TNX")

    def simulate_forward(entry_idx, max_hold, side, sl_d, tp_d, cost_usd):
        if entry_idx >= len(df):
            return None
        ent = float(df['GC_Open'].iloc[entry_idx])
        sl_p = ent - sl_d if side == 1 else ent + sl_d
        tp_p = ent + tp_d if side == 1 else ent - tp_d
        exit_p, exit_k = None, entry_idx
        end_idx = min(len(df), entry_idx + max_hold)
        for k in range(entry_idx, end_idx):
            bo = float(df['GC_Open'].iloc[k])
            bh = float(df['GC_High'].iloc[k])
            bl = float(df['GC_Low'].iloc[k])
            if side == 1:
                if bo <= sl_p:
                    exit_p, exit_k = bo, k; break
                if bo >= tp_p:
                    exit_p, exit_k = tp_p, k; break
                hit_sl, hit_tp = (bl <= sl_p), (bh >= tp_p)
            else:
                if bo >= sl_p:
                    exit_p, exit_k = bo, k; break
                if bo <= tp_p:
                    exit_p, exit_k = tp_p, k; break
                hit_sl, hit_tp = (bh >= sl_p), (bl <= tp_p)
            if hit_sl and hit_tp:
                exit_p, exit_k = sl_p, k; break
            elif hit_sl:
                exit_p, exit_k = sl_p, k; break
            elif hit_tp:
                exit_p, exit_k = tp_p, k; break
        if exit_p is None:
            exit_k = end_idx - 1
            exit_p = float(df['GC_Close'].iloc[exit_k])
        pnl = ((exit_p - ent) if side == 1 else (ent - exit_p)) - cost_usd
        return {
            'date': df.index[entry_idx].strftime('%Y-%m-%d'),
            'r': pnl / max(0.50, sl_d),
            'hold': (exit_k - entry_idx) + 1,
            'is_sample': 'IS' if df.index[entry_idx] <= pd.Timestamp('2023-12-31') else 'OOS',
            'side': side
        }

    # Pre-load Macro News empirical releases
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

    def build_news_trades(use_macro_gate=False):
        news_trades = []
        for row in all_releases:
            shock = row['shock']
            if abs(shock) < 0.50:
                continue
            p0, p_close, range_pct = row['gold_open'], row['gold_close'], row['gold_range_pct']
            is_bull = shock < 0
            # Optional macro gate on news? News is event-driven T+3m spike; keep unaltered to preserve macro alpha
            sl_dist = max(3.50, round(p0 * (0.35 / 100.0) * 1.5, 2))
            tp_dist = round(4.0 * sl_dist, 2)
            total_range_usd = p0 * (range_pct / 100.0)
            net_move_usd = (p_close - p0) if is_bull else (p0 - p_close)
            fav = max(0.0, net_move_usd) + 0.35 * (total_range_usd - abs(net_move_usd))
            adv = max(0.0, -net_move_usd) + 0.25 * (total_range_usd - abs(net_move_usd))
            cost_usd = 0.50
            if adv >= sl_dist and fav >= tp_dist:
                r_mult = -1.0 - (cost_usd / sl_dist)
            elif adv >= sl_dist:
                r_mult = -1.0 - (cost_usd / sl_dist)
            elif fav >= tp_dist:
                r_mult = 4.0 - (cost_usd / sl_dist)
            else:
                r_mult = max(-1.0, min(4.0, (net_move_usd - cost_usd) / sl_dist))
            news_trades.append({
                'date': row['date'],
                'r': r_mult,
                'is_sample': 'IS' if row['date'] <= '2023-12-31' else 'OOS',
                'track': 'news'
            })
        return news_trades

    def run_variant(use_real_yield_gate=False, min_vrp=0.0, drop_weak_tracks=False):
        def macro_ok(i, side, z20_dx):
            if not use_real_yield_gate:
                return True
            tip_px = float(df['TIP'].iloc[i])
            tip_ma = float(tip_sma50.iloc[i])
            r20_tip = float(np.sum(r_tip.iloc[i-19:i+1]))
            dxy_px = float(df['DXY'].iloc[i])
            dxy_ma = float(dxy_sma200.iloc[i])
            # Block Gold Longs when Real Yields are surging (TIP falling < SMA50 & -1% 20d) OR DXY bull surge
            if side == 1:
                real_yield_surge = (tip_px < tip_ma) and (r20_tip < -1.0)
                dxy_surge = (dxy_px > dxy_ma) and (z20_dx > 0.50)
                if real_yield_surge or dxy_surge:
                    return False
            return True

        tracks = {'scalping': [], 'intraday': [], 'daily': [], 'swing': [], 'news': build_news_trades()}

        # Scalping
        if not drop_weak_tracks:
            i = 200
            while i < len(df) - 1:
                w20 = r_gc.iloc[i-19:i+1].values
                wdx = r_dx.iloc[i-19:i+1].values
                sd20 = max(0.35, float(np.std(w20, ddof=1)))
                sddx = max(0.20, float(np.std(wdx, ddof=1)))
                r5 = float(np.sum(r_gc.iloc[i-4:i+1]))
                z20 = float(np.sum(w20)) / max(0.5, sd20 * math.sqrt(20.0))
                zdx = float(np.sum(wdx)) / max(0.5, sddx * math.sqrt(20.0))
                vrp = float(df['GVZ'].iloc[i]) - float(rv22.iloc[i])
                side = 1 if z20 > 0 else -1
                if (r5 * z20 > 0) and (abs(z20) >= 0.70) and (z20 * zdx < 0.20) and (vrp >= min_vrp) and (float(rv5.iloc[i]) <= 1.15 * float(rv22.iloc[i])) and ((float(df['GC_Close'].iloc[i]) >= float(sma200.iloc[i])) if side == 1 else (float(df['GC_Close'].iloc[i]) <= float(sma200.iloc[i]))) and macro_ok(i, side, zdx):
                    sl_d = max(3.00, 0.35 * float(atr14.iloc[i]))
                    res = simulate_forward(i+1, 1, side, sl_d, 2.0 * sl_d, 0.35)
                    if res:
                        res['track'] = 'scalping'
                        tracks['scalping'].append(res)
                        i += res['hold']
                        continue
                i += 1

            # Intraday
            i = 200
            while i < len(df) - 2:
                w20 = r_gc.iloc[i-19:i+1].values
                wdx = r_dx.iloc[i-19:i+1].values
                sd20 = max(0.35, float(np.std(w20, ddof=1)))
                sddx = max(0.20, float(np.std(wdx, ddof=1)))
                r5 = float(np.sum(r_gc.iloc[i-4:i+1]))
                z20 = float(np.sum(w20)) / max(0.5, sd20 * math.sqrt(20.0))
                zdx = float(np.sum(wdx)) / max(0.5, sddx * math.sqrt(20.0))
                vrp = float(df['GVZ'].iloc[i]) - float(rv22.iloc[i])
                side = 1 if z20 > 0 else -1
                if (r5 * z20 > 0) and (abs(z20) >= 0.70) and (z20 * zdx < 0.20) and (vrp >= min_vrp) and (float(rv5.iloc[i]) <= 1.15 * float(rv22.iloc[i])) and ((float(df['GC_Close'].iloc[i]) >= float(sma200.iloc[i])) if side == 1 else (float(df['GC_Close'].iloc[i]) <= float(sma200.iloc[i]))) and macro_ok(i, side, zdx):
                    sl_d = max(3.50, 0.50 * float(atr14.iloc[i]))
                    res = simulate_forward(i+1, 2, side, sl_d, 2.5 * sl_d, 0.50)
                    if res:
                        res['track'] = 'intraday'
                        tracks['intraday'].append(res)
                        i += res['hold']
                        continue
                i += 1

        # Daily
        i = 200
        while i < len(df) - 5:
            wi = r_gc.iloc[i-59:i+1].dropna().values
            wdx = r_dx.iloc[i-59:i+1].dropna().values
            if len(wi) >= 40:
                sdi = max(0.35, float(np.std(wi, ddof=1)))
                zi = float(np.sum(r_gc.iloc[i-19:i+1])) / max(0.5, sdi * math.sqrt(20.0))
                zdx = float(np.sum(wdx[-60:])) / max(0.5, max(0.2, float(np.std(wdx, ddof=1))) * math.sqrt(60.0))
                z20_dx = float(np.sum(r_dx.iloc[i-19:i+1])) / max(0.5, max(0.2, float(np.std(r_dx.iloc[i-19:i+1], ddof=1))) * math.sqrt(20.0))
                vrp = float(df['GVZ'].iloc[i]) - float(rv22.iloc[i])
                px, ma = float(df['GC_Close'].iloc[i]), float(sma200.iloc[i])
                side = 1 if zi > 0 else -1
                trend_ok = (px >= ma and zdx <= 0.8) if side == 1 else (px <= ma and zdx >= -0.8)
                if (abs(zi) >= 0.75) and (vrp >= min_vrp) and (float(rv5.iloc[i]) <= 1.15 * float(rv22.iloc[i])) and trend_ok and macro_ok(i, side, z20_dx):
                    atrv = float(atr14.iloc[i])
                    res = simulate_forward(i+1, 5, side, 1.00 * atrv, 2.50 * atrv, 0.80)
                    if res:
                        res['track'] = 'daily'
                        tracks['daily'].append(res)
                        i += res['hold']
                        continue
            i += 1

        # Swing
        i = 200
        while i < len(df) - 10:
            w60 = r_gc.iloc[i-59:i+1].dropna().values
            wdx = r_dx.iloc[i-59:i+1].dropna().values
            if len(w60) >= 55:
                sd60 = max(0.35, float(np.std(w60, ddof=1)))
                z60 = float(np.sum(w60[-60:])) / max(0.5, sd60 * math.sqrt(60.0))
                zdx = float(np.sum(wdx[-60:])) / max(0.5, max(0.2, float(np.std(wdx, ddof=1))) * math.sqrt(60.0))
                z20_dx = float(np.sum(r_dx.iloc[i-19:i+1])) / max(0.5, max(0.2, float(np.std(r_dx.iloc[i-19:i+1], ddof=1))) * math.sqrt(20.0))
                vrp = float(df['GVZ'].iloc[i]) - float(rv22.iloc[i])
                px, ma = float(df['GC_Close'].iloc[i]), float(sma200.iloc[i])
                side = 1 if z60 > 0 else -1
                if (abs(z60) >= 0.70) and (vrp >= min_vrp) and ((z60 * zdx) < 0.20) and ((px >= ma) if side == 1 else (px <= ma)) and macro_ok(i, side, z20_dx):
                    atrv = float(atr14.iloc[i])
                    res = simulate_forward(i+1, 10, side, 1.50 * atrv, 3.00 * atrv, 1.20)
                    if res:
                        res['track'] = 'swing'
                        tracks['swing'].append(res)
                        i += res['hold']
                        continue
            i += 1

        combined = []
        for tlist in tracks.values():
            combined.extend(tlist)
        combined.sort(key=lambda x: x['date'])
        return tracks, combined

    def eval_portfolio(trades, risk_map, initial_eq=1000.0):
        if not trades:
            return {"n": 0, "wr": 0, "mean_r": 0, "sum_r": 0, "pf": 0, "t": 0, "p": 1.0, "eq": initial_eq, "ret": 0, "dd": 0, "sharpe": 0, "rad_p": 1.0}
        r_arr = np.array([t['r'] for t in trades], dtype=float)
        # Weighted return stream (in % of equity per trade)
        w_arr = np.array([risk_map[t['track']] * t['r'] for t in trades], dtype=float)
        n = len(r_arr)
        wr = float(np.mean(r_arr > 0) * 100.0)
        mean_r = float(np.mean(r_arr))
        sum_r = float(np.sum(r_arr))
        std_r = float(np.std(r_arr, ddof=1)) if n > 1 else 1.0
        t_stat = mean_r / max(1e-6, std_r / math.sqrt(n))
        p_val = quant_model._student_t_pvalue(t_stat, max(1, n - 1))

        # Weighted PF & t-stat (actual dollar-risk weighted performance)
        pos_w = float(np.sum(w_arr[w_arr > 0])) if np.any(w_arr > 0) else 0.0
        neg_w = float(abs(np.sum(w_arr[w_arr < 0]))) if np.any(w_arr < 0) else 1e-4
        pf_w = pos_w / max(1e-4, neg_w)
        std_w = float(np.std(w_arr, ddof=1)) if n > 1 else 1.0
        t_w = float(np.mean(w_arr)) / max(1e-6, std_w / math.sqrt(n))
        p_w = quant_model._student_t_pvalue(t_w, max(1, n - 1))
        ann = math.sqrt(min(252.0, n / 5.0))
        sharpe_w = (float(np.mean(w_arr)) / max(1e-6, std_w)) * ann

        eq, peak, max_dd = initial_eq, initial_eq, 0.0
        for pct_move in w_arr:
            eq = max(10.0, eq * (1.0 + pct_move / 100.0))
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)

        rng = np.random.default_rng(42)
        flips = rng.choice([-1.0, 1.0], size=(50000, n))
        rad_p = float(np.mean(np.dot(flips, np.abs(w_arr)) >= float(np.sum(w_arr))))

        return {
            "n": n,
            "wr": round(wr, 2),
            "mean_r": round(mean_r, 4),
            "sum_r": round(sum_r, 2),
            "pf_weighted": round(pf_w, 2),
            "t_weighted": round(t_w, 2),
            "p_weighted": round(p_w, 6),
            "eq": round(eq, 2),
            "ret": round((eq - initial_eq) / initial_eq * 100.0, 2),
            "dd": round(max_dd, 2),
            "sharpe": round(sharpe_w, 2),
            "rad_p": round(rad_p, 6)
        }

    flat_075 = {'scalping': 0.75, 'intraday': 0.75, 'daily': 0.75, 'swing': 0.75, 'news': 0.75}
    flat_400 = {'scalping': 4.00, 'intraday': 4.00, 'daily': 4.00, 'swing': 4.00, 'news': 4.00}
    kelly_std = {'scalping': 0.25, 'intraday': 0.25, 'daily': 1.25, 'swing': 0.75, 'news': 1.75}
    kelly_4pct = {'scalping': 0.50, 'intraday': 0.50, 'daily': 2.50, 'swing': 1.50, 'news': 4.00}

    # Run Ablations
    _, t_v1 = run_variant(use_real_yield_gate=False, min_vrp=0.0, drop_weak_tracks=False)
    _, t_v2_ry = run_variant(use_real_yield_gate=True, min_vrp=0.0, drop_weak_tracks=False)
    _, t_v2_vrp = run_variant(use_real_yield_gate=True, min_vrp=1.5, drop_weak_tracks=False)
    tracks_v2_core, t_v2_core = run_variant(use_real_yield_gate=True, min_vrp=1.5, drop_weak_tracks=True)

    configs = [
        ("V1 Baseline (Flat 0.75%)", t_v1, flat_075),
        ("V2a + RealYield Gate (Flat 0.75%)", t_v2_ry, flat_075),
        ("V2b + RealYield + VRP>=1.5% (Flat)", t_v2_vrp, flat_075),
        ("V2c + Sharpe/Kelly Weights (All 5)", t_v2_vrp, kelly_std),
        ("V2d Core Only (Daily+Swing+News)", t_v2_core, kelly_std),
        ("V1 Baseline (Flat 4.0% Risk)", t_v1, flat_400),
        ("V2c Kelly-Scaled 4% Cap (All 5)", t_v2_vrp, kelly_4pct),
        ("V2d Core Kelly-Scaled 4% Cap", t_v2_core, kelly_4pct),
    ]

    print("\n" + "=" * 112)
    print(f"{'CONFIGURATION':<36} | {'N':<4} | {'WIN%':<6} | {'PF(W)':<6} | {'t(W)':<5} | {'SHARPE':<6} | {'RAD p':<8} | {'EQ ($1k)':<11} | {'MAX DD':<7}")
    print("-" * 112)
    summary_out = {}
    for label, tr_list, rmap in configs:
        m = eval_portfolio(tr_list, rmap)
        is_m = eval_portfolio([x for x in tr_list if x['is_sample'] == 'IS'], rmap)
        oos_m = eval_portfolio([x for x in tr_list if x['is_sample'] == 'OOS'], rmap)
        summary_out[label] = {"full": m, "in_sample": is_m, "out_of_sample": oos_m}
        print(f"{label:<36} | {m['n']:<4} | {m['wr']:>5.1f}% | {m['pf_weighted']:>5.2f}x | {m['t_weighted']:>5.2f} | {m['sharpe']:>6.2f} | {m['rad_p']:<8.6f} | {format_usd(m['eq']):<11} | {m['dd']:>6.2f}%")

    print("\n" + "=" * 112)
    print("IN-SAMPLE (2021-2023 FED HIKES) vs OUT-OF-SAMPLE (2024-2026 BLIND) ANTI-OVERFITTING CHECK")
    print("=" * 112)
    print(f"{'CONFIGURATION':<36} | {'IS N':<4} | {'IS PF':<6} | {'IS RET%':<9} | {'IS DD%':<7} || {'OOS N':<5} | {'OOS PF':<6} | {'OOS t':<5} | {'OOS SHP':<7} | {'OOS RET%':<10}")
    print("-" * 112)
    for label, _, _ in configs:
        is_m = summary_out[label]["in_sample"]
        oos_m = summary_out[label]["out_of_sample"]
        print(f"{label:<36} | {is_m['n']:<4} | {is_m['pf_weighted']:>5.2f}x | {is_m['ret']:>+8.1f}% | {is_m['dd']:>6.1f}% || {oos_m['n']:<5} | {oos_m['pf_weighted']:>5.2f}x | {oos_m['t_weighted']:>5.2f} | {oos_m['sharpe']:>7.2f} | {oos_m['ret']:>+9.1f}%")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtest_enhanced_v2_5y.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp_utc": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "data_range": {"start": start_date, "end": end_date, "bars": len(df)},
            "ablation_results": summary_out
        }, f, indent=2)
    print(f"\n[OK] Enhanced V2 ablation telemetry archived to {out_path}")


if __name__ == "__main__":
    run_enhanced_ablation()
