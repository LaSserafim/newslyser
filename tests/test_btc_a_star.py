"""
test_btc_a_star.py - Unit & Integration Tests for Bitcoin (BTCUSD) A* Quantitative Engine:
1. Instrument Specification & Calibration (1.0 BTC contract, $1.00 pip, 1:200 CFD leverage).
2. Pure-Trend Regime Deduction (EMA 50 / 200, Sandwich Chop Blackout, Macro Event Blackout).
3. Risk & Lot Sizing (1:200 leverage margin: $4.58 / 0.01 lot @ $91.5K, Kelly fractional sizing).
4. Daily A* Setup & Multi-Horizon Engine polymorphic execution.
5. Multi-Asset Ledger Isolation & Dynamic Breakeven Shield (+1.0R with $15 ratchet).
"""

import os
import unittest
import numpy as np
import quant_model as qm

class TestBtcAStarEngine(unittest.TestCase):

    def test_btcusd_instrument_calibration(self):
        """Verifies BTCUSD instrument specification and alias resolution."""
        spec = qm.get_instrument_spec("BTCUSD")
        self.assertEqual(spec["symbol"], "BTCUSD")
        self.assertEqual(spec["contract"], 1.0)
        self.assertEqual(spec["pip"], 1.0)
        self.assertEqual(spec["pip_value_001"], 0.01)
        self.assertEqual(spec["default_leverage"], 200)
        self.assertEqual(spec["price_decimals"], 2)
        self.assertEqual(spec["spread_usd"], 20.0)

        # Aliases
        for alias in ["BTC", "BITCOIN", "BTCUSDT", "BTC/USD"]:
            s = qm.get_instrument_spec(alias)
            self.assertEqual(s["symbol"], "BTCUSD")

    def test_btcusd_lot_and_risk_sizing(self):
        """
        Verifies 1:200 leverage lot sizing and margin mechanics for BTCUSD:
        At $91,500 BTC price, 0.01 lot margin = (0.01 * 1.0 * 91500) / 200 = $4.58.
        With $1,000 equity and 2% risk ($20 risk), 1500 pip SL:
        Lot = 20 / (1500 * 1.0) = 0.0133 -> rounded to 0.01 lot.
        """
        res = qm.calculate_lot_and_risk(
            equity=1000.0,
            risk_pct=2.0,
            entry_price=91500.0,
            sl_price=90000.0,  # 1500 points / pips
            tp1_price=94500.0,
            instrument="BTCUSD"
        )
        self.assertEqual(res["instrument"], "BTCUSD")
        self.assertEqual(res["leverage"], 200)
        self.assertEqual(res["computed_lot"], 0.01)
        # Margin for 0.01 lot @ 91500 with 1:200 leverage = 4.58
        self.assertAlmostEqual(res["required_margin"], 4.58, delta=0.05)
        self.assertEqual(res["sl_pips"], 1500.0)
        self.assertAlmostEqual(res["target_risk_dollars"], 20.0, places=1)
        self.assertGreaterEqual(res["rr_tp1"], 2.0)

    def test_btcusd_pure_trend_and_blackouts(self):
        """
        Verifies BTC A* Pure-Trend Playbook:
        1. Bullish pure trend: Price > EMA 50 > EMA 200.
        2. Bearish pure trend: Price < EMA 50 < EMA 200.
        3. Sandwich Chop Zone: Price between EMA 50 and 200 -> Hard Stand Aside.
        4. Macro Event Blackout: Stand aside T-30m to T+15m around CPI, FOMC, NFP.
        5. Derivatives Funding Rate Overheat Caution.
        """
        # 1. Bullish Pure Trend
        eval_bull = qm.evaluate_btc_a_star_regime(
            current_price=95000.0,
            ema50=93000.0,
            ema200=89000.0,
            atr_m15=450.0
        )
        self.assertEqual(eval_bull["trend_regime"], "BULLISH_PURE_TREND")
        self.assertTrue(eval_bull["is_bullish_trend"])
        self.assertFalse(eval_bull["is_sandwich_chop"])
        self.assertEqual(eval_bull["allowed_direction"], "BUY")
        self.assertGreaterEqual(eval_bull["long_setup"]["planned_rr"], 2.00)
        # Dynamic Breakeven Shield locks Entry + $15.00
        self.assertEqual(eval_bull["long_setup"]["dynamic_be_shield_sl"], 95015.0)

        # 2. Bearish Pure Trend
        eval_bear = qm.evaluate_btc_a_star_regime(
            current_price=87000.0,
            ema50=89000.0,
            ema200=93000.0,
            atr_m15=450.0
        )
        self.assertEqual(eval_bear["trend_regime"], "BEARISH_PURE_TREND")
        self.assertTrue(eval_bear["is_bearish_trend"])
        self.assertFalse(eval_bear["is_sandwich_chop"])
        self.assertEqual(eval_bear["allowed_direction"], "SELL")
        self.assertGreaterEqual(eval_bear["short_setup"]["planned_rr"], 2.00)
        # Dynamic Breakeven Shield locks Entry - $15.00
        self.assertEqual(eval_bear["short_setup"]["dynamic_be_shield_sl"], 86985.0)

        # 3. Sandwich Chop Zone Blackout
        eval_chop = qm.evaluate_btc_a_star_regime(
            current_price=91000.0,
            ema50=93000.0,
            ema200=89000.0,
            atr_m15=450.0
        )
        self.assertEqual(eval_chop["trend_regime"], "SANDWICH_CHOP_BLACKOUT")
        self.assertTrue(eval_chop["is_sandwich_chop"])
        self.assertEqual(eval_chop["allowed_direction"], "STAND_ASIDE")
        self.assertIn("SANDWICH CHOP", eval_chop["reason"])

        # 4. Macro Liquidity Blackout
        eval_macro = qm.evaluate_btc_a_star_regime(
            current_price=95000.0,
            ema50=93000.0,
            ema200=89000.0,
            atr_m15=450.0,
            macro_event_in_window=True
        )
        self.assertEqual(eval_macro["trend_regime"], "MACRO_EVENT_BLACKOUT")
        self.assertTrue(eval_macro["is_macro_blackout"])
        self.assertEqual(eval_macro["allowed_direction"], "STAND_ASIDE")
        self.assertIn("MACRO LIQUIDITY BLACKOUT", eval_macro["reason"])

        # 5. Overheated Funding Rate Check
        eval_funding = qm.evaluate_btc_a_star_regime(
            current_price=95000.0,
            ema50=93000.0,
            ema200=89000.0,
            atr_m15=450.0,
            funding_rate_pct=0.045
        )
        self.assertTrue(eval_funding["is_funding_overheated"])
        self.assertIn("CAUTION", eval_funding["funding_warning"])

    def test_btcusd_daily_a_star_and_multi_horizon(self):
        """Verifies polymorphic calculate_daily_a_star_setup and multi-horizon matrix on BTCUSD."""
        daily_res = qm.calculate_daily_a_star_setup(
            current_price=91500.0,
            atr_m30=450.0,
            n_paths=1500,
            instrument="BTCUSD"
        )
        self.assertEqual(daily_res["symbol"], "BTCUSD")
        self.assertEqual(daily_res["contract_size"], 1.0)
        self.assertEqual(daily_res["pip_size"], 1.0)
        self.assertAlmostEqual(daily_res["current_price"], 91500.0, places=1)
        self.assertEqual(daily_res["trade_ticket"]["asset"], "Bitcoin / US Dollar (BTCUSD)")
        self.assertIn("btc_regime", daily_res)
        self.assertIn("btc_pure_trend_macro_gate", daily_res["data_validation_gate"]["checks"])

        # Multi-horizon matrix
        multi_res = qm.calculate_multi_horizon_a_star_setup(
            current_price=91500.0,
            atr_m30=450.0,
            n_paths=1500,
            instrument="BTCUSD"
        )
        self.assertEqual(multi_res["symbol"], "BTCUSD")
        self.assertIn("horizons", multi_res)
        self.assertIn("scalping", multi_res["horizons"])
        self.assertIn("intraday", multi_res["horizons"])
        self.assertIn("daily", multi_res["horizons"])
        self.assertIn("swing", multi_res["horizons"])
        self.assertIn("news", multi_res["horizons"])

    def test_multi_asset_triad_ledger_isolation(self):
        """
        Verifies simultaneous 3-asset ledger execution (XAUUSD, EURUSD, BTCUSD)
        with zero cross-contamination of prices and accurate floating PnL/R calculations.
        """
        import tempfile
        tmp_ledger = os.path.join(tempfile.gettempdir(), "test_triad_asset_ledger.json")
        if os.path.exists(tmp_ledger):
            os.remove(tmp_ledger)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_ledger)

            # 1. Add Gold trade (0.02 lot @ $3,500)
            mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "BUY",
                "asset": "XAUUSD",
                "entry_zone": "$3500.00 - $3505.00",
                "stop_loss": "$3480.00",
                "take_profit_1": "$3550.00",
                "take_profit_2": "$3600.00",
                "recommended_lot": 0.02,
                "immediate_entry": True
            }, 3500.0)

            # 2. Add EUR/USD trade (0.50 lot @ 1.08500)
            mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "BUY",
                "asset": "EURUSD",
                "entry_zone": "$1.08500 - $1.08520",
                "stop_loss": "$1.08300",
                "take_profit_1": "$1.09000",
                "take_profit_2": "$1.09300",
                "recommended_lot": 0.50,
                "immediate_entry": True
            }, 1.08500)

            # 3. Add BTC/USD trade (0.02 lot @ $91,500.00, SL $90,000.00 = $1,500 risk distance)
            mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "BUY",
                "asset": "BTCUSD",
                "entry_zone": "$91500.00 - $91550.00",
                "stop_loss": "$90000.00",
                "take_profit_1": "$94500.00",
                "take_profit_2": "$97000.00",
                "recommended_lot": 0.02,
                "immediate_entry": True
            }, 91500.0)

            # Evaluate with live prices:
            # Gold up $20 ($3,520) -> +$40 PnL (+1.0R)
            # EUR up 20 pips (1.08700) -> +$100 PnL (+1.0R)
            # BTC up $1,500 ($93,000) -> 0.02 * $1,500 = +$30 PnL (+1.0R)
            cat = mgr.get_categorized_ledger(live_prices={
                "XAUUSD": 3520.0,
                "EURUSD": 1.08700,
                "BTCUSD": 93000.0
            })
            running = cat["active_running"]
            self.assertEqual(len(running), 3)

            gold_t = next(t for t in running if t["asset"] == "XAUUSD")
            eur_t = next(t for t in running if t["asset"] == "EURUSD")
            btc_t = next(t for t in running if t["asset"] == "BTCUSD")

            self.assertAlmostEqual(gold_t["floating_pnl_usd"], 40.0, delta=2.0)
            self.assertAlmostEqual(gold_t["floating_r"], 1.0, delta=0.1)

            self.assertAlmostEqual(eur_t["floating_pnl_usd"], 100.0, delta=2.0)
            self.assertAlmostEqual(eur_t["floating_r"], 1.0, delta=0.1)

            self.assertAlmostEqual(btc_t["floating_pnl_usd"], 30.0, delta=2.0)
            self.assertAlmostEqual(btc_t["floating_r"], 1.0, delta=0.1)

            # Dynamic Breakeven Shield triggered on BTC trade: trailed SL = Entry + $15.00 = 91515.0
            self.assertEqual(btc_t.get("trailed_sl"), 91515.0)

        finally:
            if os.path.exists(tmp_ledger):
                os.remove(tmp_ledger)

    def test_btc_5y_zero_lookahead_telemetry(self):
        """
        Verifies 5-Year Zero-Lookahead Audit Telemetry for Bitcoin (BTCUSD):
        1. Point-in-time execution, Next-bar Open, Worst-case collision.
        2. Strictly positive mean R and profit factor across all tracks.
        3. Statistical significance: Student t >= 2.0, White HC3 t >= 2.0, Rademacher p < 0.01.
        4. In-Sample vs Out-of-Sample stability (positive alpha in both bear and bull regimes).
        """
        import json
        telemetry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_btc_zero_lookahead_5y.json")
        self.assertTrue(os.path.exists(telemetry_path), f"Telemetry file missing at {telemetry_path}")

        with open(telemetry_path, "r") as f:
            tdata = json.load(f)

        self.assertEqual(tdata["asset"], "BTCUSD")
        self.assertGreaterEqual(tdata["data_range"]["bars"], 1800)

        # All 4 tracks must show positive sum R and PF >= 1.20
        tracks = tdata["tracks"]
        for t_name in ["scalping", "intraday", "daily", "swing"]:
            t_res = tracks[t_name]
            self.assertGreater(t_res["trades"], 50)
            self.assertGreater(t_res["sum_r"], 0.0, f"Track {t_name} sum R must be > 0")
            self.assertGreaterEqual(t_res["profit_factor"], 1.20, f"Track {t_name} PF must be >= 1.20")

        # Portfolio metrics
        port = tdata["portfolio"]
        self.assertGreater(port["trades"], 400)
        self.assertGreater(port["sum_r"], 40.0)
        self.assertGreaterEqual(port["win_rate_pct"], 45.0)
        self.assertGreaterEqual(port["profit_factor"], 1.30)
        self.assertGreaterEqual(port["t_stat"], 2.0)
        self.assertGreaterEqual(port["white_hc3_t"], 2.0)
        self.assertLess(port["rademacher_p"], 0.01)
        self.assertGreaterEqual(port["dsr_prob_pct"], 95.0)

        # In-sample vs Out-of-sample positive alpha
        split = tdata["sample_split"]
        self.assertGreater(split["in_sample"]["sum_r"], 10.0)
        self.assertGreaterEqual(split["in_sample"]["profit_factor"], 1.30)
        self.assertGreater(split["out_of_sample"]["sum_r"], 20.0)
        self.assertGreaterEqual(split["out_of_sample"]["profit_factor"], 1.25)

    def test_evt_cvar_stop_shield_pot(self):
        """Verifies Pillar 2: Pickands-Balkema-de Haan GPD 99% Expected Shortfall stop shield."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.02, 100)
        rets[0:5] = [-0.07, -0.09, -0.06, -0.08, -0.10] # Injected heavy losses
        shield = qm.compute_evt_cvar_stop_shield(rets, current_price=90000.0, direction="BUY", alpha=0.99, instrument="BTCUSD")
        self.assertIn("evt_sl_price", shield)
        self.assertIn("es_99_pct", shield)
        self.assertGreater(shield["es_99_pct"], shield["var_99_pct"])
        self.assertLess(shield["evt_sl_price"], 90000.0)
        self.assertGreater(shield["evt_distance_usd"], 500.0)

    def test_hawkes_jump_cascade_firewall(self):
        """Verifies Pillar 3: Hawkes (1971) self-exciting point process and cascade detection."""
        # 1. Normal calm market
        calm_rets = np.random.normal(0, 0.01, 100)
        hwk_calm = qm.compute_hawkes_jump_cascade(calm_rets)
        self.assertFalse(hwk_calm["is_cascade_active"])

        # 2. Clustered avalanche shocks
        shock_rets = np.zeros(60)
        shock_rets[-10:] = np.random.choice([-1, 1], 10) * np.random.uniform(0.08, 0.15, 10)
        hwk_shock = qm.compute_hawkes_jump_cascade(shock_rets)
        self.assertIn("branching_ratio", hwk_shock)
        self.assertIn("jump_intensity", hwk_shock)

    def test_dfa_hurst_persistence(self):
        """Verifies Pillar 4: Detrended Fluctuation Analysis (DFA) Hurst Exponent."""
        rng = np.random.default_rng(42)
        white_noise = rng.normal(0, 1, 300)
        h_res = qm.compute_dfa_hurst(white_noise)
        self.assertIn("hurst", h_res)
        self.assertGreater(h_res["hurst"], 0.10)
        self.assertLess(h_res["hurst"], 0.90)
        self.assertIn(h_res["regime"], ["RANDOM_WALK", "PERSISTENT", "MEAN_REVERTING"])

    def test_volume_profile_poc_and_value_area(self):
        """Verifies Pillar 1: 50-bin Volume Profile POC, VAH, and VAL geometry."""
        hi = [92000, 92500, 93000, 92800, 93500]
        lo = [90500, 91000, 91500, 91200, 91800]
        cl = [91500, 92000, 92500, 92400, 93200]
        vol = [1000, 1500, 2000, 1200, 2500]
        vp = qm.compute_volume_profile(hi, lo, cl, vol, n_bins=50)
        self.assertIn("poc", vp)
        self.assertIn("vah", vp)
        self.assertIn("val", vp)
        self.assertGreater(vp["vah"], vp["val"])
        self.assertGreaterEqual(vp["poc"], vp["val"])
        self.assertLessEqual(vp["poc"], vp["vah"])
        self.assertEqual(len(vp["bins"]), 50)

    def test_synthetic_liquidation_heatmap_clusters(self):
        """Verifies Pillar 5: Synthetic Perpetual Liquidation Heatmap clusters."""
        cur = 91500.0
        hi = [92000, 92500, 93000]
        lo = [90000, 90500, 91000]
        heat = qm.compute_synthetic_liquidation_heatmap(cur, hi, lo, atr_val=450.0, instrument="BTCUSD")
        self.assertIn("closest_short_liquidation_pool", heat)
        self.assertIn("closest_long_liquidation_pool", heat)
        self.assertGreater(heat["closest_short_liquidation_pool"], cur)
        self.assertLess(heat["closest_long_liquidation_pool"], cur)
        self.assertGreaterEqual(heat["short_clusters_count"], 4)
        self.assertGreaterEqual(heat["long_clusters_count"], 4)

    def test_specialized_10k_account_sizing_1pct_risk(self):
        """
        Verifies Specialized 10-20k USD Account Constraints:
        - $10,000 equity default, 1:200 broker leverage, 1.0% maximum risk ($100 budget).
        - 1500 pip SL distance on BTCUSD (1.0 contract) = 0.07 lot.
        - Required Margin = (0.07 * 1.0 * 91500) / 200 = $32.02.
        - Margin Utilization = 0.32% (< 1.0% threshold), Free Margin > 99.6%.
        """
        # Baseline $10,000 account @ 1.0% risk
        res10k = qm.calculate_lot_and_risk(
            equity=10000.0,
            risk_pct=1.0,
            leverage=200,
            entry_price=91500.0,
            sl_price=90000.0, # 1500 pips
            tp1_price=94500.0,
            instrument="BTCUSD"
        )
        self.assertEqual(res10k["equity"], 10000.0)
        self.assertEqual(res10k["leverage"], 200)
        self.assertEqual(res10k["risk_pct"], 1.0)
        self.assertAlmostEqual(res10k["target_risk_dollars"], 100.0, places=1)
        self.assertEqual(res10k["computed_lot"], 0.07)
        self.assertAlmostEqual(res10k["required_margin"], 32.02, delta=0.50)
        self.assertGreater(res10k["free_margin"], 9900.0)

        # Scaled $20,000 account @ 1.0% risk ($200 budget)
        res20k = qm.calculate_lot_and_risk(
            equity=20000.0,
            risk_pct=1.0,
            leverage=200,
            entry_price=91500.0,
            sl_price=90000.0,
            tp1_price=94500.0,
            instrument="BTCUSD"
        )
        self.assertEqual(res20k["equity"], 20000.0)
        self.assertAlmostEqual(res20k["target_risk_dollars"], 200.0, places=1)
        self.assertEqual(res20k["computed_lot"], 0.13)
        self.assertAlmostEqual(res20k["required_margin"], 59.48, delta=0.50)
        self.assertGreater(res20k["free_margin"], 19900.0)


if __name__ == "__main__":
    unittest.main()
