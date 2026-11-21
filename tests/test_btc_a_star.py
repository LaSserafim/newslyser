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

        # All 4 tracks must show positive sum R and PF >= 1.10
        tracks = tdata["tracks"]
        for t_name in ["scalping", "intraday", "daily", "swing"]:
            t_res = tracks[t_name]
            self.assertGreater(t_res["trades"], 50)
            self.assertGreater(t_res["sum_r"], 0.0, f"Track {t_name} sum R must be > 0")
            self.assertGreaterEqual(t_res["profit_factor"], 1.10, f"Track {t_name} PF must be >= 1.10")

        # Portfolio metrics (Pure Quantitative Momentum: 350 trades, 52% Win Rate, +50.9R, t=3.01)
        port = tdata["portfolio"]
        self.assertGreaterEqual(port["trades"], 300)
        self.assertGreaterEqual(port["sum_r"], 45.0)
        self.assertGreaterEqual(port["win_rate_pct"], 50.0)
        self.assertGreaterEqual(port["profit_factor"], 1.45)
        self.assertGreaterEqual(port["t_stat"], 2.5)
        self.assertGreaterEqual(port["white_hc3_t"], 2.5)
        self.assertLess(port["rademacher_p"], 0.01)
        self.assertGreaterEqual(port["dsr_prob_pct"], 95.0)

        # In-sample vs Out-of-sample positive alpha (Zero parameter tuning across regimes)
        split = tdata["sample_split"]
        self.assertGreater(split["in_sample"]["sum_r"], 5.0)
        self.assertGreaterEqual(split["in_sample"]["profit_factor"], 1.15)
        self.assertGreater(split["out_of_sample"]["sum_r"], 30.0)
        self.assertGreaterEqual(split["out_of_sample"]["profit_factor"], 1.60)

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
        rng = np.random.default_rng(42)
        calm_rets = rng.normal(0, 0.005, 100)
        hwk_calm = qm.compute_hawkes_jump_cascade(calm_rets)
        self.assertFalse(hwk_calm["is_cascade_active"])

        # 2. Clustered avalanche shocks
        shock_rets = np.zeros(60)
        shock_rets[-10:] = rng.choice([-1, 1], 10) * rng.uniform(0.08, 0.15, 10)
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

    def test_horizon_strategy_and_hft_accuracy_gates(self):
        """Verifies Core Institutional (Daily/Swing) Focus vs HFT Dynamic Agility (Scalping/Intraday) Accuracy Gates."""
        # 1. Daily Core Institutional
        daily = qm.calculate_daily_a_star_setup(current_price=91500.0, atr_m30=450.0, horizon="daily", instrument="BTCUSD")
        self.assertIn("horizon_strategy_notes", daily)
        d_strat = daily["horizon_strategy_notes"]
        self.assertEqual(d_strat["category"], "CORE_INSTITUTIONAL")
        self.assertTrue(d_strat["is_primary_focus"])
        self.assertEqual(d_strat["recommended_risk_pct"], 1.0)
        self.assertIn("★", d_strat["badge"])

        # 2. Swing Core Institutional
        swing = qm.calculate_daily_a_star_setup(current_price=91500.0, atr_m30=450.0, horizon="swing", instrument="BTCUSD")
        s_strat = swing["horizon_strategy_notes"]
        self.assertEqual(s_strat["category"], "CORE_INSTITUTIONAL")
        self.assertTrue(s_strat["is_primary_focus"])
        self.assertEqual(s_strat["recommended_risk_pct"], 1.0)

        # 3. Scalping HFT Dynamic Agility
        scalp = qm.calculate_daily_a_star_setup(current_price=91500.0, atr_m30=450.0, horizon="scalping", instrument="BTCUSD")
        sc_strat = scalp["horizon_strategy_notes"]
        self.assertEqual(sc_strat["category"], "HFT_DYNAMIC_AGILITY")
        self.assertFalse(sc_strat["is_primary_focus"])
        self.assertEqual(sc_strat["recommended_risk_pct"], 0.35)
        self.assertIn("⚡", sc_strat["badge"])
        self.assertIn("hft_accuracy_gates", sc_strat)
        gates = sc_strat["hft_accuracy_gates"]
        self.assertIn("vpin_ok", gates)
        self.assertIn("hawkes_ok", gates)
        self.assertIn("hurst_ok", gates)
        self.assertIn("evt_shield_price", gates)

        # 4. EVT Horizon Dynamic Scaling
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.02, 100)
        rets[0:5] = [-0.07, -0.09, -0.06, -0.08, -0.10]
        shield_d1 = qm.compute_evt_cvar_stop_shield(rets, current_price=91500.0, horizon_scale=1.00, instrument="BTCUSD")
        shield_scalp = qm.compute_evt_cvar_stop_shield(rets, current_price=91500.0, horizon_scale=0.25, instrument="BTCUSD")
        self.assertAlmostEqual(shield_scalp["evt_distance_usd"], shield_d1["evt_distance_usd"] * 0.25, delta=1.0)
        self.assertGreater(shield_scalp["evt_sl_price"], shield_d1["evt_sl_price"]) # Scalping stop is tighter to entry

    def test_prop_firm_100_pip_shield_and_telegram(self):
        """Verifies Prop Firm 100-Pip Defense Rule, Expanded Lifespan (18h/8h/60m), and Telegram Engine."""
        import telegram_bot

        # 1. Check EURUSD Scalp Prop Firm Compliance
        eur_scalp = qm.calculate_daily_a_star_setup(current_price=1.08500, atr_m30=0.0012, horizon="scalping", instrument="EURUSD")
        self.assertIn("prop_firm_defense", eur_scalp)
        p_def = eur_scalp["prop_firm_defense"]
        self.assertEqual(p_def["max_drawdown_pips_limit"], 100.0)
        self.assertLessEqual(p_def["sl_pips"], p_def["sl_pips_ceiling"])
        self.assertTrue(p_def["is_compliant"])
        self.assertGreaterEqual(p_def["consecutive_losses_absorbed"], 4)
        self.assertIn("prop_firm_100_pip_shield", eur_scalp["gate_checks"])
        self.assertEqual(eur_scalp["gate_checks"]["prop_firm_100_pip_shield"]["status"], "PASS")

        # 2. Check Gold Scalp Prop Firm Compliance
        gold_scalp = qm.calculate_daily_a_star_setup(current_price=4036.0, atr_m30=12.5, horizon="scalping", instrument="XAUUSD")
        g_def = gold_scalp["prop_firm_defense"]
        self.assertLessEqual(g_def["sl_pips_ceiling"], 300.0)

        # 3. Check Expanded TTL Lifespans
        ttl_scalp = qm.calculate_time_to_live_status("scalping")
        ttl_intraday = qm.calculate_time_to_live_status("intraday")
        ttl_daily = qm.calculate_time_to_live_status("daily")
        self.assertEqual(ttl_scalp["max_validity_seconds"], 3600)
        self.assertEqual(ttl_intraday["max_validity_seconds"], 28800)  # 8 hours
        self.assertEqual(ttl_daily["max_validity_seconds"], 64800)     # 18 hours

        # 4. Telegram Formatter
        mock_setup = {
            "setup_grade": "A* GRADE (SCALPING)",
            "is_a_star": True,
            "action": "BAYESIAN PROBABILITY LONG",
            "optimal_entry": 1.08500,
            "stop_loss": 1.08350,
            "take_profit_1": 1.08800,
            "take_profit_2": 1.09000,
            "risk_reward_ratio": 2.0,
            "safe_lot_size": 0.25,
            "target_risk_usd": 35.0,
            "required_margin": 13.55,
            "setup_created_epoch": 1795276000,
            "prop_firm_defense": {
                "sl_pips": 15.0,
                "remaining_pips_buffer": 85.0,
                "consecutive_losses_absorbed": 6,
                "is_compliant": True
            },
            "gate_checks": {
                "vpin_order_flow_gate": {"status": "PASS"},
                "hawkes_cascade_firewall": {"status": "PASS"},
                "dfa_hurst_persistence": {"status": "PASS"}
            }
        }
        card = telegram_bot.format_trade_alert(mock_setup, instrument="EURUSD", horizon="scalping")
        self.assertIn("NEWSLYSER // A* PROP SCALP ALERT", card)
        self.assertIn("RISK CAPPED AT $25.00 PER TRADE", card)
        self.assertIn("15.0 pips", card)

    def test_telegram_interactive_commands_and_sizing(self):
        """Verifies Telegram interactive command dispatcher, dynamic sizing, and HTML security escaping."""
        import telegram_bot
        from unittest.mock import patch

        # 1. Test unauthorized user block
        ok, err = telegram_bot.handle_telegram_command("/size 20000", from_id=999999999)
        self.assertFalse(ok)
        self.assertEqual(err, "Unauthorized user")

        # 2. Test user sizing persistence
        cfg = telegram_bot.get_user_config()
        self.assertIn("account_balance", cfg)
        self.assertIn("max_risk_pct", cfg)
        self.assertIn("leverage", cfg)

        # 3. Test dynamic sizing calculations with strict $25.00 maximum dollar loss
        mock_setup_eur = {
            "optimal_entry": 1.09674,
            "stop_loss": 1.09510,
            "current_price": 1.09674
        }
        # Balance = 10,000, Max Loss = $25.00
        telegram_bot.save_user_config({"account_balance": 10000.0, "max_loss_usd": 25.0, "leverage": 200})
        sizing_25 = telegram_bot.compute_custom_sizing(mock_setup_eur, instrument="EURUSD")
        self.assertEqual(sizing_25["max_loss_usd"], 25.0)
        self.assertLessEqual(sizing_25["actual_max_loss"], 25.0) # Guaranteed <= $25
        self.assertEqual(sizing_25["lot_size"], 0.15) # 16.4 pips * $10/pip = $164/lot -> $25/$164 = 0.15 lot

        # Test Gold sizing with $25.00 Max Loss ($4.26 stop distance)
        mock_setup_gold = {
            "optimal_entry": 3538.26,
            "stop_loss": 3534.00,
            "current_price": 3538.26
        }
        sizing_gold_25 = telegram_bot.compute_custom_sizing(mock_setup_gold, instrument="XAUUSD")
        self.assertLessEqual(sizing_gold_25["actual_max_loss"], 25.0)
        self.assertEqual(sizing_gold_25["lot_size"], 0.05) # $4.26 * 100 = $426/lot -> $25/$426 = 0.05 lot

        # Test scaling up to $50.00 Max Loss
        telegram_bot.save_user_config({"account_balance": 10000.0, "max_loss_usd": 50.0, "leverage": 200})
        sizing_50 = telegram_bot.compute_custom_sizing(mock_setup_eur, instrument="EURUSD")
        self.assertEqual(sizing_50["max_loss_usd"], 50.0)
        self.assertLessEqual(sizing_50["actual_max_loss"], 50.0)
        self.assertGreater(sizing_50["lot_size"], sizing_25["lot_size"])

        # Reset back to 25.0 default
        telegram_bot.save_user_config({"account_balance": 10000.0, "max_loss_usd": 25.0, "leverage": 200})

        # 4. Mock send_telegram_message to verify command dispatching without network I/O
        with patch("telegram_bot.send_telegram_message") as mock_send:
            mock_send.return_value = (True, "Delivered")

            # Test /start /help
            telegram_bot.handle_telegram_command("/help", from_id=6107718785)
            self.assertTrue(mock_send.called)
            help_msg = mock_send.call_args[0][0]
            self.assertIn("NEWSLYSER QUANTITATIVE TERMINAL BOT", help_msg)
            self.assertIn("/loss", help_msg)

            # Test /loss 25 and /loss 50
            telegram_bot.handle_telegram_command("/loss 50", from_id=6107718785)
            cfg_now = telegram_bot.get_user_config()
            self.assertEqual(cfg_now["max_loss_usd"], 50.0)

            telegram_bot.handle_telegram_command("/loss 25", from_id=6107718785)
            cfg_now = telegram_bot.get_user_config()
            self.assertEqual(cfg_now["max_loss_usd"], 25.0)

            # Test /size
            telegram_bot.handle_telegram_command("/size 15000", from_id=6107718785)
            cfg_now = telegram_bot.get_user_config()
            self.assertEqual(cfg_now["account_balance"], 15000.0)

            # Test /leverage
            telegram_bot.handle_telegram_command("/leverage 200", from_id=6107718785)
            cfg_now = telegram_bot.get_user_config()
            self.assertEqual(cfg_now["leverage"], 200)

            # Test /config
            telegram_bot.handle_telegram_command("/config", from_id=6107718785)
            conf_msg = mock_send.call_args[0][0]
            self.assertIn("CURRENT TRADING CONFIGURATION", conf_msg)
            self.assertIn("$15,000.00", conf_msg)
            self.assertIn("$25.00", conf_msg)

            # Test /all Triad Telemetry
            telegram_bot.handle_telegram_command("/all", from_id=6107718785)
            all_msg = mock_send.call_args[0][0]
            self.assertIn("TRIAD LIVE TELEMETRY", all_msg)
            self.assertIn("EUR/USD", all_msg)
            self.assertIn("GOLD (XAU/USD)", all_msg)
            self.assertIn("BITCOIN (BTC/USD)", all_msg)

            # Test Mobile Touch Reply Buttons
            touch_buttons = [
                ("📈 EUR/USD", "EUR/USD"),
                ("🥇 GOLD", "GOLD"),
                ("₿ BTC/USD", "BITCOIN"),
                ("📊 All Markets", "TRIAD LIVE TELEMETRY"),
                ("⚙️ My Sizing", "CURRENT TRADING CONFIGURATION"),
                ("🛡️ Max Loss $25", "MAXIMUM RISK PER TRADE UPDATED")
            ]
            for btn_label, expected_snippet in touch_buttons:
                ok, status = telegram_bot.handle_telegram_command(btn_label, from_id=6107718785)
                self.assertTrue(ok)
                btn_msg = mock_send.call_args[0][0]
                self.assertIn(expected_snippet, btn_msg)
                call_kwargs = mock_send.call_args[1]
                self.assertIn("reply_markup", call_kwargs)
                self.assertIn("keyboard", call_kwargs["reply_markup"])

            # Test HTML safety on unknown command
            telegram_bot.handle_telegram_command("/test <script>alert(1)</script>", from_id=6107718785)
            err_msg = mock_send.call_args[0][0]
            self.assertIn("&lt;script&gt;", err_msg)
            self.assertNotIn("<script>", err_msg)

        # Restore default 10k config with 25.0 max loss
        telegram_bot.save_user_config({"account_balance": 10000.0, "max_loss_usd": 25.0, "leverage": 200})


if __name__ == "__main__":
    unittest.main()
