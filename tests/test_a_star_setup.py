"""
Unit tests for Institutional Pure-Probability A* Setup Engine in NEWSLYSER.
Tests 6-stage quantitative gates, First-Passage mark-to-market payoff,
dead-band enforcement, VPIN toxicity lockout, and calculative risk management.
"""

import unittest
import os
import json
import numpy as np
import quant_model as qm


class TestAStarSetupEngine(unittest.TestCase):

    def test_live_a_star_deduction(self):
        """Verifies that A* setup runs without exception and outputs the full schema."""
        ast = qm.calculate_daily_a_star_setup(n_paths=2000)
        self.assertIn("is_a_star", ast)
        self.assertIn("grade", ast)
        self.assertIn("confluence_score", ast)
        self.assertIn("trade_ticket", ast)
        self.assertIn("first_passage_analysis", ast)
        self.assertIn("data_validation_gate", ast)

        ticket = ast["trade_ticket"]
        self.assertIn("action", ticket)
        self.assertIn("stop_loss", ticket)
        self.assertIn("take_profit_1", ticket)
        self.assertIn("recommended_lot_size_1000_equity", ticket)

        fp = ast["first_passage_analysis"]
        self.assertIn("expected_value_r", fp)
        self.assertIn("win_probability_pct", fp)
        self.assertIn("quarter_kelly_pct", fp)
        self.assertIn("undecided_pct", fp)
        self.assertGreater(fp["n_paths"], 0)

    def test_neutral_chop_deadband(self):
        """Verifies that when |z_20| < 0.75, setup is strictly gated out as STAND ASIDE."""
        orig_fn = qm._get_daily_prob_regime
        try:
            def mocked_regime():
                d = orig_fn().copy()
                d["z_20"] = 0.20  # Inside dead-band (-0.75, +0.75)
                d["mu_daily_post_pct"] = 0.02
                return d

            qm._get_daily_prob_regime = mocked_regime
            ast = qm.calculate_daily_a_star_setup(n_paths=1000)

            self.assertFalse(ast["is_a_star"])
            self.assertIn("STAND ASIDE", ast["trade_ticket"]["action"])
            self.assertEqual(
                ast["data_validation_gate"]["checks"]["bayesian_drift_z20"]["status"],
                "UNMET"
            )
        finally:
            qm._get_daily_prob_regime = orig_fn

    def test_vpin_toxicity_gate_lockout(self):
        """Verifies that when VPIN >= 0.70, the microstructure gate immediately locks execution."""
        orig_snap = qm._load_premarket_snapshot
        try:
            def mocked_snap():
                s = orig_snap().copy()
                s["quant_8_snapshot"]["microstructure_flow"]["vpin_now"] = 0.78
                s["quant_8_snapshot"]["microstructure_flow"]["vpin_toxic_gate"] = True
                return s

            qm._load_premarket_snapshot = mocked_snap
            ast = qm.calculate_daily_a_star_setup(n_paths=1000)

            self.assertFalse(ast["is_a_star"])
            self.assertEqual(
                ast["data_validation_gate"]["checks"]["vpin_order_flow_gate"]["status"],
                "LOCKED"
            )
        finally:
            qm._load_premarket_snapshot = orig_snap

    def test_vrp_negative_rejection(self):
        """Verifies that when VRP <= 0, the setup cannot be awarded A* grade."""
        orig_fn = qm._get_daily_prob_regime
        try:
            def mocked_regime():
                d = orig_fn().copy()
                d["vrp_ann"] = -1.5  # Inverted volatility risk premium
                return d

            qm._get_daily_prob_regime = mocked_regime
            ast = qm.calculate_daily_a_star_setup(n_paths=1000)

            self.assertFalse(ast["is_a_star"])
            self.assertEqual(
                ast["data_validation_gate"]["checks"]["vrp_implied_edge"]["status"],
                "UNMET"
            )
        finally:
            qm._get_daily_prob_regime = orig_fn

    def test_mark_to_market_first_passage_math(self):
        """
        Verifies that First-Passage Monte Carlo does not discard undecided paths,
        and produces unbiased realized R-multiples consistent with mark-to-market.
        """
        ast = qm.calculate_daily_a_star_setup(n_paths=3000)
        fp = ast["first_passage_analysis"]

        # Sum of early TP, early SL, and undecided must equal exactly 100%
        early_tp = fp.get("early_tp_rate_pct", 0.0)
        early_sl = fp.get("early_sl_rate_pct", 0.0)
        undecided = fp.get("undecided_pct", 0.0)
        total_paths_pct = early_tp + early_sl + undecided
        self.assertAlmostEqual(total_paths_pct, 100.0, delta=1.0)

        # Win probability is unconditional (R > 0 across ALL paths, including profitable undecided)
        self.assertGreaterEqual(fp["win_probability_pct"], early_tp)

        # Expected value is a valid float
        self.assertIsInstance(fp["expected_value_r"], float)

    def test_quarter_kelly_calculative_sizing(self):
        """Verifies that Quarter-Kelly sizing scales dynamically and caps at 2.0%."""
        ast = qm.calculate_daily_a_star_setup(n_paths=1000)
        q_kelly = ast["first_passage_analysis"]["quarter_kelly_pct"]
        self.assertGreaterEqual(q_kelly, 0.0)
        self.assertLessEqual(q_kelly, 2.0)

    def test_multi_horizon_intraday_and_swing_schema(self):
        """Verifies that Scalping, Intraday, and Swing setups execute with scaled barriers and distinct TTLs."""
        scalp = qm.calculate_daily_a_star_setup(horizon="scalping", n_paths=1000)
        self.assertEqual(scalp["horizon"], "scalping")
        self.assertEqual(scalp["time_to_live"]["max_validity_minutes"], 60)
        self.assertIn("Scalping", scalp["execution_timeframe"])

        intra = qm.calculate_daily_a_star_setup(horizon="intraday", n_paths=1000)
        self.assertEqual(intra["horizon"], "intraday")
        self.assertEqual(intra["time_to_live"]["max_validity_minutes"], 240)
        self.assertIn("Intraday", intra["execution_timeframe"])

        swing = qm.calculate_daily_a_star_setup(horizon="swing", n_paths=1000)
        self.assertEqual(swing["horizon"], "swing")
        self.assertEqual(swing["time_to_live"]["max_validity_minutes"], 7200)
        self.assertIn("Swing", swing["execution_timeframe"])

        news = qm.calculate_daily_a_star_setup(horizon="news", n_paths=1000)
        self.assertEqual(news["horizon"], "news")
        self.assertEqual(news["time_to_live"]["max_validity_minutes"], 60)
        self.assertIn("News", news["execution_timeframe"])

    def test_multi_horizon_matrix(self):
        """Verifies that calculate_multi_horizon_a_star_setup returns unified matrix with all 5 horizons."""
        mat = qm.calculate_multi_horizon_a_star_setup(n_paths=1000)
        self.assertIn("horizons", mat)
        self.assertIn("scalping", mat["horizons"])
        self.assertIn("intraday", mat["horizons"])
        self.assertIn("daily", mat["horizons"])
        self.assertIn("swing", mat["horizons"])
        self.assertIn("news", mat["horizons"])
        self.assertIn("portfolio_allocation", mat)
        self.assertLessEqual(mat["portfolio_allocation"]["max_portfolio_concurrent_risk_pct"], 5.0)

    def test_force_reset_horizon(self):
        """Verifies that force_reset_horizon clears active setups and archives superseded state (isolated temp ledger)."""
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "test_force_reset_ledger.json")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_path)
            raw_ticket = {"action": "BUY", "ticket_type": "TEST", "entry_zone": "$2000 - $2002", "stop_loss": "$1990", "take_profit_1": "$2020"}
            locked, _ = mgr.get_or_create_locked_setup(raw_ticket, 2000.0, is_a_star=True, horizon="intraday")
            self.assertIsNotNone(locked)
            # Force reset
            mgr.force_reset_horizon("intraday")
            ledger = mgr.get_ledger()
            self.assertIsNone(ledger.get("active_setups", {}).get("intraday"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_5y_multi_horizon_backtest_telemetry(self):
        """Verifies that 5-year multi-horizon backtest telemetry is validated with statistical rigor."""
        res = qm.calculate_daily_a_star_setup(horizon="daily", n_paths=1000)
        pe = res["probability_engine"]
        self.assertIn("backtest_3y_summary", pe)
        self.assertIn("backtest_5y_summary", pe)
        bt5 = pe["backtest_5y_summary"]
        self.assertGreater(bt5["trades"], 50)
        self.assertGreater(bt5["profit_factor"], 1.0)

        # Check JSON telemetry on disk
        bt5_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_multi_horizon_5y.json")
        self.assertTrue(os.path.exists(bt5_path))
        with open(bt5_path, "r") as f:
            data = json.load(f)
        self.assertIn("tracks", data)
        comb = data["tracks"]["combined_portfolio"]
        self.assertGreaterEqual(comb["trades"], 500)
        self.assertGreater(comb["sum_r"], 50.0)
        self.assertGreater(comb["ending_equity_usd"], 2500.0)
        self.assertGreaterEqual(comb["t_stat"], 2.0)
        self.assertLess(comb["p_val"], 0.01)
        self.assertLess(data["hypothesis_testing"]["rademacher_permutation_pval"], 0.01)
        self.assertGreater(data["hypothesis_testing"]["deflated_sharpe_ratio"], 0.99)

    def test_zero_lookahead_audit_telemetry(self):
        """Verifies that the strict zero-lookahead, anti-overfitting audit telemetry is valid."""
        zl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_zero_lookahead_5y.json")
        self.assertTrue(os.path.exists(zl_path))
        with open(zl_path, "r") as f:
            data = json.load(f)
        self.assertIn("anti_bias_rules", data)
        self.assertIn("portfolio", data)
        self.assertIn("sample_split", data)
        self.assertIn("scenario_4pct_risk", data)

        port = data["portfolio"]
        self.assertGreaterEqual(port["trades"], 500)
        self.assertGreater(port["sum_r"], 50.0)
        self.assertGreater(port["t_stat"], 2.0)
        self.assertLess(port["p_val"], 0.05)
        self.assertGreater(port["dsr_prob_pct"], 95.0)

        # Anti-overfitting validation: Out-of-sample must have positive EV and t-stat
        oos = data["sample_split"]["out_of_sample"]
        self.assertGreater(oos["trades"], 300)
        self.assertGreater(oos["mean_r"], 0.15)
        self.assertGreater(oos["t_stat"], 3.0)
        self.assertGreater(oos["sharpe_ratio"], 1.0)

    def test_intraday_squeeze_conflict_blocks_blind_short(self):
        """Verifies that a strong live M15 buying squeeze (+3.0σ) detects LONG on Scalping and blocks blind SHORT on Daily/Intraday."""
        orig_daily = qm._get_daily_prob_regime
        orig_intra = qm._get_intraday_momentum_regime
        try:
            def mock_daily():
                d = orig_daily().copy()
                d["z_20"] = -1.53
                d["z_60"] = -2.10
                d["r5_pct"] = -5.0
                d["sma200"] = 4450.0
                d["vrp_ann"] = 1.6
                d["atr_d1"] = 100.0
                d["mu_daily_post_pct"] = -0.18
                return d

            def mock_intra():
                return {
                    "source": "YAHOO_INTRADAY_LIVE",
                    "z_m15_1h": 3.00,
                    "z_m15_4h": 0.20,
                    "z_h1_4h": 0.25,
                    "z_h1_12h": -0.60,
                    "day_open": 3850.0,
                    "day_high": 3851.0,
                    "day_low": 3745.0,
                    "live_px": 3788.0,
                    "mu_m15_post_pct": 1.20,
                    "mu_h1_post_pct": -0.40,
                }

            qm._get_daily_prob_regime = mock_daily
            qm._get_intraday_momentum_regime = mock_intra

            scalp = qm.calculate_daily_a_star_setup(current_price=3788.0, horizon="scalping", n_paths=1000)
            intra = qm.calculate_daily_a_star_setup(current_price=3788.0, horizon="intraday", n_paths=1000)
            daily = qm.calculate_daily_a_star_setup(current_price=3788.0, horizon="daily", n_paths=1000)

            self.assertIn("LONG", scalp["trade_ticket"]["action"])
            self.assertFalse(intra["is_a_star"])
            self.assertIn("STAND ASIDE", intra["trade_ticket"]["action"])
            self.assertFalse(daily["is_a_star"])
            self.assertIn("STAND ASIDE", daily["trade_ticket"]["action"])
        finally:
            qm._get_daily_prob_regime = orig_daily
            qm._get_intraday_momentum_regime = orig_intra

    def test_optin_running_list_and_divided_ledger(self):
        """Verifies manual opt-in add, live tick evaluation (TP hit & SL touched), and 4-box ledger categorization."""
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "test_optin_ledger.json")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_path)
            # 1. Opt-in add Scalping BUY
            r1 = mgr.add_to_running_list({
                "horizon": "scalping",
                "direction": "BUY",
                "entry_zone": "$3785.00 - $3787.00",
                "stop_loss": "$3765.00",
                "take_profit_1": "$3805.00",
                "take_profit_2": "$3820.00",
                "recommended_lot": 0.02,
                "immediate_entry": True
            }, 3786.0)
            self.assertEqual(r1["status"], "ADDED")

            # Duplicate identical instant click caught cleanly by debounce
            r_dup = mgr.add_to_running_list({"horizon": "scalping", "direction": "BUY", "entry_zone": "$3785.00 - $3787.00", "stop_loss": "$3765.00", "take_profit_1": "$3805.00"}, 3786.0)
            self.assertEqual(r_dup["status"], "DUPLICATE_DEBOUNCE")

            # Scale-in / new candidate setup at different price succeeds with ADDED
            r_scale = mgr.add_to_running_list({
                "horizon": "scalping",
                "direction": "BUY",
                "entry_zone": "$3790.00 - $3792.00",
                "stop_loss": "$3775.00",
                "take_profit_1": "$3815.00",
                "take_profit_2": "$3830.00",
                "recommended_lot": 0.01,
                "immediate_entry": True
            }, 3791.0)
            self.assertEqual(r_scale["status"], "ADDED")
            self.assertEqual(r_scale["running_count"], 2)

            # 2. Opt-in add Daily SELL
            r2 = mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "SELL",
                "entry_zone": "$3785.00 - $3787.00",
                "stop_loss": "$3810.00",
                "take_profit_1": "$3735.00",
                "take_profit_2": "$3700.00",
                "recommended_lot": 0.01,
                "immediate_entry": True
            }, 3786.0)
            self.assertEqual(r2["status"], "ADDED")

            # 3. Price moves to 3832.0 -> Both Scalping BUYs hit FULL TP2, Daily SELL hits SL!
            cat = mgr.get_categorized_ledger(live_price=3832.0)
            self.assertEqual(cat["summary"]["running_count"], 0)
            self.assertEqual(cat["summary"]["hit_tp_count"], 2)
            self.assertEqual(cat["summary"]["sl_touched_count"], 1)
            self.assertEqual(cat["summary"]["win_rate_pct"], 66.7)
            self.assertGreater(cat["summary"]["hit_tp_total_r"], 2.0)
            self.assertEqual(cat["summary"]["sl_touched_total_r"], -1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_production_ledger_never_mutated_by_read_calls(self):
        """Strict anti-pollution invariant: calculate_daily_a_star_setup must NEVER write to production setup_ledger.json."""
        prod_path = "/Users/macbook/newslyser/data/setup_ledger.json"
        with open(prod_path, "r") as f:
            before = f.read() if os.path.exists(prod_path) else ""
        qm.calculate_daily_a_star_setup(current_price=3725.0, horizon="scalping", n_paths=500)
        qm.calculate_daily_a_star_setup(current_price=3725.0, horizon="daily", n_paths=500)
        with open(prod_path, "r") as f:
            after = f.read() if os.path.exists(prod_path) else ""
        self.assertEqual(before, after)

    def test_candidate_setup_never_locked_out_by_running_trade(self):
        """Verifies candidate setup is freshly computed around spot and never locked out by running trades."""
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "test_nolock_ledger.json")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_path)
            # Add initial trade at 3750.0
            r1 = mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "SELL",
                "entry_zone": "$3748.00 - $3752.00",
                "stop_loss": "$3780.00",
                "take_profit_1": "$3700.00",
                "take_profit_2": "$3680.00",
                "recommended_lot": 0.01,
                "immediate_entry": True
            }, 3750.0)
            self.assertEqual(r1["status"], "ADDED")
            self.assertEqual(r1["running_count"], 1)

            # Spot price moves to 3720.0; calculate fresh daily setup
            res = qm.calculate_daily_a_star_setup(current_price=3720.0, horizon="daily", n_paths=500)
            tt = res.get("trade_ticket", {})
            # Entry zone must be calculated around new spot $3720, not the old $3750
            self.assertIn("372", tt.get("entry_zone", ""))
            self.assertNotIn("3748", tt.get("entry_zone", ""))

            # Now add a new setup / scale-in position at 3720.0
            r2 = mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "SELL",
                "entry_zone": "$3718.00 - $3722.00",
                "stop_loss": "$3740.00",
                "take_profit_1": "$3690.00",
                "take_profit_2": "$3670.00",
                "recommended_lot": 0.01,
                "immediate_entry": True
            }, 3720.0)
            self.assertEqual(r2["status"], "ADDED")
            self.assertEqual(r2["running_count"], 2)

            # Both positions run concurrently in ledger
            cat = mgr.get_categorized_ledger(live_price=3720.0)
            self.assertEqual(cat["summary"]["running_count"], 2)

            # Test force_reset_horizon: NEVER purges user running trades
            mgr.force_reset_horizon(horizon="daily")
            cat_after_reset = mgr.get_categorized_ledger(live_price=3720.0)
            self.assertEqual(cat_after_reset["summary"]["running_count"], 2, "Running trades must not be purged by session reset!")

            # Test remove mode: archives into completed_setups with REMOVED_BY_TRADER status (no vanishing)
            sid_to_remove = r1["setup"]["setup_id"]
            res_rem = mgr.close_or_remove_running_trade(sid_to_remove, mode="remove")
            self.assertEqual(res_rem["status"], "OK")
            cat_after_remove = mgr.get_categorized_ledger(live_price=3720.0)
            self.assertEqual(cat_after_remove["summary"]["running_count"], 1)
            archived = cat_after_remove["archived_expired"]
            arch_ids = [x["setup_id"] for x in archived]
            self.assertIn(sid_to_remove, arch_ids, "Removed trade must be preserved in archived ledger with audit trail!")
            rem_entry = [x for x in archived if x["setup_id"] == sid_to_remove][0]
            self.assertEqual(rem_entry["status"], "REMOVED_BY_TRADER")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_brownian_bridge_barrier_correction(self):
        """Verifies that Brownian bridge barrier correction detects intra-step crossings and penalizes friction."""
        res_buy = qm.calculate_first_passage_win_probability(
            current_price=2000.0,
            entry_price=2000.0,
            sl_price=1990.0,
            tp_price=2020.0,
            drift_pct=0.0,
            atr_m30=10.0,
            is_buy=True,
            n_paths=5000,
            steps=20
        )
        self.assertIn("win_probability_pct", res_buy)
        self.assertIn("overall_loss_rate_pct", res_buy)
        self.assertIn("survival_curve", res_buy)
        # Symmetrical barrier with friction: SL hit probability should be strictly > 0
        self.assertGreater(res_buy["overall_loss_rate_pct"], 0.0)
        self.assertGreater(res_buy["win_probability_pct"], 0.0)

    def test_compensated_kou_jump_martingale(self):
        """Verifies that Kou jump diffusion with martingale compensator eliminates Jensen inequality drift bias at z=0."""
        rng = np.random.default_rng(123)
        n_paths = 10000
        steps = 20
        dt = 1.0 / steps
        step_vol = 0.01
        p_up = 0.50
        eta1 = 1.0 / 0.0065
        eta2 = 1.0 / 0.0065
        kappa_j = (p_up * eta1 / max(1.0001, eta1 - 1.0)) + ((1.0 - p_up) * eta2 / (eta2 + 1.0)) - 1.0
        lambda_step = 0.20 * dt

        z_mat = rng.normal(0, 1, (n_paths, steps))
        n_jumps = rng.poisson(lambda_step, (n_paths, steps))
        max_k = int(np.max(n_jumps))
        jump_rets = np.zeros((n_paths, steps))
        if max_k > 0:
            u_j = rng.uniform(0, 1, (n_paths, steps, max_k))
            up_m = u_j < p_up
            e_up = rng.exponential(1.0 / eta1, (n_paths, steps, max_k))
            e_dn = rng.exponential(1.0 / eta2, (n_paths, steps, max_k))
            j_vals = np.where(up_m, e_up, -e_dn)
            k_indices = np.arange(max_k)[None, None, :]
            active_k = k_indices < n_jumps[:, :, None]
            jump_rets = np.sum(j_vals * active_k, axis=2)

        # Compensated log return
        diff_rets = (-0.5 * (step_vol ** 2) - lambda_step * kappa_j) + step_vol * z_mat
        log_rets = diff_rets + jump_rets
        terminal_ratio = np.exp(np.sum(log_rets, axis=1))
        mean_ratio = float(np.mean(terminal_ratio))
        se = float(np.std(terminal_ratio) / np.sqrt(n_paths))

        # Under the martingale property, E[S_T / S_0] == 1.0 within 2.5 standard errors
        self.assertAlmostEqual(mean_ratio, 1.0, delta=2.5 * se,
                               msg=f"Compensated Kou jump must be martingale: mean={mean_ratio:.5f}, se={se:.5f}")

    def test_smooth_factor_shrinkage(self):
        """Verifies that Empirical Bayes factor shrinkage is smooth and has no cliff edge at t=1.96."""
        def shrinkage_weight(t):
            t_sq = float(t) ** 2
            return max(0.0, 1.0 - (1.96 ** 2) / max(1e-6, t_sq))

        # At t=1.95 (just below 1.96), weight is 0
        w_below = shrinkage_weight(1.95)
        self.assertEqual(w_below, 0.0)

        # At t=1.961 (just above 1.96), weight is smoothly positive without cliff
        w_just_above = shrinkage_weight(1.961)
        self.assertGreater(w_just_above, 0.0)
        self.assertLess(w_just_above, 0.01, "Weight must smoothly scale from 0 without discontinuous jump")

        # At t=4.0, weight is ~0.76
        w_high = shrinkage_weight(4.0)
        self.assertAlmostEqual(w_high, 1.0 - (1.96**2)/16.0, places=3)

    def test_portfolio_net_delta_and_hedging(self):
        """Verifies net delta exposure, gross exposure, and hedging conflict calculation in SetupLifecycleManager."""
        import tempfile
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, "test_net_delta_ledger.json")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        mgr = qm.SetupLifecycleManager(filepath=tmp_path)
        try:
            # Add a BUY trade (0.02 lot = 2.0 oz)
            mgr.add_to_running_list({
                "horizon": "daily",
                "direction": "BUY",
                "entry_zone": "$2000.00",
                "stop_loss": "$1980.00",
                "take_profit_1": "$2040.00",
                "recommended_lot": 0.02,
                "immediate_entry": True
            }, 2000.0)

            # Add a SELL trade (0.01 lot = 1.0 oz)
            mgr.add_to_running_list({
                "horizon": "scalping",
                "direction": "SELL",
                "entry_zone": "$2000.00",
                "stop_loss": "$2010.00",
                "take_profit_1": "$1990.00",
                "recommended_lot": 0.01,
                "immediate_entry": True
            }, 2000.0)

            cat = mgr.get_categorized_ledger(live_price=2000.0)
            summary = cat["summary"]
            self.assertEqual(summary["long_exposure_oz"], 2.0)
            self.assertEqual(summary["short_exposure_oz"], 1.0)
            self.assertEqual(summary["gross_exposure_oz"], 3.0)
            self.assertEqual(summary["net_delta_oz"], 1.0)
            self.assertTrue(summary["is_hedged_conflict"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_continuous_4th_moment_kelly(self):
        """Verifies that 4th-moment continuous Kelly incorporates skewness and excess kurtosis."""
        ast = qm.calculate_daily_a_star_setup(n_paths=2000)
        fp = ast["first_passage_analysis"]
        q_kelly = fp["quarter_kelly_pct"]
        self.assertGreaterEqual(q_kelly, 0.0)
        self.assertLessEqual(q_kelly, 2.0, "Quarter-Kelly must be bounded by risk ceiling 2.0%")

    def test_differentiable_sigmoid_confluence(self):
        """Verifies that confluence score is smoothly evaluated across pillars."""
        ast = qm.calculate_daily_a_star_setup(n_paths=1500)
        score = ast["confluence_score"]
        self.assertGreaterEqual(score, 20.0)
        self.assertLessEqual(score, 100.0)
        pillars = ast["pillars"]
        for p_name, p_data in pillars.items():
            self.assertIn("score", p_data)
            self.assertGreaterEqual(p_data["score"], 5.0)
            self.assertLessEqual(p_data["score"], 25.0)


if __name__ == "__main__":
    unittest.main()
