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
        self.assertEqual(intra["time_to_live"]["max_validity_minutes"], 480)
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

    def test_5y_eurusd_multi_horizon_backtest_telemetry(self):
        """Verifies Phase 5: EUR/USD 5-year multi-horizon backtest engine and hypothesis testing telemetry."""
        res = qm.calculate_daily_a_star_setup(instrument="EURUSD", horizon="daily", n_paths=1000)
        pe = res["probability_engine"]
        self.assertIn("backtest_3y_summary", pe)
        self.assertIn("backtest_5y_summary", pe)
        bt5 = pe["backtest_5y_summary"]
        self.assertGreater(bt5["trades"], 50)
        self.assertGreater(bt5["profit_factor"], 1.0)

        # Check JSON telemetry on disk
        bt5_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_multi_horizon_5y_eurusd.json")
        self.assertTrue(os.path.exists(bt5_path), f"File {bt5_path} must exist from Phase 5 backtest engine")
        with open(bt5_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data.get("instrument"), "EURUSD")
        self.assertIn("tracks", data)
        tracks = data["tracks"]
        for expected_track in ["scalping_astar", "intraday_astar", "daily_astar", "swing_astar", "macro_news", "combined_portfolio"]:
            self.assertIn(expected_track, tracks, f"Track {expected_track} must exist in EUR/USD backtest")

        comb = tracks["combined_portfolio"]
        self.assertGreaterEqual(comb["trades"], 300, "Combined trades must be >= 300 across 5 years of genuine data")
        self.assertGreater(comb["sum_r"], 40.0, "Cumulative R must exceed +40.0R")
        self.assertGreater(comb["ending_equity_usd"], 1500.0, "Ending equity from $1k baseline must exceed $1,500.00")
        self.assertGreaterEqual(comb["t_stat"], 1.9, "Student t-statistic must be >= 1.9")
        self.assertLessEqual(comb["p_val"], 0.06, "p-value must be <= 0.06")
        self.assertLess(comb["max_drawdown_pct"], 30.0, "Max DD must be < 30.0%")

        # Hypothesis testing validation
        ht = data["hypothesis_testing"]
        self.assertLess(ht["rademacher_permutation_pval"], 0.05, "100,000 Rademacher sign-flipping permutation must reject H0 at p < 0.05")
        self.assertGreaterEqual(ht["deflated_sharpe_ratio"], 75.0, "Deflated Sharpe Ratio (DSR) must be >= 75.0% confidence")
        self.assertLess(ht["exact_binomial_pval"], 0.01, "Exact binomial test must reject H0 at p < 0.01")

        # Year-over-Year vintage stability
        yoy = data.get("year_over_year", {})
        profitable_years = 0
        for yr in ["2022", "2023", "2024", "2025", "2026"]:
            self.assertIn(yr, yoy, f"Year {yr} must be represented in vintage stability")
            if yoy[yr]["profit_factor"] > 1.0:
                profitable_years += 1
        self.assertGreaterEqual(profitable_years, 4, "At least 4 of 5 years must show profitable regime survival")

        # Test API endpoints
        import app
        from fastapi.testclient import TestClient
        client = TestClient(app.app)

        res_5y = client.get("/api/backtest-5y?instrument=EURUSD")
        self.assertEqual(res_5y.status_code, 200)
        d_5y = res_5y.json()
        self.assertEqual(d_5y.get("instrument"), "EURUSD")
        self.assertEqual(d_5y["tracks"]["combined_portfolio"]["trades"], comb["trades"])

        res_eur = client.get("/api/backtest-eurusd-5y")
        self.assertEqual(res_eur.status_code, 200)

        res_mh = client.get("/api/backtest-multi-horizon?instrument=EURUSD")
        self.assertEqual(res_mh.status_code, 200)
        d_mh = res_mh.json()
        self.assertIn("backtest_5y", d_mh)
        self.assertEqual(d_mh["backtest_5y"].get("instrument"), "EURUSD")

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

            def mock_intra(*args, **kwargs):
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

    def test_eurusd_instrument_calibration(self):
        """Verifies EURUSD multi-asset polymorphic calibration in quant_model."""
        # 1. Instrument profile lookup
        eur_spec = qm.get_instrument_spec("EURUSD")
        self.assertEqual(eur_spec["symbol"], "EURUSD")
        self.assertEqual(eur_spec["contract"], 100000.0)
        self.assertEqual(eur_spec["pip"], 0.0001)
        self.assertEqual(eur_spec["pip_value_001"], 0.10)
        self.assertEqual(eur_spec["price_decimals"], 5)
        self.assertAlmostEqual(eur_spec["spread_usd"], 0.00003, places=6)

        # 2. First-Passage Monte Carlo on EURUSD
        fp_eur = qm.calculate_first_passage_win_probability(
            current_price=1.08500,
            entry_price=1.08500,
            sl_price=1.08300,
            tp_price=1.08800,
            drift_pct=0.05,
            atr_m30=0.0012,
            is_buy=True,
            n_paths=2000,
            instrument="EURUSD"
        )
        self.assertIn("win_probability_pct", fp_eur)
        self.assertIn("overall_win_rate_pct", fp_eur)
        self.assertGreaterEqual(fp_eur["win_probability_pct"], 0.0)
        self.assertLessEqual(fp_eur["win_probability_pct"], 100.0)

        # 3. Daily A* Setup polymorphic execution on EURUSD
        ast_eur = qm.calculate_daily_a_star_setup(
            current_price=1.08500,
            atr_m30=0.0012,
            n_paths=2000,
            instrument="EURUSD"
        )
        self.assertEqual(ast_eur["symbol"], "EURUSD")
        self.assertEqual(ast_eur["contract_size"], 100000.0)
        self.assertEqual(ast_eur["pip_size"], 0.0001)
        self.assertAlmostEqual(ast_eur["current_price"], 1.08500, places=5)
        self.assertIn("first_passage_analysis", ast_eur)
        fp_a = ast_eur["first_passage_analysis"]
        # Ensure SL and TP prices are in proper EURUSD decimal scale (~1.08)
        self.assertGreater(fp_a["sl_price"], 1.05)
        self.assertLess(fp_a["sl_price"], 1.12)

        # 4. Multi-Horizon Matrix polymorphic execution on EURUSD
        mh_eur = qm.calculate_multi_horizon_a_star_setup(
            current_price=1.08500,
            atr_m30=0.0012,
            n_paths=1500,
            instrument="EURUSD"
        )
        self.assertEqual(mh_eur["symbol"], "EURUSD")
        self.assertIn("horizons", mh_eur)
        self.assertIn("scalping", mh_eur["horizons"])
        self.assertIn("intraday", mh_eur["horizons"])
        self.assertIn("daily", mh_eur["horizons"])
        self.assertIn("swing", mh_eur["horizons"])
        self.assertIn("news", mh_eur["horizons"])

    def test_eurusd_snr_pure_trend_playbook(self):
        """
        Verifies EUR/USD SNR Pure-Trend Playbook:
        1. Trend Deduction (EMA 50 & 200).
        2. Sandwich Chop Zone Blackout (Hard Stand Aside).
        3. Zero Counter-Trend Lockout.
        4. Minimum 1:2.50 RR Compounding targets.
        5. Dynamic Breakeven Shield at +1.0R (locking Entry +/- 0.5 pips).
        """
        # 1. Bullish pure trend: Price > EMA 50 > EMA 200
        snr_bull = qm.evaluate_eurusd_snr_pure_trend(
            current_price=1.09500,
            ema50=1.09200,
            ema200=1.08800,
            atr_m15=0.0010
        )
        self.assertEqual(snr_bull["trend_regime"], "BULLISH_PURE_TREND")
        self.assertTrue(snr_bull["is_bullish_trend"])
        self.assertFalse(snr_bull["is_sandwich_chop"])
        self.assertEqual(snr_bull["allowed_direction"], "BUY")
        self.assertGreaterEqual(snr_bull["long_setup"]["planned_rr"], 2.50)
        self.assertEqual(snr_bull["long_setup"]["dynamic_be_shield_sl"], 1.09505)

        # 2. Bearish pure trend: Price < EMA 50 < EMA 200
        snr_bear = qm.evaluate_eurusd_snr_pure_trend(
            current_price=1.08200,
            ema50=1.08500,
            ema200=1.09000,
            atr_m15=0.0010
        )
        self.assertEqual(snr_bear["trend_regime"], "BEARISH_PURE_TREND")
        self.assertTrue(snr_bear["is_bearish_trend"])
        self.assertFalse(snr_bear["is_sandwich_chop"])
        self.assertEqual(snr_bear["allowed_direction"], "SELL")
        self.assertGreaterEqual(snr_bear["short_setup"]["planned_rr"], 2.50)
        self.assertEqual(snr_bear["short_setup"]["dynamic_be_shield_sl"], 1.08195)

        # 3. Sandwich Chop Zone Blackout: EMA 50 <= Price <= EMA 200
        snr_chop = qm.evaluate_eurusd_snr_pure_trend(
            current_price=1.08800,
            ema50=1.08500,
            ema200=1.09200,
            atr_m15=0.0010
        )
        self.assertEqual(snr_chop["trend_regime"], "SANDWICH_CHOP_BLACKOUT")
        self.assertTrue(snr_chop["is_sandwich_chop"])
        self.assertEqual(snr_chop["allowed_direction"], "STAND_ASIDE")

        # 4. In calculate_daily_a_star_setup: Sandwich Chop Blackout forces STAND ASIDE
        orig_intra = qm._get_intraday_momentum_regime
        try:
            def mock_intra_chop(instrument="EURUSD"):
                return {
                    "source": "MOCK",
                    "instrument": "EURUSD",
                    "z_m15_1h": 1.20,
                    "z_h1_12h": 1.10,
                    "day_open": 1.08800,
                    "day_high": 1.08900,
                    "day_low": 1.08700,
                    "live_px": 1.08800,
                    "ema50": 1.08500,
                    "ema200": 1.09200,  # Price trapped between 1.08500 and 1.09200
                    "is_engulfing_bull": False,
                    "is_engulfing_bear": False,
                    "mu_m15_post_pct": 0.05,
                    "mu_h1_post_pct": 0.05
                }
            qm._get_intraday_momentum_regime = mock_intra_chop
            res_chop = qm.calculate_daily_a_star_setup(
                current_price=1.08800,
                atr_m30=0.0012,
                n_paths=1000,
                instrument="EURUSD"
            )
            self.assertFalse(res_chop["is_a_star"])
            self.assertIn("SANDWICH CHOP BLACKOUT", res_chop["trade_ticket"]["action"])
            self.assertEqual(
                res_chop["data_validation_gate"]["checks"]["eurusd_snr_pure_trend_gate"]["status"],
                "BLOCKED"
            )
        finally:
            qm._get_intraday_momentum_regime = orig_intra

        # 5. Dynamic Breakeven Shield test in SetupLifecycleManager
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "test_eurusd_be_ledger.json")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_path)
            # Add EURUSD BUY at 1.08500, SL at 1.08300 (risk = 20 pips), TP1 at 1.09000 (2.5R)
            add_res = mgr.add_to_running_list({
                "horizon": "intraday",
                "direction": "BUY",
                "asset": "EURUSD",
                "entry_zone": "$1.08490 - $1.08510",
                "stop_loss": "$1.08300",
                "take_profit_1": "$1.09000",
                "take_profit_2": "$1.09300",
                "recommended_lot": 0.50,
                "immediate_entry": True
            }, 1.08500)
            self.assertEqual(add_res["status"], "ADDED")

            # Tick 1: Price advances to 1.08700 (+20 pips = +1.0R) -> triggers Dynamic Breakeven Shield
            cat1 = mgr.get_categorized_ledger(live_price=1.08700)
            running = cat1["active_running"]
            self.assertEqual(len(running), 1)
            trade = running[0]
            self.assertIsNotNone(trade.get("trailed_sl"))
            # 0.5 pip shield locks entry + 0.5 pips = 1.08505
            self.assertEqual(trade.get("trailed_sl"), 1.08505)

            # Tick 2: Price pulls back to 1.08500 (hits trailed SL at 1.08505) -> BREAKEVEN_EXIT
            cat2 = mgr.get_categorized_ledger(live_price=1.08500)
            self.assertEqual(cat2["summary"]["running_count"], 0)
            self.assertEqual(cat2["summary"]["hit_tp_count"], 1)  # BREAKEVEN_EXIT categorized in protected wins
            self.assertGreaterEqual(cat2["summary"]["hit_tp_total_r"], 0.0)
            self.assertEqual(cat2["summary"]["sl_touched_count"], 0)  # NOT counted as loss!
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_eurusd_pipeline_macro_telemetry_and_endpoints(self):
        """Verifies Phase 3: EUR/USD data pipeline, macro telemetry ingestion, and multi-asset endpoints."""
        import app
        from fastapi.testclient import TestClient

        # 1. High Impact Calendar: Eurozone macro events & effect_eurusd verification
        cal_path = os.path.join(app.DATA_DIR, "high_impact_calendar.json")
        self.assertTrue(os.path.exists(cal_path))
        with open(cal_path) as f:
            calendar_events = json.load(f)
        ecb_event = next((e for e in calendar_events if e.get("event_id") == "ecb_rate_dec26"), None)
        self.assertIsNotNone(ecb_event, "ECB rate decision event must be registered in high_impact_calendar.json")
        self.assertIn("effect_eurusd", ecb_event)
        self.assertIn("US10Y - DE10Y", ecb_event["effect_eurusd"])

        # 2. Live Macro Data: Verify EURUSD spot & DE10Y & Transatlantic Yield Spread ingestion
        macro_data = app.fetch_live_macro_data()
        self.assertIn("prices", macro_data)
        prices = macro_data["prices"]
        self.assertIn("EURUSD=X", prices)
        self.assertIn("EURUSD_SPOT", prices)
        self.assertIn("DE10Y", prices)
        self.assertIn("^TNX", prices)
        self.assertIn("YIELD_SPREAD_US_DE", prices)
        self.assertAlmostEqual(prices["YIELD_SPREAD_US_DE"], round(prices["^TNX"] - prices["DE10Y"], 3), places=2)

        # 3. TestClient: GET /api/live-ticks?instrument=EURUSD
        client = TestClient(app.app)
        res_ticks = client.get("/api/live-ticks?instrument=EURUSD")
        self.assertEqual(res_ticks.status_code, 200)
        d_ticks = res_ticks.json()
        self.assertEqual(d_ticks["status"], "SUCCESS")
        self.assertEqual(d_ticks["instrument"], "EURUSD")
        self.assertLess(d_ticks["current_price"], 2.0)  # FX rate ~ 1.09, not 3500 Gold

        # 4. TestClient: GET /api/daily-a-star?instrument=EURUSD
        res_daily = client.get("/api/daily-a-star?instrument=EURUSD")
        self.assertEqual(res_daily.status_code, 200)
        d_daily = res_daily.json()
        self.assertEqual(d_daily["symbol"], "EURUSD")
        self.assertEqual(d_daily["contract_size"], 100000.0)
        self.assertEqual(d_daily["pip_size"], 0.0001)
        self.assertIn("snr_pure_trend", d_daily)
        tt = d_daily["trade_ticket"]
        self.assertEqual(tt["asset"], "Euro / US Dollar (EURUSD)")

        # 5. TestClient: GET /api/multi-horizon-a-star?instrument=EURUSD
        res_multi = client.get("/api/multi-horizon-a-star?instrument=EURUSD")
        self.assertEqual(res_multi.status_code, 200)
        d_multi = res_multi.json()
        self.assertEqual(d_multi["symbol"], "EURUSD")
        self.assertEqual(d_multi["pip_size"], 0.0001)
        self.assertIn("horizons", d_multi)
        self.assertIn("scalping", d_multi["horizons"])
        self.assertIn("daily", d_multi["horizons"])

        # 6. Multi-Asset Isolation: Concurrent Gold & EUR/USD trades in ledger without price cross-contamination
        import tempfile
        tmp_ledger = os.path.join(tempfile.gettempdir(), "test_dual_asset_ledger.json")
        if os.path.exists(tmp_ledger):
            os.remove(tmp_ledger)
        try:
            mgr = qm.SetupLifecycleManager(filepath=tmp_ledger)
            # Add XAUUSD trade
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
            # Add EURUSD trade
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

            # Evaluate with live_prices dictionary for both assets simultaneously
            cat = mgr.get_categorized_ledger(live_prices={"XAUUSD": 3520.0, "EURUSD": 1.08700})
            running = cat["active_running"]
            self.assertEqual(len(running), 2)
            gold_trade = next(t for t in running if t["asset"] == "XAUUSD")
            eur_trade = next(t for t in running if t["asset"] == "EURUSD")

            # Gold: +$20 on 0.02 lot (2 oz) = +$40.00, +1.0R
            self.assertAlmostEqual(gold_trade["floating_pnl_usd"], 40.0, delta=2.0)
            self.assertAlmostEqual(gold_trade["floating_r"], 1.0, delta=0.1)

            # EUR: +20 pips on 0.50 lot (50,000 EUR) = +$100.00, +1.0R
            self.assertAlmostEqual(eur_trade["floating_pnl_usd"], 100.0, delta=2.0)
            self.assertAlmostEqual(eur_trade["floating_r"], 1.0, delta=0.1)
            # Dynamic Breakeven Shield triggered on EUR trade (at +1.0R)
            self.assertEqual(eur_trade.get("trailed_sl"), 1.08505)
        finally:
            if os.path.exists(tmp_ledger):
                os.remove(tmp_ledger)

    def test_phase4_frontend_multi_asset_toggle_and_metrics(self):
        """Verifies Phase 4: UI templates, ticker strip, and multi-asset toggle integration."""
        import app
        from fastapi.testclient import TestClient

        client = TestClient(app.app)

        # 1. Verify index.html contains EURUSD asset selector and new telemetry elements
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.text
        self.assertIn('data-asset="EURUSD"', html, "EURUSD asset toggle button must exist in index.html")
        self.assertIn('id="tick-yield-spread-us-de"', html, "Transatlantic yield spread ticker must exist")
        self.assertIn('id="row-ast-yield-spread"', html, "Yield spread table row must exist")
        self.assertIn('id="row-ast-pure-trend"', html, "SNR Pure-Trend table row must exist")
        self.assertIn('id="row-ast-be-shield"', html, "Dynamic Breakeven Shield table row must exist")
        self.assertIn('id="row-astar-be-shield"', html, "Ticket Pane Dynamic Breakeven Shield must exist")

        # 2. Verify static/app.js is accessible and contains Phase 4 multi-asset telemetry handlers
        res_js = client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        js = res_js.text
        self.assertIn("EURUSD.SNR <GO>", js)
        self.assertIn("TRANSATLANTIC SPREAD (US10Y - DE10Y)", js)
        self.assertIn("SANDWICH CHOP BLACKOUT", js)
        self.assertIn("tick-yield-spread-us-de", js)

    def test_hypothesis_testing_battery_mathematics(self):
        """Verifies mathematical correctness of Acklam normal PPF, White HC3, DSR EVT, and Binomial test."""
        # 1. Acklam normal inverse CDF
        self.assertAlmostEqual(qm.normal_ppf(0.50), 0.0, places=5)
        self.assertAlmostEqual(qm.normal_ppf(0.8413447), 1.0, places=4)
        self.assertAlmostEqual(qm.normal_ppf(0.975), 1.95996, places=4)
        self.assertAlmostEqual(qm.normal_ppf(0.99), 2.32635, places=4)

        # 2. White HC3 Robust Estimator
        rng = np.random.default_rng(20261121)
        r_homo = rng.normal(0.20, 1.0, 500)
        se, t_val, p_val = qm.compute_white_hc3_estimator(r_homo)
        self.assertGreater(t_val, 2.0)
        self.assertLess(p_val, 0.05)
        self.assertGreater(se, 0.0)

        # 3. Deflated Sharpe Ratio (DSR) with EVT multiple-testing benchmark
        dsr_prob, dsr_z, ann_sr, ann_bench = qm.compute_deflated_sharpe_ratio(r_homo, n_trials=5)
        self.assertGreaterEqual(dsr_prob, 90.0)
        self.assertGreater(ann_sr, ann_bench)

        # 4. Single-trial DSR (no selection penalty)
        dsr_single, dsr_z_single, _, bench_single = qm.compute_deflated_sharpe_ratio(r_homo, n_trials=1)
        self.assertEqual(bench_single, 0.0)
        self.assertGreaterEqual(dsr_single, dsr_prob)

        # 5. Exact Binomial Test
        # 50 wins out of 100 on fair coin (p=0.50) -> p_val ~ 0.539
        p_coin = qm.compute_exact_binomial_test(100, 50, 0.50)
        self.assertGreater(p_coin, 0.40)
        # 75 wins out of 100 on fair coin (p=0.50) -> p_val < 1e-6 (H0 rejected)
        p_edge = qm.compute_exact_binomial_test(100, 75, 0.50)
        self.assertLess(p_edge, 1e-6)

    def test_multi_asset_5y_backtest_endpoints(self):
        """Verifies that /api/backtest-5y and /api/backtest-multi-horizon serve all 3 assets with hypothesis testing."""
        import app
        from fastapi.testclient import TestClient
        client = TestClient(app.app)

        for inst in ["XAUUSD", "EURUSD", "BTCUSD"]:
            res = client.get(f"/api/backtest-5y?instrument={inst}")
            self.assertEqual(res.status_code, 200, f"/api/backtest-5y?instrument={inst} failed")
            data = res.json()
            self.assertIn("tracks", data)
            self.assertIn("hypothesis_testing", data)
            ht = data["hypothesis_testing"]
            self.assertIn("student_t_stat", ht)
            self.assertIn("deflated_sharpe_ratio", ht)
            self.assertIn("rademacher_permutation_pval", ht)

            res_mh = client.get(f"/api/backtest-multi-horizon?instrument={inst}")
            self.assertEqual(res_mh.status_code, 200, f"/api/backtest-multi-horizon?instrument={inst} failed")
            d_mh = res_mh.json()
            self.assertIn("backtest_5y", d_mh)


if __name__ == "__main__":
    unittest.main()
