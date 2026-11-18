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
        """Verifies that force_reset_horizon clears active setups and archives superseded state."""
        mgr = qm.SetupLifecycleManager()
        raw_ticket = {"action": "BUY", "ticket_type": "TEST", "entry_zone": "$2000 - $2002", "stop_loss": "$1990", "take_profit_1": "$2020"}
        locked, _ = mgr.get_or_create_locked_setup(raw_ticket, 2000.0, is_a_star=True, horizon="intraday")
        self.assertIsNotNone(locked)
        # Force reset
        mgr.force_reset_horizon("intraday")
        ledger = mgr.get_ledger()
        self.assertIsNone(ledger.get("active_setups", {}).get("intraday"))

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
            mgr = qm.SetupLifecycleManager()
            for h in ("scalping", "intraday", "daily"):
                mgr.force_reset_horizon(h)

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


if __name__ == "__main__":
    unittest.main()
