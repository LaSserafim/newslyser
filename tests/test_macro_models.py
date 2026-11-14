"""
Unit tests for Econometric & Macro Transmission Engines in NEWSLYSER.
Tests White (1980) HC3 OLS, Stage-B Shrinkage, Lo-MacKinlay VR(5),
CPI/FOMC/NFP/PPI Pre-News & Post-News models, and Lot Sizer precedence.
"""

import unittest
import numpy as np
import quant_model as qm


class TestMacroEconometricModels(unittest.TestCase):

    def test_white_hc3_ols_estimator(self):
        """Verifies that run_regression_ols returns accurate HC3 standard errors and leverage."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        true_beta = np.array([1.5, -2.0, 0.8])
        X_design = np.column_stack([np.ones(100), X])
        # Heteroskedastic errors scaling with X[:, 0]^2
        errors = np.random.normal(0, 0.5 + 0.5 * X[:, 0] ** 2)
        y = X_design @ true_beta + errors

        res = qm.run_regression_ols(X, y)
        self.assertEqual(len(res["beta"]), 3)
        self.assertEqual(len(res["se_hc3"]), 3)
        self.assertAlmostEqual(res["beta"][0], true_beta[0], delta=0.5)
        self.assertAlmostEqual(res["beta"][1], true_beta[1], delta=0.5)
        self.assertAlmostEqual(res["beta"][2], true_beta[2], delta=0.5)
        self.assertGreater(res["r2"], 0.0)
        self.assertGreater(res["f_stat"], 0.0)

    def test_stage_b_shrinkage_on_noise(self):
        """
        Verifies that compute_stage_b_gold_direction strictly shrinks insignificant
        direct slopes (p >= 0.05) to 50/50 instead of projecting sample-mean noise.
        """
        # For PPI, direct slope p-value is 0.7335 (insignificant)
        res_ppi = qm.compute_stage_b_gold_direction("ppi", z_pre=-0.07)
        self.assertEqual(res_ppi["p_gold_up"], 50.0)
        self.assertEqual(res_ppi["p_gold_down"], 50.0)
        self.assertEqual(res_ppi["expected_gold_pct"], 0.0)
        self.assertIn("NO_DIRECT_EDGE", res_ppi["edge_note"])

        # For FOMC, direct slope p-value is 0.3248 (insignificant)
        res_fomc = qm.compute_stage_b_gold_direction("fomc", z_pre=1.20)
        self.assertEqual(res_fomc["p_gold_up"], 50.0)
        self.assertEqual(res_fomc["p_gold_down"], 50.0)
        self.assertEqual(res_fomc["expected_gold_pct"], 0.0)

        # For CPI, direct slope p-value is 0.0002 (highly significant at 99%)
        res_cpi = qm.compute_stage_b_gold_direction("cpi", z_pre=-0.37)
        self.assertNotEqual(res_cpi["expected_gold_pct"], 0.0)
        self.assertGreater(res_cpi["p_gold_up"], 55.0)
        self.assertIn("STATISTICALLY_VALID_SLOPE", res_cpi["edge_note"])

    def test_lo_mackinlay_variance_ratio(self):
        """Verifies that Lo-MacKinlay VR(5) computes z* scaled by sqrt(n)."""
        fp = qm.compute_algorithmic_footprint("daily", current_price=3880.0, atr_m30=12.0)
        self.assertIn("lo_mackinlay_vr5", fp)
        self.assertIn("variance_ratio_z", fp)
        self.assertIn("p_value", fp)
        self.assertIsInstance(fp["is_algorithmic_flow"], bool)
        self.assertGreater(fp["n_bars"], 15)

    def test_cpi_bayesian_model(self):
        """Verifies CPI pre-news Bayesian update and post-news simulation."""
        meta = {"forecast_mom": 0.3, "forecast_core_yoy": 2.9, "cleveland_fed_nowcast_mom": 0.31, "survey_dispersion_sd": 0.075}
        pre = qm.calculate_pre_news_bayesian_model(meta, current_price=3880.0, atr_m30=11.0, include_mc=False)
        self.assertEqual(pre["event_type"], "cpi")
        self.assertIn("z_pre", pre)
        self.assertIn("p_hawkish_beat", pre)
        self.assertIn("p_dovish_miss", pre)

        # Post-news simulation
        sim = qm.simulate_tomorrow_scenario(0.4, 2.9, meta, current_price=3880.0)
        self.assertIn("model_forecast", sim)
        self.assertIn("expected_gold_return_pct", sim["model_forecast"])

    def test_fomc_model(self):
        """Verifies FOMC pre-news Softmax and post-news Kuttner surprise decomposition."""
        pre = qm.calculate_fomc_prenew_model(current_price=3880.0, atr_m30=11.0, include_mc=False)
        total_p = pre["p_hawkish_beat"] + pre["p_inline"] + pre["p_dovish_miss"]
        self.assertAlmostEqual(total_p, 100.0, delta=0.5)

        post = qm.calculate_fomc_model(4.00, 3.90, dots_shift_bps=10.0, current_price=3880.0)
        self.assertEqual(post["metrics"]["rate_shock_bps"], 10.0)
        self.assertEqual(post["metrics"]["total_policy_shock_bps"], 15.0)

    def test_nfp_model(self):
        """Verifies NFP labor composite model."""
        pre = qm.calculate_nfp_prenew_model(current_price=3880.0, atr_m30=11.0, include_mc=False)
        self.assertEqual(pre["event_type"], "nfp")
        self.assertIn("z_pre", pre)

        post = qm.calculate_nfp_model(180.0, 135.0, 4.0, 4.1, 0.4, 0.3, current_price=3880.0)
        self.assertGreater(post["metrics"]["z_score_composite_labor"], 0)

    def test_ppi_retail_model(self):
        """Verifies PPI & Retail demand model with BLS Oct 2026 data point."""
        pre = qm.calculate_ppi_prenew_model(current_price=3880.0, atr_m30=11.0, include_mc=False)
        self.assertEqual(pre["event_type"], "ppi")
        self.assertIn("z_pre", pre)

        post = qm.calculate_ppi_retail_model(0.35, 0.40, 0.40, 0.40, current_price=3880.0)
        self.assertAlmostEqual(post["metrics"]["ppi_surprise_pp"], -0.05, delta=0.01)

    def test_lot_sizer_precedence(self):
        """Verifies that sl_price takes exact precedence over sl_pips defaults."""
        res_price = qm.calculate_lot_and_risk(equity=1000.0, risk_mode="pct", risk_pct=2.0, entry_price=3900.0, sl_price=3895.0, instrument="XAUUSD")
        self.assertEqual(res_price["sl_pips"], 50.0)
        self.assertEqual(res_price["actual_risk_dollars"], 20.0)

        res_pips = qm.calculate_lot_and_risk(equity=1000.0, risk_mode="pct", risk_pct=2.0, entry_price=3900.0, sl_pips=100.0, instrument="XAUUSD")
        self.assertEqual(res_pips["sl_pips"], 100.0)


if __name__ == "__main__":
    unittest.main()
