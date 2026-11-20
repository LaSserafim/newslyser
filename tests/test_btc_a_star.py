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

if __name__ == "__main__":
    unittest.main()
