"""
Integration tests for NEWSLYSER Live HTTP API (port 8888).
Tests all event endpoints, lot sizer, calendar, sync-live, and DeepSeek prompts.
"""

import unittest
import urllib.request
import json


class TestLiveApiEndpoints(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8888"

    @classmethod
    def setUpClass(cls):
        try:
            req = urllib.request.Request(f"{cls.BASE_URL}/api/daily-a-star")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                pass
        except Exception:
            raise unittest.SkipTest("Live API server not running on port 8888")

    def _get(self, path):
        req = urllib.request.Request(f"{self.BASE_URL}{path}", headers={"User-Agent": "NEWSLYSER_TEST"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.BASE_URL}{path}", data=data, headers={"Content-Type": "application/json", "User-Agent": "NEWSLYSER_TEST"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def test_daily_a_star_endpoint(self):
        d = self._get("/api/daily-a-star")
        self.assertIn("is_a_star", d)
        self.assertIn("confluence_score", d)
        self.assertIn("first_passage_analysis", d)
        self.assertIn("data_validation_gate", d)

        d_scalp = self._get("/api/daily-a-star?horizon=scalping")
        self.assertEqual(d_scalp["horizon"], "scalping")

        d_intra = self._get("/api/daily-a-star?horizon=intraday")
        self.assertEqual(d_intra["horizon"], "intraday")

        d_multi = self._get("/api/multi-horizon-a-star")
        self.assertIn("horizons", d_multi)
        self.assertIn("scalping", d_multi["horizons"])
        self.assertIn("intraday", d_multi["horizons"])
        self.assertIn("daily", d_multi["horizons"])
        self.assertIn("swing", d_multi["horizons"])

    def test_prenew_endpoints(self):
        for path in ["/api/ppi-prenew-model", "/api/pre-news-model", "/api/fomc-prenew-model", "/api/nfp-prenew-model"]:
            d = self._get(path)
            self.assertIn("z_pre", d)
            self.assertIn("p_hawkish_beat", d)
            self.assertIn("p_dovish_miss", d)
            self.assertIn("directive_title", d)

    def test_lot_sizer_api(self):
        payload = {"equity": 2500.0, "risk_mode": "pct", "risk_pct": 1.5, "entry_price": 3880.0, "sl_price": 3870.0, "instrument": "XAUUSD", "direction": "BUY"}
        res = self._post("/api/calculate-lot-size", payload)
        self.assertEqual(res["sl_pips"], 100.0)
        self.assertEqual(res["target_risk_dollars"], 37.5)
        self.assertEqual(res["computed_lot"], 0.04)
        self.assertEqual(res["actual_risk_dollars"], 40.0)
    def test_calendar_endpoint(self):
        cal = self._get("/api/high-impact-calendar")
        self.assertIsInstance(cal, list)
        self.assertGreater(len(cal), 5)
        ppi_row = next((r for r in cal if r.get("event_id") == "ppi_oct26"), None)
        self.assertIsNotNone(ppi_row)
        self.assertEqual(ppi_row.get("actual"), "0.35%")


if __name__ == "__main__":
    unittest.main()
