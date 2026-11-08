"""Offline clock/provider fixtures: no fabricated fixtures are published."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import numpy as np
import pandas as pd

import market_data as md
from indicators import wilder_rsi
from backtest_engine import calc_rsi
from market_ui import quote_label, fmt
from screener_engine import filter_records, scan


def at(value):
    return datetime.fromisoformat(value)


def quote(stamp="2026-09-29T16:45:00+00:00"):
    return {"ticker": "TEST", "regular": {"price": 100., "source_time": stamp},
            "pre": {"price": None, "source_time": None},
            "post": {"price": 101., "source_time": None},
            "volume": 100., "change_pct": None, "source": "fixture"}


class MarketUpgradeTests(unittest.TestCase):
    def tearDown(self):
        md.clear_market_cache()

    def test_nyse_sessions_dst_holiday_weekend_and_early_close(self):
        examples = {
            "2026-09-29T13:29:59+00:00": "盤前",
            "2026-09-29T13:30:00+00:00": "盤中",
            "2026-09-29T20:00:00+00:00": "盤後",
            "2026-09-30T00:00:00+00:00": "休市",
            "2026-11-09T14:30:00+00:00": "盤中",
            "2026-07-03T14:00:00+00:00": "休市",
            "2026-11-07T15:00:00+00:00": "休市",
            "2026-11-27T18:01:00+00:00": "盤後",
        }
        for stamp, label in examples.items():
            self.assertEqual(md.market_status(at(stamp))["label"], label, stamp)

    def test_source_clock_unknown_stale_and_cached_are_independent(self):
        q = quote("2026-09-29T14:00:00+00:00")
        q["cache_hit"] = True
        text = quote_label(q, now=at("2026-09-29T14:06:00+00:00"))
        self.assertIn("10:00:00", text)
        self.assertIn("6 分鐘前", text)
        self.assertIn("可能過期", text)
        self.assertIn("快取", text)
        self.assertIn("資料不足", quote_label(quote(None)))
        self.assertNotIn("可能過期", quote_label(q, now=at("2026-09-30T01:00:00+00:00")))

    def test_cache_layers_expiry_and_force(self):
        with patch.object(md.provider, "quote", return_value=quote()) as fetch, \
             patch.object(md.provider, "fundamentals", return_value={"eps": None}) as facts:
            a = md.get_quote("TEST")
            b = md.get_quote("TEST")
            self.assertFalse(a["cache_hit"])
            self.assertTrue(b["cache_hit"])
            self.assertEqual(a["regular"]["source_time"], b["regular"]["source_time"])
            self.assertEqual(fetch.call_count, 1)
            key = ("quote", "TEST")
            stamp, data = md._cache[key]
            md._cache[key] = (stamp - timedelta(seconds=61), data)
            md.get_quote("TEST")
            self.assertEqual(fetch.call_count, 2)
            md.get_quote("TEST", force=True)
            self.assertEqual(fetch.call_count, 3)
            md.get_fundamentals("TEST")
            self.assertTrue(md.get_fundamentals("TEST")["cache_hit"])
            self.assertEqual(facts.call_count, 1)

    def test_missing_quote_time_uses_original_fetch_time_and_fetch_age(self):
        q = quote(None)
        q["received_at"] = "2026-09-29T14:05:00+00:00"
        q["cache_hit"] = True
        text = quote_label(q, now=at("2026-09-29T14:07:00+00:00"))
        self.assertIn("抓取時間 2026-09-29 10:05:00 ET（非報價時間）", text)
        self.assertIn("距抓取 2 分鐘", text)
        self.assertIn("快取", text)
        self.assertIn("報價來源時間未知", text)
        # Recent retrieval must never hide a stale genuine quote.
        q["regular"]["source_time"] = "2026-09-29T14:00:00+00:00"
        text = quote_label(q, now=at("2026-09-29T14:07:00+00:00"))
        self.assertIn("報價時間", text)
        self.assertIn("7 分鐘前", text)
        self.assertIn("可能過期", text)
        self.assertNotIn("抓取時間", text)

    def test_cached_fallback_age_does_not_reset_on_read(self):
        with patch.object(md.provider, "quote", return_value=quote(None)):
            first = md.get_quote("TEST")
            cached = md.get_quote("TEST")
        self.assertEqual(first["received_at"], cached["received_at"])
        now = datetime.fromisoformat(first["received_at"]) + timedelta(seconds=42)
        self.assertIn("距抓取 42 秒", quote_label(cached, now=now))

    def test_failed_fetch_not_cached_or_replaced_by_zero(self):
        with patch.object(md.provider, "quote", side_effect=ValueError("no quote")) as fetch:
            for _ in range(2):
                with self.assertRaises(ValueError):
                    md.get_quote("TEST")
            self.assertEqual(fetch.call_count, 2)
        self.assertIsNone(md.number(None))
        self.assertIsNone(md.number(float("nan")))
        self.assertEqual(md.number(0), 0.)
        self.assertEqual(fmt(None), "資料不足")

    def test_wilder_seed_and_recursive_formula(self):
        prices = pd.Series([10., 11., 10., 12., 11., 13.])
        out = wilder_rsi(prices, 3)
        self.assertTrue(out.iloc[:3].isna().all())
        self.assertAlmostEqual(out.iloc[3], 75.)
        self.assertAlmostEqual(out.iloc[4], 54.54545454545)
        pd.testing.assert_series_equal(out, calc_rsi(prices, 3))
        self.assertEqual(wilder_rsi(pd.Series(range(20), dtype=float)).iloc[-1], 100.)
        self.assertEqual(wilder_rsi(pd.Series([1.]*20)).iloc[-1], 50.)
        self.assertEqual(wilder_rsi(pd.Series(range(20, 0, -1), dtype=float)).iloc[-1], 0.)

    def test_intraday_volume_uses_source_elapsed_and_complete_baseline(self):
        q = quote()
        # 12:45 ET is half the 6.5-hour session. 100 / (200 * 0.5) = 1.
        self.assertAlmostEqual(md.quote_volume_ratio(q, 200., at("2026-09-29T17:00:00+00:00")), 1.)
        self.assertIsNone(md.quote_volume_ratio(q, 200., at("2026-09-30T17:00:00+00:00")))
        self.assertIsNone(md.quote_volume_ratio(quote(None), 200.))
        self.assertIsNone(md.relative_volume(None, 200.))

    def test_filter_all_new_fields_missing_values_and_sorting(self):
        rows = [dict(ticker="A", price=10., change_pct=2., volume=200.,
                     volume_ratio=1.5, rsi=40., forward_pe=12., market_cap=1e10),
                dict(ticker="B", price=20., change_pct=3., volume=100.,
                     volume_ratio=None, rsi=60., forward_pe=20., market_cap=2e10)]
        for key in ("price", "change_pct", "volume", "volume_ratio", "rsi", "forward_pe"):
            self.assertEqual(filter_records(rows, {key: (rows[0][key], rows[0][key])})[0]["ticker"], "A")
        self.assertEqual(filter_records(rows, {}, "change_pct")[0]["ticker"], "B")
        self.assertEqual(filter_records(rows, {}, "rsi", False)[0]["ticker"], "A")
        self.assertEqual(len(filter_records(rows, {"volume_ratio": (1., None)})), 1)

    def test_scan_reports_every_failed_symbol_and_progress(self):
        def fetch(ticker, force):
            if ticker == "FAIL":
                raise ValueError("source unavailable")
            return {"ticker": ticker}
        progress = []
        with patch("screener_engine.build_record", side_effect=fetch):
            records, failures = scan(["OK", "FAIL", "OK2"], lambda d, t, s: progress.append((d, t)), batch_size=2)
        self.assertEqual(len(records), 2)
        self.assertEqual(list(failures), ["FAIL"])
        self.assertEqual(progress[-1], (3, 3))

    def test_social_failure_is_not_neutral(self):
        import data_fetcher as df
        df.get_social_sentiment.clear()
        with patch("data_fetcher.requests.get", side_effect=OSError("offline")):
            result = df.get_social_sentiment("TEST_OFFLINE")
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "無數據")
        df.get_social_sentiment.clear()


if __name__ == "__main__":
    unittest.main()
