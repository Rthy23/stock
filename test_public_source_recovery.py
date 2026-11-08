"""Recovery tests use isolated caches; real-source rendering is verified separately."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

import public_data_cache as cache


class PublicRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patcher = patch.object(cache, "CACHE_DIR", self.root)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_failed_and_empty_fetch_preserve_success_date_and_decimals(self):
        first = cache.last_success("holdings", lambda: {"shares": Decimal("123.5")})
        for loader in (Mock(side_effect=ValueError("offline")), lambda: None):
            recovered = cache.last_success("holdings", loader, force=True)
            self.assertTrue(recovered["stale"])
            self.assertTrue(recovered["cache_hit"])
            self.assertEqual(recovered["fetched_at"], first["fetched_at"])
            self.assertEqual(recovered["data"]["shares"], Decimal("123.5"))
            self.assertEqual(cache.read_success("holdings")["data"], first["data"])

    def test_no_live_and_no_cache_raises(self):
        with self.assertRaises(ValueError):
            cache.last_success("empty", lambda: [])

    def test_transient_http_retry_uses_backoff_but_403_does_not(self):
        failed = Mock(status_code=503, headers={})
        good = Mock(status_code=200)
        with patch("public_data_cache.requests.get", side_effect=[failed, failed, good]) as get, \
             patch("public_data_cache.time.sleep") as sleep:
            self.assertIs(cache.request_get("https://source.invalid"), good)
        self.assertEqual(get.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [.5, 1.])
        denied = Mock(status_code=403)
        denied.raise_for_status.side_effect = requests.HTTPError("403")
        with patch("public_data_cache.requests.get", return_value=denied) as get, \
             patch("public_data_cache.time.sleep") as sleep, self.assertRaises(requests.HTTPError):
            cache.request_get("https://source.invalid")
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_retry_call_does_not_substitute_a_failed_result(self):
        fetch = Mock(side_effect=[ValueError(), ValueError(), {"live": 1}])
        with patch("public_data_cache.time.sleep"):
            self.assertEqual(cache.retry_call(fetch), {"live": 1})
        self.assertEqual(fetch.call_count, 3)

    def test_ark_without_database_still_loads_official_source_and_cached_failure(self):
        from ark_holdings import parse_holdings, ArkStorageError, ArkHoldingsError
        from page_modules.ark_holdings import load_ark_view
        from test_ark_holdings import official_csv
        snapshot = parse_holdings(official_csv())
        with patch("page_modules.ark_holdings.CACHE_DIR", self.root), \
             patch("ark_holdings.PostgresArkStore.status", side_effect=ArkStorageError("DB missing")), \
             patch("ark_holdings.fetch_holdings", return_value=snapshot):
            first = load_ark_view()
        self.assertEqual(first["current"]["date"], snapshot["date"])
        self.assertIn("不能保證跨重新發布保留", first["storage_note"])
        # Expire the failed-source retry clock without changing the snapshot date.
        from ark_holdings import FileArkStore
        store = FileArkStore(self.root / "ark")
        store.save_status({"checked_at": (datetime.now(timezone.utc)-timedelta(days=2)).isoformat(),
                           "error": None, "unchanged_on_refresh": False})
        with patch("page_modules.ark_holdings.CACHE_DIR", self.root), \
             patch("ark_holdings.PostgresArkStore.status", side_effect=ArkStorageError("DB missing")), \
             patch("ark_holdings.fetch_holdings", side_effect=ArkHoldingsError("offline")):
            recovered = load_ark_view()
        self.assertEqual(recovered["current"], first["current"])
        self.assertIn("offline", recovered["error"])

    def test_pre_post_and_closed_polling_cadence(self):
        for label, interval in (("盤前", 60), ("盤後", 60), ("休市", 300)):
            now = datetime.now(timezone.utc)
            app = AppTest.from_string("from market_ui import render_market_header\nrender_market_header()")
            app.session_state["quote_request_at"] = (now-timedelta(seconds=interval+1)).isoformat()
            def fetched(tickers, force=False):
                import streamlit as st
                st.session_state["quote_request_at"] = now.isoformat()
            with patch("market_ui.market_status", return_value={"label": label, "open": False, "now": now}), \
                 patch("market_ui._refresh", side_effect=fetched) as refresh:
                app.run()
            self.assertFalse(app.exception)
            refresh.assert_called_once()
            self.assertTrue(any(f"每 {interval} 秒" in c.value for c in app.caption))

    def test_feed_parser_requires_real_source_dates_and_links(self):
        from industry_sources import fetch_articles
        response = Mock(content=b"""<rss><channel>
          <item><title>Real research</title><link>https://publisher.org/article</link>
          <pubDate>Fri, 06 Nov 2026 14:00:00 GMT</pubDate><description>&lt;b&gt;Source text&lt;/b&gt;</description></item>
          <item><title>Undated</title><link>https://publisher.org/undated</link></item>
        </channel></rss>""")
        with patch("industry_sources.request_get", return_value=response):
            rows = fetch_articles("https://publisher.org/rss")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["published_at"], "2026-11-06T14:00:00+00:00")
        self.assertEqual(rows[0]["summary"], "Source text")

    def test_recommendation_ui_keeps_missing_event_targets_as_dash(self):
        import pandas as pd
        script = ("import streamlit as st\nfrom analyst_views import render_recommendation_details\n"
                  "render_recommendation_details(st.session_state['data'], 'TEST')")
        app = AppTest.from_string(script)
        app.session_state["data"] = {
            "recs_df": pd.DataFrame([{"period": "0m", "strongBuy": 2, "buy": 3,
                                      "hold": 1, "sell": 0, "strongSell": 0}]),
            "changes_df": pd.DataFrame([{"GradeDate": "2026-11-06", "Firm": "Actual firm",
                                        "Action": "up", "currentPriceTarget": 0}]),
            "source_meta": {}}
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[-1].value.iloc[0]["本次目標價"], "—")
        self.assertEqual(app.dataframe[-1].value.iloc[0]["動作"], "升級")


if __name__ == "__main__":
    unittest.main()
