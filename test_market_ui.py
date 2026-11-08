"""Render and interact with Streamlit, using clearly isolated source fixtures."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from test_market_upgrade import quote


class MarketUITests(unittest.TestCase):
    def test_closed_session_no_automatic_quote_fetch(self):
        now = datetime.now(timezone.utc)
        status = {"label": "休市", "open": False, "now": now}
        app = AppTest.from_string("from market_ui import render_market_header\nrender_market_header()")
        app.session_state["quote_request_at"] = (now - timedelta(minutes=10)).isoformat()
        with patch("market_ui.market_status", return_value=status), patch("market_ui._refresh") as refresh:
            app.run()
        self.assertFalse(app.exception)
        refresh.assert_not_called()
        self.assertTrue(any("停止自動" in c.value for c in app.caption))

    def test_open_session_refreshes_then_countdown(self):
        now = datetime.now(timezone.utc)
        app = AppTest.from_string("from market_ui import render_market_header\nrender_market_header()")
        app.session_state["quote_request_at"] = (now - timedelta(seconds=61)).isoformat()
        def fetched(tickers, force=False):
            import streamlit as st
            st.session_state["quote_request_at"] = datetime.now(timezone.utc).isoformat()
        with patch("market_ui.market_status", return_value={"label": "盤中", "open": True, "now": now}), \
             patch("market_ui._refresh", side_effect=fetched) as refresh:
            app.run()
        self.assertFalse(app.exception)
        refresh.assert_called_once()
        self.assertTrue(any("倒數" in c.value for c in app.caption))

    def test_force_button_bypasses_quote_and_application_caches(self):
        from market_ui import _refresh
        import streamlit as st
        with patch("market_ui.st.cache_data.clear") as streamlit_clear, \
             patch("market_ui.clear_market_cache") as provider_clear, \
             patch("market_ui.get_quote", return_value=quote()) as fetch:
            _refresh(["TEST"], force=True)
        streamlit_clear.assert_called_once()
        provider_clear.assert_called_once()
        fetch.assert_called_once_with("TEST", force=True)

    def test_screener_renders_coverage_new_fields_and_all_actions(self):
        row = {"ticker": "TEST", "name": "Fixture", "market_cap": 1e10,
               "net_margin": .2, "revenue_growth": .2, "pe_ratio": 15.,
               "forward_pe": None, "price": 100., "rsi": 40., "volume": 100.,
               "volume_ratio": None, "change_pct": 1., "quote": quote(),
               "partial_errors": [], "eps": None}
        universe = {"symbols": ["TEST", "FAIL"], "source": "Fixture", "received_at": "fixture"}
        script = ("from screener_ui import render_screener\n"
                  "render_screener((1,10,15,30), lambda r: None, lambda w: None)")
        app = AppTest.from_string(script)
        app.session_state["scan_records"] = [row]
        app.session_state["scan_failures"] = {"FAIL": "offline"}
        app.session_state["scan_universe"] = universe
        with patch("screener_ui.get_sp500", return_value=universe):
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("成功 1 支／失敗 1 支" in v.value for v in app.info))
            self.assertTrue(any("覆蓋率：50.00%" in v.value for v in app.info))
            self.assertEqual(app.dataframe[0].value.iloc[0]["Forward PE"], "資料不足")
            labels = [button.label for button in app.button]
            for label in ("個股診斷", "切換收藏", "送前五支到回測", "開始掃描 S&P 500"):
                self.assertIn(label, labels)
            app.checkbox(key="filter_on_forward_pe").check().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("沒有符合" in v.value for v in app.info))


if __name__ == "__main__":
    unittest.main()
