"""Regression checks for missingness, neutral scores, comparisons and honest UI."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis import calculate_seven_factors, build_factor_prompt, plot_factor_radar
from research_context import (
    relative_returns, trend_structure, summarize_breadth, breadth_record,
    historical_forward_pe, vix_percentile, valuation_comparison,
)


def history(n=300):
    close = np.linspace(100, 180, n)
    return pd.DataFrame({"Open": close-1, "Close": close, "High": close+2, "Low": close-2,
                         "Volume": np.full(n, 1000)},
                        index=pd.bdate_range("2025-01-01", periods=n, tz="America/New_York"))


def facts():
    return dict(pe_ratio=15, pb_ratio=5, ev_ebitda=12, ps_ratio=3, dividend_yield=1,
                roe=15, roa=8, gross_margin=40, op_margin=20, debt_equity=50,
                current_ratio=2, rev_growth=15, eps_growth=20, fwd_eps=6,
                trailing_eps=5, beta=1.11, short_pct=3, inst_ownership=68.76,
                insider_ownership=5, rec_mean=2.22, financial_period="2026-06-30",
                received_at="2026-11-08T12:00:00+00:00")


class ResearchMethodsTests(unittest.TestCase):
    def test_all_missing_has_no_score_or_hold(self):
        f = calculate_seven_factors({}, None, {})
        self.assertIsNone(f["composite"])
        self.assertEqual(f["signal"], "數據不足")
        self.assertEqual(f["completeness"], 0)
        self.assertIsNone(f["Value"]["score"])
        self.assertEqual(f["Value"]["effective_weight"], 0)
        self.assertIn("數據不足", build_factor_prompt("TEST", "Test", f))
        self.assertTrue(plot_factor_radar(f).data)

    def test_neutral_zero_is_valid_and_missing_excluded(self):
        f = calculate_seven_factors({}, None, {"pe_ratio": 15, "pb_ratio": 5})
        self.assertEqual(f["Value"]["score"], 1.5)  # valid +3 and valid zero
        self.assertEqual(f["Value"]["valid_count"], 2)
        self.assertEqual(f["Value"]["items"][1]["score"], 0)
        self.assertIsNone(f["Value"]["items"][2]["score"])
        self.assertIsNone(f["composite"])  # too few fields, not HOLD

    def test_missing_group_reweights_and_full_model_weights_unchanged(self):
        raw = facts()
        f = calculate_seven_factors({}, history(), raw)
        self.assertEqual(f["completeness"], 1)
        self.assertEqual(f["Growth"]["effective_weight"], .20)
        for name in ("pe_ratio", "pb_ratio", "ev_ebitda", "ps_ratio", "dividend_yield"):
            raw[name] = None
        f = calculate_seven_factors({}, history(), raw)
        self.assertIsNone(f["Value"]["score"])
        self.assertAlmostEqual(f["Growth"]["effective_weight"], .2/.85)
        keys = ("Momentum", "Value", "Quality", "Growth", "Volatility", "Sentiment", "Macro")
        self.assertAlmostEqual(sum(f[k]["effective_weight"] for k in keys), 1)
        expected = sum(f[k]["score"] * f[k]["effective_weight"]
                       for k in keys if f[k]["score"] is not None)
        self.assertEqual(f["composite"], round(expected, 2))

    def test_nonfinite_is_missing_and_duplicate_inputs_removed(self):
        raw = facts()
        raw["roe"] = float("nan")
        raw["roa"] = float("inf")
        f = calculate_seven_factors({}, history(), raw)
        self.assertIsNone(f["Quality"]["items"][0]["score"])
        self.assertIsNone(f["Quality"]["items"][1]["score"])
        groups = [g for g in f.values() if isinstance(g, dict) and "items" in g]
        for value in (2.22, 1.11, 68.76):
            self.assertEqual(sum(i["value_raw"] == value for g in groups for i in g["items"]), 1)
        self.assertTrue(all("input_time" in i for g in groups for i in g["items"]))

    def test_relative_returns_use_full_common_intervals(self):
        stock = history()
        benchmark = history()
        benchmark["Close"] = 100.
        result = relative_returns(stock, benchmark)
        self.assertAlmostEqual(result[1], (180/stock.Close.iloc[-22]-1)*100)
        self.assertIsNone(relative_returns(stock.tail(20), benchmark)[1])
        self.assertEqual(relative_returns(stock, stock), {1: 0., 3: 0., 6: 0.})

    def test_trend_uses_slopes_and_honest_non_daily_label(self):
        result = trend_structure(history(), "1h")
        self.assertEqual(result["unit"], "根 K 線")
        self.assertGreater(result[200]["slope"], 0)
        self.assertIn("上升", result["trend"])
        self.assertEqual(trend_structure(history(30))["trend"], "數據不足")

    def test_breadth_excludes_stale_sessions_and_missing_ma(self):
        h = history()
        records = [breadth_record("A", h), breadth_record("B", h.tail(100)),
                   breadth_record("OLD", h.iloc[:-1])]
        result = summarize_breadth(records, 5)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["coverage"], .4)
        self.assertEqual(result["count200"], 1)
        self.assertEqual(result["above200"], 1)
        self.assertEqual(result["advances"], 2)

    def test_vix_preceding_year_percentile_no_lookahead(self):
        series = pd.Series(np.arange(260, dtype=float))
        ranked = vix_percentile(series)
        self.assertTrue(ranked.iloc[:252].isna().all())
        self.assertEqual(ranked.iloc[-1], 100)
        self.assertEqual(vix_percentile(pd.Series([20.]*253)).iloc[-1], 50)
        extended = pd.concat([series, pd.Series([1e8])], ignore_index=True)
        self.assertEqual(vix_percentile(extended).iloc[259], ranked.iloc[259])

    def test_forward_pe_parser_accepts_yahoo_labels_not_trailing_or_current(self):
        table = pd.DataFrame({"Current": [30, 25], "9/30/2026": [28, 23],
                              "6/30/2026": [27, 24]},
                             index=["Trailing P/E", "Forward P/E"])
        series = historical_forward_pe(table)
        self.assertEqual(sorted(series.tolist()), [23, 24])
        self.assertTrue(historical_forward_pe(table.iloc[:1]).empty)

    def test_sector_valuation_median_positive_sample_and_history(self):
        valuation_comparison.clear()
        table = pd.DataFrame({"9/30/2026": [23], "6/30/2026": [24]}, index=["Forward P/E"])
        def fact(t):
            return {"forward_pe": {"A": 10, "B": 20, "C": 30, "D": 40, "E": 50, "NEG": -1}[t],
                    "received_at": "2026-11-08T12:00:00+00:00"}
        with patch("research_context.get_sp500", return_value={
            "sectors": {t: "Information Technology" for t in ("A", "B", "C", "D", "E", "NEG")}}), \
             patch("research_context.get_fundamentals", side_effect=fact), \
             patch("research_context.yf.Ticker") as ticker:
            ticker.return_value.get_valuation_measures.return_value = table
            result = valuation_comparison("A", "Technology")
        self.assertEqual(result["median"], 30)
        self.assertEqual(result["sample"], 5)
        self.assertEqual(result["pool"], 6)
        self.assertAlmostEqual(result["premium"], (10/30-1)*100)
        self.assertEqual(len(result["history"]), 2)
        valuation_comparison.clear()

    def test_stocktwits_uses_effective_sample_and_keeps_source_times(self):
        import data_fetcher as df
        df.get_social_sentiment.clear()
        payload = {"messages": [
            {"created_at": "2026-11-08T10:00:00Z", "entities": {"sentiment": {"basic": "Bullish"}}},
            {"created_at": "2026-11-08T10:01:00Z", "entities": {"sentiment": {"basic": "Bearish"}}},
        ] + [{"created_at": "2026-11-08T10:02:00Z", "entities": {}}]*28}
        response = Mock(ok=True)
        response.json.return_value = payload
        with patch("data_fetcher.requests.get", return_value=response):
            result = df.get_social_sentiment("TEST")
        self.assertEqual(result["score"], 50)
        self.assertEqual(result["tagged_count"], 2)
        self.assertEqual(result["total"], 30)
        self.assertIn("10:01", result["source_time"])
        self.assertNotIn("10:02", result["source_time"])
        df.get_social_sentiment.clear()

    def test_stock_context_ui_keeps_missing_visible(self):
        script = ("from research_ui import render_stock_context\n"
                  "render_stock_context('TEST', {'sector':'Unknown'}, None, '1h')")
        with patch("research_ui.earnings_event", return_value={"start": None}), \
             patch("research_ui.get_history", side_effect=ValueError("offline")):
            app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("根 K 線", captions)
        self.assertIn("資料不足", captions)

    def test_market_context_ui_breadth_denominators(self):
        app = AppTest.from_string("from research_ui import render_market_context\nrender_market_context()")
        app.session_state["market_breadth"] = {
            "summary": summarize_breadth([breadth_record("A", history())], 503),
            "errors": {"FAIL": "offline"}, "received_at": "2026-11-08T12:00:00Z"}
        app.run()
        self.assertFalse(app.exception)
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("1/503", captions)
        self.assertIn("有效分母 1", captions)

    def test_full_stock_page_complete_and_empty_factor_results(self):
        from test_market_upgrade import quote
        h = history()
        q = quote("2026-11-06T21:00:00+00:00")
        stock = dict(ticker="TEST", name="Test", price=180., market_cap=1e11,
                     pe_ratio=15., net_margin=.2, revenue_growth=.2, eps=5,
                     beta=1, sector="Technology", dividend_yield=.01, quote=q)
        bm = dict(ticker="VOO", label="VOO", price=180, sma50=170, sma200=150,
                  golden_cross=True, hist=h, perf_1y=20)
        sentiment = {"combined": None, "news_score": None, "social_score": None, "social_data": {}}
        ad = {"target_mean": 250., "target_median": 200., "target_low": 150.,
              "target_high": 300., "num_analysts": None, "recommendation": "buy",
              "rec_mean": 2, "recs_df": None}
        with ExitStack() as context:
            for target, value in [
                ("data_fetcher.get_market_benchmark", bm),
                ("data_fetcher.get_combined_sentiment", sentiment),
                ("data_fetcher.get_analyst_data", ad),
                ("data_fetcher.get_stock_news", []),
                ("data_fetcher.get_historical_data", h),
                ("research_ui.get_history", h),
                ("research_ui.earnings_event", {"start": None}),
                ("market_ui.get_quote", q),
            ]:
                context.enter_context(patch(target, return_value=value))
            for empty in (True, False):
                app = AppTest.from_file("app.py", default_timeout=90)
                app.session_state["nav_page"] = "🔬 個股診斷 (Micro)"
                app.session_state["diag_ticker"] = "TEST"
                app.session_state["diag_stock_info"] = stock
                app.session_state["diag_hist"] = h
                app.session_state["factor7_TEST"] = (
                    calculate_seven_factors({}, None, {}) if empty else
                    calculate_seven_factors(stock, h, facts()))
                app.session_state["factor7_data_TEST"] = {} if empty else facts()
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(any("渲染異常" in x.value or "MODULE_ERROR" in x.value
                                     for x in app.error))
                captions = " ".join(c.value for c in app.caption)
                self.assertIn("距目標中位數上行空間：+11.1%", captions)
                self.assertIn("數據完整度", captions)
                rendered = " ".join(m.value for m in app.markdown)
                self.assertIn("共 — 位分析師", rendered)
                if empty:
                    self.assertIn("訊號偏向：數據不足", rendered)

    def test_macro_ui_low_effective_sample_even_with_many_messages(self):
        sentiment = {"combined": 50., "news_score": 50., "social_score": 50.,
                     "news_data": {"sample_count": 7, "source_time": "2026-11-08"},
                     "social_data": {"bull_count": 1, "bear_count": 1, "total": 30,
                                     "tagged_count": 2, "source_time": "2026-11-08"}}
        script = ("from ui_components import render_macro_sentiment_dashboard\n"
                  "render_macro_sentiment_dashboard()")
        with patch("ui_components.get_combined_sentiment", return_value=sentiment), \
             patch("ui_components.get_vix_history", return_value=pd.DataFrame()):
            app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("有效標籤樣本少於 10 則（2 則）", captions)
        self.assertIn("樣本 7 篇", captions)


if __name__ == "__main__":
    unittest.main()
