"""Synthetic calculations are isolated from real history; never inserted into production/dev."""
import copy
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from scientific_factors import FACTORS, benefit, calibrate, calibrate_reference, recombine
from trade_reference import wilder_atr, trade_reference, position_size
from factor_validation import forward_pairs, validation_report, ic_composite, attach_ic_weighting
from research_alerts import generate_alerts
from research_pipeline import closed_prices

UTC = timezone.utc


def model(value=0):
    return recombine({k:dict(label=k, items=[dict(name=k, metric="momentum_1m", value_raw=value, score=value)],
                            score=value,valid_count=1,total_count=1) for k in FACTORS})


def cohort_rows():
    return [dict(ticker=f"{s}{i:02}",gics_sector=s,factors=model(i+(100 if s=="B" else 0)))
            for s in ("A","B") for i in range(1,11)]


def bars():
    return pd.DataFrame({"Close":[100.]*30,"High":[102.]*30,"Low":[98.]*30,"Volume":[10000.]*30},
                        index=pd.date_range("2026-01-01",periods=30,tz="America/New_York"))


def validation_fixture(alternate=False, groups=12):
    scores, prices = [], []
    for cohort in range(groups):
        entry = datetime(2026,1,5,21,tzinfo=UTC)+timedelta(weeks=cohort)
        for i in range(40):
            ticker=f"C{cohort}T{i}"
            f=model((i-19.5)/4)
            scores.append(dict(ticker=ticker,time=(entry-timedelta(hours=7)).isoformat(),factors=f))
            sign=-1 if alternate and cohort%2 else 1
            prices.extend([dict(ticker=ticker,time=entry.isoformat(),close=100.),
                           dict(ticker=ticker,time=(entry+timedelta(weeks=1)).isoformat(),close=100+i*.1*sign)])
    return scores,prices


class CalibrationTests(unittest.TestCase):
    def test_sector_first_blend_exact_midrank_and_contributions(self):
        result=calibrate(cohort_rows())
        stock=next(r for r in result if r["ticker"]=="A05")
        self.assertAlmostEqual(stock["factors"]["Momentum"]["score"],10*(.7*.45+.3*.225)-5)
        self.assertAlmostEqual(sum(stock["factors"][k]["contribution"] for k in FACTORS),
                               stock["factors"]["composite"])
        self.assertTrue(all(-5<=r["factors"]["composite"]<=5 for r in result))

    def test_zero_is_valid_ties_neutral_and_missing_reweights(self):
        rows=cohort_rows()
        for row in rows:
            row["factors"]["Value"]["items"][0]["value_raw"]=None
            row["factors"]["Momentum"]["items"][0]["value_raw"]=0.
        result=calibrate(rows)[0]["factors"]
        self.assertEqual(result["Momentum"]["score"],0.)
        self.assertIsNone(result["Value"]["score"])
        self.assertEqual(result["Value"]["effective_weight"],0)
        self.assertAlmostEqual(result["Quality"]["effective_weight"],.2/.85)
        self.assertEqual(result["valid_count"],6)

    def test_insufficient_universe_no_hold_no_default_zero(self):
        stock=calibrate(cohort_rows()[:3])[0]["factors"]
        self.assertIsNone(stock["composite"])
        self.assertEqual(stock["signal"],"數據不足")
        self.assertEqual(stock["completeness"],0.)

    def test_unknown_or_small_sector_falls_back_to_universe(self):
        rows=cohort_rows()
        rows[0]["gics_sector"]="tiny"
        item=calibrate(rows)[0]["factors"]["Momentum"]["items"][0]
        self.assertEqual(item["calibration"],"全池（板塊樣本不足）")
        self.assertAlmostEqual(item["score"],10*.025-5)

    def test_raw_orientation_and_nonpositive_multiples(self):
        self.assertIsNone(benefit(dict(metric="pe",value_raw=-3)))
        self.assertEqual(benefit(dict(metric="pe",value_raw=10)),-10)
        self.assertGreater(benefit(dict(metric="rsi",value_raw=60)),benefit(dict(metric="rsi",value_raw=90)))
        self.assertEqual(benefit(dict(metric="momentum_1m",value_raw=0)),0)

    def test_inputs_not_mutated(self):
        rows=cohort_rows()
        original=copy.deepcopy(rows)
        calibrate(rows)
        self.assertEqual(rows,original)

    def test_outside_watchlist_uses_pool_not_itself_and_never_changes_reference(self):
        peers=calibrate(cohort_rows())
        original=copy.deepcopy(peers)
        row=calibrate_reference(dict(ticker="OUT",gics_sector="A",factors=model(1000)),peers)
        self.assertAlmostEqual(row["factors"]["composite"],5.)
        self.assertFalse(row["factors"]["universe_member"])
        self.assertEqual(peers,original)


class TradeTests(unittest.TestCase):
    def test_atr_seed_wilder_update_stop_zone_and_capital_cap(self):
        hist=bars()
        self.assertEqual(wilder_atr(hist),4.)
        hist.loc[hist.index[-1],"High"]=110.
        self.assertAlmostEqual(wilder_atr(hist),(13*4+12)/14)
        plan=trade_reference(bars(),99,101)
        self.assertEqual(plan["stop"],92.)
        self.assertEqual((plan["entry_low"],plan["entry_high"]),(98.,100.))
        self.assertEqual(plan["spread_pct"],2.)
        self.assertEqual(plan["avg_dollar_volume20"],1e6)
        self.assertTrue(plan["low_liquidity"])
        self.assertEqual(position_size(plan,100000,1),125)
        self.assertEqual(position_size(plan,1000,100),10)

    def test_missing_short_history_spread_and_invalid_stop(self):
        self.assertIsNone(trade_reference(None)["stop"])
        self.assertIsNone(trade_reference(bars().head(10))["atr14"])
        self.assertIsNone(trade_reference(bars(),101,99)["spread_pct"])
        self.assertIsNone(position_size(dict(entry=10,stop=-1)))
        missing=bars()
        missing.loc[missing.index[-1],"High"]=np.nan
        self.assertIsNone(wilder_atr(missing))

    def test_calendar_closes_not_midnight_and_early_close(self):
        h=bars().tail(1).copy()
        h.index=pd.DatetimeIndex(["2026-11-27"],tz="America/New_York")
        self.assertEqual(closed_prices("X",h,datetime(2026,11,27,17,tzinfo=UTC)),[])
        values=closed_prices("X",h,datetime(2026,11,27,19,tzinfo=UTC))
        self.assertEqual(values[0][1].hour,18)  # Thanksgiving Friday early close.


class ValidationTests(unittest.TestCase):
    def test_forward_entry_strictly_after_score_calendar_horizons_and_no_future(self):
        f=model(1)
        scores=[dict(ticker="A",time="2026-01-05T22:00:00Z",factors=f)]
        prices=[dict(ticker="A",time=t,close=p) for t,p in [
            ("2026-01-05T21:00:00Z",10),("2026-01-06T21:00:00Z",100),
            ("2026-01-13T21:00:00Z",110),("2026-02-03T21:00:00Z",120),
            ("2026-03-31T20:00:00Z",150),("2026-04-01T20:00:00Z",160)]]
        self.assertEqual(forward_pairs(scores,prices,1,"2026-01-10T00:00:00Z"),[])
        self.assertAlmostEqual(forward_pairs(scores,prices,1,"2026-04-10T00:00:00Z")[0]["return_pct"],10)
        self.assertAlmostEqual(forward_pairs(scores,prices,4,"2026-04-10T00:00:00Z")[0]["return_pct"],20)
        # Calendar target must not lose an hour when US daylight-saving time changes.
        self.assertAlmostEqual(forward_pairs(scores,prices,12,"2026-04-10T00:00:00Z")[0]["return_pct"],50)

    def test_rank_ic_spread_sample_gate_and_ic_weighted_composite(self):
        scores,prices=validation_fixture()
        report=validation_report(scores,prices,"2026-12-31T00:00:00Z")
        one=report[0]
        self.assertEqual(one["cohorts"],12)
        self.assertEqual(one["samples"],480)
        self.assertAlmostEqual(one["ic"],1.)
        self.assertAlmostEqual(one["spread"],3.2)
        self.assertTrue(one["eligible"])
        self.assertAlmostEqual(ic_composite(model(2),report),2.)

    def test_repeated_intraday_scores_do_not_inflate_sample(self):
        scores,prices=validation_fixture()
        report=validation_report(scores+scores+scores,prices,"2026-12-31T00:00:00Z")
        self.assertEqual(report[0]["samples"],480)
        self.assertEqual(report[0]["cohorts"],12)

    def test_small_and_zero_uncertain_ic_are_explicitly_unvalidated(self):
        scores,prices=validation_fixture(groups=3)
        report=validation_report(scores,prices,"2026-12-31T00:00:00Z")
        self.assertIn("尚未驗證",report[0]["status"])
        self.assertIsNone(ic_composite(model(1),report))
        scores,prices=validation_fixture(alternate=True)
        report=validation_report(scores,prices,"2026-12-31T00:00:00Z")
        self.assertIn("效能未證實",report[0]["status"])
        self.assertFalse(report[0]["eligible"])

    def test_ties_and_constant_return_never_manufacture_ic(self):
        scores,prices=validation_fixture()
        for score in scores:
            score["factors"]=model(0)
        report=validation_report(scores,prices,"2026-12-31T00:00:00Z")
        self.assertIsNone(report[0]["ic"])
        self.assertFalse(report[0]["eligible"])

    def test_learned_weights_frozen_at_scoring_not_retroactively_replaced(self):
        scores,prices=validation_fixture()
        report=validation_report(scores,prices,"2026-12-31T00:00:00Z")
        row={"factors":model(2)}
        attach_ic_weighting([row],report,"2026-12-31T00:00:00Z")
        frozen=copy.deepcopy(row["factors"])
        for item in report:
            item["ic"]=-1.
        self.assertEqual(row["factors"],frozen)
        self.assertAlmostEqual(sum(frozen["ic_weights"].values()),1.)
        self.assertEqual(frozen["ic_composite"],2.)

    def test_outside_watchlist_not_added_to_sp500_validation(self):
        scores,prices=validation_fixture()
        for s in scores:
            s["factors"]["universe_member"]=False
        self.assertEqual(forward_pairs(scores,prices,1,"2026-12-31T00:00:00Z"),[])


class AlertWorkerTests(unittest.TestCase):
    def test_score_rs_sentiment_earnings_and_dedup_keys(self):
        now=datetime(2026,11,8,tzinfo=UTC)
        old=dict(records=[dict(ticker="A",factors=model(0),rs1m=-1)],sentiment={"combined":20})
        row=dict(ticker="A",factors=model(2),rs1m=1,
                 earnings={"start":(now+timedelta(days=3)).timestamp()})
        alerts=generate_alerts(old,[row],{"combined":50},{},now)
        self.assertEqual({a["kind"] for a in alerts},{"分數變動","相對強弱穿越","業績臨近","情緒急變"})
        again=generate_alerts(old,[row],{"combined":50},{},now+timedelta(minutes=1))
        self.assertEqual(next(a["key"] for a in alerts if a["kind"]=="業績臨近"),
                         next(a["key"] for a in again if a["kind"]=="業績臨近"))

    def test_missing_values_and_first_run_no_fabricated_changes(self):
        missing=model()
        missing["composite"]=None
        alerts=generate_alerts({"records":[{"ticker":"A","factors":missing}]},
                               [{"ticker":"A","factors":model(3),"rs1m":1}],{"combined":None},{})
        self.assertEqual(alerts,[])

    def test_worker_runs_without_browser_and_respects_advisory_lock(self):
        from research_worker import run_once
        store=MagicMock()
        lock=store.connect.return_value.__enter__.return_value
        lock.execute.return_value.fetchone.return_value=(False,)
        self.assertTrue(run_once(store)["skipped"])
        store.mark_worker.assert_not_called()
        lock.execute.return_value.fetchone.return_value=(True,)
        store.settings.return_value={}
        with patch("research_worker.get_sp500",return_value={"symbols":["A"]}), \
             patch("research_worker.market_sentiment",return_value={"combined":None}), \
             patch("research_worker.scan",return_value=([{"ticker":"A"}],{})), \
             patch("research_worker.score_records",return_value=([{"ticker":"A"}],[])), \
             patch("research_worker.persist_scan",return_value="saved") as save:
            self.assertEqual(run_once(store)["time"],"saved")
            self.assertTrue(save.call_args.kwargs["background"])

    def test_worker_failure_does_not_mark_success(self):
        from research_worker import run_once
        store=MagicMock()
        store.connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value=(True,)
        with patch("research_worker.market_sentiment",return_value={}), \
             patch("research_worker.get_sp500",side_effect=RuntimeError("upstream")):
            with self.assertRaises(RuntimeError):
                run_once(store)
        self.assertIsNotNone(store.mark_worker.call_args.kwargs.get("error"))

    def test_global_due_time_prevents_new_instance_rescanning_immediately(self):
        from research_worker import run_once
        store=MagicMock()
        store.connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value=(True,)
        store.status.return_value={"completed_at":datetime.now(UTC).isoformat()}
        self.assertTrue(run_once(store)["skipped"])
        store.mark_worker.assert_not_called()

    def test_sentiment_failure_preserves_original_success_time_not_a_fake_new_observation(self):
        from research_pipeline import market_sentiment
        old=dict(combined=60.,combined_fetched_at="2026-11-05T00:00:00Z",news_score=50,
                 social_score=70,vix_greed=20.,vix_time="2026-11-04",vix_fetched_at="2026-11-05T00:00:00Z")
        with patch("research_pipeline.get_combined_sentiment",return_value={"combined":None}), \
             patch("research_pipeline.get_vix_history",return_value=pd.DataFrame()):
            value=market_sentiment(old)
        self.assertEqual(value["combined"],60.)
        self.assertEqual(value["combined_fetched_at"],old["combined_fetched_at"])
        self.assertEqual(value["vix_time"],old["vix_time"])
        self.assertTrue(value["combined_stale"])
        self.assertTrue(value["vix_stale"])

    def test_all_failed_scan_does_not_overwrite_last_success(self):
        from research_pipeline import persist_scan
        store=MagicMock()
        with self.assertRaises(ValueError):
            persist_scan([],[],{"symbols":["A"]},{"A":"offline"},store=store)
        store.append.assert_not_called()

    def test_trade_and_contribution_ui_missing_and_valid(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string(
            "from scientific_ui import render_trade_block, render_score_details\n"
            "import streamlit as st\n"
            "render_trade_block('A',st.session_state['plan'],100000.,1.)\n"
            "render_score_details(st.session_state['row'])")
        app.session_state["plan"]=trade_reference(bars())
        app.session_state["row"]={"factors":model(0)}
        with patch("scientific_ui.research_state",return_value={"report":[]}):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value),7)
        self.assertTrue(any("尚未驗證／效能未證實" in c.value for c in app.caption))


if __name__=="__main__":
    unittest.main()
