"""ARK official daily data: fail closed and compare only saved distinct dates."""

import csv
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from ark_holdings import (
    ArkHoldingsError, ArkStorageError, SOURCE_URL, compare_snapshots,
    get_ark_holdings, parse_holdings,
)
from page_modules.ark_holdings import render_ark_holdings
from kol_whitelist import PICKS_DATA, build_consensus_table, call_gemini_consensus, score_picks
import picks_store
import user_config


def official_csv(day="09/28/2026", *, shares="1,000", cusip="88160R101",
                 header=None, fund="ARKK"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header or [
        "date", "fund", "company", "ticker", "cusip", "shares",
        "market value ($)", "weight (%)",
    ])
    writer.writerow([day, fund, "TESLA INC", "TSLA", cusip, shares, "$42,000.00", "9.12%"])
    writer.writerow(["Investors should carefully consider the risks of an ARK ETF."])
    return output.getvalue().encode()


class ArkHoldingsTests(unittest.TestCase):
    def test_real_csv_shape_and_legal_footer(self):
        data = parse_holdings(official_csv(), today=datetime(2026, 9, 28, 17, tzinfo=timezone.utc))
        self.assertEqual((data["date"], data["fund"], data["source_url"]),
                         ("2026-09-28", "ARKK", SOURCE_URL))
        self.assertEqual(len(data["holdings"]), 1)
        self.assertEqual(data["holdings"][0]["shares"], "1000")

    def test_bad_source_content_is_not_presented_as_holdings(self):
        variants = [
            b"", b"<html>Cloudflare challenge</html>",
            official_csv(header=["wrong", "columns"]),
            official_csv(day="broken"),
            official_csv(fund="ARKF"),
            official_csv(shares="not-number"),
            official_csv(day="09/27/2026"),
        ]
        for content in variants:
            with self.subTest(content=content[:40]), self.assertRaises(ArkHoldingsError):
                parse_holdings(content, today=datetime(2026, 9, 28, 17, tzinfo=timezone.utc))

    def test_first_day_repeat_date_and_two_real_dates(self):
        monday = datetime(2026, 9, 28, 17, tzinfo=timezone.utc)
        tuesday = datetime(2026, 9, 29, 17, tzinfo=timezone.utc)
        wednesday = datetime(2026, 9, 30, 17, tzinfo=timezone.utc)
        fetched = []

        def fetch(day, shares):
            fetched.append(day)
            return parse_holdings(official_csv(day, shares=shares), today=wednesday)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = get_ark_holdings(cache_dir=directory, now=monday,
                                     fetcher=lambda: fetch("09/28/2026", "1,000"))
            self.assertIsNone(first["previous"])
            self.assertIn("無法比較", first["comparison_note"])
            self.assertEqual(len(list(directory.glob("ARKK_*.json"))), 1)

            # Same UTC day is a cache hit, not a second "yesterday" snapshot.
            second = get_ark_holdings(cache_dir=directory, now=monday,
                                      fetcher=lambda: self.fail("cache must be used"))
            self.assertEqual(second["current"]["date"], "2026-09-28")
            self.assertEqual(len(fetched), 1)

            unchanged = get_ark_holdings(cache_dir=directory, now=tuesday,
                                         fetcher=lambda: fetch("09/28/2026", "1,000"))
            self.assertTrue(unchanged["unchanged_on_refresh"])
            self.assertIsNone(unchanged["previous"])
            self.assertEqual(len(list(directory.glob("ARKK_*.json"))), 1)

            changed = get_ark_holdings(cache_dir=directory, now=wednesday,
                                       fetcher=lambda: fetch("09/30/2026", "1,250"))
            self.assertEqual(changed["previous"]["date"], "2026-09-28")
            self.assertEqual(changed["current"]["date"], "2026-09-30")
            self.assertEqual(changed["changes"][0]["cusip"], "88160R101")
            self.assertEqual(changed["changes"][0]["change_shares"], "250")
            self.assertEqual(len(list(directory.glob("ARKK_*.json"))), 2)

    def test_source_failure_keeps_last_success_and_retry_is_throttled(self):
        monday = datetime(2026, 9, 28, 17, tzinfo=timezone.utc)
        tuesday = datetime(2026, 9, 29, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            get_ark_holdings(cache_dir=directory, now=monday,
                             fetcher=lambda: parse_holdings(official_csv(), today=monday))
            error = get_ark_holdings(cache_dir=directory, now=tuesday,
                                     fetcher=lambda: parse_holdings(b"<html>oops</html>"))
            self.assertEqual(error["current"]["date"], "2026-09-28")
            self.assertIn("欄位格式已變更", error["error"])
            self.assertEqual(len(list(directory.glob("ARKK_*.json"))), 1)
            retried = get_ark_holdings(cache_dir=directory, now=tuesday,
                                       fetcher=lambda: self.fail("retry must wait"))
            self.assertEqual(retried["error"], error["error"])

    def test_no_previous_or_missing_identifiers_never_infers_trades(self):
        monday = parse_holdings(official_csv(cusip=""), today=datetime(
            2026, 9, 30, 17, tzinfo=timezone.utc))
        tuesday = parse_holdings(official_csv(day="09/29/2026", shares="1,200"), today=datetime(
            2026, 9, 30, 17, tzinfo=timezone.utc))
        changes, note = compare_snapshots(tuesday, monday)
        self.assertEqual(changes, [])
        self.assertIn("無法可靠比較", note)
        self.assertIn("無法比較", compare_snapshots(tuesday, None)[1])

    def test_older_official_date_keeps_last_success(self):
        tuesday = datetime(2026, 9, 29, 17, tzinfo=timezone.utc)
        wednesday = datetime(2026, 9, 30, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            get_ark_holdings(cache_dir=directory, now=tuesday,
                             fetcher=lambda: parse_holdings(
                                 official_csv(day="09/29/2026"), today=tuesday))
            result = get_ark_holdings(cache_dir=directory, now=wednesday,
                                      fetcher=lambda: parse_holdings(
                                          official_csv(day="09/28/2026"), today=wednesday))
            self.assertIn("日期早於", result["error"])
            self.assertEqual(result["current"]["date"], "2026-09-29")
            self.assertEqual(len(list(directory.glob("ARKK_*.json"))), 1)

    def test_database_failure_never_falls_back_to_local_snapshot(self):
        # Even with a legacy snapshot present, a DB failure must be visible.
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = parse_holdings(official_csv(), today=datetime(
                2026, 9, 28, 17, tzinfo=timezone.utc))
            path = Path(tmp) / "ARKK_2026-09-28.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch("ark_holdings.CACHE_DIR", Path(tmp)), patch(
                "ark_holdings.PostgresArkStore"
            ) as postgres:
                postgres.return_value.status.side_effect = ArkStorageError("資料庫不可用")
                with self.assertRaisesRegex(ArkStorageError, "資料庫不可用"):
                    get_ark_holdings(fetcher=lambda: self.fail("cannot fetch without DB"))

    def test_real_section_labels_and_first_day_no_change_table(self):
        snapshot = parse_holdings(official_csv(), today=datetime(
            2026, 9, 28, 17, tzinfo=timezone.utc))
        with patch("page_modules.ark_holdings.get_ark_holdings", return_value={
            "current": snapshot, "previous": None, "changes": [],
            "comparison_note": "無法比較", "error": None,
            "checked_at": "2026-09-28T17:00:00+00:00",
            "unchanged_on_refresh": False,
        }), patch("page_modules.ark_holdings.st") as ui:
            render_ark_holdings()
        self.assertIn("資料來源：ARK Invest 官方每日持倉揭露", ui.info.call_args_list[0].args[0])
        self.assertIn("2026-09-28", ui.markdown.call_args_list[-1].args[0])
        self.assertTrue(ui.dataframe.called)
        self.assertFalse(ui.error.called)

    def test_cathie_legacy_picks_are_preserved_but_never_ranked_or_added(self):
        self.assertFalse({"cathie_wood"} & {
            analyst_id for p in score_picks(PICKS_DATA) for analyst_id in p["kol_ids"]
        })
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                picks_store, "PICKS_FILE", os.path.join(tmp, "picks.json")):
            records = [
                {"kol_id": "cathie_wood", "ticker": "TSLA", "date": "2026-09-28",
                 "argument_quality": 3, "thesis": "invented"},
                {"kol_id": "howard_marks", "ticker": "HYG", "date": "2026-09-28",
                 "argument_quality": 3, "thesis": "unverified"},
            ]
            picks_store.save_picks(records)
            self.assertEqual(picks_store.get_picks_with_status(), [])
            self.assertEqual(build_consensus_table(), [])
            self.assertEqual([r["ticker"] for r in build_consensus_table(
                picks=records, whitelist=[{"id": "cathie_wood", "rep": 5, "name": "Cathie Wood"}]
            )], [])
            with self.assertRaises(ValueError):
                picks_store.add_pick(records[0])
            with self.assertRaises(ValueError):
                picks_store.update_pick(0, {"ticker": "NVDA"})
            picks_store.purge_expired_picks(days=0)
            with open(picks_store.PICKS_FILE, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)[0], records[0])
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                picks_store, "PICKS_FILE", os.path.join(tmp, "new.json")):
            self.assertNotIn("cathie_wood",
                             {p["kol_id"] for p in picks_store.load_picks()})
        with patch.object(user_config, "load_config", return_value={
            "analyst_whitelist": ["@CathieWood", "@howard_marks"]}):
            self.assertEqual(user_config.load_kol_whitelist(), ["@howard_marks"])
            self.assertFalse(user_config.add_kol("@cathie_wood")[0])

    def test_cathie_is_stripped_even_from_external_gemini_payload(self):
        mixed = {"ticker": "TSLA", "experts": ["Cathie Wood", "Howard Marks"],
                 "kol_ids": ["cathie_wood", "howard_marks"],
                 "theses": ["invented personal recommendation", "other demo"],
                 "consensus": 2}
        with patch("gemini_helper.call_gemini_cached",
                   return_value='{"summary":"x","confidence":3,"reason":"y"}') as ai:
            result = call_gemini_consensus([mixed], "test")
        self.assertEqual(result, [])
        ai.assert_not_called()


if __name__ == "__main__":
    unittest.main()