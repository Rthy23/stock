"""C-class commentary is not published until a manual source-checking workflow exists."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from kol_config import ANALYST_DIRECTORY, COMMENTARY_IDS
from kol_whitelist import (PICKS_DATA, WHITELIST, build_consensus_table,
                           call_gemini_consensus, render_industry_commentary_reference,
                           score_picks)
import picks_store
import user_config
from sec_edgar_fetcher import MANAGER_IDS


def pick(kol_id, ticker="AAPL", source=False):
    data = {"kol_id": kol_id, "ticker": ticker, "date": "2026-09-28",
            "argument_quality": 3, "thesis": "unverified example"}
    if source:
        data["source_url"] = "https://publisher.example/article"
    return data


class CommentaryQuarantineTests(unittest.TestCase):
    def test_all_14_commentary_names_are_in_the_directory_but_not_scoring_whitelist(self):
        self.assertEqual(len(COMMENTARY_IDS), 14)
        self.assertLessEqual(COMMENTARY_IDS, {a["id"] for a in ANALYST_DIRECTORY})
        self.assertFalse(COMMENTARY_IDS & {a["id"] for a in WHITELIST})

    def test_static_and_explicit_legacy_picks_are_excluded_from_both_rankings(self):
        ranked = score_picks(picks=PICKS_DATA)
        self.assertFalse(COMMENTARY_IDS & set(
            analyst_id for item in ranked for analyst_id in item["kol_ids"]))
        records = [pick("dan_ives", "TSLA"), pick("goldman_global_research", "SPY", True),
                   pick("warren_buffett", "KO"), pick("howard_marks", "HYG")]
        for whitelist in (ANALYST_DIRECTORY, WHITELIST):
            ranked = build_consensus_table(picks=records, whitelist=whitelist)
            self.assertEqual(ranked, [])
        alias = {"id": "@DanIves", "name": "Dan Ives", "rep": 5}
        self.assertEqual(score_picks(picks=[pick("@DanIves")], whitelist=[alias]), [])

    def test_old_json_is_preserved_and_management_index_remains_original(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                picks_store, "PICKS_FILE", os.path.join(tmp, "old.json")):
            records = [pick("dan_ives"), pick("warren_buffett"), pick("howard_marks", "HYG"),
                       pick("wsj_markets", "MSFT", True)]
            picks_store.save_picks(records)
            visible = picks_store.get_picks_with_status()
            self.assertEqual(visible, [])
            self.assertEqual(score_picks(), [])
            with self.assertRaises(ValueError):
                picks_store.add_pick(pick("dan_ives"))
            with self.assertRaises(ValueError):
                picks_store.add_pick(pick("@DanIves"))
            with self.assertRaises(ValueError):
                picks_store.update_pick(0, {"ticker": "NEW"})
            with self.assertRaises(ValueError):
                picks_store.update_pick(2, {"kol_id": "wsj_markets"})
            picks_store.purge_expired_picks(days=0)
            with open(picks_store.PICKS_FILE, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.assertEqual([item["kol_id"] for item in raw if item["kol_id"] in
                              COMMENTARY_IDS | MANAGER_IDS],
                             ["dan_ives", "warren_buffett", "wsj_markets"])

    def test_first_install_never_seeds_commentary_or_13f_picks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                picks_store, "PICKS_FILE", os.path.join(tmp, "new.json")):
            records = picks_store.load_picks()
            self.assertEqual(records, [])

    def test_gemini_defensively_skips_untraceable_and_strips_blocked_input(self):
        mixed = {
            "ticker": "AAPL",
            "experts": ["Dan Ives", "Howard Marks"],
            "kol_ids": ["dan_ives", "howard_marks"],
            "theses": ["false statement", "other demo"],
            "consensus": 2,
        }
        with patch("gemini_helper.call_gemini_cached", return_value='{"summary":"x","confidence":3,"reason":"y"}') as ai:
            result = call_gemini_consensus([{"ticker": "TSLA", "experts": ["Dan Ives"],
                                             "theses": ["fake"], "consensus": 1},
                                            mixed], "test")
        self.assertEqual(result, [])
        ai.assert_not_called()

    def test_placeholder_lists_names_only(self):
        with patch("kol_whitelist.st") as ui:
            render_industry_commentary_reference()
        self.assertIn("產業評論參考", ui.markdown.call_args.args[0])
        self.assertIn("此功能開發中，暫不提供", ui.info.call_args.args[0])
        displayed = ui.write.call_args.args[0]
        self.assertIn("Dan Ives", displayed)
        self.assertIn("Morningstar Quantitative Research", displayed)
        self.assertNotIn("AAPL", displayed)

    def test_sidebar_whitelist_cannot_reintroduce_commentary(self):
        with patch.object(user_config, "load_config", return_value={
                "analyst_whitelist": ["@Dan Ives", "@howard_marks", "@blackrock_institute"]}):
            self.assertEqual(user_config.load_kol_whitelist(), ["@howard_marks"])
            self.assertFalse(user_config.add_kol("@dan_ives")[0])
            self.assertFalse(user_config.add_kol("@DanIves")[0])
            self.assertFalse(user_config.add_kol("@GoldmanSachs")[0])


if __name__ == "__main__":
    unittest.main()