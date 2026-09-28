"""SEC fixtures and safety boundaries; no network required."""

from decimal import Decimal
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import sec_edgar_fetcher as sec
from kol_whitelist import score_picks
import picks_store


def xml_table(*rows: str) -> bytes:
    return ('<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">'
            + ''.join(rows) + '</informationTable>').encode()


def row(cusip="123456789", shares="10", value="500", issuer="Example Inc",
        title="COM", option="") -> str:
    return (f"<infoTable><nameOfIssuer>{issuer}</nameOfIssuer><titleOfClass>{title}</titleOfClass>"
            f"<cusip>{cusip}</cusip><value>{value}</value>"
            f"<shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt>"
            "<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>"
            f"<putCall>{option}</putCall></infoTable>")


class SecTests(unittest.TestCase):
    def test_parser_aggregates_duplicate_same_security(self):
        result = sec.parse_information_table(xml_table(row() + row(shares="4", value="100")))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["quantity"], Decimal(14))
        self.assertEqual(result[0]["value_reported"], Decimal(600))

    def test_parser_does_not_conflate_option_and_shares(self):
        result = sec.parse_information_table(xml_table(row() + row(option="PUT")))
        self.assertEqual(len(result), 2)

    def test_missing_and_wrong_xml_fail_closed(self):
        for payload in (b"bad xml", b"<form13F/>", xml_table(row(shares="-1"))):
            with self.subTest(payload=payload), self.assertRaises(sec.EdgarError):
                sec.parse_information_table(payload)

    def test_all_change_types(self):
        before = sec.parse_information_table(xml_table(
            row("111111111", shares="2") + row("222222222", shares="8") +
            row("333333333", shares="3") + row("444444444", shares="3")))
        after = sec.parse_information_table(xml_table(
            row("111111111", shares="4") + row("222222222", shares="1") +
            row("444444444", shares="3") + row("555555555", shares="2")))
        changes = sec.compare_holdings(after, before)
        self.assertEqual({r["cusip"]: r["status"] for r in changes}, {
            "111111111": "加碼", "222222222": "減碼", "333333333": "出清",
            "444444444": "未變", "555555555": "新增",
        })

    def test_periods_are_distinct_and_amendments_ordered(self):
        data = {"filings": {"recent": {
            "form": ["13F-HR", "13F-HR/A", "13F-HR"],
            "accessionNumber": ["a", "b", "c"],
            "filingDate": ["2026-05-15", "2026-05-18", "2026-02-14"],
            "reportDate": ["2026-03-31", "2026-03-31", "2025-12-31"],
            "primaryDocument": ["primary_doc.xml"] * 3,
        }}}
        groups = sec._candidate_periods(data, "Test test@valid.org")
        self.assertEqual([r["accessionNumber"] for r in groups[0]], ["b", "a"])
        self.assertEqual(groups[1][0]["reportDate"], "2025-12-31")

    def test_additive_amendment_withholds_incomplete_quarter(self):
        filings = [{"form": "13F-HR/A", "accessionNumber": "000001-26-000001",
                    "filingDate": "2026-05-18", "reportDate": "2026-03-31",
                    "primaryDocument": "actual_primary.xml"},
                   {"form": "13F-HR", "accessionNumber": "000001-26-000000",
                    "filingDate": "2026-05-15", "reportDate": "2026-03-31",
                    "primaryDocument": "primary_doc.xml"}]
        class Response:
            def json(self):
                return {"directory": {"item": [{"name": "primary_doc.xml"},
                                               {"name": "holdings.xml"}]}}

            content = b"<edgarSubmission><amendmentType>NEW HOLDINGS</amendmentType></edgarSubmission>"
        with patch.object(sec, "_get", return_value=Response()), self.assertRaisesRegex(
                sec.EdgarError, "未合併前不提供完整持倉"):
            sec._read_period("1067983", filings, "Test test@valid.org")

    def test_restatement_uses_actual_primary_filename(self):
        filings = [{"form": "13F-HR/A", "accessionNumber": "000001-26-000001",
                    "filingDate": "2026-05-18", "reportDate": "2026-03-31",
                    "primaryDocument": "cover_actual.xml"},
                   {"form": "13F-HR", "accessionNumber": "000001-26-000000",
                    "filingDate": "2026-05-15", "reportDate": "2026-03-31",
                    "primaryDocument": "primary_doc.xml"}]
        class Response:
            def __init__(self, content=b"", data=None):
                self.content, self._data = content, data
            def json(self):
                return self._data
        def fake_get(url, agent):
            if url.endswith("/cover_actual.xml"):
                return Response(b"<edgarSubmission><amendmentType>RESTATEMENT</amendmentType></edgarSubmission>")
            if url.endswith("/index.json"):
                return Response(data={"directory": {"item": [
                    {"name": "cover_actual.xml"}, {"name": "table.xml"}]}})
            if url.endswith("/table.xml"):
                return Response(xml_table(row(shares="22")))
            raise AssertionError(url)
        with patch.object(sec, "_get", side_effect=fake_get):
            filing, _ = sec._read_period("1067983", filings, "Test contact@valid.org")
        self.assertEqual(filing["form"], "13F-HR/A")
        self.assertEqual(filing["holdings"][0]["quantity"], Decimal(22))

    def test_cik_name_mismatch_prevents_unverified_holdings(self):
        class Response:
            def json(self):
                return {"name": "Some other manager", "cik": 1067983}
        with patch.object(sec, "_get", return_value=Response()), self.assertRaises(sec.EdgarError):
            sec.fetch_manager(sec.MANAGERS[0], "App contact@valid.org")

    def test_no_fake_agent_and_no_simulated_manager_scoring(self):
        with self.assertRaises(sec.EdgarError):
            sec.validate_user_agent("Demo contact@example.com")
        picks = [
            {"kol_id": "warren_buffett", "ticker": "AAPL", "date": "2026-09-28",
             "argument_quality": 3, "thesis": "fictional"},
            {"kol_id": "howard_marks", "ticker": "MSFT", "date": "2026-09-28",
             "argument_quality": 2, "thesis": "unverified"},
        ]
        ranked = score_picks(picks=picks)
        self.assertEqual([p["ticker"] for p in ranked], ["MSFT"])

    def test_existing_manager_records_preserved_but_hidden_and_index_stable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(picks_store, "PICKS_FILE", os.path.join(tmp, "picks.json")):
            rows = [
                {"kol_id": "warren_buffett", "ticker": "AAPL", "date": "2026-09-28", "argument_quality": 3, "thesis": "fake"},
                {"kol_id": "howard_marks", "ticker": "MSFT", "date": "2026-09-28", "argument_quality": 3, "thesis": "demo"},
            ]
            picks_store.save_picks(rows)
            visible = picks_store.get_picks_with_status()
            self.assertEqual(len(visible), 1)
            self.assertEqual(visible[0]["_storage_index"], 1)
            with self.assertRaises(ValueError):
                picks_store.add_pick(rows[0])
            with self.assertRaises(ValueError):
                picks_store.update_pick(0, {"thesis": "still fake"})
            picks_store.delete_pick(visible[0]["_storage_index"])
            with open(picks_store.PICKS_FILE, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), rows[:1])

    def test_full_fetch_uses_actual_periods_and_sec_links(self):
        class Response:
            def __init__(self, data=None, content=b""):
                self._data, self.content = data, content

            def json(self):
                return self._data

        data = {"name": "Berkshire Hathaway Inc.", "cik": 1067983,
                "filings": {"recent": {
                    "form": ["13F-HR", "13F-HR"], "accessionNumber": ["0001067983-26-000002", "0001067983-26-000001"],
                    "filingDate": ["2026-05-15", "2026-02-15"],
                    "reportDate": ["2026-03-31", "2025-12-31"],
                    "primaryDocument": ["primary_doc.xml", "primary_doc.xml"],
                }}}
        def fake_get(url, agent):
            self.assertEqual(agent, "Test contact@valid.org")
            if url.endswith("CIK0001067983.json"):
                return Response(data)
            if url.endswith("index.json"):
                return Response({"directory": {"item": [{"name": "primary_doc.xml"},
                                                         {"name": "info.xml"}]}})
            if "/000106798326000002/info.xml" in url:
                return Response(content=xml_table(row(shares="20")))
            if "/000106798326000001/info.xml" in url:
                return Response(content=xml_table(row(shares="10")))
            raise AssertionError(url)

        with patch.object(sec, "_get", side_effect=fake_get):
            result = sec.fetch_manager(sec.MANAGERS[0], "Test contact@valid.org")
        self.assertEqual(result["current"]["report_date"], "2026-03-31")
        self.assertEqual(result["previous"]["report_date"], "2025-12-31")
        self.assertEqual(result["changes"][0]["status"], "加碼")
        self.assertTrue(result["current"]["source_url"].startswith("https://www.sec.gov/"))


if __name__ == "__main__":
    unittest.main()