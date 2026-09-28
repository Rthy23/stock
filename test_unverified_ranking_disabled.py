"""Only official filings and fund holdings are rendered; examples stay archived."""

import unittest
from unittest.mock import patch

from kol_whitelist import render_kol_section
from page_modules.analyst_consensus import render_analyst_consensus_page


class PublicSectionsTests(unittest.TestCase):
    def test_holdings_page_never_runs_broker_or_example_leaderboards(self):
        with patch("page_modules.analyst_consensus.st") as ui, \
             patch("page_modules.analyst_consensus.render_sec_holdings") as sec, \
             patch("page_modules.analyst_consensus.render_ark_holdings") as ark, \
             patch("page_modules.analyst_consensus.render_industry_commentary_reference"), \
             patch("page_modules.analyst_consensus._render_curated_consensus",
                   side_effect=AssertionError("example ranking called")), \
             patch("page_modules.analyst_consensus.fetch_market_recommendations",
                   side_effect=AssertionError("broker ranking called")):
            render_analyst_consensus_page()

        sec.assert_called_once()
        ark.assert_called_once()
        self.assertIn("官方機構與基金持倉", ui.title.call_args.args[0])
        self.assertIn("不是即時選股訊號", ui.caption.call_args.args[0])
        self.assertIn("暫不提供", ui.info.call_args.args[0])
        ui.error.assert_not_called()
        ui.plotly_chart.assert_not_called()

    def test_macro_shows_disclosures_not_example_rankings_or_ai(self):
        with patch("kol_whitelist.st") as ui, \
             patch("page_modules.sec_holdings.render_sec_holdings") as sec, \
             patch("kol_whitelist.render_ark_holdings") as ark, \
             patch("kol_whitelist.render_industry_commentary_reference"), \
             patch("kol_whitelist.build_consensus_table",
                   side_effect=AssertionError("example ranking called")), \
             patch("kol_whitelist.call_gemini_consensus",
                   side_effect=AssertionError("AI consensus called")):
            render_kol_section(api_key="unused")

        sec.assert_called_once()
        ark.assert_called_once()
        self.assertIn("暫不提供", ui.info.call_args.args[0])
        ui.error.assert_not_called()
        ui.button.assert_not_called()


if __name__ == "__main__":
    unittest.main()