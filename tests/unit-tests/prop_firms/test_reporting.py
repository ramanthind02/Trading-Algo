import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prop_firms import (
    PortfolioSimulationConfig,
    create_apex_portfolio_simulator,
    create_lucid_portfolio_simulator,
    generate_portfolio_report,
)


class TestPropFirmReporting(unittest.TestCase):
    def test_generate_portfolio_report_writes_markdown_html_and_csvs(self) -> None:
        simulator = create_lucid_portfolio_simulator()
        returns = pd.Series(
            [0.026, 0.024, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=pd.bdate_range("2026-01-02", periods=7),
        )
        result = simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="unit_test_report",
                save_csvs=True,
            )

            self.assertTrue(artifacts.report_markdown_path.exists())
            self.assertTrue(artifacts.report_html_path.exists())
            self.assertTrue(artifacts.daily_timeline_csv_path.exists())
            self.assertTrue(artifacts.monthly_summary_csv_path.exists())
            self.assertTrue(artifacts.yearly_summary_csv_path.exists())
            self.assertTrue(artifacts.account_summaries_csv_path.exists())
            self.assertTrue(artifacts.events_csv_path.exists())

            report_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Lucid Portfolio Report", report_text)
            self.assertIn("Net cashflow", report_text)
            self.assertIn("Funded Closure Reasons", report_text)
            self.assertIn("No funded closures", report_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("<html", html_text)
            self.assertIn("EV Statistics", html_text)
            self.assertIn("Charts", html_text)
            self.assertIn("Companion CSV Files", html_text)
            self.assertIn("Net cashflow", html_text)
            self.assertIn("Funded Closure Reasons", html_text)
            self.assertIn("data:image/png;base64,", html_text)

    def test_generate_portfolio_report_uses_apex_provider_title(self) -> None:
        simulator = create_apex_portfolio_simulator()
        returns = pd.Series(
            [0.061, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=pd.bdate_range("2026-01-05", periods=6),
        )
        result = simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(account_code="50000"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="apex_unit_test_report",
                save_csvs=False,
            )

            report_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Apex Portfolio Report", report_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("Apex Portfolio Report", html_text)

    def test_generate_portfolio_report_includes_expired_apex_challenges(self) -> None:
        simulator = create_apex_portfolio_simulator()
        returns = pd.Series(
            [0.0] * len(pd.bdate_range("2026-01-05", "2026-02-04")),
            index=pd.bdate_range("2026-01-05", "2026-02-04"),
        )
        result = simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(account_code="50000"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="apex_expiry_report",
                save_csvs=False,
            )

            report_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("challenge_expired", report_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("challenge_expired", html_text)


if __name__ == "__main__":
    unittest.main()
