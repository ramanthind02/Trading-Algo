import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from quantfoundry_core.prop_firm import PortfolioSimulationConfig, create_simulator_for_firm

from research.portfolio.prop_firm_report_builder import (
    PropFirmReportArtifacts,
    generate_portfolio_report,
)


class TestPropFirmReportBuilder(unittest.TestCase):
    def test_generate_portfolio_report_writes_markdown_html_and_csvs(self) -> None:
        simulator = create_simulator_for_firm("fundednext")
        returns = pd.Series(
            [0.026, 0.024, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=pd.bdate_range("2026-01-02", periods=7),
        )
        result = simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(account_code="50000"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="unit_test_report",
                save_csvs=True,
            )

            self.assertIsInstance(artifacts, PropFirmReportArtifacts)
            self.assertTrue(artifacts.report_markdown_path.exists())
            self.assertTrue(artifacts.report_html_path.exists())
            self.assertTrue(artifacts.daily_timeline_csv_path.exists())
            self.assertTrue(artifacts.monthly_summary_csv_path.exists())
            self.assertTrue(artifacts.yearly_summary_csv_path.exists())
            self.assertTrue(artifacts.account_summaries_csv_path.exists())
            self.assertTrue(artifacts.events_csv_path.exists())

            report_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Portfolio Report", report_text)
            self.assertIn("Net cashflow", report_text)
            self.assertIn("Funded Closure Reasons", report_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("<html", html_text)
            self.assertIn("EV Statistics", html_text)
            self.assertIn("Charts", html_text)
            self.assertIn("Companion CSV Files", html_text)
            self.assertIn("Net cashflow", html_text)
            self.assertIn("data:image/png;base64,", html_text)

    def test_generate_portfolio_report_save_csvs_false(self) -> None:
        simulator = create_simulator_for_firm("fundednext")
        returns = pd.Series(
            [0.026, 0.024, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=pd.bdate_range("2026-01-02", periods=7),
        )
        result = simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(account_code="50000"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="no_csv_report",
                save_csvs=False,
            )

            self.assertTrue(artifacts.report_markdown_path.exists())
            self.assertTrue(artifacts.report_html_path.exists())
            self.assertIsNone(artifacts.daily_timeline_csv_path)

    def test_generate_portfolio_report_writes_monthly_breakdown_and_rolling(self) -> None:
        simulator = create_simulator_for_firm("fundednext")
        returns = pd.Series(
            0.0004,
            index=pd.bdate_range("2021-01-04", "2024-12-31"),
        )
        config = PortfolioSimulationConfig(account_code="50000")
        result = simulator.simulate(returns=returns, config=config)
        rolling = simulator.simulate_rolling(
            returns=returns,
            config=config,
            window_months=12,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_portfolio_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="qf_rolling_report",
                save_csvs=True,
                rolling=rolling,
            )

            self.assertTrue(artifacts.monthly_breakdown_csv_path is not None)
            self.assertTrue(artifacts.monthly_breakdown_csv_path.exists())
            self.assertTrue(artifacts.rolling_pooled_monthly_stats_csv_path is not None)
            self.assertTrue(artifacts.rolling_pooled_monthly_stats_csv_path.exists())
            self.assertTrue(artifacts.rolling_batch_statistics_json_path is not None)
            batch_payload = json.loads(
                artifacts.rolling_batch_statistics_json_path.read_text(encoding="utf-8")
            )
            self.assertGreater(batch_payload["n_runs"], 0)

            report_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Rolling Window EV", report_text)
            self.assertIn("Steady-State Monthly EV", report_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("Rolling Window EV", html_text)
            self.assertIn("data:image/png;base64,", html_text)


if __name__ == "__main__":
    unittest.main()
