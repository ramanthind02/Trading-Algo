import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from prop_firms import (
    PortfolioSimulationConfig,
    create_apex_portfolio_simulator,
    create_lucid_portfolio_simulator,
    generate_portfolio_report,
)
from quantfoundry_core.prop_firm import create_simulator_for_firm


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
