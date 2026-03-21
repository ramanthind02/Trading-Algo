"""Config-driven Lucid portfolio report runner."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    search_root = start if start.is_dir() else start.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


_repo_root = _find_repo_root(Path(__file__).resolve())
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from prop_firms import build_return_series, create_lucid_portfolio_simulator
from prop_firms.report_config import LucidPortfolioReportConfig, load_report_config
from prop_firms.reporting import generate_portfolio_report


def run_lucid_portfolio_report(
    config: LucidPortfolioReportConfig | None = None,
) -> None:
    """Run the Lucid portfolio simulator and write a readable report."""

    resolved_config = load_report_config() if config is None else config
    returns = build_return_series(
        config=resolved_config.simulation.return_engine,
        data_path=resolved_config.data_path,
    )
    simulator = create_lucid_portfolio_simulator()
    result = simulator.simulate(
        returns=returns,
        config=resolved_config.simulation,
    )
    artifacts = generate_portfolio_report(
        result=result,
        output_dir=resolved_config.output_dir,
        report_stem=resolved_config.report_stem,
        save_csvs=resolved_config.save_csvs,
    )

    print("=" * 70)
    print("LUCID PORTFOLIO REPORT")
    print("=" * 70)
    print(f"Markdown report: {artifacts.report_markdown_path}")
    print(f"HTML report: {artifacts.report_html_path}")
    if artifacts.daily_timeline_csv_path is not None:
        print(f"Daily timeline CSV: {artifacts.daily_timeline_csv_path}")
    if artifacts.monthly_summary_csv_path is not None:
        print(f"Monthly summary CSV: {artifacts.monthly_summary_csv_path}")
    if artifacts.yearly_summary_csv_path is not None:
        print(f"Yearly summary CSV: {artifacts.yearly_summary_csv_path}")
    if artifacts.account_summaries_csv_path is not None:
        print(f"Account summaries CSV: {artifacts.account_summaries_csv_path}")
    if artifacts.events_csv_path is not None:
        print(f"Events CSV: {artifacts.events_csv_path}")


if __name__ == "__main__":
    run_lucid_portfolio_report()
