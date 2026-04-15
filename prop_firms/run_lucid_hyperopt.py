"""Config-driven Lucid hyperparameter optimizer runner."""

from __future__ import annotations

from pathlib import Path

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from prop_firms.lucid import create_lucid_portfolio_simulator
from prop_firms.optimization import run_lucid_hyperopt
from prop_firms.optimization_reporting import generate_optimization_report
from prop_firms.report_config import (
    LucidPortfolioHyperoptReportConfig,
    load_hyperopt_config,
)


def run_lucid_hyperopt_report(
    config: LucidPortfolioHyperoptReportConfig | None = None,
) -> None:
    """Run the Lucid optimizer and write optimization artifacts."""

    resolved_config = load_hyperopt_config() if config is None else config
    simulator = create_lucid_portfolio_simulator()
    result = run_lucid_hyperopt(
        config=resolved_config.optimization,
        simulator=simulator,
    )
    artifacts = generate_optimization_report(
        result=result,
        output_dir=resolved_config.output_dir,
        report_stem=resolved_config.report_stem,
        save_csvs=resolved_config.save_csvs,
    )

    print("=" * 70)
    print("LUCID HYPEROPT REPORT")
    print("=" * 70)
    print(f"Markdown report: {artifacts.report_markdown_path}")
    print(f"HTML report: {artifacts.report_html_path}")
    if artifacts.trials_csv_path is not None:
        print(f"Trials CSV: {artifacts.trials_csv_path}")
    if artifacts.seed_runs_csv_path is not None:
        print(f"Seed runs CSV: {artifacts.seed_runs_csv_path}")


if __name__ == "__main__":
    run_lucid_hyperopt_report()
