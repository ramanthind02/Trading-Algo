"""End-to-end portfolio holdout pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from portfolio_research.config import PortfolioResearchConfig
from portfolio_research.holdout.portfolio_holdout_runner import run_portfolio_holdout_report
from portfolio_research.holdout.rolling_eval import run_holdout_evaluation
from portfolio_research.holdout.strategy_monitoring import run_strategy_holdout_monitoring


def run_portfolio_holdout_pipeline(
    config: PortfolioResearchConfig | None = None,
    *,
    emit_tearsheets: bool = True,
) -> dict[str, Any]:
    """Rolling holdout eval → strategy monitoring → portfolio holdout report."""

    cfg = config if config is not None else __import__(
        "portfolio_research.config", fromlist=["load_config"]
    ).load_config()

    rolling = run_holdout_evaluation(cfg, emit_tearsheets=emit_tearsheets)
    strategy_artifacts = run_strategy_holdout_monitoring(
        cfg,
        research_strategy_returns=rolling.strategy_returns_research,
        holdout_strategy_returns=rolling.strategy_returns_holdout,
    )
    portfolio_report, portfolio_artifacts = run_portfolio_holdout_report(
        cfg,
        research_strategy_returns=rolling.strategy_returns_research,
        holdout_strategy_returns=rolling.strategy_returns_holdout,
        research_portfolio_returns=rolling.portfolio_returns_research,
        holdout_portfolio_returns=rolling.portfolio_returns_holdout,
    )
    return {
        "fold_manifest": rolling.fold_manifest_path,
        "strategy_artifacts": strategy_artifacts,
        "portfolio_report": portfolio_report,
        "portfolio_artifacts": portfolio_artifacts,
    }


def main() -> None:
    from portfolio_research.config import load_config

    result = run_portfolio_holdout_pipeline(load_config())
    print(f"Portfolio holdout complete. Artifacts under {load_config().output_root / 'holdout'}")


if __name__ == "__main__":
    main()
