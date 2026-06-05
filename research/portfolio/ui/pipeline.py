"""End-to-end portfolio research orchestration for the workspace UI."""
from __future__ import annotations

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.holdout.portfolio_holdout_runner import run_portfolio_holdout_report
from research.portfolio.holdout.rolling_eval import run_holdout_evaluation
from research.portfolio.holdout.strategy_monitoring import run_strategy_holdout_monitoring
from research.portfolio.pipelines.portfolio_test import (
    run_portfolio_research_cache_preflight,
    run_portfolio_test_pipeline,
)


def run_full_portfolio_research_pipeline(config: PortfolioResearchConfig) -> None:
    """Run portfolio test, strategy monitoring, and portfolio holdout in one pass."""

    run_portfolio_research_cache_preflight(config)

    print("\n" + "=" * 64)
    print("Stage 1/3 — Portfolio test (tearsheets + FundedNext prop-firm reports)")
    print("=" * 64 + "\n")
    run_portfolio_test_pipeline(config)

    print("\n" + "=" * 64)
    print("Stage 2/3 — Holdout evaluation and strategy monitoring")
    print("=" * 64 + "\n")
    rolling = run_holdout_evaluation(config, emit_tearsheets=True)
    run_strategy_holdout_monitoring(
        config,
        research_strategy_returns=rolling.strategy_returns_research,
        holdout_strategy_returns=rolling.strategy_returns_holdout,
    )

    print("\n" + "=" * 64)
    print("Stage 3/3 — Portfolio holdout analytics")
    print("=" * 64 + "\n")
    run_portfolio_holdout_report(
        config,
        research_strategy_returns=rolling.strategy_returns_research,
        holdout_strategy_returns=rolling.strategy_returns_holdout,
        research_portfolio_returns=rolling.portfolio_returns_research,
        holdout_portfolio_returns=rolling.portfolio_returns_holdout,
    )

    print(
        f"\nFull pipeline complete. Artifacts: {config.output_root} "
        "(train/validation/test tearsheets, prop_firm/fundednext/ reports, "
        "holdout/strategies, holdout/visualization/portfolio)\n"
    )
