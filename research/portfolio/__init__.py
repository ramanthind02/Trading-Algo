"""Portfolio research: config-driven script for vault portfolio fit and tearsheets.

Two datasets: in-sample walkforward (bounds from feature_research window formula)
and OOS (explicit OOSWindowConfig). Use load_config() and run_portfolio_test() from here
or run_portfolio_test.py as script.
"""
from research.portfolio.config import PortfolioResearchConfig, load_config
from research.portfolio.holdout.pipeline import run_portfolio_holdout_pipeline
from research.portfolio.run_portfolio_test import run_portfolio_test

__all__ = [
    "PortfolioResearchConfig",
    "load_config",
    "run_portfolio_holdout_pipeline",
    "run_portfolio_test",
]
