"""Portfolio test entrypoint.

Loads portfolio research config and runs the test pipeline.
See portfolio_research/pipelines/portfolio_test.py for implementation.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python portfolio_research/run_portfolio_test.py

Artifacts are written to config.output_root, including per-phase and combined
``weight_layer_weights*.csv`` files (long format for Power BI; stream/model ids
are not hardcoded). Optional HTML tearsheets: set
``export_per_timeframe_tearsheets`` / ``export_per_ensemble_tearsheets`` in
``portfolio_research.config.load_config()`` to False to skip slower detail reports
(combined phase portfolio tearsheets are still written).
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    """Allow ``python path/to/run_portfolio_test.py`` without PYTHONPATH."""
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_prepend_repo_root_to_syspath()

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from portfolio_research.config import PortfolioResearchConfig, load_config
from portfolio_research.pipelines.portfolio_test import (
    _build_daily_dates_per_ticker,
    _group_ensembles_by_timeframe,
    run_portfolio_test_pipeline,
)


def run_portfolio_test(config: PortfolioResearchConfig | None = None) -> None:
    """Load config (if not provided) and run the portfolio test pipeline."""
    if config is None:
        config = load_config()
    run_portfolio_test_pipeline(config)


if __name__ == "__main__":
    run_portfolio_test()
