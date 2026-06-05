"""Persist stitched holdout return matrices for downstream monitoring."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.portfolio.shared.visualization_paths import holdout_returns_dir


def write_return_matrices(
    *,
    output_root: Path,
    research_strategy_returns: pd.DataFrame,
    holdout_strategy_returns: pd.DataFrame,
    holdout_portfolio_returns: pd.Series,
    research_portfolio_returns: pd.Series | None = None,
) -> dict[str, Path]:
    """Write canonical CSV return artifacts under ``holdout/returns/``."""

    out = holdout_returns_dir(output_root)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "strategy_returns_research": out / "strategy_returns_research.csv",
        "strategy_returns_holdout": out / "strategy_returns_holdout.csv",
        "portfolio_returns_holdout": out / "portfolio_returns_holdout.csv",
    }
    research_strategy_returns.sort_index().to_csv(paths["strategy_returns_research"])
    holdout_strategy_returns.sort_index().to_csv(paths["strategy_returns_holdout"])
    holdout_portfolio_returns.sort_index().to_frame(name="portfolio").to_csv(
        paths["portfolio_returns_holdout"]
    )
    if research_portfolio_returns is not None:
        research_path = out / "portfolio_returns_research.csv"
        research_portfolio_returns.sort_index().to_frame(name="portfolio").to_csv(research_path)
        paths["portfolio_returns_research"] = research_path
    return paths


def load_return_matrices(output_root: Path) -> dict[str, pd.DataFrame | pd.Series]:
    """Load persisted holdout return artifacts."""

    out = holdout_returns_dir(output_root)
    strategy_research = pd.read_csv(out / "strategy_returns_research.csv", index_col=0, parse_dates=True)
    strategy_holdout = pd.read_csv(out / "strategy_returns_holdout.csv", index_col=0, parse_dates=True)
    portfolio_holdout = pd.read_csv(out / "portfolio_returns_holdout.csv", index_col=0, parse_dates=True)
    portfolio_series = portfolio_holdout.iloc[:, 0]
    portfolio_series.name = "portfolio"
    return {
        "strategy_research": strategy_research,
        "strategy_holdout": strategy_holdout,
        "portfolio_holdout": portfolio_series,
    }
