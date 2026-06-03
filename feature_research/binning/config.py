"""Standalone configuration for research-only continuous-feature binning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.core.enums import Direction, Ticker, TimeFrame


@dataclass(frozen=True)
class BinningResearchConfig:
    """Inputs for continuous-feature binning exports used in research."""

    enabled: bool
    tickers: list[Ticker]
    start: datetime
    end: datetime
    timeframe: TimeFrame
    bias_spec: dict[str, Any]
    target_col: str
    reports_dir: Path
    reports_subdir_name: str = "binning_phase"
    n_bins: int = 10
    strategy: Direction = Direction.LONG
    rolling_window: int = 252
    rolling_min_periods: int = 126
    feature_col_substr: str | None = None
    strategy_bias_spec: dict[str, Any] | None = None
    """When set (e.g. locked ``cumulative_rsi_signal``), decile metrics use strategy PnL."""


def load_binning_research_config() -> BinningResearchConfig:
    """Return the standalone continuous-node research config.

    This config is intentionally independent from ``feature_research.config.load_config()``.
    The main feature-research config now describes the signed-signal trading pipeline,
    while this file is the single source of truth for continuous-node binning / EDA work.

    Cyclical RSI (4, 120, 2) — 10 quantile bins; LONG uses bin index 0 (lowest decile).
    """
    tickers = [Ticker.ES, Ticker.NQ]
    start = datetime(2000, 1, 1)
    end = datetime(2019, 1, 1)
    timeframe = TimeFrame.D
    bias_spec: dict[str, Any] = {
        "module_name": "cyclical_rsi",
        "timeframes": [timeframe],
        "params": {
            "short_period": 4,
            "long_period": 120,
            "rsi_period": 2,
        },
    }
    target_col = "log_return_ewsd"
    reports_dir = Path("feature_research") / "in_sample" / "results" / "continuous" / "cyclical_rsi_4_120_2"
    return BinningResearchConfig(
        enabled=True,
        tickers=tickers,
        start=start,
        end=end,
        timeframe=timeframe,
        bias_spec=bias_spec,
        target_col=target_col,
        reports_dir=reports_dir,
        reports_subdir_name="binning_phase",
        n_bins=10,
        strategy=Direction.LONG,
        rolling_window=252,
        rolling_min_periods=126,
        feature_col_substr=None,
    )
