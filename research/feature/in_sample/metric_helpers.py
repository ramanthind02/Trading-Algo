"""Shared metric helpers for in-sample research notebooks and scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.validation.objective_metrics import (
    metric_calmar,
    metric_mean_return,
    metric_profit_factor,
    metric_sharpe,
    metric_sortino,
    metric_t_stat,
)
from lib.core.enums import TimeFrame

SUPPORTED_PARAM_SENSITIVITY_METRICS: frozenset[str] = frozenset(
    {"sharpe", "mean", "t_stat", "sortino", "calmar", "profit_factor"}
)


def compute_param_sensitivity_metric(
    returns: pd.Series | np.ndarray,
    metric_name: str,
    timeframe: TimeFrame = TimeFrame.D,
) -> float:
    """Compute a named metric for param sensitivity using shared logic.

    Callers (e.g. in-sample EDA) typically pass **per-bar strategy returns**
    ``signal * target`` over the full aligned index so features are comparable on the same
    calendar.     ``sharpe`` and ``sortino`` annualize using ``sqrt(timeframe.bars_per_year)`` (same
    ``TimeFrame`` convention as elsewhere). ``mean`` and ``t_stat`` stay on the raw
    per-bar return scale. Adds ``calmar``/``profit_factor`` via shared objective metrics.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Return series for metric computation.
    metric_name : str
        Name of metric: sharpe, mean, t_stat, sortino, calmar, profit_factor.
    timeframe : TimeFrame
        Timeframe for annualization (sharpe, sortino, calmar). Defaults to TimeFrame.D (252 bars/year).
    """
    clean = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if clean.empty:
        return 0.0

    metric_name = str(metric_name)
    if metric_name == "mean":
        return metric_mean_return(clean)

    if metric_name == "sharpe":
        return metric_sharpe(
            clean,
            annualization_factor=float(timeframe.bars_per_year),
        )

    if metric_name == "t_stat":
        return metric_t_stat(clean)

    if metric_name == "sortino":
        return metric_sortino(
            clean,
            annualization_factor=float(timeframe.bars_per_year),
        )

    if metric_name == "calmar":
        return metric_calmar(
            clean,
            annualization_factor=float(timeframe.bars_per_year),
        )

    if metric_name == "profit_factor":
        return metric_profit_factor(clean)

    supported = ", ".join(sorted(SUPPORTED_PARAM_SENSITIVITY_METRICS))
    raise ValueError(
        f"Unsupported param sensitivity metric: {metric_name}. Supported: {supported}."
    )
