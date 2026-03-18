"""Shared metric helpers for in-sample research notebooks and scripts."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from feature_selection.validation.objective_metrics import metric_calmar, metric_profit_factor
from utils.core.enums import TimeFrame

SUPPORTED_PARAM_SENSITIVITY_METRICS: frozenset[str] = frozenset(
    {"sharpe", "mean", "t_stat", "sortino", "calmar", "profit_factor"}
)


def compute_param_sensitivity_metric(
    returns: pd.Series | np.ndarray,
    metric_name: str,
    timeframe: TimeFrame = TimeFrame.D,
) -> float:
    """Compute a named metric for param sensitivity using shared logic.

    This preserves existing binning semantics for ``sharpe``/``mean``/``t_stat``/``sortino``
    and adds ``calmar``/``profit_factor`` using the shared objective-metric implementations.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Return series for metric computation.
    metric_name : str
        Name of metric: sharpe, mean, t_stat, sortino, calmar, profit_factor.
    timeframe : TimeFrame
        Timeframe for annualization (sortino, calmar). Defaults to TimeFrame.D (252 bars/year).
    """
    s = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if s.empty:
        return 0.0

    metric_name = str(metric_name)
    mean_return = float(s.mean())
    n_obs = int(len(s))
    volatility = float(s.std())

    if metric_name == "mean":
        return mean_return

    if metric_name == "sharpe":
        if not np.isfinite(volatility) or volatility <= 1e-12:
            if abs(mean_return) <= 1e-12:
                return 0.0
            return float(np.sign(mean_return) * 10.0)
        return float(mean_return / volatility)

    if metric_name == "t_stat":
        if not np.isfinite(volatility) or volatility <= 1e-12:
            if abs(mean_return) <= 1e-12:
                return 0.0
            return float(np.sign(mean_return) * math.sqrt(n_obs))
        return float(mean_return / (volatility / math.sqrt(n_obs)))

    if metric_name == "sortino":
        annualization_factor = float(timeframe.bars_per_year)
        downside = s[s < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        if (not np.isfinite(downside_std)) or downside_std <= 1e-12:
            return float(mean_return * math.sqrt(annualization_factor)) if mean_return > 0 else 0.0
        return float(mean_return / downside_std) * math.sqrt(annualization_factor)

    if metric_name == "calmar":
        annualization_factor = float(timeframe.bars_per_year)
        return float(metric_calmar(s, annualization_factor=annualization_factor))

    if metric_name == "profit_factor":
        return float(metric_profit_factor(s))

    supported = ", ".join(sorted(SUPPORTED_PARAM_SENSITIVITY_METRICS))
    raise ValueError(
        f"Unsupported param sensitivity metric: {metric_name}. Supported: {supported}."
    )
