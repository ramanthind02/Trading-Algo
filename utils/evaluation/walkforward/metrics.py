from __future__ import annotations

from typing import Callable

import pandas as pd

def _mean_return_metric(returns: pd.Series) -> float:
    mean_return = returns.mean()
    if pd.isna(mean_return):
        return 0.0
    return float(mean_return)


def _sharpe_metric(returns: pd.Series) -> float:
    standard_deviation = returns.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return 0.0
    return float(returns.mean() / standard_deviation)


def _sortino_metric(returns: pd.Series) -> float:
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std(ddof=0)
    if pd.isna(downside_deviation) or downside_deviation == 0:
        return 0.0
    return float(returns.mean() / downside_deviation)


def _t_stat_metric(returns: pd.Series) -> float:
    """t-statistic for mean return: mean / (std / sqrt(n)).

    Scale grows with sqrt(n): for the same series, t_stat = Sharpe * sqrt(n).
    Typical values can be in the tens when n is large (e.g. hundreds of OOS days).
    """
    clean = returns.dropna()
    n = clean.size
    if n < 2:
        return 0.0
    standard_deviation = clean.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return 0.0
    standard_error = standard_deviation / float(n**0.5)
    if standard_error == 0:
        return 0.0
    return float(clean.mean() / standard_error)


_OBJECTIVE_METRIC_RESOLVER: dict[str, Callable[[pd.Series], float]] = {
    "sharpe": _sharpe_metric,
    "sortino": _sortino_metric,
    "mean_return": _mean_return_metric,
    "t_stat": _t_stat_metric,
}

SUPPORTED_OBJECTIVE_METRICS: tuple[str, ...] = tuple(_OBJECTIVE_METRIC_RESOLVER)


def resolve_objective_metric(metric_name: str) -> Callable[[pd.Series], float]:
    if metric_name not in _OBJECTIVE_METRIC_RESOLVER:
        raise ValueError(f"Unsupported objective metric: {metric_name}")
    return _OBJECTIVE_METRIC_RESOLVER[metric_name]
