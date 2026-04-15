from __future__ import annotations

from typing import Callable

import pandas as pd

from feature_selection.validation.objective_metrics import (
    ObjectiveMetricSpec,
    resolve_objective_metric as resolve_objective_metric_spec,
)


def _mean_return_metric(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    mean_return = clean.mean()
    return 0.0 if pd.isna(mean_return) else float(mean_return)


_OBJECTIVE_METRIC_SPECS: dict[str, ObjectiveMetricSpec] = {
    "sharpe": ObjectiveMetricSpec(builtin="sharpe"),
    "sortino": ObjectiveMetricSpec(builtin="sortino"),
    "calmar": ObjectiveMetricSpec(builtin="calmar"),
    "t_stat": ObjectiveMetricSpec(builtin="t_stat"),
    "profit_factor": ObjectiveMetricSpec(builtin="profit_factor"),
}

SUPPORTED_OBJECTIVE_METRICS: tuple[str, ...] = (
    *tuple(_OBJECTIVE_METRIC_SPECS),
    "mean_return",
)


def resolve_objective_metric(metric_name: str) -> Callable[[pd.Series], float]:
    if metric_name == "mean_return":
        return _mean_return_metric

    spec = _OBJECTIVE_METRIC_SPECS.get(metric_name)
    if spec is None:
        raise ValueError(f"Unsupported objective metric: {metric_name}")
    return resolve_objective_metric_spec(spec)
