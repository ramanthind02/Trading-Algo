"""Objective metric specification and resolver helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Callable, Literal, Mapping, cast

import numpy as np
import pandas as pd
from quantfoundry_core.metrics import (
    MetricName,
    ReturnsCompounding,
    ReturnsValidationError,
    compute_scalar_metric,
)

BuiltinMetricName = Literal['sharpe', 'sortino', 'calmar', 't_stat', 'profit_factor', 'mean_return', 'always_zero']
ObjectiveMetricCallable = Callable[[pd.Series], float]
_MetricFactoryCallable = Callable[..., float]
_SCALAR_INDEX_START = "2000-01-01"


def metric_sharpe(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """Sharpe ratio via ``quantfoundry_core.metrics`` with local edge-case fallback."""
    clean = _prepare_scalar_returns(returns)
    if clean.empty:
        return 0.0
    annual_risk_free_rate = risk_free_rate * max(float(annualization_factor), 0.0)
    core_value = _compute_supported_core_metric(
        clean,
        metric=MetricName.SHARPE,
        annualization_factor=annualization_factor,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    return (
        core_value
        if core_value is not None
        else _fallback_builtin_metric(
            clean,
            MetricName.SHARPE,
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        )
    )


def metric_sortino(
    returns: pd.Series,
    *,
    target_return: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """Sortino ratio via ``quantfoundry_core.metrics`` with local edge-case fallback."""
    clean = _prepare_scalar_returns(returns)
    if clean.empty:
        return 0.0
    annual_target_return = target_return * max(float(annualization_factor), 0.0)
    core_value = _compute_supported_core_metric(
        clean,
        metric=MetricName.SORTINO,
        annualization_factor=annualization_factor,
        annual_risk_free_rate=annual_target_return,
    )
    return (
        core_value
        if core_value is not None
        else _fallback_builtin_metric(
            clean,
            MetricName.SORTINO,
            annualization_factor=annualization_factor,
            target_return=target_return,
        )
    )


def metric_calmar(
    returns: pd.Series,
    *,
    annualization_factor: float = 1.0,
) -> float:
    """Calmar ratio via ``quantfoundry_core.metrics`` with local edge-case fallback."""
    clean = _prepare_scalar_returns(returns)
    if clean.empty:
        return 0.0
    core_value = _compute_supported_core_metric(
        clean,
        metric=MetricName.CALMAR,
        annualization_factor=annualization_factor,
    )
    return (
        core_value
        if core_value is not None
        else _fallback_builtin_metric(
            clean,
            MetricName.CALMAR,
            annualization_factor=annualization_factor,
        )
    )


def metric_t_stat(returns: pd.Series) -> float:
    """One-sample t-statistic of mean returns against zero."""
    clean = _prepare_scalar_returns(returns)
    if clean.empty or len(clean) < 2:
        return 0.0
    mean_return = float(clean.mean())
    sample_std = float(clean.std(ddof=1))
    standard_error = sample_std / math.sqrt(float(len(clean)))
    return _deterministic_ratio(numerator=mean_return, denominator=standard_error)


def metric_profit_factor(returns: pd.Series) -> float:
    """Profit factor as gross gains divided by gross losses."""
    clean = _prepare_scalar_returns(returns)
    if clean.empty:
        return 0.0
    core_value = _compute_supported_core_metric(
        clean,
        metric=MetricName.PROFIT_FACTOR,
        annualization_factor=1.0,
    )
    return (
        core_value
        if core_value is not None
        else _fallback_builtin_metric(clean, MetricName.PROFIT_FACTOR)
    )


def metric_mean_return(returns: pd.Series) -> float:
    """Arithmetic mean return with NaN/inf cleanup."""
    clean = _prepare_scalar_returns(returns)
    return float(clean.mean()) if not clean.empty else 0.0


def metric_always_zero(_returns: pd.Series) -> float:
    """Always returns zero - useful for forcing test failures."""
    return 0.0


_BUILTIN_OBJECTIVE_METRICS: Mapping[BuiltinMetricName, _MetricFactoryCallable] = {
    'sharpe': metric_sharpe,
    'sortino': metric_sortino,
    'calmar': metric_calmar,
    't_stat': metric_t_stat,
    'profit_factor': metric_profit_factor,
    'mean_return': metric_mean_return,
    'always_zero': metric_always_zero,
}

SUPPORTED_OBJECTIVE_METRIC_NAMES: tuple[str, ...] = tuple(_BUILTIN_OBJECTIVE_METRICS)


@dataclass(frozen=True)
class ObjectiveMetricSpec:
    """Configuration for selecting objective metric callable."""

    builtin: BuiltinMetricName | None = None
    callable_path: str | None = None
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_builtin = self.builtin is not None
        has_callable_path = self.callable_path is not None
        if has_builtin == has_callable_path:
            raise ValueError('ObjectiveMetricSpec requires exactly one of builtin/callable_path.')

        if self.builtin is not None and self.builtin not in _BUILTIN_OBJECTIVE_METRICS:
            supported = ', '.join(sorted(_BUILTIN_OBJECTIVE_METRICS))
            raise ValueError(
                f'Unsupported builtin objective metric {self.builtin!r}. '
                f'Supported builtins: {supported}.',
            )

        if self.callable_path is None:
            return

        module_name, separator, function_name = self.callable_path.partition(':')
        if separator != ':' or not module_name or not function_name:
            raise ValueError('callable_path must use module:function format.')


def apply_objective_metric(spec: ObjectiveMetricSpec, returns: pd.Series) -> float:
    """Evaluate a frozen objective spec on returns (safe to use inside multiprocessing workers)."""
    metric_callable = (
        _BUILTIN_OBJECTIVE_METRICS[spec.builtin]
        if spec.builtin is not None
        else _import_metric_callable(spec.callable_path)
    )
    kwargs = dict(spec.kwargs)
    if not kwargs:
        return float(metric_callable(returns))
    return float(metric_callable(returns, **kwargs))


def resolve_objective_metric(spec: ObjectiveMetricSpec) -> ObjectiveMetricCallable:
    """Resolve objective metric callable from builtin or import path."""
    return lambda returns: apply_objective_metric(spec, returns)


def resolve_objective_metric_name(metric_name: str) -> ObjectiveMetricCallable:
    """Resolve a builtin metric callable from its stable string name."""
    spec = _BUILTIN_OBJECTIVE_METRICS.get(metric_name)
    if spec is None:
        raise ValueError(f"Unsupported objective metric: {metric_name}")
    return lambda returns: float(spec(returns))


def _import_metric_callable(callable_path: str | None) -> _MetricFactoryCallable:
    if callable_path is None:
        raise ValueError('callable_path is required for custom objective metrics.')

    module_name, _, function_name = callable_path.partition(':')
    module = importlib.import_module(module_name)
    metric_object = getattr(module, function_name)
    if not callable(metric_object):
        raise TypeError(f'Imported object at {callable_path!r} is not callable.')
    return cast(_MetricFactoryCallable, metric_object)


def _compute_supported_core_metric(
    returns: pd.Series,
    *,
    metric: MetricName,
    annualization_factor: float,
    annual_risk_free_rate: float = 0.0,
) -> float | None:
    periods_per_year = _periods_per_year(annualization_factor)
    if periods_per_year is None:
        return None
    try:
        return float(
            compute_scalar_metric(
                returns,
                metric,
                periods_per_year=periods_per_year,
                risk_free_rate=annual_risk_free_rate,
                compounding=ReturnsCompounding.SIMPLE,
            )
        )
    except ReturnsValidationError:
        return None


def _prepare_scalar_returns(returns: pd.Series | np.ndarray) -> pd.Series:
    raw = returns if isinstance(returns, pd.Series) else pd.Series(returns)
    numeric = pd.to_numeric(raw, errors='coerce').dropna()
    if numeric.empty:
        return pd.Series(dtype='float64')
    values = numeric.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return pd.Series(dtype='float64')
    synthetic_index = pd.date_range(_SCALAR_INDEX_START, periods=len(finite_values), freq='D')
    return pd.Series(finite_values, index=synthetic_index, dtype='float64')


def _periods_per_year(annualization_factor: float) -> int | None:
    if not math.isfinite(annualization_factor) or annualization_factor <= 0.0:
        return None
    return max(1, int(round(float(annualization_factor))))


def _fallback_builtin_metric(
    returns: pd.Series,
    metric: MetricName,
    *,
    annualization_factor: float = 1.0,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """Local edge-case policy when Core rejects or cannot annualize the series."""
    match metric:
        case MetricName.SHARPE:
            excess = returns - risk_free_rate
            volatility = float(excess.std(ddof=0))
            annualization_scale = math.sqrt(max(annualization_factor, 0.0))
            numerator = float(excess.mean() * annualization_scale)
            return _deterministic_ratio(numerator=numerator, denominator=volatility)
        case MetricName.SORTINO:
            excess = returns - target_return
            downside = excess[excess < 0.0]
            downside_risk = float(downside.std(ddof=0))
            annualization_scale = math.sqrt(max(annualization_factor, 0.0))
            numerator = float(excess.mean() * annualization_scale)
            return _deterministic_ratio(numerator=numerator, denominator=downside_risk)
        case MetricName.CALMAR:
            equity_curve = (1.0 + returns).cumprod()
            drawdown = equity_curve / equity_curve.cummax() - 1.0
            max_drawdown = abs(float(drawdown.min()))
            annualized_return = float(returns.mean() * annualization_factor)
            return _deterministic_ratio(numerator=annualized_return, denominator=max_drawdown)
        case MetricName.PROFIT_FACTOR:
            gross_gain = float(returns[returns > 0.0].sum())
            gross_loss = abs(float(returns[returns < 0.0].sum()))
            return _deterministic_ratio(numerator=gross_gain, denominator=gross_loss)
        case _:
            raise ValueError(f"Unsupported fallback metric: {metric!r}")


def _deterministic_ratio(*, numerator: float, denominator: float) -> float:
    if math.isfinite(denominator) and denominator > 0.0:
        return numerator / denominator
    if numerator > 0.0:
        return math.inf
    if numerator < 0.0:
        return -math.inf
    return 0.0
