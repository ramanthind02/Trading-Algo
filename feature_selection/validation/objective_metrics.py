"""Objective metric specification and resolver helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Callable, Literal, Mapping, cast

import numpy as np
import pandas as pd

BuiltinMetricName = Literal['sharpe', 'sortino', 'calmar', 't_stat', 'profit_factor', 'always_zero']
ObjectiveMetricCallable = Callable[[pd.Series], float]
_MetricFactoryCallable = Callable[..., float]


def metric_sharpe(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """Sharpe ratio with optional annualization."""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    excess = clean - risk_free_rate
    volatility = float(excess.std(ddof=0))
    annualization_scale = math.sqrt(max(annualization_factor, 0.0))
    numerator = float(excess.mean() * annualization_scale)
    return _deterministic_ratio(numerator=numerator, denominator=volatility)


def metric_sortino(
    returns: pd.Series,
    *,
    target_return: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """Sortino ratio using downside volatility only."""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    excess = clean - target_return
    downside = excess[excess < 0.0]
    downside_risk = float(downside.std(ddof=0))
    annualization_scale = math.sqrt(max(annualization_factor, 0.0))
    numerator = float(excess.mean() * annualization_scale)
    return _deterministic_ratio(numerator=numerator, denominator=downside_risk)


def metric_calmar(
    returns: pd.Series,
    *,
    annualization_factor: float = 1.0,
) -> float:
    """Calmar ratio as annualized mean return over max drawdown."""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    equity_curve = (1.0 + clean).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = abs(float(drawdown.min()))
    annualized_return = float(clean.mean() * annualization_factor)
    return _deterministic_ratio(numerator=annualized_return, denominator=max_drawdown)


def metric_t_stat(returns: pd.Series) -> float:
    """One-sample t-statistic of mean returns against zero."""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    n_obs = int(clean.shape[0])
    if n_obs < 2:
        return 0.0
    mean_return = float(clean.mean())
    sample_std = float(clean.std(ddof=1))
    standard_error = sample_std / math.sqrt(float(n_obs))
    return _deterministic_ratio(numerator=mean_return, denominator=standard_error)


def metric_profit_factor(returns: pd.Series) -> float:
    """Profit factor as gross gains divided by gross losses."""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    gross_gain = float(clean[clean > 0.0].sum())
    gross_loss = abs(float(clean[clean < 0.0].sum()))
    return _deterministic_ratio(numerator=gross_gain, denominator=gross_loss)


def metric_always_zero(_returns: pd.Series) -> float:
    """Always returns zero - useful for forcing test failures."""
    return 0.0


_BUILTIN_OBJECTIVE_METRICS: Mapping[BuiltinMetricName, _MetricFactoryCallable] = {
    'sharpe': metric_sharpe,
    'sortino': metric_sortino,
    'calmar': metric_calmar,
    't_stat': metric_t_stat,
    'profit_factor': metric_profit_factor,
    'always_zero': metric_always_zero,
}


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


def resolve_objective_metric(spec: ObjectiveMetricSpec) -> ObjectiveMetricCallable:
    """Resolve objective metric callable from builtin or import path."""
    metric_callable = (
        _BUILTIN_OBJECTIVE_METRICS[spec.builtin]
        if spec.builtin is not None
        else _import_metric_callable(spec.callable_path)
    )
    return _bind_metric_kwargs(metric_callable, spec.kwargs)


def _import_metric_callable(callable_path: str | None) -> _MetricFactoryCallable:
    if callable_path is None:
        raise ValueError('callable_path is required for custom objective metrics.')

    module_name, _, function_name = callable_path.partition(':')
    module = importlib.import_module(module_name)
    metric_object = getattr(module, function_name)
    if not callable(metric_object):
        raise TypeError(f'Imported object at {callable_path!r} is not callable.')
    return cast(_MetricFactoryCallable, metric_object)


def _bind_metric_kwargs(
    metric_callable: _MetricFactoryCallable,
    kwargs: Mapping[str, object],
) -> ObjectiveMetricCallable:
    if not kwargs:
        return lambda returns: float(metric_callable(returns))

    bound_kwargs = dict(kwargs)
    return lambda returns: float(metric_callable(returns, **bound_kwargs))


def _clean_returns(returns: pd.Series) -> pd.Series:
    numeric_returns = pd.to_numeric(returns, errors='coerce').dropna()
    if numeric_returns.empty:
        return numeric_returns
    finite_mask = np.isfinite(numeric_returns.to_numpy(dtype=float))
    return numeric_returns.loc[finite_mask]


def _deterministic_ratio(*, numerator: float, denominator: float) -> float:
    """Return stable ratio values when denominator is zero or non-finite."""
    if math.isfinite(denominator) and denominator > 0.0:
        return numerator / denominator
    if numerator > 0.0:
        return math.inf
    if numerator < 0.0:
        return -math.inf
    return 0.0
