"""Unit tests for objective metric resolver and config spec."""
from __future__ import annotations

import sys
import types
from typing import Callable, cast

import pandas as pd
import pytest

from feature_selection.validation.objective_metrics import (
    BuiltinMetricName,
    ObjectiveMetricSpec,
    metric_calmar,
    metric_profit_factor,
    metric_sharpe,
    metric_sortino,
    metric_t_stat,
    resolve_objective_metric,
)


def test_spec_requires_exactly_one_source() -> None:
    """ObjectiveMetricSpec accepts exactly one metric source."""
    with pytest.raises(ValueError, match='exactly one'):
        ObjectiveMetricSpec()

    with pytest.raises(ValueError, match='exactly one'):
        ObjectiveMetricSpec(builtin='sharpe', callable_path='mod:fn')


def test_resolve_builtin_metric_with_kwargs() -> None:
    """Resolver binds kwargs for builtin metrics."""
    returns = pd.Series([1.0, -1.0, 2.0, -0.5])
    spec = ObjectiveMetricSpec(builtin='sharpe', kwargs={'risk_free_rate': 0.125})

    metric = resolve_objective_metric(spec)

    assert metric(returns) == pytest.approx(metric_sharpe(returns, risk_free_rate=0.125))


def test_resolve_callable_path_metric_with_kwargs() -> None:
    """Resolver loads callable from import path and binds kwargs."""
    module_name = 'temp_metric_module'
    module = types.ModuleType(module_name)

    def custom_metric(returns: pd.Series, *, scale: float = 1.0) -> float:
        return float(returns.mean() * scale)

    module.custom_metric = custom_metric  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    try:
        returns = pd.Series([1.0, 3.0, 5.0])
        spec = ObjectiveMetricSpec(
            callable_path=f'{module_name}:custom_metric',
            kwargs={'scale': 2.0},
        )

        metric = resolve_objective_metric(spec)

        assert metric(returns) == pytest.approx(6.0)
    finally:
        del sys.modules[module_name]


def test_invalid_callable_path_format_raises() -> None:
    """callable_path must use module:function format."""
    with pytest.raises(ValueError, match='module:function'):
        ObjectiveMetricSpec(callable_path='bad.path')


def test_invalid_builtin_name_raises_with_supported_names() -> None:
    """Invalid builtin names return an explicit supported-name error."""
    with pytest.raises(
        ValueError,
        match='Supported builtins: always_zero, calmar, mean_return, profit_factor, sharpe, sortino, t_stat',
    ):
        ObjectiveMetricSpec(builtin=cast(BuiltinMetricName, 'not_a_metric'))


@pytest.mark.parametrize(
    ('metric_func', 'returns', 'expected'),
    (
        (metric_sharpe, pd.Series([0.01, 0.01, 0.01]), float('inf')),
        (metric_sharpe, pd.Series([-0.01, -0.01, -0.01]), float('-inf')),
        (metric_sharpe, pd.Series([0.0, 0.0, 0.0]), 0.0),
        (metric_sortino, pd.Series([0.01, 0.01, 0.01]), float('inf')),
        (metric_sortino, pd.Series([-0.01, -0.01, -0.01]), float('-inf')),
        (metric_sortino, pd.Series([0.0, 0.0, 0.0]), 0.0),
        (metric_calmar, pd.Series([0.01, 0.01, 0.01]), float('inf')),
        (metric_calmar, pd.Series([0.0, 0.0, 0.0]), 0.0),
        (metric_t_stat, pd.Series([0.01, 0.01, 0.01]), float('inf')),
        (metric_t_stat, pd.Series([-0.01, -0.01, -0.01]), float('-inf')),
        (metric_t_stat, pd.Series([0.0, 0.0, 0.0]), 0.0),
        (metric_profit_factor, pd.Series([0.01, 0.01, 0.01]), float('inf')),
        (metric_profit_factor, pd.Series([0.0, 0.0, 0.0]), 0.0),
    ),
)
def test_zero_risk_denominator_policy_is_deterministic(
    metric_func: Callable[[pd.Series], float],
    returns: pd.Series,
    expected: float,
) -> None:
    """Zero-risk denominators map to deterministic signed/neutral outputs."""
    result = metric_func(returns)
    if expected == 0.0:
        assert result == 0.0
    else:
        assert result == pytest.approx(expected)
