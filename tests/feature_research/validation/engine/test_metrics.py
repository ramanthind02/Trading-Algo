from __future__ import annotations

import pandas as pd
from pathlib import Path
import pytest
from quantfoundry_core.metrics import MetricName, compute_scalar_metric

from feature_selection.validation.objective_metrics import (
    resolve_objective_metric_name as resolve_objective_metric,
)


def test_resolve_mean_return_metric() -> None:
    metric = resolve_objective_metric("mean_return")
    returns = pd.Series([0.01, -0.02, 0.03, 0.02])

    assert metric(returns) == pytest.approx(returns.mean())


def test_resolve_sharpe_metric() -> None:
    metric = resolve_objective_metric("sharpe")
    returns = pd.Series([0.02, -0.01, 0.015, -0.005])
    indexed = returns.set_axis(pd.date_range("2024-01-01", periods=len(returns), freq="D"))
    expected = compute_scalar_metric(indexed, MetricName.SHARPE, periods_per_year=1)

    assert metric(returns) == pytest.approx(expected)


def test_resolve_sortino_metric() -> None:
    metric = resolve_objective_metric("sortino")
    returns = pd.Series([0.03, -0.01, 0.02, -0.02])
    indexed = returns.set_axis(pd.date_range("2024-01-01", periods=len(returns), freq="D"))
    expected = compute_scalar_metric(indexed, MetricName.SORTINO, periods_per_year=1)

    assert metric(returns) == pytest.approx(expected)


@pytest.mark.parametrize("metric_name", ["Sharpe", "SORTINO", "omega"])
def test_unknown_metric_name_raises_value_error(metric_name: str) -> None:
    with pytest.raises(ValueError, match=metric_name):
        resolve_objective_metric(metric_name)


def test_empty_metric_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported objective metric"):
        resolve_objective_metric("")


def test_metric_selection_is_deterministic() -> None:
    """Callables may be fresh lambdas; behavior on the same inputs must match."""
    s = pd.Series([0.02, -0.01, 0.015, -0.005])
    for name in ("sharpe", "sortino", "mean_return"):
        a = resolve_objective_metric(name)
        b = resolve_objective_metric(name)
        assert a(s) == b(s)


def test_mean_return_empty_series_returns_zero() -> None:
    metric = resolve_objective_metric("mean_return")

    assert metric(pd.Series(dtype=float)) == 0.0


def test_mean_return_all_nan_series_returns_zero() -> None:
    metric = resolve_objective_metric("mean_return")

    assert metric(pd.Series([float("nan"), float("nan")])) == 0.0


def test_sharpe_zero_variance_returns_infinity() -> None:
    """Positive mean with zero volatility yields an infinite Sharpe under the ratio helper."""
    metric = resolve_objective_metric("sharpe")

    assert metric(pd.Series([0.01, 0.01, 0.01])) == float("inf")


def test_sortino_no_downside_returns_infinity() -> None:
    """No below-target returns → zero downside vol → ratio convention is ``inf``."""
    metric = resolve_objective_metric("sortino")

    assert metric(pd.Series([0.01, 0.02, 0.03])) == float("inf")
