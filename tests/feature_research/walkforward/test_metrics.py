from __future__ import annotations

import pandas as pd
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.metrics import resolve_objective_metric


def test_resolve_mean_return_metric() -> None:
    metric = resolve_objective_metric("mean_return")
    returns = pd.Series([0.01, -0.02, 0.03, 0.02])

    assert metric(returns) == pytest.approx(returns.mean())


def test_resolve_sharpe_metric() -> None:
    metric = resolve_objective_metric("sharpe")
    returns = pd.Series([0.02, -0.01, 0.015, -0.005])
    expected = float(returns.mean() / returns.std(ddof=0))

    assert metric(returns) == pytest.approx(expected)


def test_resolve_sortino_metric() -> None:
    metric = resolve_objective_metric("sortino")
    returns = pd.Series([0.03, -0.01, 0.02, -0.02])
    downside = returns[returns < 0]
    expected = float(returns.mean() / downside.std(ddof=0))

    assert metric(returns) == pytest.approx(expected)


@pytest.mark.parametrize("metric_name", ["Sharpe", "SORTINO", "omega"])
def test_unknown_metric_name_raises_value_error(metric_name: str) -> None:
    with pytest.raises(ValueError, match=metric_name):
        resolve_objective_metric(metric_name)


def test_empty_metric_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported objective metric"):
        resolve_objective_metric("")


def test_metric_selection_is_deterministic() -> None:
    assert resolve_objective_metric("sharpe") is resolve_objective_metric("sharpe")
    assert resolve_objective_metric("sortino") is resolve_objective_metric("sortino")
    assert resolve_objective_metric("mean_return") is resolve_objective_metric("mean_return")


def test_mean_return_empty_series_returns_zero() -> None:
    metric = resolve_objective_metric("mean_return")

    assert metric(pd.Series(dtype=float)) == 0.0


def test_mean_return_all_nan_series_returns_zero() -> None:
    metric = resolve_objective_metric("mean_return")

    assert metric(pd.Series([float("nan"), float("nan")])) == 0.0


def test_sharpe_zero_variance_returns_zero() -> None:
    metric = resolve_objective_metric("sharpe")

    assert metric(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_sortino_no_downside_returns_zero() -> None:
    metric = resolve_objective_metric("sortino")

    assert metric(pd.Series([0.01, 0.02, 0.03])) == 0.0
