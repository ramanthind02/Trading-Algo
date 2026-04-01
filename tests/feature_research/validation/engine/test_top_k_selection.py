from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from utils.evaluation.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
)
from utils.evaluation.walkforward.top_k_selection import (
    EnhancedSelectionResult,
    compute_trade_frequency,
    compute_all_trade_frequencies,
    _extract_series_from_evaluator_result,
    run_enhanced_selection,
)


def _make_signal(values: list[float]) -> pd.Series:
    index = pd.date_range("2000-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=index)


def test_trade_frequency_all_nonzero() -> None:
    signal = _make_signal([1.0, -1.0, 1.0, 1.0])

    assert compute_trade_frequency(signal) == pytest.approx(1.0)


def test_trade_frequency_half() -> None:
    signal = _make_signal([1.0, 0.0, 1.0, 0.0])

    assert compute_trade_frequency(signal) == pytest.approx(0.5)


def test_trade_frequency_all_zero() -> None:
    signal = _make_signal([0.0, 0.0, 0.0])

    assert compute_trade_frequency(signal) == pytest.approx(0.0)


def test_trade_frequency_empty_series() -> None:
    signal = pd.Series([], dtype=float)

    assert compute_trade_frequency(signal) == pytest.approx(0.0)


def _make_training_data(n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2000-01-01", periods=n, freq="B")
    candles = pd.DataFrame(
        {"close": np.random.default_rng(0).normal(100.0, 1.0, n)},
        index=index,
    )
    target = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, n), index=index)
    return candles, target


def _dummy_evaluate(
    _candles: pd.DataFrame,
    target: pd.Series,
    params: dict[str, object],
) -> pd.Series:
    lookback = cast(int, params.get("lookback", 1))
    values = np.random.default_rng(seed=lookback).choice(
        [-1.0, 0.0, 1.0],
        size=len(target),
        p=[0.3, 0.2, 0.5],
    )
    return pd.Series(values, index=target.index) * 0.01


def test_run_enhanced_selection_returns_result() -> None:
    candles, target = _make_training_data()
    param_grid: list[dict[str, object]] = [
        {"lookback": value} for value in range(3, 10)
    ]
    smoothed = {f"lookback={value}": 0.5 + value * 0.01 for value in range(3, 10)}
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2010, 1, 1),
        selection_method=WalkforwardSelectionMethod.ENHANCED,
    )

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=_dummy_evaluate,
        smoothed_objectives=smoothed,
        config=config,
    )

    assert isinstance(result, EnhancedSelectionResult)
    assert len(result.selected_labels) <= config.top_k
    assert all(label in smoothed for label in result.selected_labels)


def test_run_enhanced_selection_respects_trade_freq_min() -> None:
    candles, target = _make_training_data(200)
    param_grid: list[dict[str, object]] = [{"lookback": 3}]
    smoothed = {"lookback=3": 0.5}
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2005, 1, 1),
        selection_method=WalkforwardSelectionMethod.ENHANCED,
        trade_freq_min=0.99,
    )

    def zero_signal(
        _candles: pd.DataFrame,
        target_series: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        return pd.Series(0.0, index=target_series.index)

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=zero_signal,
        smoothed_objectives=smoothed,
        config=config,
    )

    assert result.selected_labels == []


def test_run_enhanced_selection_uses_smoothed_objective_only_for_ranking() -> None:
    candles, target = _make_training_data(120)
    param_grid: list[dict[str, object]] = [{"lookback": 3}, {"lookback": 4}, {"lookback": 5}]
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2005, 1, 1),
        selection_method=WalkforwardSelectionMethod.ENHANCED,
        top_k=2,
    )

    smoothed = {"lookback=3": 0.3, "lookback=4": 0.9, "lookback=5": 0.7}

    def nonzero_signal(
        _candles: pd.DataFrame,
        target_series: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        return pd.Series(0.01, index=target_series.index)

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=nonzero_signal,
        smoothed_objectives=smoothed,
        config=config,
    )

    assert result.selected_labels == ["lookback=4", "lookback=5"]


def test_extract_series_from_evaluator_result_plain_series() -> None:
    """_extract_series_from_evaluator_result returns series unchanged when not a tuple."""
    s = _make_signal([1.0, 0.0, 1.0])
    assert _extract_series_from_evaluator_result(s) is s


def test_extract_series_from_evaluator_result_tuple() -> None:
    """_extract_series_from_evaluator_result returns first element when tuple."""
    s = _make_signal([1.0, 0.0, 1.0])
    out = _extract_series_from_evaluator_result((s, {"selected_long_bin": 2}))
    pd.testing.assert_series_equal(out, s)


def test_compute_all_trade_frequencies_handles_tuple_return() -> None:
    """compute_all_trade_frequencies unpacks (series, meta) from evaluator."""
    candles, target = _make_training_data(100)
    param_grid: list[dict[str, object]] = [{"lookback": 3}]

    def evaluator_returns_tuple(
        _candles: pd.DataFrame,
        target_series: pd.Series,
        _params: dict[str, object],
    ) -> tuple[pd.Series, dict[str, object]]:
        return (pd.Series(0.01, index=target_series.index), {"selected_long_bin": 2})

    freqs = compute_all_trade_frequencies(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=evaluator_returns_tuple,
    )
    assert "lookback=3" in freqs
    assert freqs["lookback=3"] == pytest.approx(1.0)


def test_run_enhanced_selection_long_t_stat_filters_negative_objective() -> None:
    """When strategy is long and objective is t_stat, only positive smoothed_objective params are considered."""
    candles, target = _make_training_data(120)
    param_grid: list[dict[str, object]] = [
        {"lookback": 3},
        {"lookback": 4},
        {"lookback": 5},
    ]
    # One negative, two positive
    smoothed = {"lookback=3": -0.5, "lookback=4": 0.9, "lookback=5": 0.7}
    config = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2005, 1, 1),
        selection_method=WalkforwardSelectionMethod.ENHANCED,
        top_k=2,
        objective_metric_name="t_stat",
    )

    def nonzero_signal(
        _candles: pd.DataFrame,
        target_series: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        return pd.Series(0.01, index=target_series.index)

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=nonzero_signal,
        smoothed_objectives=smoothed,
        config=config,
        strategy="long",
    )
    # lookback=3 has negative objective so must be excluded; only 4 and 5 qualify
    assert "lookback=3" not in result.selected_labels
    assert set(result.selected_labels) <= {"lookback=4", "lookback=5"}
    assert len(result.selected_labels) == 2
