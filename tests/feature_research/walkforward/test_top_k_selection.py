from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.top_k_selection import (
    EnhancedSelectionResult,
    compute_trade_frequency,
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
        use_enhanced_selection=True,
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
        use_enhanced_selection=True,
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
        use_enhanced_selection=True,
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
