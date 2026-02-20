from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.top_k_selection import (
    EnhancedSelectionResult,
    compute_quality_scores,
    compute_robustness_score,
    compute_signal_correlation_matrix,
    compute_trade_frequency,
    greedy_diversity_select,
    run_enhanced_selection,
)
from feature_research.walkforward.config import WalkforwardResearchConfig


def _make_signal(values: list[float]) -> pd.Series:
    index = pd.date_range("2000-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=index)


def _make_signals(matrix: dict[str, list[float]]) -> dict[str, pd.Series]:
    index = pd.date_range("2000-01-01", periods=len(next(iter(matrix.values()))), freq="B")
    return {label: pd.Series(values, index=index) for label, values in matrix.items()}


def _make_corr(labels: list[str], values: dict[tuple[str, str], float]) -> pd.DataFrame:
    corr = pd.DataFrame(1.0, index=labels, columns=labels)
    for (left, right), corr_value in values.items():
        corr.loc[left, right] = corr_value
        corr.loc[right, left] = corr_value
    return corr


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


def test_robustness_score_perfectly_consistent() -> None:
    assert compute_robustness_score([0.6, 0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.6)


def test_robustness_score_high_variance_penalised() -> None:
    low_var = compute_robustness_score([0.6, 0.58, 0.62, 0.59, 0.61])
    high_var = compute_robustness_score([0.6, 0.10, 1.10, 0.20, 1.00])

    assert low_var > high_var


def test_robustness_score_zero_mean_returns_zero() -> None:
    assert compute_robustness_score([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_robustness_score_negative_mean_returns_zero() -> None:
    assert compute_robustness_score([-0.2, -0.3, -0.1]) == pytest.approx(0.0)


def test_robustness_score_single_block() -> None:
    assert compute_robustness_score([0.5]) == pytest.approx(0.5)


def test_corr_matrix_identical_signals() -> None:
    signals = _make_signals({"A": [1.0, -1.0, 1.0, 1.0], "B": [1.0, -1.0, 1.0, 1.0]})

    corr = compute_signal_correlation_matrix(signals)

    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_corr_matrix_opposite_signals() -> None:
    signals = _make_signals({"A": [1.0, -1.0, 1.0], "B": [-1.0, 1.0, -1.0]})

    corr = compute_signal_correlation_matrix(signals)

    assert corr.loc["A", "B"] == pytest.approx(-1.0)


def test_corr_matrix_diagonal_is_one() -> None:
    signals = _make_signals({"A": [0.5, 0.3, -0.2], "B": [0.1, 0.8, 0.4]})

    corr = compute_signal_correlation_matrix(signals)

    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_corr_matrix_is_symmetric() -> None:
    signals = _make_signals({"A": [1.0, 0.5, -0.2], "B": [0.3, 0.8, 0.1], "C": [-0.1, 0.2, 0.9]})

    corr = compute_signal_correlation_matrix(signals)

    pd.testing.assert_frame_equal(corr, corr.T)


def test_corr_matrix_single_param() -> None:
    signals = _make_signals({"A": [1.0, 0.5, -0.2]})

    corr = compute_signal_correlation_matrix(signals)

    assert corr.shape == (1, 1)
    assert corr.loc["A", "A"] == pytest.approx(1.0)


def test_quality_scores_normalised_to_unit_interval() -> None:
    smoothed = {"A": 0.8, "B": 0.6, "C": 0.4}
    robustness = {"A": 0.5, "B": 0.7, "C": 0.3}

    scores = compute_quality_scores(
        smoothed,
        robustness,
        weight_stability=0.5,
        weight_robustness=0.5,
    )

    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_quality_scores_best_param_scores_one() -> None:
    smoothed = {"A": 1.0, "B": 0.5}
    robustness = {"A": 1.0, "B": 0.5}

    scores = compute_quality_scores(
        smoothed,
        robustness,
        weight_stability=0.5,
        weight_robustness=0.5,
    )

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)


def test_quality_scores_all_equal_inputs_zero() -> None:
    smoothed = {"A": 0.6, "B": 0.6}
    robustness = {"A": 0.4, "B": 0.4}

    scores = compute_quality_scores(
        smoothed,
        robustness,
        weight_stability=0.5,
        weight_robustness=0.5,
    )

    assert scores["A"] == pytest.approx(0.0)
    assert scores["B"] == pytest.approx(0.0)


def test_quality_scores_weights_applied() -> None:
    smoothed = {"A": 1.0, "B": 0.0}
    robustness = {"A": 0.0, "B": 1.0}

    scores_stability = compute_quality_scores(
        smoothed,
        robustness,
        weight_stability=1.0,
        weight_robustness=0.0,
    )
    assert scores_stability["A"] > scores_stability["B"]

    scores_robustness = compute_quality_scores(
        smoothed,
        robustness,
        weight_stability=0.0,
        weight_robustness=1.0,
    )
    assert scores_robustness["B"] > scores_robustness["A"]


def test_greedy_select_returns_k_items() -> None:
    quality = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}
    corr = _make_corr(
        ["A", "B", "C", "D"],
        {
            ("A", "B"): 0.3,
            ("A", "C"): 0.2,
            ("A", "D"): 0.1,
            ("B", "C"): 0.3,
            ("B", "D"): 0.2,
            ("C", "D"): 0.3,
        },
    )

    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.4)

    assert len(selected) == 3


def test_greedy_select_first_is_highest_quality() -> None:
    quality = {"A": 0.9, "B": 0.8, "C": 0.7}
    corr = _make_corr(
        ["A", "B", "C"],
        {("A", "B"): 0.5, ("A", "C"): 0.5, ("B", "C"): 0.5},
    )

    selected = greedy_diversity_select(quality, corr, k=2, diversity_weight=0.4)

    assert selected[0] == "A"


def test_greedy_select_high_corr_penalised() -> None:
    quality = {"A": 1.0, "B": 0.9, "C": 0.55}
    corr = _make_corr(
        ["A", "B", "C"],
        {("A", "B"): 0.99, ("A", "C"): 0.0, ("B", "C"): 0.0},
    )

    selected = greedy_diversity_select(quality, corr, k=2, diversity_weight=0.4)

    assert selected[0] == "A"
    assert selected[1] == "C"


def test_greedy_select_k_larger_than_candidates() -> None:
    quality = {"A": 0.9, "B": 0.8}
    corr = _make_corr(["A", "B"], {("A", "B"): 0.3})

    selected = greedy_diversity_select(quality, corr, k=5, diversity_weight=0.4)

    assert len(selected) == 2


def test_greedy_select_no_diversity_weight_equals_quality_rank() -> None:
    quality = {"A": 0.9, "B": 0.8, "C": 0.7}
    corr = _make_corr(
        ["A", "B", "C"],
        {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0},
    )

    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.0)

    assert selected == ["A", "B", "C"]


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

    def objective(series: pd.Series) -> float:
        std_dev = float(series.std())
        if std_dev == 0.0:
            return 0.0
        return float(series.mean() / std_dev)

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=_dummy_evaluate,
        smoothed_objectives=smoothed,
        objective_metric=objective,
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
        objective_metric=lambda _series: 0.0,
        config=config,
    )

    assert result.selected_labels == []


def test_enhanced_selection_avoids_redundant_highly_correlated_pair() -> None:
    quality = {"A": 1.0, "B": 0.9, "C": 0.89, "D": 0.6}
    corr = _make_corr(
        ["A", "B", "C", "D"],
        {
            ("A", "B"): 0.2,
            ("A", "C"): 0.2,
            ("A", "D"): 0.1,
            ("B", "C"): 0.99,
            ("B", "D"): 0.1,
            ("C", "D"): 0.1,
        },
    )

    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.4)

    assert "A" in selected
    assert "B" in selected
    assert "C" not in selected
    assert "D" in selected
