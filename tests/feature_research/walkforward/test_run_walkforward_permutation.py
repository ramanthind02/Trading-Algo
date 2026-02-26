"""Tests for run_walkforward_permutation script and vector/candle shuffle paths."""
from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.permutation_helpers import (
    aggregate_oos_metric_from_report,
    two_unit_masks_from_fold_rows,
)
from feature_research.walkforward.runner import _build_fold_rows
from feature_research.walkforward.permutation_core import (
    _one_vector_shuffle_rep_fixed_signal,
    run_vector_shuffle_null,
)
from feature_research.walkforward.run_walkforward_permutation import _two_unit_train_windows


def _minimal_candles_and_target() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    target = pd.Series(0.01, index=index, name="walkforward_target")
    candles = pd.DataFrame({"close": target}, index=index)
    return candles, target


def _minimal_research_config() -> object:
    """Minimal research_config for run_portfolio_simulation (no selected params path)."""
    return types.SimpleNamespace(
        walkforward=types.SimpleNamespace(objective_metric_name="mean_return"),
        bias_spec={"timeframes": [None]},
    )


def _minimal_initial_report(fold_rows: list) -> object:
    """Initial report with selection_summary_df that has no selected params (yields 0 Sharpe)."""
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": int(f["fold_id"]),
                "selected_feature": "",
                "selected_raw_objective": 0.0,
                "selected_smoothed_objective": 0.0,
                "top_k_features": "[]",
            }
            for f in fold_rows
        ],
        columns=[
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "top_k_features",
        ],
    )
    return types.SimpleNamespace(selection_summary_df=selection_summary_df)


def test_two_unit_train_windows() -> None:
    """_two_unit_train_windows returns two (start, end) pairs; second end is exclusive."""
    fold_rows = [
        {
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-02-10"),
            "test_start": pd.Timestamp("2020-02-11"),
        }
    ]
    last_ts = pd.Timestamp("2020-06-01")
    windows = _two_unit_train_windows(fold_rows, last_ts)
    assert len(windows) == 2
    assert windows[0][0] == pd.Timestamp("2020-01-01")
    assert windows[0][1] == pd.Timestamp("2020-02-10")
    assert windows[1][0] == pd.Timestamp("2020-02-11")
    assert windows[1][1] == last_ts + pd.Timedelta(days=1)


def test_run_vector_shuffle_null_returns_correct_shape() -> None:
    """run_vector_shuffle_null returns array of length nreps (single WF run, re-eval with permuted target)."""
    candles, target = _minimal_candles_and_target()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=3,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    fold_rows = _build_fold_rows(target.index, config)
    assert len(fold_rows) >= 1
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(target.index, fold_rows)
    initial_report = _minimal_initial_report(fold_rows)
    research_config = _minimal_research_config()
    # No selected params in selection_summary -> aggregate = 0 for all replicates
    null_metrics = run_vector_shuffle_null(
        reference_candles=candles,
        reference_target=target,
        fold_rows=fold_rows,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        nreps=3,
        random_seed=42,
        initial_report=initial_report,
        research_config=research_config,
        feature_data_by_combo=None,
        portfolio_candles_df=None,
    )
    assert null_metrics.shape == (3,)
    assert np.all(np.isfinite(null_metrics))


def test_run_vector_shuffle_null_n_jobs_2_returns_correct_shape() -> None:
    """run_vector_shuffle_null with n_jobs=2 returns same shape and finite metrics."""
    candles, target = _minimal_candles_and_target()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=3,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    fold_rows = _build_fold_rows(target.index, config)
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(target.index, fold_rows)
    initial_report = _minimal_initial_report(fold_rows)
    research_config = _minimal_research_config()
    null_metrics = run_vector_shuffle_null(
        reference_candles=candles,
        reference_target=target,
        fold_rows=fold_rows,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        nreps=4,
        random_seed=123,
        initial_report=initial_report,
        research_config=research_config,
        feature_data_by_combo=None,
        portfolio_candles_df=None,
        n_jobs=2,
    )
    assert null_metrics.shape == (4,)
    assert np.all(np.isfinite(null_metrics))


def test_one_vector_shuffle_rep_fixed_signal_returns_finite_and_reproducible() -> None:
    """Fixed-signal path: one replicate returns finite float; same seed gives same result."""
    candles, target = _minimal_candles_and_target()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=3,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    fold_rows = _build_fold_rows(target.index, config)
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(target.index, fold_rows)
    # One fold with dummy OOS signal (same index as a slice of target)
    fold_id = int(fold_rows[0]["fold_id"])
    test_start = pd.Timestamp(fold_rows[0]["test_start"])
    test_end = pd.Timestamp(fold_rows[0]["test_end"])
    oos_index = target.index[(target.index >= test_start) & (target.index <= test_end)]
    fixed_oos_signal_by_fold = {
        fold_id: pd.Series(0.5, index=oos_index),
    }
    result = _one_vector_shuffle_rep_fixed_signal(
        seed=42,
        reference_target=target,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
        fold_rows=fold_rows,
        objective_metric_name="sharpe",
    )
    assert np.isfinite(result)
    result2 = _one_vector_shuffle_rep_fixed_signal(
        seed=42,
        reference_target=target,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
        fold_rows=fold_rows,
        objective_metric_name="sharpe",
    )
    assert result == result2


def test_run_vector_shuffle_null_with_fixed_signal_returns_correct_shape() -> None:
    """run_vector_shuffle_null with fixed_oos_signal_by_fold uses no-refit path and returns correct shape."""
    candles, target = _minimal_candles_and_target()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=3,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    fold_rows = _build_fold_rows(target.index, config)
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(target.index, fold_rows)
    fold_id = int(fold_rows[0]["fold_id"])
    test_start = pd.Timestamp(fold_rows[0]["test_start"])
    test_end = pd.Timestamp(fold_rows[0]["test_end"])
    oos_index = target.index[(target.index >= test_start) & (target.index <= test_end)]
    fixed_oos_signal_by_fold = {fold_id: pd.Series(0.5, index=oos_index)}
    initial_report = _minimal_initial_report(fold_rows)
    research_config = _minimal_research_config()
    null_metrics = run_vector_shuffle_null(
        reference_candles=candles,
        reference_target=target,
        fold_rows=fold_rows,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        nreps=3,
        random_seed=100,
        initial_report=initial_report,
        research_config=research_config,
        feature_data_by_combo=None,
        portfolio_candles_df=None,
        fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
        objective_metric_name="sharpe",
    )
    assert null_metrics.shape == (3,)
    assert np.all(np.isfinite(null_metrics))


def test_run_vector_shuffle_null_reproducible_with_n_jobs() -> None:
    """Same seed yields same null_metrics for n_jobs=1 and n_jobs=2."""
    candles, target = _minimal_candles_and_target()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=3,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    fold_rows = _build_fold_rows(target.index, config)
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(target.index, fold_rows)
    initial_report = _minimal_initial_report(fold_rows)
    research_config = _minimal_research_config()
    common = dict(
        reference_candles=candles,
        reference_target=target,
        fold_rows=fold_rows,
        unit1_mask=unit1_mask,
        unit2_mask=unit2_mask,
        nreps=3,
        random_seed=99,
        initial_report=initial_report,
        research_config=research_config,
        feature_data_by_combo=None,
        portfolio_candles_df=None,
    )
    null_1 = run_vector_shuffle_null(**common, n_jobs=1)
    null_2 = run_vector_shuffle_null(**common, n_jobs=2)
    np.testing.assert_array_almost_equal(null_1, null_2)


def test_p_value_in_valid_range() -> None:
    """p-value from (1 + n_ge) / (nreps + 1) is in [0, 1]."""
    nreps = 10
    original = 0.5
    null_metrics = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_ge = int((null_metrics >= original).sum())
    p_value = float(1 + n_ge) / float(nreps + 1)
    assert 0 <= p_value <= 1
    assert n_ge == 6
    assert p_value == pytest.approx(7 / 11)  # (1 + 6) / (10 + 1)
