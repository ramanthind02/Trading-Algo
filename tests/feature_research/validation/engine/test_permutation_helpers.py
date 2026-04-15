"""Unit tests for walkforward permutation helpers (two-unit target permute, masks, aggregate)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.evaluation.permutation_test.permutation_nulls import (
    aggregate_oos_metric_from_report,
    permute_target_in_two_units,
    two_unit_masks_from_fold_rows,
)


def test_permute_target_in_two_units_preserves_values_and_length() -> None:
    """Permuted series has same index, same set of values per unit, length unchanged."""
    rng = np.random.default_rng(42)
    n = 100
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    target = pd.Series(np.linspace(0, 1, n), index=index, name="target")
    unit1_mask = pd.Series(index < index[40], index=index)
    unit2_mask = pd.Series(index >= index[40], index=index)

    out = permute_target_in_two_units(target, unit1_mask, unit2_mask, rng)

    assert out.index.equals(target.index)
    assert len(out) == len(target)
    u1_vals_orig = target.loc[unit1_mask].sort_values().values
    u1_vals_out = out.loc[unit1_mask].sort_values().values
    np.testing.assert_allclose(u1_vals_orig, u1_vals_out)
    u2_vals_orig = target.loc[unit2_mask].sort_values().values
    u2_vals_out = out.loc[unit2_mask].sort_values().values
    np.testing.assert_allclose(u2_vals_orig, u2_vals_out)


def test_permute_target_in_two_units_deterministic_with_same_seed() -> None:
    """Same seed produces same permutation."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    target = pd.Series(np.random.randn(50), index=index)
    unit1 = index < index[20]
    unit2 = index >= index[20]
    u1_mask = pd.Series(unit1, index=index)
    u2_mask = pd.Series(unit2, index=index)

    out1 = permute_target_in_two_units(target, u1_mask, u2_mask, rng1)
    out2 = permute_target_in_two_units(target, u1_mask, u2_mask, rng2)

    pd.testing.assert_series_equal(out1, out2)


def test_permute_target_in_two_units_single_element_units() -> None:
    """Units with 0 or 1 element do not raise; values unchanged for single-element."""
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    target = pd.Series(np.arange(10.0), index=index)
    unit1_mask = pd.Series(False, index=index)
    unit1_mask.iloc[0] = True
    unit2_mask = pd.Series(True, index=index)
    unit2_mask.iloc[0] = False

    out = permute_target_in_two_units(target, unit1_mask, unit2_mask, rng)

    assert out.iloc[0] == target.iloc[0]
    assert set(out.iloc[1:].values) == set(target.iloc[1:].values)


def test_two_unit_masks_from_fold_rows_empty_folds() -> None:
    """Empty fold_rows returns all-False masks."""
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    unit1, unit2 = two_unit_masks_from_fold_rows(index, [])
    assert not unit1.any()
    assert not unit2.any()
    assert len(unit1) == len(index) and len(unit2) == len(index)


def test_two_unit_masks_from_fold_rows_one_fold() -> None:
    """First fold train = unit1, from test_start to end = unit2."""
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    train_start = pd.Timestamp("2020-01-01")
    train_end = pd.Timestamp("2020-02-10")
    test_start = pd.Timestamp("2020-02-11")
    test_end = pd.Timestamp("2020-03-20")
    fold_rows = [
        {
            "fold_id": 0,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "_train_mask": (index >= train_start) & (index < train_end),
            "_test_mask": (index >= test_start) & (index < test_end),
        }
    ]
    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(index, fold_rows)

    assert unit1_mask.sum() == ((index >= train_start) & (index < train_end)).sum()
    last_ts = index.max()
    assert unit2_mask.sum() == ((index >= test_start) & (index <= last_ts)).sum()
    assert not (unit1_mask & unit2_mask).any()


def test_aggregate_oos_metric_from_report_empty_df() -> None:
    """Empty portfolio_results_df returns 0."""
    class EmptyReport:
        portfolio_results_df = None
    assert aggregate_oos_metric_from_report(EmptyReport(), treat_no_selection_as_zero=True) == 0.0

    class EmptyDfReport:
        portfolio_results_df = pd.DataFrame()
    assert aggregate_oos_metric_from_report(EmptyDfReport(), treat_no_selection_as_zero=True) == 0.0


def test_aggregate_oos_metric_from_report_treats_no_selection_as_zero() -> None:
    """Folds with no_selected_params contribute 0 when treat_no_selection_as_zero=True."""
    df = pd.DataFrame({
        "fold_id": [0, 1],
        "oos_portfolio_sharpe": [np.nan, 0.5],
        "error": ["no_selected_params", ""],
    })
    class Report:
        portfolio_results_df = df

    out = aggregate_oos_metric_from_report(Report(), treat_no_selection_as_zero=True)
    assert out == 0.25  # (0 + 0.5) / 2
