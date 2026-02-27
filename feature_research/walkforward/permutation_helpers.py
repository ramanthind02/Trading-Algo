"""Helpers for walkforward permutation testing.

Two-unit blocking: first fold train vs all remaining data.
Used by run_walkforward_permutation.py for vector (target) and candle shuffle.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


@contextlib.contextmanager
def _joblib_tqdm(total: int, desc: str, unit: str = "rep"):
    """Context manager that patches joblib to update a tqdm progress bar as batches complete."""
    import joblib.parallel

    pbar = tqdm(total=total, desc=desc, unit=unit)
    _pbar_ref: list[Optional[tqdm]] = [pbar]

    class _TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def _dispatch_new(self) -> None:
            super()._dispatch_new()
            if _pbar_ref[0] is not None:
                _pbar_ref[0].update(n=self.batch_size)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCallback
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        _pbar_ref[0] = None
        pbar.close()


def permute_target_in_two_units(
    target: pd.Series,
    unit1_mask: pd.Series,
    unit2_mask: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """Permute target values within each of two disjoint index units.

    Values in unit1 are permuted among unit1 indices; values in unit2
    among unit2 indices. Breaks feature-target alignment while preserving
    marginal distribution per unit. Same null as permuting each feature
    vector; one shuffle per replicate can be shared across all param combos.

    Parameters
    ----------
    target : pd.Series
        Target return series (e.g. walkforward_target). Must share index with masks.
    unit1_mask : pd.Series
        Boolean series, True for indices in first unit (e.g. first fold train).
    unit2_mask : pd.Series
        Boolean series, True for indices in second unit (e.g. rest of data).
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    pd.Series
        New series with same index as target; values permuted within unit1 and unit2.
    """
    out = target.copy()
    idx = target.index
    if not unit1_mask.index.equals(idx) or not unit2_mask.index.equals(idx):
        unit1_mask = unit1_mask.reindex(idx).fillna(False)
        unit2_mask = unit2_mask.reindex(idx).fillna(False)
    # Use iloc to avoid duplicate-index assignment issues (multi-ticker)
    u1_pos = np.where(unit1_mask.values)[0]
    u2_pos = np.where(unit2_mask.values)[0]
    if len(u1_pos) > 1:
        perm1 = rng.permutation(len(u1_pos))
        out.iloc[u1_pos] = target.iloc[u1_pos].values[perm1]
    if len(u2_pos) > 1:
        perm2 = rng.permutation(len(u2_pos))
        out.iloc[u2_pos] = target.iloc[u2_pos].values[perm2]
    return out


def two_unit_masks_from_fold_rows(
    index: pd.DatetimeIndex,
    fold_rows: list[dict[str, object]],
) -> tuple[pd.Series, pd.Series]:
    """Build unit1 (first fold train) and unit2 (rest) boolean masks from fold rows.

    Unit 1: train_start_0 <= t < train_end_0.
    Unit 2: test_start_0 <= t <= last_timestamp (first fold test through end of data).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full datetime index (e.g. candles_df.index or target.index).
    fold_rows : list[dict]
        From runner._build_fold_rows; each has train_start, train_end, test_start, test_end.

    Returns
    -------
    unit1_mask, unit2_mask : pd.Series
        Boolean series aligned to index.
    """
    if not fold_rows:
        unit1 = pd.Series(False, index=index)
        unit2 = pd.Series(False, index=index)
        return (unit1, unit2)
    first = fold_rows[0]
    train_start = pd.Timestamp(first["train_start"])
    train_end = pd.Timestamp(first["train_end"])
    test_start = pd.Timestamp(first["test_start"])
    last_ts = pd.Timestamp(index.max())
    unit1_mask = (index >= train_start) & (index < train_end)
    unit2_mask = (index >= test_start) & (index <= last_ts)
    return (
        pd.Series(unit1_mask, index=index),
        pd.Series(unit2_mask, index=index),
    )


def aggregate_oos_metric_from_report(
    report: object,
    treat_no_selection_as_zero: bool = True,
    metric_fn: Callable[[pd.Series], float] | None = None,
) -> float:
    """Aggregate walkforward OOS metric from report for permutation null.

    When metric_fn is provided and report has non-empty aggregate_oos_returns,
    returns metric_fn(aggregate_oos_returns) (one metric over all OOS folds).
    Otherwise uses mean of per-fold OOS portfolio Sharpe (fallback). Folds with
    no selected params or missing/error are treated as 0 when
    treat_no_selection_as_zero is True (required for a well-defined null).

    For permutation testing, the original metric must be computed with this
    same function so that the test statistic is consistent: both the original
    and all replicates use either the aggregate statistic or the per-fold mean
    fallback. Mixing aggregate for the original with per-fold mean for the
    null distribution would invalidate the p-value.

    Parameters
    ----------
    report : WalkforwardRunReport
        From run_walkforward_research.
    treat_no_selection_as_zero : bool
        If True, folds with no_selected_params or nan metric contribute 0
        (only when using per-fold mean fallback).
    metric_fn : callable, optional
        If provided and report.aggregate_oos_returns is non-empty, applied to
        the concatenated OOS returns; otherwise per-fold mean is used.

    Returns
    -------
    float
        Single scalar for this replicate (aggregate or mean fold).
    """
    agg_returns = getattr(report, "aggregate_oos_returns", None)
    portfolio_results_df = getattr(report, "portfolio_results_df", None)
    if metric_fn is not None and agg_returns is not None and not agg_returns.empty:
        clean = agg_returns.dropna()
        if clean.empty:
            return 0.0
        val = metric_fn(clean)
        out = float(val) if np.isfinite(val) else 0.0
        return out

    if portfolio_results_df is None or portfolio_results_df.empty:
        return 0.0
    df = portfolio_results_df
    sharpes = df["oos_portfolio_sharpe"].astype(float)
    errors = df.get("error", pd.Series("", index=df.index))
    if treat_no_selection_as_zero:
        no_sel = (errors == "no_selected_params") | (errors == "missing_selection_summary")
        sharpes = sharpes.copy()
        sharpes.loc[no_sel] = 0.0
        sharpes = sharpes.fillna(0.0)
    fallback_mean = float(sharpes.mean())
    return fallback_mean
