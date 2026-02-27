"""Shared permutation test logic for walkforward and OOS.

Two-unit blocking; fixed feature vector (continuous) or refit (rule-based) null.
Used by run_walkforward_permutation.py and run_oos_permutation.py.
"""
from __future__ import annotations

import types
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_research.config import FeatureType
from feature_research.walkforward.permutation_helpers import (
    _joblib_tqdm,
    aggregate_oos_metric_from_report,
    permute_target_in_two_units,
)
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_research.walkforward.portfolio_evaluator import (
    _build_one_base_model_with_members,
    _normalize_strategy,
    _normalize_timeframe,
    build_research_portfolio,
)
from feature_research.walkforward.runner import (
    _parse_top_k_param_labels,
    _resolve_feature_type,
    run_portfolio_simulation,
)
from utils.core.enums import TimeFrame


def run_vector_shuffle_null_vectorized(
    reference_target: pd.Series,
    unit1_mask: pd.Series,
    unit2_mask: pd.Series,
    fixed_oos_signal_by_fold: dict[int, pd.Series],
    fold_rows: list[dict],
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
    canonical_oos_index: pd.Index | None = None,
) -> np.ndarray:
    """Vectorized null distribution: all nreps permutations at once, no loop overhead.

    Generates all nreps permuted targets as a (nreps, T) matrix, computes all
    (nreps, n_oos) return matrices via broadcast multiply, and applies the
    objective metric vectorized across axis=1. ~10–100x faster than looping
    over _one_vector_shuffle_rep_fixed_signal for nreps >= 100.

    Parameters
    ----------
    reference_target : pd.Series
        Target return series (walkforward_target, possibly multi-ticker with dups).
    unit1_mask, unit2_mask : pd.Series
        Boolean masks defining two-unit blocking (e.g., first fold train vs rest).
    fixed_oos_signal_by_fold : dict[int, pd.Series]
        OOS signals per fold (pre-fitted, no refit in null loop).
    fold_rows : list[dict]
        Fold metadata with train_start, train_end, test_start, test_end.
    nreps : int
        Number of null replicates.
    random_seed : int or None
        Seed for reproducibility.
    objective_metric_name : str
        Metric name: "sharpe", "mean_return", "t_stat", "sortino".
    canonical_oos_index : pd.Index or None
        If provided, align aggregate returns to this index (padding with 0s).

    Returns
    -------
    np.ndarray
        Shape (nreps,); metric value for each replicate.
    """
    # Prepare target (handle multi-ticker duplicates)
    target_unique = (
        reference_target.groupby(level=0).first()
        if reference_target.index.duplicated().any()
        else reference_target
    )
    T = len(target_unique)
    target_vals = target_unique.values.copy()  # shape (T,)
    target_idx = target_unique.index

    # Compute permutation positions (once, outside rep loop)
    unit1_mask_aligned = unit1_mask.reindex(target_idx).fillna(False)
    unit2_mask_aligned = unit2_mask.reindex(target_idx).fillna(False)
    u1_pos = np.where(unit1_mask_aligned.values)[0]
    u2_pos = np.where(unit2_mask_aligned.values)[0]

    # Determine output OOS index
    if canonical_oos_index is not None and len(canonical_oos_index) > 0:
        oos_index = canonical_oos_index
    else:
        # Build OOS index from fold test windows
        collected: list[pd.Series] = []
        for fold_row in fold_rows:
            if not all(k in fold_row for k in ("test_start", "test_end")):
                continue
            test_start = pd.Timestamp(fold_row["test_start"])
            test_end = pd.Timestamp(fold_row["test_end"])
            fold_test_idx = target_idx[(target_idx >= test_start) & (target_idx <= test_end)]
            if len(fold_test_idx) > 0:
                collected.append(pd.Series(0, index=fold_test_idx))

        if not collected:
            return np.zeros(nreps, dtype=float)
        oos_index = pd.concat(collected).index.unique().sort_values()

    n_oos_cols = len(oos_index)

    # Generate all nreps permutations at once (vectorized)
    rng = np.random.default_rng(random_seed)
    if len(u1_pos) > 1:
        perm1 = rng.random((nreps, len(u1_pos))).argsort(axis=1)  # (nreps, |u1|)
    else:
        perm1 = np.zeros((nreps, len(u1_pos)), dtype=np.int64)

    if len(u2_pos) > 1:
        perm2 = rng.random((nreps, len(u2_pos))).argsort(axis=1)  # (nreps, |u2|)
    else:
        perm2 = np.zeros((nreps, len(u2_pos)), dtype=np.int64)

    # All permuted targets: (nreps, T)
    all_targets = np.tile(target_vals, (nreps, 1))  # broadcast copy
    if len(u1_pos) > 0:
        all_targets[:, u1_pos] = target_vals[u1_pos][perm1]
    if len(u2_pos) > 0:
        all_targets[:, u2_pos] = target_vals[u2_pos][perm2]

    # Allocate output matrix: (nreps, n_oos_cols)
    all_returns = np.zeros((nreps, n_oos_cols), dtype=np.float64)

    # Populate OOS returns by fold
    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        oos_signal = fixed_oos_signal_by_fold.get(fold_id)

        if not all(k in fold_row for k in ("test_start", "test_end")):
            continue

        test_start = pd.Timestamp(fold_row["test_start"])
        test_end = pd.Timestamp(fold_row["test_end"])
        full_test_index = target_idx[(target_idx >= test_start) & (target_idx <= test_end)]

        if len(full_test_index) == 0:
            continue

        if oos_signal is None or oos_signal.empty:
            # No signal for this fold: contribute zeros (already zero-initialized)
            continue

        # Align signal with target (same as aggregate_signal_target_returns)
        oos_target_at_signal = target_unique.reindex(oos_signal.index).dropna()
        oos_signal_aligned = oos_signal.reindex(oos_target_at_signal.index).dropna()

        if oos_signal_aligned.empty:
            continue

        # Find integer positions in target_unique and oos_index
        oos_int_pos = target_idx.get_indexer(oos_signal_aligned.index)
        oos_col_pos = oos_index.get_indexer(oos_signal_aligned.index)

        # Keep only indices valid in BOTH target and oos_index
        valid_both = (oos_int_pos >= 0) & (oos_col_pos >= 0)
        if not valid_both.any():
            continue

        oos_int_pos_valid = oos_int_pos[valid_both]
        oos_col_pos_valid = oos_col_pos[valid_both]
        signal_vals_valid = oos_signal_aligned.values[valid_both]

        # Broadcast multiply: (nreps, len(valid)) * (1, len(valid))
        all_returns[:, oos_col_pos_valid] = all_targets[:, oos_int_pos_valid] * signal_vals_valid[np.newaxis, :]

    # Apply metric vectorized (no loop over reps)
    if objective_metric_name == "sharpe":
        means = all_returns.mean(axis=1)
        stds = all_returns.std(axis=1, ddof=0)
        null_metrics = np.where(stds > 0, means / stds, 0.0)
    elif objective_metric_name == "mean_return":
        null_metrics = all_returns.mean(axis=1)
    elif objective_metric_name == "t_stat":
        means = all_returns.mean(axis=1)
        stds = all_returns.std(axis=1, ddof=0)
        n = all_returns.shape[1]
        null_metrics = np.where(stds > 0, means / stds * np.sqrt(n), 0.0)
    elif objective_metric_name == "sortino":
        means = all_returns.mean(axis=1)
        neg_mask = all_returns < 0
        downside_sq = np.where(neg_mask, all_returns ** 2, 0.0).mean(axis=1)
        downside_std = np.sqrt(downside_sq)
        null_metrics = np.where(downside_std > 0, means / downside_std, 0.0)
    else:
        # Fallback: apply metric_fn to each row (slow, but correct)
        metric_fn = resolve_objective_metric(objective_metric_name)
        null_metrics = np.array(
            [metric_fn(pd.Series(all_returns[i])) for i in range(nreps)],
            dtype=float
        )

    # Handle infinities/NaNs
    null_metrics = np.where(np.isfinite(null_metrics), null_metrics, 0.0)
    return null_metrics


def _compute_fixed_oos_signal_by_fold(
    fold_rows: list[dict],
    selection_summary_df: pd.DataFrame,
    reference_target: pd.Series,
    research_config: object,
    feature_data_by_combo: dict,
) -> dict[int, pd.Series]:
    """Compute per-fold OOS averaged binned signal (fixed feature vector) from original run.

    Fits binning once per fold with real target; returns averaged member signals on test only.
    Used for vector-shuffle null: no refit in the null loop.
    """
    result: dict[int, pd.Series] = {}
    all_index = reference_target.index
    if getattr(all_index, "tz", None) is not None:
        all_index = all_index.tz_localize(None)
    # Reindex requires unique index; multi-ticker target can have duplicate datetimes.
    target_unique = (
        reference_target.groupby(level=0).first()
        if reference_target.index.duplicated().any()
        else reference_target
    )
    binning_config = getattr(research_config, "binning_params", None)
    if binning_config is None:
        return result
    tickers = getattr(research_config, "tickers", [])
    bias_spec = getattr(research_config, "bias_spec", {}) or {}
    module_name = str(bias_spec.get("module_name", "rsi")) if hasattr(bias_spec, "get") else "rsi"
    timeframes = bias_spec.get("timeframes", [None]) if hasattr(bias_spec, "get") else [None]
    tf_raw = timeframes[0] if timeframes else None
    trading_timeframe = (
        _normalize_timeframe(tf_raw) if tf_raw is not None else TimeFrame.D
    )
    strategy = _normalize_strategy(str(getattr(binning_config, "strategy", "long")))

    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        summary = selection_summary_df.loc[selection_summary_df["fold_id"] == fold_id]
        if summary.empty:
            continue
        selected_params = _parse_top_k_param_labels(str(summary.iloc[0]["top_k_features"]))
        if not selected_params:
            continue
        if all(k in fold_row for k in ("train_start", "train_end", "test_start", "test_end")):
            train_start = pd.Timestamp(fold_row["train_start"])
            train_end = pd.Timestamp(fold_row["train_end"])
            test_start = pd.Timestamp(fold_row["test_start"])
            test_end = pd.Timestamp(fold_row["test_end"])
            train_mask = (all_index >= train_start) & (all_index <= train_end)
            test_mask = (all_index >= test_start) & (all_index <= test_end)
        else:
            train_mask = cast(pd.Series, fold_row["_train_mask"])
            test_mask = cast(pd.Series, fold_row["_test_mask"])
        train_index = all_index[train_mask]
        test_index = all_index[test_mask]
        train_index_unique = pd.DatetimeIndex(train_index.unique()).sort_values()
        test_index_unique = pd.DatetimeIndex(test_index.unique()).sort_values()
        train_target = target_unique.reindex(train_index_unique).dropna()

        try:
            feature_type = _resolve_feature_type(research_config)
            base_model, _train_df, test_features_df = _build_one_base_model_with_members(
                selected_params=selected_params,
                binning_config=binning_config,
                tickers=list(tickers),
                trading_timeframe=trading_timeframe,
                module_name=module_name,
                feature_data_by_combo=feature_data_by_combo,
                train_index=train_index_unique,
                test_index=test_index_unique,
                train_target=train_target,
                feature_type=feature_type,
            )
        except Exception:
            continue
        member_signals: list[pd.Series] = []
        for name, bm in base_model.members:
            col = getattr(base_model, "_member_feature_columns", {}).get(
                name, base_model.feature_column
            )
            if col is None or col not in test_features_df.columns:
                continue
            test_ser = test_features_df[col].dropna()
            if test_ser.empty:
                continue
            sig = bm.predict(test_ser, strategy=strategy)
            member_signals.append(sig)
        if not member_signals:
            continue
        aligned = pd.concat(member_signals, axis=1)
        aligned = aligned.dropna(how="all")
        if aligned.empty:
            continue
        avg_signal = aligned.mean(axis=1)
        result[fold_id] = avg_signal

    return result


def _compute_rule_based_oos_signal_by_fold(
    fold_rows: list[dict],
    reference_candles: pd.DataFrame,
    reference_target: pd.Series,
    selection_summary_df: pd.DataFrame,
    research_config: object,
    portfolio_candles_df: pd.DataFrame | None = None,
) -> dict[int, pd.Series]:
    """Per-fold OOS position fractions from fitted rule-based portfolio (one-time, no null loop refit).

    Fits the portfolio ONCE per fold with real target; extracts position_fraction
    on the test window (averaged across tickers). Returned dict is passed to
    run_vector_shuffle_null as fixed_oos_signal_by_fold, triggering the fast
    vectorized null path (~50k reps/sec).

    Parameters
    ----------
    fold_rows, reference_target, selection_summary_df, research_config : as in _compute_fixed_oos_signal_by_fold
    reference_candles : pd.DataFrame
        Low-level candles (may have limited columns). Used only for slicing logic;
        actual fit/predict uses candles_for_fitting (see below).
    portfolio_candles_df : pd.DataFrame or None
        Full OHLCV candles with all columns required by portfolio.fit_from_candles.
        If provided, used for fit/predict instead of reference_candles.
    """
    result: dict[int, pd.Series] = {}
    binning_config = getattr(research_config, "binning_params", None)
    if binning_config is None:
        return result
    tickers = list(getattr(research_config, "tickers", []))
    bias_spec = getattr(research_config, "bias_spec", {}) or {}
    module_name = str(bias_spec.get("module_name", "rsi")) if hasattr(bias_spec, "get") else "rsi"
    timeframes = bias_spec.get("timeframes", [None]) if hasattr(bias_spec, "get") else [None]
    tf_raw = timeframes[0] if timeframes else None
    trading_timeframe = _normalize_timeframe(tf_raw) if tf_raw is not None else TimeFrame.D

    # Use portfolio candles for fitting if available (has all required columns)
    candles_for_fitting = portfolio_candles_df if portfolio_candles_df is not None else reference_candles

    target_unique = (
        reference_target.groupby(level=0).first()
        if reference_target.index.duplicated().any()
        else reference_target
    )

    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        summary = selection_summary_df.loc[selection_summary_df["fold_id"] == fold_id]
        if summary.empty:
            continue
        selected_params = _parse_top_k_param_labels(str(summary.iloc[0]["top_k_features"]))
        if not selected_params:
            continue

        # Slice candles using stored masks (same as run_portfolio_simulation), else timestamp fallback
        if "_train_mask" in fold_row and "_test_mask" in fold_row:
            train_candles = candles_for_fitting[fold_row["_train_mask"]]
            test_candles = candles_for_fitting[fold_row["_test_mask"]]
        else:
            train_start = pd.Timestamp(fold_row["train_start"])
            train_end = pd.Timestamp(fold_row["train_end"])
            test_start = pd.Timestamp(fold_row["test_start"])
            test_end = pd.Timestamp(fold_row["test_end"])
            idx = candles_for_fitting.index
            train_candles = candles_for_fitting.loc[(idx >= train_start) & (idx <= train_end)]
            test_candles = candles_for_fitting.loc[(idx >= test_start) & (idx <= test_end)]

        train_end_ts = pd.Timestamp(fold_row["train_end"])
        train_target = target_unique.loc[:train_end_ts].dropna()

        try:
            portfolio = build_research_portfolio(
                selected_params=selected_params,
                binning_config=binning_config,
                tickers=tickers,
                trading_timeframe=trading_timeframe,
                module_name=module_name,
                feature_type=FeatureType.RULE_BASED,
            )
            portfolio.fit_from_candles(train_candles, target_data=train_target)
            predictions = portfolio.predict_from_candles(test_candles)
            portfolio_preds = predictions["portfolio"] if isinstance(predictions, dict) else predictions
            if "position_fraction" not in portfolio_preds.columns:
                continue
            if "datetime" in portfolio_preds.columns:
                pos_df = portfolio_preds.set_index("datetime")
            else:
                pos_df = portfolio_preds
            # Average position_fraction across tickers per datetime
            pos_signal = pos_df.groupby(level=0)["position_fraction"].mean()
            if pos_signal.empty:
                continue
            result[fold_id] = pos_signal
        except Exception:
            continue

    return result


def aggregate_signal_target_returns(
    fixed_oos_signal_by_fold: dict[int, pd.Series],
    target: pd.Series,
    fold_rows: list[dict],
) -> pd.Series:
    """Build concatenated (signal * target) OOS returns across folds.

    Uses full OOS period per fold (zeros when inactive) so the test statistic has
    ~full bar count, not just active periods. Same construction as the fixed-signal null.
    """
    target_unique = (
        target.groupby(level=0).first()
        if target.index.duplicated().any()
        else target
    )
    collected: list[pd.Series] = []
    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        if not all(k in fold_row for k in ("test_start", "test_end")):
            continue
        test_start = pd.Timestamp(fold_row["test_start"])
        test_end = pd.Timestamp(fold_row["test_end"])
        full_test_index = target_unique.index[
            (target_unique.index >= test_start) & (target_unique.index <= test_end)
        ]
        if len(full_test_index) == 0:
            continue
        oos_signal = fixed_oos_signal_by_fold.get(fold_id)
        if oos_signal is None or oos_signal.empty:
            collected.append(pd.Series(0.0, index=full_test_index))
            continue
        oos_target = target_unique.reindex(oos_signal.index).dropna()
        oos_signal = oos_signal.reindex(oos_target.index).dropna()
        if oos_signal.empty:
            collected.append(pd.Series(0.0, index=full_test_index))
            continue
        ret = oos_signal * oos_target
        ret_full = ret.reindex(full_test_index).fillna(0.0)
        collected.append(ret_full)
    if not collected:
        return pd.Series(dtype=float)
    return pd.concat(collected, axis=0).sort_index()


def run_return_shuffle_null_vectorized(
    aggregate_oos_returns: pd.Series,
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
) -> np.ndarray:
    """Vectorized return shuffle: all nreps shuffles at once, ~50k reps/sec.

    Generates all nreps shuffled return series as a (nreps, N) matrix and
    applies the objective metric vectorized across axis=1. No model fits,
    no signals, no joblib needed. Suitable for rule-based features where
    the test hypothesis is: "does the temporal order of OOS returns matter?"

    Parameters
    ----------
    aggregate_oos_returns : pd.Series
        Pre-computed OOS returns from the original walkforward run.
    nreps : int
        Number of null replicates.
    random_seed : int or None
        Seed for reproducibility.
    objective_metric_name : str
        Metric to apply to each shuffled return series.

    Returns
    -------
    np.ndarray
        Shape (nreps,); metric value for each replicate.
    """
    clean = aggregate_oos_returns.dropna()
    if clean.empty:
        return np.zeros(nreps, dtype=float)

    oos_vals = clean.values.copy()  # shape (N,)
    n = len(oos_vals)

    rng = np.random.default_rng(random_seed)
    # All shuffles at once: (nreps, N) — each row is a random permutation of oos_vals
    perm = rng.random((nreps, n)).argsort(axis=1)  # (nreps, N) permutation indices
    all_returns = oos_vals[perm]  # (nreps, N) — all shuffled return series

    # Apply metric vectorized (same as run_vector_shuffle_null_vectorized)
    if objective_metric_name == "sharpe":
        means = all_returns.mean(axis=1)
        stds = all_returns.std(axis=1, ddof=0)
        null_metrics = np.where(stds > 0, means / stds, 0.0)
    elif objective_metric_name == "mean_return":
        null_metrics = all_returns.mean(axis=1)
    elif objective_metric_name == "t_stat":
        means = all_returns.mean(axis=1)
        stds = all_returns.std(axis=1, ddof=0)
        null_metrics = np.where(stds > 0, means / stds * np.sqrt(n), 0.0)
    elif objective_metric_name == "sortino":
        means = all_returns.mean(axis=1)
        neg_mask = all_returns < 0
        downside_sq = np.where(neg_mask, all_returns ** 2, 0.0).mean(axis=1)
        downside_std = np.sqrt(downside_sq)
        null_metrics = np.where(downside_std > 0, means / downside_std, 0.0)
    else:
        # Fallback: apply metric_fn to each row (slower)
        metric_fn = resolve_objective_metric(objective_metric_name)
        null_metrics = np.array(
            [metric_fn(pd.Series(all_returns[i])) for i in range(nreps)],
            dtype=float
        )

    null_metrics = np.where(np.isfinite(null_metrics), null_metrics, 0.0)
    return null_metrics


def _one_vector_shuffle_rep_fixed_signal(
    seed: int,
    reference_target: pd.Series,
    unit1_mask: pd.Series,
    unit2_mask: pd.Series,
    fixed_oos_signal_by_fold: dict[int, pd.Series],
    fold_rows: list[dict],
    objective_metric_name: str,
    canonical_oos_index: pd.Index | None = None,
) -> float:
    """One null replicate with fixed feature vector: permute target, then metric(aggregate OOS returns).

    Concatenates (signal * permuted_target) across all folds and applies the objective
    metric once. No refit. Module-level for joblib pickling when n_jobs > 1.
    When canonical_oos_index is provided, aligns aggregate to that index so original and null use the same n.
    """
    rng = np.random.default_rng(seed)
    permuted_target = permute_target_in_two_units(
        reference_target, unit1_mask, unit2_mask, rng
    )
    permuted_unique = (
        permuted_target.groupby(level=0).first()
        if permuted_target.index.duplicated().any()
        else permuted_target
    )
    metric_fn = resolve_objective_metric(objective_metric_name)
    collected_returns: list[pd.Series] = []
    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        if not all(k in fold_row for k in ("test_start", "test_end")):
            continue
        test_start = pd.Timestamp(fold_row["test_start"])
        test_end = pd.Timestamp(fold_row["test_end"])
        full_test_index = permuted_unique.index[
            (permuted_unique.index >= test_start) & (permuted_unique.index <= test_end)
        ]
        if len(full_test_index) == 0:
            continue
        oos_signal = fixed_oos_signal_by_fold.get(fold_id)
        if oos_signal is None or oos_signal.empty:
            collected_returns.append(pd.Series(0.0, index=full_test_index))
            continue
        oos_target = permuted_unique.reindex(oos_signal.index).dropna()
        oos_signal = oos_signal.reindex(oos_target.index).dropna()
        if oos_signal.empty:
            collected_returns.append(pd.Series(0.0, index=full_test_index))
            continue
        null_return = oos_signal * oos_target
        null_return_full = null_return.reindex(full_test_index).fillna(0.0)
        collected_returns.append(null_return_full)
    if not collected_returns:
        return 0.0
    aggregate = pd.concat(collected_returns, axis=0).sort_index()
    if canonical_oos_index is not None and len(canonical_oos_index) > 0:
        aggregate = aggregate.reindex(canonical_oos_index).fillna(0.0)
        if aggregate.empty:
            return 0.0
    val = metric_fn(aggregate)
    return float(val) if np.isfinite(val) else 0.0


def _one_vector_shuffle_rep(
    seed: int,
    reference_target: pd.Series,
    unit1_mask: pd.Series,
    unit2_mask: pd.Series,
    candles_for_simulation: pd.DataFrame,
    fold_rows: list[dict],
    selection_summary_df: pd.DataFrame,
    research_config: object,
    feature_data_by_combo: dict | None,
    objective_metric_name: str = "sharpe",
) -> float:
    """One null replicate (legacy): permute target, run portfolio simulation, return aggregate metric.

    Used when fixed_oos_signal_by_fold is not available (e.g. rule-based). Module-level for joblib.
    Uses metric on concatenated OOS returns when available, else mean of per-fold metric.
    """
    rng = np.random.default_rng(seed)
    permuted_target = permute_target_in_two_units(
        reference_target, unit1_mask, unit2_mask, rng
    )
    portfolio_results_df, _, aggregate_oos_returns = run_portfolio_simulation(
        candles_df=candles_for_simulation,
        target=permuted_target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
        feature_data_by_combo=feature_data_by_combo,
        tearsheets_dir=None,
    )
    minimal_report = types.SimpleNamespace(
        portfolio_results_df=portfolio_results_df,
        aggregate_oos_returns=aggregate_oos_returns,
    )
    metric_fn = resolve_objective_metric(objective_metric_name)
    return aggregate_oos_metric_from_report(
        minimal_report, treat_no_selection_as_zero=True, metric_fn=metric_fn
    )


def _one_return_shuffle_rep(
    seed: int,
    oos_values: np.ndarray,
    index_values: np.ndarray,
    objective_metric_name: str,
) -> float:
    """One return-shuffle null replicate. Module-level for joblib pickling when n_jobs > 1."""
    rng = np.random.default_rng(seed)
    shuffled = pd.Series(rng.permutation(oos_values), index=pd.DatetimeIndex(index_values))
    metric_fn = resolve_objective_metric(objective_metric_name)
    val = metric_fn(shuffled)
    return float(val) if np.isfinite(val) else 0.0


def run_return_shuffle_null(
    aggregate_oos_returns: pd.Series,
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
    n_jobs: int = 1,
) -> np.ndarray:
    """Null distribution by shuffling pre-computed aggregate OOS returns.

    No refit, no signal computation. Works for both rule-based and continuous features.
    Each replicate randomly reorders the OOS return values and applies the objective metric.
    The null hypothesis: the temporal ordering of OOS returns does not matter — any
    arrangement of the same return magnitudes would produce this metric.

    Parameters
    ----------
    aggregate_oos_returns : pd.Series
        Concatenated OOS returns from the original walkforward run (report0.aggregate_oos_returns).
    nreps : int
        Number of null replicates.
    random_seed : int or None
        Seed for reproducibility.
    objective_metric_name : str
        Metric to evaluate on each shuffled series.
    n_jobs : int
        Parallel jobs (1 = sequential; -1 = all CPUs; joblib loky backend).
    """
    clean = aggregate_oos_returns.dropna()
    oos_values = clean.values.copy()
    index_values = clean.index.values.copy()
    rng = np.random.default_rng(random_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(nreps)]

    if n_jobs == 1:
        null_metrics = np.empty(nreps, dtype=float)
        for i in tqdm(range(nreps), desc="Vector shuffle (return shuffle)", unit="rep"):
            null_metrics[i] = _one_return_shuffle_rep(
                seeds[i], oos_values, index_values, objective_metric_name
            )
        return null_metrics

    from multiprocessing import cpu_count
    from joblib import Parallel, delayed

    n_jobs_actual = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    with _joblib_tqdm(nreps, desc="Vector shuffle (return shuffle)", unit="rep"):
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_one_return_shuffle_rep)(seed, oos_values, index_values, objective_metric_name)
            for seed in seeds
        )
    return np.array(results, dtype=float)


def run_vector_shuffle_null(
    reference_candles: pd.DataFrame,
    reference_target: pd.Series,
    fold_rows: list[dict],
    unit1_mask: pd.Series,
    unit2_mask: pd.Series,
    nreps: int,
    random_seed: int | None,
    initial_report: object,
    research_config: object,
    feature_data_by_combo: dict | None,
    portfolio_candles_df: pd.DataFrame | None,
    n_jobs: int = -1,
    fixed_oos_signal_by_fold: dict[int, pd.Series] | None = None,
    objective_metric_name: str = "sharpe",
    canonical_oos_index: pd.Index | None = None,
) -> np.ndarray:
    """Run nreps null replicates; return null distribution.

    When fixed_oos_signal_by_fold is provided (continuous path): no refit; each replicate
    permutes target and computes mean fold Sharpe(fixed_signal * permuted_target) on OOS.
    Otherwise (e.g. rule-based): calls run_portfolio_simulation per replicate (refits).
    When n_jobs > 1, replicates run in parallel via joblib (loky backend).
    """
    rng = np.random.default_rng(random_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(nreps)]
    use_fixed_signal = fixed_oos_signal_by_fold is not None and len(fixed_oos_signal_by_fold) > 0

    if use_fixed_signal:
        # Use fully vectorized approach: all nreps permutations at once, no loop overhead
        # This is ~10–100x faster than the serial/parallel loop approaches
        obj_name = objective_metric_name or "sharpe"
        return run_vector_shuffle_null_vectorized(
            reference_target=reference_target,
            unit1_mask=unit1_mask,
            unit2_mask=unit2_mask,
            fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
            fold_rows=fold_rows,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=obj_name,
            canonical_oos_index=canonical_oos_index,
        )

    selection_summary_df = getattr(initial_report, "selection_summary_df", None)
    if selection_summary_df is None:
        raise ValueError("initial_report must have selection_summary_df")
    candles_for_simulation = (
        portfolio_candles_df if portfolio_candles_df is not None else reference_candles
    )

    obj_name = objective_metric_name or "sharpe"
    if n_jobs == 1:
        null_metrics = np.empty(nreps, dtype=float)
        for i in tqdm(range(nreps), desc="Vector shuffle (target permute)", unit="rep"):
            null_metrics[i] = _one_vector_shuffle_rep(
                seeds[i],
                reference_target,
                unit1_mask,
                unit2_mask,
                candles_for_simulation,
                fold_rows,
                selection_summary_df,
                research_config,
                feature_data_by_combo,
                objective_metric_name=obj_name,
            )
        return null_metrics

    from multiprocessing import cpu_count
    from joblib import Parallel, delayed

    n_jobs_actual = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    with _joblib_tqdm(nreps, desc="Vector shuffle (target permute)", unit="rep"):
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_one_vector_shuffle_rep)(
                seed,
                reference_target,
                unit1_mask,
                unit2_mask,
                candles_for_simulation,
                fold_rows,
                selection_summary_df,
                research_config,
                feature_data_by_combo,
                obj_name,
            )
            for seed in seeds
        )
    return np.array(results, dtype=float)
