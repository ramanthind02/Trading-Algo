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

from feature_research.walkforward.permutation_helpers import (
    aggregate_oos_metric_from_report,
    permute_target_in_two_units,
)
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_research.walkforward.portfolio_evaluator import (
    _build_one_base_model_with_members,
    _ensure_cross_ticker_data,
    _normalize_strategy,
    _normalize_timeframe,
)
from feature_research.walkforward.runner import (
    _parse_top_k_param_labels,
    _resolve_feature_type,
    run_portfolio_simulation,
)
from utils.core.enums import TimeFrame


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
            _ensure_cross_ticker_data(selected_params, [trading_timeframe])
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
        obj_name = objective_metric_name or "sharpe"
        if n_jobs == 1:
            null_metrics = np.empty(nreps, dtype=float)
            for i in tqdm(range(nreps), desc="Vector shuffle (target permute)", unit="rep"):
                null_metrics[i] = _one_vector_shuffle_rep_fixed_signal(
                    seeds[i],
                    reference_target,
                    unit1_mask,
                    unit2_mask,
                    fixed_oos_signal_by_fold,
                    fold_rows,
                    obj_name,
                    canonical_oos_index=canonical_oos_index,
                )
            return null_metrics
        from multiprocessing import cpu_count
        from joblib import Parallel, delayed

        n_jobs_actual = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_one_vector_shuffle_rep_fixed_signal)(
                seed,
                reference_target,
                unit1_mask,
                unit2_mask,
                fixed_oos_signal_by_fold,
                fold_rows,
                obj_name,
                canonical_oos_index=canonical_oos_index,
            )
            for seed in seeds
        )
        return np.array(results, dtype=float)

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
