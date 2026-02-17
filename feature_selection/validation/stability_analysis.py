"""
T015: Walkforward Stability Analysis

Divides in-sample data into non-overlapping folds, evaluates ALL parameter
combinations per fold with neighbor smoothing, selects top-K per fold, then
measures consistency of selections across folds.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from feature_selection.validation.reports import FoldResult, WalkforwardStabilityReport


def _param_combo_name(params: Dict) -> str:
    """Convert param dict to canonical string key: {'lookback': 5} -> 'lookback_5'."""
    return '_'.join(f'{k}_{v}' for k, v in sorted(params.items()))


def _compute_neighbor_smoothed_objectives(
    param_names: List[str],
    param_grid: List[Dict],
    raw_objectives: Dict[str, float],
) -> Dict[str, float]:
    """Axis-aligned 1-step neighbor smoothing.

    smoothed(P) = mean([obj(P)] + [obj(N) for N in 1-step_axis_neighbors(P)])

    1-step neighbors: differ in exactly one param dimension by one grid step
    (adjacent values in the sorted unique value list for that dimension).
    """
    # Sorted unique values per param dimension
    param_value_sets: Dict[str, List] = {}
    for col in param_names:
        vals = sorted(set(p[col] for p in param_grid))
        param_value_sets[col] = vals

    smoothed: Dict[str, float] = {}
    for params in param_grid:
        combo = _param_combo_name(params)
        values = [raw_objectives.get(combo, 0.0)]

        for col in param_names:
            sorted_vals = param_value_sets[col]
            try:
                idx = sorted_vals.index(params[col])
            except ValueError:
                continue

            for neighbor_idx in (idx - 1, idx + 1):
                if 0 <= neighbor_idx < len(sorted_vals):
                    neighbor_params = {**params, col: sorted_vals[neighbor_idx]}
                    neighbor_key = _param_combo_name(neighbor_params)
                    if neighbor_key in raw_objectives:
                        values.append(raw_objectives[neighbor_key])

        smoothed[combo] = float(np.mean(values))

    return smoothed


def _compute_consistency_metrics(
    fold_results: List[FoldResult], top_k: int
) -> Dict[str, float]:
    """Compute overlap_rate and related metrics across consecutive folds."""
    if not fold_results:
        return {'overlap_rate': 0.0, 'n_folds': 0.0, 'n_consecutive_pairs': 0.0}
    if len(fold_results) == 1:
        return {'overlap_rate': 1.0, 'n_folds': 1.0, 'n_consecutive_pairs': 0.0}

    overlap_rates: List[float] = []
    for i in range(len(fold_results) - 1):
        a = set(fold_results[i].top_k_params)
        b = set(fold_results[i + 1].top_k_params)
        if not a and not b:
            overlap_rates.append(1.0)
        elif not a or not b:
            overlap_rates.append(0.0)
        else:
            overlap_rates.append(len(a & b) / max(len(a), len(b)))

    return {
        'overlap_rate': float(np.mean(overlap_rates)),
        'n_folds': float(len(fold_results)),
        'n_consecutive_pairs': float(len(overlap_rates)),
    }


def run_walkforward_stability(
    candles_df: pd.DataFrame,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    target: pd.Series,
    objective_func: Callable[[pd.Series], float],
    param_grid: List[Dict],
    fold_structure: List[Tuple[pd.Timestamp, pd.Timestamp]],
    top_k: int = 3,
    permutation_passers: Optional[Set[str]] = None,
    feature_type: str = 'continuous',
    feature_name: str = 'unknown',
    binning_model_factory: Optional[Callable[[Dict], object]] = None,
) -> WalkforwardStabilityReport:
    """Stage 3: Walkforward Stability Analysis.

    Args:
        candles_df: OHLCV DataFrame with DatetimeIndex.
        extractor_func: (candles_df, params) -> named pd.Series of feature values.
        target: Forward-return series aligned to candles.
        objective_func: (returns: pd.Series) -> float.
        param_grid: ALL parameter combinations (needed for neighbor smoothing).
        fold_structure: List of (start, end) Timestamp tuples for each fold.
        top_k: Number of top params to select per fold.
        permutation_passers: Optional set of param_combo_names that passed Stage 1-2.
        feature_type: 'continuous' or 'rule_based'.
        feature_name: Display name of the feature.
        binning_model_factory: Optional (params) -> BinningModelBase factory for
            continuous features. If None, rule-based computation (no binning).
    """
    param_names = sorted(param_grid[0].keys()) if param_grid else []
    fold_results: List[FoldResult] = []

    for fold_idx, (fold_start, fold_end) in enumerate(fold_structure):
        fold_mask = (candles_df.index >= fold_start) & (candles_df.index < fold_end)
        fold_candles = candles_df[fold_mask]
        fold_target = target.reindex(fold_candles.index).dropna()

        if len(fold_candles) < 10:
            continue

        raw_objectives: Dict[str, float] = {}
        for params in param_grid:
            combo_name = _param_combo_name(params)
            try:
                feature = extractor_func(fold_candles, params)
                feature = feature.reindex(fold_target.index).dropna()
                fold_target_aligned = fold_target.reindex(feature.index).dropna()
                feature = feature.reindex(fold_target_aligned.index)

                if binning_model_factory is not None and feature_type == 'continuous':
                    model = binning_model_factory(params)
                    model.fit(feature, fold_target_aligned)
                    signals = model.predict(feature, strategy='long')
                    active = signals[signals != 0]
                    if len(active) == 0:
                        obj = 0.0
                    else:
                        returns = fold_target_aligned.reindex(active.index) * active
                        obj = objective_func(returns.dropna())
                else:
                    returns = fold_target_aligned * feature
                    obj = objective_func(returns.dropna()) if len(returns) > 0 else 0.0
            except Exception:
                obj = 0.0
            raw_objectives[combo_name] = obj

        smoothed = _compute_neighbor_smoothed_objectives(param_names, param_grid, raw_objectives)
        top_k_actual = min(top_k, len(smoothed))
        top_k_params = sorted(smoothed, key=lambda k: smoothed[k], reverse=True)[:top_k_actual]
        overlay = [
            (param in permutation_passers) if permutation_passers is not None else False
            for param in top_k_params
        ]

        fold_results.append(FoldResult(
            fold_id=f'fold_{fold_idx}',
            fold_period=(str(fold_start.date()), str(fold_end.date())),
            top_k_params=top_k_params,
            smoothed_objectives=smoothed,
            passed_permutation_overlay=overlay,
        ))

    consistency_metrics = _compute_consistency_metrics(fold_results, top_k)
    overlap_rate = consistency_metrics.get('overlap_rate', 0.0)
    is_stable = overlap_rate >= 0.5

    verdict = (
        f"STABLE: Top-K params show consistent selection (overlap_rate={overlap_rate:.2f})"
        if is_stable
        else f"UNSTABLE: Top-K params show high variability (overlap_rate={overlap_rate:.2f})"
    )

    return WalkforwardStabilityReport(
        feature_name=feature_name,
        feature_type=feature_type,  # type: ignore[arg-type]
        fold_results=fold_results,
        consistency_metrics=consistency_metrics,
        is_stable=is_stable,
        stability_verdict=verdict,
        top_k=top_k,
    )
