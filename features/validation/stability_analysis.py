"""
T015: Walkforward stability analysis for frozen signed signals.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from features.validation.permutation_tests import _compute_metric_from_signals
from features.validation.reports import FoldResult, WalkforwardStabilityReport


def _param_combo_name(params: Dict) -> str:
    return "_".join(f"{k}_{v}" for k, v in sorted(params.items()))


def _compute_neighbor_smoothed_objectives(
    param_names: List[str],
    param_grid: List[Dict],
    raw_objectives: Dict[str, float],
) -> Dict[str, float]:
    param_value_sets = {col: sorted({p[col] for p in param_grid}) for col in param_names}
    return {
        _param_combo_name(params): float(
            np.mean(
                [
                    raw_objectives.get(_param_combo_name(params), 0.0),
                    *[
                        raw_objectives[_param_combo_name({**params, col: sorted_vals[neighbor_idx]})]
                        for col in param_names
                        for sorted_vals in (param_value_sets[col],)
                        for idx in [sorted_vals.index(params[col])]
                        for neighbor_idx in (idx - 1, idx + 1)
                        if 0 <= neighbor_idx < len(sorted_vals)
                        if _param_combo_name({**params, col: sorted_vals[neighbor_idx]}) in raw_objectives
                    ],
                ]
            )
        )
        for params in param_grid
    }


def compute_jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(len(a), len(b)))


def _compute_consistency_metrics(fold_results: List[FoldResult], top_k: int) -> Dict[str, float]:
    _ = top_k
    if not fold_results:
        return {"overlap_rate": 0.0, "n_folds": 0.0, "n_consecutive_pairs": 0.0}
    if len(fold_results) == 1:
        return {"overlap_rate": 1.0, "n_folds": 1.0, "n_consecutive_pairs": 0.0}

    overlap_rates = [
        compute_jaccard_overlap(set(a.top_k_params), set(b.top_k_params))
        for a, b in zip(fold_results, fold_results[1:])
    ]
    return {
        "overlap_rate": float(np.mean(overlap_rates)),
        "n_folds": float(len(fold_results)),
        "n_consecutive_pairs": float(len(overlap_rates)),
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
    feature_name: str = "unknown",
) -> WalkforwardStabilityReport:
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
                signal = extractor_func(fold_candles, params)
                signal = signal.reindex(fold_target.index).dropna()
                aligned_target = fold_target.reindex(signal.index).dropna()
                signal = signal.reindex(aligned_target.index)
                raw_objectives[combo_name], _ = _compute_metric_from_signals(
                    signal,
                    aligned_target,
                    objective_func,
                )
            except Exception:
                raw_objectives[combo_name] = 0.0

        smoothed = _compute_neighbor_smoothed_objectives(param_names, param_grid, raw_objectives)
        top_k_params = sorted(smoothed, key=smoothed.get, reverse=True)[: min(top_k, len(smoothed))]
        fold_results.append(
            FoldResult(
                fold_id=f"fold_{fold_idx}",
                fold_period=(str(fold_start.date()), str(fold_end.date())),
                top_k_params=top_k_params,
                smoothed_objectives=smoothed,
                passed_permutation_overlay=[
                    param in permutation_passers if permutation_passers is not None else False
                    for param in top_k_params
                ],
            )
        )

    consistency_metrics = _compute_consistency_metrics(fold_results, top_k)
    overlap_rate = consistency_metrics.get("overlap_rate", 0.0)
    is_stable = overlap_rate >= 0.5
    verdict = (
        f"STABLE: Top-K params show consistent selection (overlap_rate={overlap_rate:.2f})"
        if is_stable
        else f"UNSTABLE: Top-K params show high variability (overlap_rate={overlap_rate:.2f})"
    )

    return WalkforwardStabilityReport(
        feature_name=feature_name,
        feature_type="signed_signal",
        fold_results=fold_results,
        consistency_metrics=consistency_metrics,
        is_stable=is_stable,
        stability_verdict=verdict,
        top_k=top_k,
    )
