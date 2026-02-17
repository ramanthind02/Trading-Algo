"""
T016: Early Stopping Orchestration

Coordinates the three-stage permutation testing funnel with early stopping:
  Stage 1 (vector shuffle) -> filter passers
  Stage 2 (pipeline permutation) -> filter passers (only from Stage 1 passers)
  Stage 3 (walkforward stability) -> uses ALL params for neighbor smoothing
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.permutation_tests import (
    run_pipeline_permutation_continuous,
    run_pipeline_permutation_rule_based,
    run_vector_shuffle_test,
)
from feature_selection.validation.reports import (
    FunnelStatistics,
    PermutationTestSuite,
    PipelinePermutationReport,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)
from feature_selection.validation.stability_analysis import (
    _param_combo_name,
    run_walkforward_stability,
)


def _build_summary(
    feature_name: str,
    param_grid: List[Dict],
    stage1_reports: Dict[str, VectorShuffleReport],
    stage2_reports: Dict[str, PipelinePermutationReport],
    stage3_report: WalkforwardStabilityReport,
    funnel_stats: FunnelStatistics,
    ensemble_candidates: List[str],
) -> str:
    n_total = funnel_stats.total_params
    n1 = funnel_stats.stage1_pass
    n2 = funnel_stats.stage2_pass
    n_stable = funnel_stats.stable_params
    n_cand = funnel_stats.ensemble_candidates
    savings = funnel_stats.computational_savings_pct

    lines = [
        f"Feature: {feature_name}",
        f"Parameter grid: {n_total} combinations",
        "",
        f"Stage 1 (Vector Shuffle): {n_total} tested -> {n1} passed ({100*n1//max(n_total,1)}%)",
        f"Stage 2 (Pipeline Permutation): {n1} tested -> {n2} passed",
        f"Stage 3 (Walkforward Stability): {n_total} evaluated -> {n_stable} stable params",
        "",
        f"Ensemble candidates: {ensemble_candidates}",
        f"Computational savings: {savings:.1f}%",
        "",
        f"Stability verdict: {stage3_report.stability_verdict}",
    ]
    return '\n'.join(lines)


def run_permutation_test_suite(
    candles_df: pd.DataFrame,
    feature_spec: Dict,
    target: pd.Series,
    param_grid: List[Dict],
    objective_func: Callable[[pd.Series], float],
    fold_structure: List[Tuple[pd.Timestamp, pd.Timestamp]],
    config: PermutationTestConfig,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    binning_model_factory: Optional[Callable[[Dict], object]] = None,
    feature_type: str = 'continuous',
    feature_name: str = 'unknown',
) -> PermutationTestSuite:
    """Run the full three-stage permutation testing funnel with early stopping.

    Args:
        candles_df: OHLCV DataFrame with DatetimeIndex.
        feature_spec: Dict with 'module_name' and 'params' keys (used as label).
        target: Forward-return Series.
        param_grid: All parameter combinations (as list of dicts).
        objective_func: (returns: pd.Series) -> float.
        fold_structure: List of (start, end) Timestamp tuples for Stage 3.
        config: PermutationTestConfig with nreps, alpha, etc.
        extractor_func: (candles_df, params) -> named pd.Series.
        binning_model_factory: Optional (params) -> BinningModelBase factory.
        feature_type: 'continuous' or 'rule_based'.
        feature_name: Display name.

    Returns:
        PermutationTestSuite with all stage reports and funnel statistics.
    """
    # ---------- Stage 1: Vector Shuffle ----------
    print(f"\n{'='*60}")
    print(f"Stage 1: Vector Shuffle — testing {len(param_grid)} param combos")
    print(f"{'='*60}")

    stage1_reports: Dict[str, VectorShuffleReport] = {}
    for params in param_grid:
        combo_name = _param_combo_name(params)
        try:
            # Extract feature + fit binning to get position multipliers
            feature = extractor_func(candles_df, params)
            feature = feature.reindex(target.index).dropna()
            aligned_target = target.reindex(feature.index)

            if binning_model_factory is not None and feature_type == 'continuous':
                model = binning_model_factory(params)
                model.fit(feature, aligned_target)
                fitted_vec = model.get_fitted_vector(strategy='long')
            else:
                fitted_vec = feature  # rule-based: use signal directly

            report = run_vector_shuffle_test(
                fitted_feature=fitted_vec,
                target=aligned_target,
                objective_func=objective_func,
                nreps=config.nreps,
                alpha=config.alpha,
                random_seed=config.random_seed,
                param_combo=combo_name,
            )
        except Exception as e:
            # Create a failed report
            report = VectorShuffleReport(
                param_combo=combo_name,
                original_metric=0.0,
                null_distribution=np.zeros(config.nreps),
                critical_value=0.0,
                p_value=1.0,
                passed=False,
                alpha=config.alpha,
                nreps=config.nreps,
            )

        stage1_reports[combo_name] = report
        status = 'PASS' if report.passed else 'FAIL'
        print(f"  {combo_name}: p={report.p_value:.3f} -> {status}")

    stage1_passers: Set[str] = {k for k, r in stage1_reports.items() if r.passed}
    print(f"\nStage 1: {len(stage1_passers)}/{len(param_grid)} passed")

    # ---------- Stage 2: Pipeline Permutation (only Stage 1 passers) ----------
    print(f"\n{'='*60}")
    print(f"Stage 2: Pipeline Permutation — testing {len(stage1_passers)} passers")
    print(f"{'='*60}")

    stage2_reports: Dict[str, PipelinePermutationReport] = {}
    for params in param_grid:
        combo_name = _param_combo_name(params)
        if combo_name not in stage1_passers:
            continue

        try:
            if feature_type == 'continuous':
                def _extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                    return extractor_func(df, _params)

                model_template = (
                    binning_model_factory(params) if binning_model_factory else None
                )
                if model_template is None:
                    raise ValueError('binning_model_factory required for continuous features')

                report = run_pipeline_permutation_continuous(
                    candles_df=candles_df,
                    bias_node_extractor=_extractor_for_combo,
                    binning_model=model_template,
                    target=target,
                    objective_func=objective_func,
                    permutation_mode=config.permutation_mode_stage2,
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                    param_combo=combo_name,
                )
            else:
                def _rule_extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                    return extractor_func(df, _params)

                report = run_pipeline_permutation_rule_based(
                    candles_df=candles_df,
                    rule_extractor=_rule_extractor_for_combo,
                    target=target,
                    objective_func=objective_func,
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                    param_combo=combo_name,
                )
        except Exception:
            report = PipelinePermutationReport(
                param_combo=combo_name,
                feature_type=feature_type,  # type: ignore[arg-type]
                permutation_mode=config.permutation_mode_stage2,
                original_metric=0.0,
                null_distribution=np.zeros(config.nreps),
                critical_value=0.0,
                p_value=1.0,
                passed=False,
                alpha=config.alpha,
                nreps=config.nreps,
                no_trade_permutations=config.nreps,
            )

        stage2_reports[combo_name] = report
        status = 'PASS' if report.passed else 'FAIL'
        print(f"  {combo_name}: p={report.p_value:.3f} -> {status}")

    stage2_passers: Set[str] = {k for k, r in stage2_reports.items() if r.passed}
    print(f"\nStage 2: {len(stage2_passers)}/{len(stage1_passers)} passed")

    # ---------- Stage 3: Walkforward Stability (ALL params) ----------
    print(f"\n{'='*60}")
    print(f"Stage 3: Walkforward Stability — evaluating all {len(param_grid)} params")
    print(f"{'='*60}")

    stage3_report = run_walkforward_stability(
        candles_df=candles_df,
        extractor_func=extractor_func,
        target=target,
        objective_func=objective_func,
        param_grid=param_grid,
        fold_structure=fold_structure,
        top_k=config.top_k,
        permutation_passers=stage2_passers,
        feature_type=feature_type,
        feature_name=feature_name,
        binning_model_factory=binning_model_factory,
    )

    # Collect stable params (top-K in >= min_folds_stable folds)
    param_fold_counts: Dict[str, int] = {}
    for fold_result in stage3_report.fold_results:
        for param in fold_result.top_k_params:
            param_fold_counts[param] = param_fold_counts.get(param, 0) + 1

    stable_params: Set[str] = {
        p for p, count in param_fold_counts.items()
        if count >= config.min_folds_stable
    }

    # Ensemble candidates: passed Stage 2 AND in stable region
    ensemble_candidates = sorted(stage2_passers & stable_params)

    # Funnel statistics
    n_total = len(param_grid)
    n1 = len(stage1_passers)
    n2 = len(stage2_passers)
    n_stable = len(stable_params)
    n_cand = len(ensemble_candidates)

    # Savings = Stage 2 evaluations saved by Stage 1 early stopping
    total_without = n_total * config.nreps * 2 + n_total * len(fold_structure)
    actual = n_total * config.nreps + n1 * config.nreps + n_total * len(fold_structure)
    savings_pct = max(0.0, (total_without - actual) / max(total_without, 1) * 100.0)

    funnel_stats = FunnelStatistics(
        total_params=n_total,
        stage1_pass=n1,
        stage2_pass=n2,
        stable_params=n_stable,
        ensemble_candidates=n_cand,
        computational_savings_pct=savings_pct,
    )

    summary = _build_summary(
        feature_name, param_grid, stage1_reports, stage2_reports,
        stage3_report, funnel_stats, ensemble_candidates,
    )

    print(f"\n{summary}")

    return PermutationTestSuite(
        feature_name=feature_name,
        feature_type=feature_type,  # type: ignore[arg-type]
        stage1_reports=stage1_reports,
        stage2_reports=stage2_reports,
        stage3_report=stage3_report,
        funnel_stats=funnel_stats,
        ensemble_candidates=ensemble_candidates,
        summary=summary,
    )
