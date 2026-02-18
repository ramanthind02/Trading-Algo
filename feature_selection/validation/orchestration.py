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

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.permutation_tests import (
    run_oos_permutation_for_param,
    run_pipeline_permutation_continuous,
    run_pipeline_permutation_rule_based,
    run_vector_shuffle_test,
)
from feature_selection.validation.reports import (
    ComboDecisionRecord,
    FunnelStatistics,
    OutOfSamplePermutationReport,
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
    phase3_oos_reports: Dict[str, OutOfSamplePermutationReport],
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
        f"Stage 3 (OOS Permutation): {len(phase3_oos_reports)} tested -> {sum(r.passed for r in phase3_oos_reports.values())} passed",
        "",
        f"Ensemble candidates: {ensemble_candidates}",
        f"Computational savings: {savings:.1f}%",
        "",
        f"Stability verdict: {stage3_report.stability_verdict}",
    ]
    return '\n'.join(lines)


def _extract_fitted_feature(
    candles_df: pd.DataFrame,
    target: pd.Series,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    params: Dict,
    feature_type: str,
    binning_model_factory: Optional[Callable[[Dict], BinningModelBase]],
) -> pd.Series:
    feature = extractor_func(candles_df, params)
    feature = feature.reindex(target.index).dropna()
    aligned_target = target.reindex(feature.index)

    if binning_model_factory is not None and feature_type == 'continuous':
        model = binning_model_factory(params)
        model.fit(feature, aligned_target)
        return model.get_fitted_vector(strategy='long')

    return feature


def _build_failed_oos_report(
    combo_name: str,
    nreps: int,
    alpha: float,
) -> OutOfSamplePermutationReport:
    return OutOfSamplePermutationReport(
        param_combo=combo_name,
        vector_report=VectorShuffleReport(
            param_combo=combo_name,
            original_metric=0.0,
            null_distribution=np.zeros(nreps),
            critical_value=0.0,
            p_value=1.0,
            passed=False,
            alpha=alpha,
            nreps=nreps,
        ),
        candle_report=None,
        passed=False,
    )


def run_permutation_test_suite(
    candles_df: pd.DataFrame,
    feature_spec: Dict,
    target: pd.Series,
    param_grid: List[Dict],
    objective_func: Callable[[pd.Series], float],
    fold_structure: List[Tuple[pd.Timestamp, pd.Timestamp]],
    config: PermutationTestConfig,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    binning_model_factory: Optional[Callable[[Dict], BinningModelBase]] = None,
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

    # ---------- Phase 3: OOS permutation on selected candidates ----------
    candidate_source = config.out_of_sample.candidate_source
    oos_candidates = (
        stage2_passers
        if candidate_source == 'stage2_passers'
        else stage2_passers & stable_params
    )

    phase3_oos_reports: Dict[str, OutOfSamplePermutationReport] = {}
    oos_objective_func = resolve_objective_metric(config.out_of_sample.objective_metric)
    params_by_combo = {_param_combo_name(params): params for params in param_grid}
    for combo_name in sorted(oos_candidates):
        params = params_by_combo.get(combo_name)
        if params is None:
            continue
        try:
            fitted_feature = _extract_fitted_feature(
                candles_df=candles_df,
                target=target,
                extractor_func=extractor_func,
                params=params,
                feature_type=feature_type,
                binning_model_factory=binning_model_factory,
            )

            if feature_type == 'continuous':
                if binning_model_factory is None:
                    raise ValueError('binning_model_factory required for continuous features')

                def _extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                    return extractor_func(df, _params)

                report = run_oos_permutation_for_param(
                    param_combo=combo_name,
                    feature_type='continuous',
                    fitted_feature=fitted_feature,
                    candles_df=candles_df,
                    target=target,
                    objective_func=oos_objective_func,
                    bias_node_extractor=_extractor_for_combo,
                    binning_model=binning_model_factory(params),
                    permutation_mode=config.permutation_mode_stage2,
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                )
            else:
                def _rule_extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                    return extractor_func(df, _params)

                report = run_oos_permutation_for_param(
                    param_combo=combo_name,
                    feature_type='rule_based',
                    fitted_feature=fitted_feature,
                    candles_df=candles_df,
                    target=target,
                    objective_func=oos_objective_func,
                    rule_extractor=_rule_extractor_for_combo,
                    permutation_mode='candle_shuffle',
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                )
        except Exception:
            report = _build_failed_oos_report(
                combo_name=combo_name,
                nreps=config.nreps,
                alpha=config.alpha,
            )

        phase3_oos_reports[combo_name] = report

    oos_passers = {combo for combo, report in phase3_oos_reports.items() if report.passed}
    final_candidates = sorted(set(ensemble_candidates) & oos_passers)
    final_candidate_set = set(final_candidates)

    combo_decisions = {
        combo_name: ComboDecisionRecord(
            param_combo=combo_name,
            stage1_passed=combo_name in stage1_passers,
            stage2_passed=combo_name in stage2_passers,
            walkforward_stable=combo_name in stable_params,
            oos_passed=combo_name in oos_passers,
            final_status=(
                'candidate'
                if combo_name in final_candidate_set
                else 'needs_review'
                if combo_name in stage2_passers
                else 'rejected'
            ),
        )
        for combo_name in (_param_combo_name(params) for params in param_grid)
    }

    # Funnel statistics
    n_total = len(param_grid)
    n1 = len(stage1_passers)
    n2 = len(stage2_passers)
    n_stable = len(stable_params)
    n_cand = len(final_candidates)

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
        stage3_report, funnel_stats, final_candidates, phase3_oos_reports,
    )

    print(f"\n{summary}")

    return PermutationTestSuite(
        feature_name=feature_name,
        feature_type=feature_type,  # type: ignore[arg-type]
        stage1_reports=stage1_reports,
        stage2_reports=stage2_reports,
        stage3_report=stage3_report,
        funnel_stats=funnel_stats,
        ensemble_candidates=final_candidates,
        summary=summary,
        phase3_oos_reports=phase3_oos_reports,
        combo_decisions=combo_decisions,
    )
