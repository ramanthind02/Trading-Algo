"""
T016: Early stopping orchestration for frozen signed signals.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.permutation_tests import (
    _SignedSignalPermutationBatchItem,
    _run_pipeline_permutation_batch,
    run_oos_permutation_for_param,
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
from feature_selection.validation.stability_analysis import _param_combo_name


def _align_signal_to_target(signal: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned_signal = signal.reindex(target.index).dropna()
    aligned_target = target.reindex(aligned_signal.index).dropna()
    aligned_signal = aligned_signal.reindex(aligned_target.index)
    return aligned_signal, aligned_target


def _build_summary(
    feature_name: str,
    param_grid: List[Dict],
    stage1_reports: Dict[str, VectorShuffleReport],
    stage2_reports: Dict[str, PipelinePermutationReport],
    stage3_report: WalkforwardStabilityReport,
    funnel_stats: FunnelStatistics,
    ensemble_candidates: List[str],
    phase3_oos_reports: Dict[str, OutOfSamplePermutationReport],
    run_oos_permutation: bool = False,
) -> str:
    n_total = funnel_stats.total_params
    n1 = funnel_stats.stage1_pass
    n2 = funnel_stats.stage2_pass
    savings = funnel_stats.computational_savings_pct
    oos_line = (
        f"OOS Permutation: {len(phase3_oos_reports)} tested -> {sum(r.passed for r in phase3_oos_reports.values())} passed"
        if run_oos_permutation
        else "OOS Permutation: skipped (in-sample / walkforward-only phase)"
    )
    return "\n".join(
        [
            f"Feature: {feature_name}",
            f"Parameter grid: {n_total} combinations",
            "",
            f"Stage 1 (Vector Shuffle): {n_total} tested -> {n1} passed ({100*n1//max(n_total,1)}%)",
            f"Stage 2 (Pipeline Permutation): {n1} tested -> {n2} passed",
            "Stage 3 (Walkforward Stability): removed",
            oos_line,
            "",
            f"Ensemble candidates: {ensemble_candidates}",
            f"Computational savings: {savings:.1f}%",
            "",
            f"Stability verdict: {stage3_report.stability_verdict}",
        ]
    )


def _build_failed_oos_report(combo_name: str, nreps: int, alpha: float) -> OutOfSamplePermutationReport:
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
    config: PermutationTestConfig,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    feature_name: str = "unknown",
    fold_structure: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> PermutationTestSuite:
    """Run the signed-signal permutation funnel."""
    _ = feature_spec
    fold_structure = fold_structure or []

    def _signal_for_params(df: pd.DataFrame, params: Dict) -> pd.Series:
        return extractor_func(df, params)

    stage1_reports: Dict[str, VectorShuffleReport] = {}
    if config.run_stage1:
        print(f"\n{'='*60}")
        print(f"Stage 1: Vector Shuffle — testing {len(param_grid)} param combos")
        print(f"{'='*60}")
        for params in tqdm(param_grid, desc="Stage 1 (vector shuffle)", unit="combo"):
            combo_name = _param_combo_name(params)
            try:
                fitted_feature, aligned_target = _align_signal_to_target(
                    _signal_for_params(candles_df, params),
                    target,
                )
                stage1_reports[combo_name] = run_vector_shuffle_test(
                    fitted_feature=fitted_feature,
                    target=aligned_target,
                    objective_func=objective_func,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                    param_combo=combo_name,
                )
            except Exception:
                stage1_reports[combo_name] = VectorShuffleReport(
                    param_combo=combo_name,
                    original_metric=0.0,
                    null_distribution=np.zeros(config.nreps),
                    critical_value=0.0,
                    p_value=1.0,
                    passed=False,
                    alpha=config.alpha,
                    nreps=config.nreps,
                )
        stage1_passers: Set[str] = {k for k, r in stage1_reports.items() if r.passed}
        print(f"Stage 1: {len(stage1_passers)}/{len(param_grid)} passed")
    else:
        stage1_passers = {_param_combo_name(params) for params in param_grid}
        print("\nStage 1: skipped (disabled in config)")

    stage2_reports: Dict[str, PipelinePermutationReport] = {}
    params_to_run_stage2 = [p for p in param_grid if _param_combo_name(p) in stage1_passers]
    if not config.run_stage2:
        print("\nStage 2: skipped (disabled in config)")
        stage2_passers = set(stage1_passers)
    elif params_to_run_stage2:
        print(f"\n{'='*60}")
        print(f"Stage 2: Pipeline Permutation (signed signal) — {len(params_to_run_stage2)} passers")
        print(f"{'='*60}")
        try:
            batch_items = [
                _SignedSignalPermutationBatchItem(
                    param_combo=_param_combo_name(params),
                    signal_extractor=lambda df, _params=params: _signal_for_params(df, _params),
                )
                for params in params_to_run_stage2
            ]
            stage2_reports = _run_pipeline_permutation_batch(
                candles_df=candles_df,
                items=batch_items,
                target=target,
                objective_func=objective_func,
                permutation_mode=config.permutation_mode_stage2,
                metric_threshold=config.metric_threshold,
                nreps=config.nreps,
                alpha=config.alpha,
                random_seed=config.random_seed,
                n_jobs_reps=config.n_jobs_stage2_reps,
            )
        except Exception:
            for params in params_to_run_stage2:
                combo_name = _param_combo_name(params)
                stage2_reports[combo_name] = PipelinePermutationReport(
                    param_combo=combo_name,
                    feature_type="signed_signal",
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

        print(
            f"Stage 2: {len([r for r in stage2_reports.values() if r.passed])}/{len(params_to_run_stage2)} passed"
        )
        stage2_passers = {k for k, r in stage2_reports.items() if r.passed}
    else:
        print("\nStage 2: skipped (0 Stage 1 passers)")
        stage2_passers = set()

    stable_params: Set[str] = set()
    ensemble_candidates = sorted(stage2_passers)
    stage3_report = WalkforwardStabilityReport(
        feature_name=feature_name,
        feature_type="signed_signal",
        fold_results=[],
        consistency_metrics={},
        is_stable=False,
        stability_verdict="Stage 3 (walkforward permutation) removed.",
        top_k=0,
    )
    print("\nStage 3: removed (walkforward permutation no longer run)")

    phase3_oos_reports: Dict[str, OutOfSamplePermutationReport] = {}
    oos_passers: Set[str] = set()
    if config.run_oos_permutation:
        candidate_source = config.out_of_sample.candidate_source
        oos_candidates = (
            stage2_passers if candidate_source == "stage2_passers" else stage2_passers & stable_params
        )
        oos_objective_func = resolve_objective_metric(config.out_of_sample.objective_metric)
        params_by_combo = {_param_combo_name(params): params for params in param_grid}
        for combo_name in sorted(oos_candidates):
            params = params_by_combo.get(combo_name)
            if params is None:
                continue
            try:
                fitted_feature, aligned_target = _align_signal_to_target(
                    _signal_for_params(candles_df, params),
                    target,
                )
                phase3_oos_reports[combo_name] = run_oos_permutation_for_param(
                    param_combo=combo_name,
                    fitted_feature=fitted_feature,
                    candles_df=candles_df,
                    target=aligned_target,
                    objective_func=oos_objective_func,
                    signal_extractor=lambda df, _params=params: _signal_for_params(df, _params),
                    permutation_mode=config.permutation_mode_stage2,
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                )
            except Exception:
                phase3_oos_reports[combo_name] = _build_failed_oos_report(
                    combo_name=combo_name,
                    nreps=config.nreps,
                    alpha=config.alpha,
                )
        oos_passers = {combo for combo, report in phase3_oos_reports.items() if report.passed}

    final_candidates = (
        sorted(set(ensemble_candidates) & oos_passers)
        if config.run_oos_permutation
        else sorted(ensemble_candidates)
    )
    final_candidate_set = set(final_candidates)

    combo_decisions = {
        combo_name: ComboDecisionRecord(
            param_combo=combo_name,
            stage1_passed=combo_name in stage1_passers,
            stage2_passed=combo_name in stage2_passers,
            walkforward_stable=combo_name in stable_params,
            oos_passed=combo_name in oos_passers,
            final_status=(
                "candidate"
                if combo_name in final_candidate_set
                else "needs_review"
                if combo_name in stage2_passers
                else "rejected"
            ),
        )
        for combo_name in (_param_combo_name(params) for params in param_grid)
    }

    n_total = len(param_grid)
    n1 = len(stage1_passers)
    n2 = len(stage2_passers)
    n_stable = len(stable_params)
    n_cand = len(final_candidates)
    total_without = n_total * config.nreps * 2
    actual_stage1_cost = (n_total * config.nreps) if config.run_stage1 else 0
    actual_stage2_cost = (n1 * config.nreps) if config.run_stage2 else 0
    savings_pct = max(0.0, (total_without - (actual_stage1_cost + actual_stage2_cost)) / max(total_without, 1) * 100.0)

    funnel_stats = FunnelStatistics(
        total_params=n_total,
        stage1_pass=n1,
        stage2_pass=n2,
        stable_params=n_stable,
        ensemble_candidates=n_cand,
        computational_savings_pct=savings_pct,
    )
    summary = _build_summary(
        feature_name,
        param_grid,
        stage1_reports,
        stage2_reports,
        stage3_report,
        funnel_stats,
        final_candidates,
        phase3_oos_reports,
        run_oos_permutation=config.run_oos_permutation,
    )
    print(f"\n{summary}")

    return PermutationTestSuite(
        feature_name=feature_name,
        feature_type="signed_signal",
        stage1_reports=stage1_reports,
        stage2_reports=stage2_reports,
        stage3_report=stage3_report,
        funnel_stats=funnel_stats,
        ensemble_candidates=final_candidates,
        summary=summary,
        phase3_oos_reports=phase3_oos_reports,
        combo_decisions=combo_decisions,
    )
