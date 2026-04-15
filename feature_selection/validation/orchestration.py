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
    run_oos_permutation_for_param,
    run_vector_shuffle_target_perm_batch,
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
from utils.core.signal_alignment import align_signal_to_target


def _failed_vector_shuffle_report(combo_name: str, nreps: int, alpha: float) -> VectorShuffleReport:
    return VectorShuffleReport(
        param_combo=combo_name,
        original_metric=0.0,
        null_distribution=np.zeros(nreps),
        critical_value=0.0,
        p_value=1.0,
        passed=False,
        alpha=alpha,
        nreps=nreps,
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
    run_oos_permutation: bool = False,
) -> str:
    n_total = funnel_stats.total_params
    n1 = funnel_stats.stage1_pass
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
            f"Vector shuffle: {n_total} tested -> {n1} passed ({100*n1//max(n_total,1)}%)",
            "Pipeline permutation: removed",
            "Walkforward stability (Stage 3): removed",
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
        vector_report=_failed_vector_shuffle_report(combo_name, nreps, alpha),
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
    *,
    aligned_signals_by_combo: Optional[Dict[str, Tuple[pd.Series, pd.Series]]] = None,
) -> PermutationTestSuite:
    """Run the signed-signal permutation funnel.

    aligned_signals_by_combo
        When provided, each combo's ``(feature, target)`` is already aligned; stage 1 runs
        ``run_vector_shuffle_target_perm_batch``: a **batched target-permutation** null
        (one random reorder of ``target`` per rep, shared across combos). This is **not**
        the same null as ``run_vector_shuffle_test`` (permute **feature** values, fixed
        target). Prefer omitting this for standard "vector shuffle" signal alignment.
    """
    _ = feature_spec
    fold_structure = fold_structure or []

    def _signal_for_params(df: pd.DataFrame, params: Dict) -> pd.Series:
        return extractor_func(df, params)

    def _stage1_report_for_params(params: Dict) -> Tuple[str, VectorShuffleReport]:
        combo_name = _param_combo_name(params)
        try:
            fitted_feature, aligned_target = align_signal_to_target(
                _signal_for_params(candles_df, params),
                target,
            )
            report = run_vector_shuffle_test(
                fitted_feature=fitted_feature,
                target=aligned_target,
                objective_func=objective_func,
                nreps=config.nreps,
                alpha=config.alpha,
                random_seed=config.random_seed,
                param_combo=combo_name,
            )
            return combo_name, report
        except Exception:
            return combo_name, _failed_vector_shuffle_report(
                combo_name, config.nreps, config.alpha
            )

    stage1_reports: Dict[str, VectorShuffleReport] = {}
    if config.run_stage1:
        print(f"\n{'='*60}")
        print(f"Vector shuffle — testing {len(param_grid)} param combos")
        print(f"{'='*60}")
        if aligned_signals_by_combo is not None:
            ordered = [_param_combo_name(params) for params in param_grid]
            features_by_combo = {cn: aligned_signals_by_combo[cn][0] for cn in ordered}
            ref_index = features_by_combo[ordered[0]].index
            for cn in ordered:
                if not features_by_combo[cn].index.equals(ref_index):
                    raise ValueError(
                        f"Permutation batch requires identical index for all combos; mismatch at {cn!r}",
                    )
            shared_target = target.reindex(ref_index)
            print(
                "Vector shuffle: batched target permutation — "
                f"{len(ordered)} combos × {config.nreps} reps (one target shuffle per rep)",
            )
            stage1_reports = run_vector_shuffle_target_perm_batch(
                ordered_combo_names=ordered,
                features_by_combo=features_by_combo,
                target=shared_target,
                metric_spec=config.objective_metric,
                nreps=config.nreps,
                alpha=config.alpha,
                random_seed=config.random_seed,
            )
        else:
            for params in tqdm(param_grid, desc="Vector shuffle", unit="combo"):
                combo_name, report = _stage1_report_for_params(params)
                stage1_reports[combo_name] = report
        stage1_passers: Set[str] = {k for k, r in stage1_reports.items() if r.passed}
        print(f"Vector shuffle: {len(stage1_passers)}/{len(param_grid)} passed")
    else:
        stage1_passers = {_param_combo_name(params) for params in param_grid}
        print("\nVector shuffle: skipped (disabled in config)")

    stage2_reports: Dict[str, PipelinePermutationReport] = {}
    stage2_passers: Set[str] = set(stage1_passers)

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
    print("\nWalkforward stability stage: removed")

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
                fitted_feature, aligned_target = align_signal_to_target(
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
    savings_pct = 0.0

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
