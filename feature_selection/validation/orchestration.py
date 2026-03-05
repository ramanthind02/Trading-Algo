"""
T016: Early Stopping Orchestration

Coordinates the two-stage permutation testing funnel with early stopping:
  Stage 1 (vector shuffle) -> filter passers
  Stage 2 (pipeline permutation) -> filter passers (only from Stage 1 passers)
  Stage 3 (walkforward stability) has been removed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.permutation_tests import (
    _ContinuousPermutationBatchItem,
    _RuleBasedPermutationBatchItem,
    _run_pipeline_permutation_continuous_batch,
    _run_pipeline_permutation_rule_based_batch,
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
from feature_selection.validation.stability_analysis import _param_combo_name


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
    n_stable = funnel_stats.stable_params
    n_cand = funnel_stats.ensemble_candidates
    savings = funnel_stats.computational_savings_pct

    oos_line = (
        f"OOS Permutation: {len(phase3_oos_reports)} tested -> {sum(r.passed for r in phase3_oos_reports.values())} passed"
        if run_oos_permutation
        else "OOS Permutation: skipped (in-sample / walkforward-only phase)"
    )
    lines = [
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
    config: PermutationTestConfig,
    extractor_func: Callable[[pd.DataFrame, Dict], pd.Series],
    binning_model_factory: Optional[Callable[[Dict], BinningModelBase]] = None,
    feature_type: str = 'continuous',
    feature_name: str = 'unknown',
    fold_structure: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> PermutationTestSuite:
    """Run the two-stage permutation testing funnel (Stage 1 + 2; Stage 3 removed).

    Args:
        candles_df: OHLCV DataFrame with DatetimeIndex.
        feature_spec: Dict with 'module_name' and 'params' keys (used as label).
        target: Forward-return Series.
        param_grid: All parameter combinations (as list of dicts).
        objective_func: (returns: pd.Series) -> float.
        config: PermutationTestConfig with nreps, alpha, etc.
        extractor_func: (candles_df, params) -> named pd.Series.
        binning_model_factory: Optional (params) -> BinningModelBase factory.
        feature_type: 'continuous' or 'rule_based'.
        feature_name: Display name.
        fold_structure: Unused (Stage 3 removed). Kept for backward compatibility.

    Returns:
        PermutationTestSuite with all stage reports and funnel statistics.
    """
    fold_structure = fold_structure or []
    # ---------- Stage 1: Vector Shuffle (all param combos) ----------
    stage1_reports: Dict[str, VectorShuffleReport] = {}
    if config.run_stage1:
        print(f"\n{'='*60}")
        print(f"Stage 1: Vector Shuffle — testing {len(param_grid)} param combos")
        print(f"{'='*60}")

        for params in tqdm(param_grid, desc="Stage 1 (vector shuffle)", unit="combo"):
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
            except Exception:
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

        stage1_passers: Set[str] = {k for k, r in stage1_reports.items() if r.passed}
        print(f"Stage 1: {len(stage1_passers)}/{len(param_grid)} passed")
    else:
        stage1_passers = {_param_combo_name(params) for params in param_grid}
        print("\nStage 1: skipped (disabled in config)")

    # ---------- Stage 2: Pipeline / Candle Permutation (only Stage 1 passers) ----------
    stage2_reports: Dict[str, PipelinePermutationReport] = {}
    params_to_run_stage2 = [p for p in param_grid if _param_combo_name(p) in stage1_passers]

    if not config.run_stage2:
        print("\nStage 2: skipped (disabled in config)")
        stage2_passers = set(stage1_passers)
    elif params_to_run_stage2:
        print(f"\n{'='*60}")
        print(f"Stage 2: Pipeline Permutation (candle shuffle) — {len(params_to_run_stage2)} passers")
        print(f"{'='*60}")
        try:
            if feature_type == 'continuous':
                batch_items: list[_ContinuousPermutationBatchItem] = []
                for params in params_to_run_stage2:
                    combo_name = _param_combo_name(params)

                    def _extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                        return extractor_func(df, _params)

                    model_template = binning_model_factory(params) if binning_model_factory else None
                    if model_template is None:
                        raise ValueError('binning_model_factory required for continuous features')

                    batch_items.append(
                        _ContinuousPermutationBatchItem(
                            param_combo=combo_name,
                            bias_node_extractor=_extractor_for_combo,
                            binning_model=model_template,
                        )
                    )

                stage2_reports = _run_pipeline_permutation_continuous_batch(
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
            else:
                batch_items_rb: list[_RuleBasedPermutationBatchItem] = []
                for params in params_to_run_stage2:
                    combo_name = _param_combo_name(params)

                    def _rule_extractor_for_combo(df: pd.DataFrame, _params=params) -> pd.Series:
                        return extractor_func(df, _params)

                    batch_items_rb.append(
                        _RuleBasedPermutationBatchItem(
                            param_combo=combo_name,
                            rule_extractor=_rule_extractor_for_combo,
                        )
                    )

                stage2_reports = _run_pipeline_permutation_rule_based_batch(
                    candles_df=candles_df,
                    items=batch_items_rb,
                    target=target,
                    objective_func=objective_func,
                    metric_threshold=config.metric_threshold,
                    nreps=config.nreps,
                    alpha=config.alpha,
                    random_seed=config.random_seed,
                )
        except Exception:
            for params in params_to_run_stage2:
                combo_name = _param_combo_name(params)
                stage2_reports[combo_name] = PipelinePermutationReport(
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

        print(f"Stage 2: {len([r for r in stage2_reports.values() if r.passed])}/{len(params_to_run_stage2)} passed")
        stage2_passers = {k for k, r in stage2_reports.items() if r.passed}
    else:
        print(f"\nStage 2: skipped (0 Stage 1 passers)")
        stage2_passers = set()

    # ---------- Stage 3: removed (was walkforward stability) ----------
    stable_params: Set[str] = set()
    ensemble_candidates = sorted(stage2_passers)
    stage3_report = WalkforwardStabilityReport(
        feature_name=feature_name,
        feature_type=feature_type,  # type: ignore[arg-type]
        fold_results=[],
        consistency_metrics={},
        is_stable=False,
        stability_verdict="Stage 3 (walkforward permutation) removed.",
        top_k=0,
    )
    print("\nStage 3: removed (walkforward permutation no longer run)")

    # ---------- Phase 3: OOS permutation (optional; skipped in in-sample phase) ----------
    phase3_oos_reports: Dict[str, OutOfSamplePermutationReport] = {}
    oos_passers: Set[str] = set()
    if config.run_oos_permutation:
        candidate_source = config.out_of_sample.candidate_source
        oos_candidates = (
            stage2_passers
            if candidate_source == 'stage2_passers'
            else stage2_passers & stable_params
        )
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

    # Savings = Stage 2 evaluations saved by Stage 1 early stopping (Stage 3 removed)
    total_without = n_total * config.nreps * 2
    actual_stage1_cost = (n_total * config.nreps) if config.run_stage1 else 0
    actual_stage2_cost = (n1 * config.nreps) if config.run_stage2 else 0
    actual = actual_stage1_cost + actual_stage2_cost
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
        run_oos_permutation=config.run_oos_permutation,
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
