# feature_research/rule_based/pipeline.py
"""Core EDA pipeline for rule-based feature research.

Entry point for tests and scripts alike — import ``run_rule_based_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.in_sample.rule_based.config import RuleBasedResearchConfig
    from feature_research.walkforward.runner import WalkforwardRunReport
    from feature_selection.validation.reports import PermutationTestSuite

from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.orchestration import run_permutation_test_suite
from feature_research.in_sample.rule_based.data_loader import (
    expand_bias_specs,
    load_candles_for_config,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.walkforward.io import write_walkforward_artifacts
from feature_research.walkforward.runner import run_walkforward_research
from feature_research.walkforward.visualization import plot_fold_timeline, plot_selection_stability
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import run_eda_for_rule_based_feature, save_eda_report
from utils.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def _combo_key(params: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(params.items(), key=lambda item: item[0]))


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    datetime_index = pd.DatetimeIndex(index)
    return datetime_index.tz_localize(None) if datetime_index.tz is not None else datetime_index


def _normalize_series_datetime_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    normalized_series = series.copy()
    normalized_series.index = _normalize_datetime_index(series.index)
    return normalized_series


def _build_walkforward_evaluator(
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series],
) -> Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]:
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        returns = combo_returns[_combo_key(params)]
        fold_returns = returns.reindex(fold_candles.index).dropna()
        return fold_returns if not fold_returns.empty else pd.Series(dtype=float)

    return evaluate_param_combo


def run_rule_based_eda_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full rule-based feature EDA pipeline for every param combo in config.

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run ``run_eda_for_rule_based_feature`` to produce a ``RuleBasedEDAReport``.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : RuleBasedResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports. Created if it does not exist.
        Each param combo writes to ``output_dir / param_label /``.

    Returns
    -------
    dict[str, Path]
        Mapping of ``param_label`` → saved report path for each successful combo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    tf = _normalize_timeframe(config.bias_spec)

    results: dict[str, Path] = {}
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None

    print(f"\n{'='*64}")
    print(f"Rule-Based EDA Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, feature_col = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature = paired["feature"]
        target = paired["target"]
        timestamps = pd.DatetimeIndex(feature.index)

        rolling_window = max(20, min(252, len(feature) // 4))

        metadata = EDAMetadata(
            feature_name=feature_col,
            param_combo=combo,
            timeframe=tf,
            ticker=config.tickers[0],
            timestamp=datetime.now(),
        )
        eda_config = EDAConfig(rolling_window=rolling_window, bootstrap_iterations=500)

        report = run_eda_for_rule_based_feature(feature, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        results[label] = saved_path

        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)

        stats_by_level = report.rule_stats.per_level_stats.stats_by_level
        level_parts = "  ".join(
            f"L[{lvl}]: sharpe={stats_by_level[lvl].sharpe:+.2f}"
            if lvl in stats_by_level
            else f"L[{lvl}]: n/a"
            for lvl in [-1, 0, 1]
        )
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings_count = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  {level_parts}  {viable}  warnings={warnings_count}"
        )

    if config.walkforward.enabled and reference_index is not None and successful_param_grid:
        reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        walkforward_report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="rule_based",
            module_name=str(config.bias_spec["module_name"]),
            config=config.walkforward,
            param_grid=successful_param_grid,
            evaluate_param_combo=_build_walkforward_evaluator(combo_returns),
        )
        stability_figure, _ = plot_selection_stability(
            selection_summary_df=walkforward_report.selection_summary_df,
            top_k=config.walkforward.top_k,
        )
        timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
        write_walkforward_artifacts(
            report=walkforward_report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="rule_based",
            module_name=str(config.bias_spec["module_name"]),
            root_dir=config.walkforward.output_root,
            research_context={
                "tickers": [ticker.name for ticker in config.tickers],
                "period_start": str(config.start.date()),
                "period_end": str(config.end.date()),
                "target_col": config.target_col,
                "strategy": config.strategy,
                "walkforward_test_step": config.walkforward.test_step,
                "walkforward_num_steps": config.walkforward.num_steps,
                "walkforward_top_k": config.walkforward.top_k,
                "walkforward_use_enhanced_selection": config.walkforward.use_enhanced_selection,
            },
        )
        plt.close(stability_figure)
        plt.close(timeline_figure)

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results


def run_rule_based_walkforward_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    """Run walkforward research only (no EDA) for rule-based features.

    Loads feature data for all param combos, builds the return-series evaluator,
    runs walkforward research with optional enhanced selection, and writes artifacts
    to ``config.walkforward.output_root``.

    Parameters
    ----------
    config : RuleBasedResearchConfig
        Research settings. Set ``config.walkforward.use_enhanced_selection = True``
        to activate the three-objective enhanced selection algorithm.
    output_dir : Path
        Created if it does not exist. Not used for artifact output - artifacts
        are written to ``config.walkforward.output_root`` via
        ``write_walkforward_artifacts``.

    Returns
    -------
    WalkforwardRunReport
        Folds, fold scores (with enhanced columns if enabled), and selection summary.

    Raises
    ------
    ValueError
        If no param combos load successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)

    print(f"\n{'='*64}")
    print(f"Rule-Based Walkforward Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(
        f"Enhanced selection: {getattr(config.walkforward, 'use_enhanced_selection', False)}"
    )
    print(f"{'='*64}\n")

    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature = paired["feature"]
        target = paired["target"]
        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)
        print(f"  [{label}] loaded n={len(feature):,}")

    if not successful_param_grid or reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)

    walkforward_report = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type="rule_based",
        module_name=str(config.bias_spec["module_name"]),
        config=config.walkforward,
        param_grid=successful_param_grid,
        evaluate_param_combo=_build_walkforward_evaluator(combo_returns),
    )
    if walkforward_report.folds_df.empty:
        first_ts = pd.Timestamp(reference_index.min())
        last_ts = pd.Timestamp(reference_index.max())
        raise ValueError(
            "No walkforward folds were generated. "
            f"Data window={first_ts.date()}..{last_ts.date()}, "
            f"walkforward train_start={config.walkforward.train_start.date()}, "
            f"train_end={config.walkforward.train_end.date()}, "
            f"test_step={config.walkforward.test_step}, num_steps={config.walkforward.num_steps}."
        )
    stability_figure, _ = plot_selection_stability(
        selection_summary_df=walkforward_report.selection_summary_df,
        top_k=config.walkforward.top_k,
    )
    timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
    write_walkforward_artifacts(
        report=walkforward_report,
        walkforward_stability_figure=stability_figure,
        fold_timeline_figure=timeline_figure,
        feature_type="rule_based",
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
        research_context={
            "tickers": [ticker.name for ticker in config.tickers],
            "period_start": str(config.start.date()),
            "period_end": str(config.end.date()),
            "target_col": config.target_col,
            "strategy": config.strategy,
            "walkforward_test_step": config.walkforward.test_step,
            "walkforward_num_steps": config.walkforward.num_steps,
            "walkforward_top_k": config.walkforward.top_k,
            "walkforward_use_enhanced_selection": config.walkforward.use_enhanced_selection,
        },
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)

    print(
        f"\nDone. {len(successful_param_grid)}/{len(expanded)} combos "
        f"-> walkforward artifacts written to {config.walkforward.output_root}\n"
    )
    return walkforward_report


def _build_fold_structure(
    start: datetime,
    end: datetime,
    fold_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    step_years = max(1, fold_years)
    fold_start = pd.Timestamp(start).normalize()
    end_exclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while fold_start < end_exclusive:
        fold_end = min(fold_start + pd.DateOffset(years=step_years), end_exclusive)
        if fold_end <= fold_start:
            break
        folds.append((fold_start, fold_end))
        fold_start = fold_end

    return folds


def run_rule_based_permutation_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> "PermutationTestSuite":
    """Run the shared permutation suite using the rule-based research adapter."""
    if not config.permutation_suite.enabled:
        raise ValueError("Permutation suite is disabled; set config.permutation_suite.enabled=True.")

    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for permutation suite.")

    candles_df = load_candles_for_config(config)
    seed_feature_data = load_features_for_combo(expanded[0], config, candles_override=candles_df)
    if seed_feature_data is None:
        raise ValueError("Unable to load feature/target data. Ensure cache and candles are available.")

    _, target, feature_col = seed_feature_data
    param_grid = [single_spec["params"] for single_spec in expanded]
    module_name = config.bias_spec["module_name"]
    timeframe = _normalize_timeframe(config.bias_spec)

    def extractor_func(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        single_spec = {
            "module_name": module_name,
            "timeframes": [timeframe],
            "params": params,
        }
        loaded = load_features_for_combo(single_spec, config, candles_override=df)
        if loaded is None:
            raise ValueError(f"Feature extraction returned no data for params={params}.")
        feature, _, feature_name = loaded
        return feature.rename(feature_name)

    permutation_config = PermutationTestConfig(
        nreps=config.permutation_suite.nreps,
        alpha=config.permutation_suite.alpha,
        metric_threshold=config.permutation_suite.metric_threshold,
        top_k=config.permutation_suite.top_k,
        random_seed=config.permutation_suite.random_seed,
        permutation_mode_stage2=config.permutation_suite.permutation_mode_stage2,
        min_folds_stable=config.permutation_suite.min_folds_stable,
        objective_metric=config.permutation_suite.objective_metric,
    )
    objective_func = resolve_objective_metric(config.permutation_suite.objective_metric)
    fold_structure = _build_fold_structure(
        config.start,
        config.end,
        config.permutation_suite.fold_years,
    )

    return run_permutation_test_suite(
        candles_df=candles_df,
        feature_spec=config.bias_spec,
        target=target,
        param_grid=param_grid,
        objective_func=objective_func,
        fold_structure=fold_structure,
        config=permutation_config,
        extractor_func=extractor_func,
        feature_type="rule_based",
        feature_name=feature_col,
    )
