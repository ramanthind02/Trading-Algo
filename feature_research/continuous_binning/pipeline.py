# feature_research/continuous_binning/pipeline.py
"""Core EDA pipeline for continuous binning research.

Entry point for tests and scripts alike — import ``run_continuous_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.continuous_binning.config import ResearchConfig
    from feature_research.walkforward.runner import WalkforwardRunReport
    from feature_selection.validation.reports import PermutationTestSuite

from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.orchestration import run_permutation_test_suite
from feature_research.continuous_binning.data_loader import (
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
from feature_selection.eda.eda_reporter import run_eda_for_continuous_feature, save_eda_report
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
    combo_feature_target: dict[tuple[tuple[str, object], ...], pd.DataFrame],
    config: "ResearchConfig",
) -> Callable[..., pd.Series]:
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> pd.Series:
        combo_data = combo_feature_target[_combo_key(params)]
        fold_data = combo_data.reindex(_normalize_datetime_index(fold_candles.index)).dropna()
        if fold_data.empty:
            return pd.Series(dtype=float)

        train_cutoff = pd.Timestamp(train_end) if train_end is not None else pd.Timestamp(fold_data.index.max())
        train_data = fold_data.loc[fold_data.index <= train_cutoff]
        if train_data.empty:
            return pd.Series(dtype=float)

        bin_count = int(cast(int, params.get("bin_count", config.binning_params.bin_counts[0])))
        model = ContinuousBinningModel(
            n_bins=bin_count,
            bin_counts=[bin_count],
            selection_metric=config.binning_params.selection_metric,
            strategy=config.binning_params.strategy,
            metric_threshold=config.binning_params.metric_threshold,
            t_threshold=config.binning_params.t_threshold,
            min_region_width=config.binning_params.min_region_width,
            shrinkage_k=config.binning_params.shrinkage_k,
            long_clip_min=config.binning_params.long_clip_min,
            long_clip_max=config.binning_params.long_clip_max,
            short_clip_min=config.binning_params.short_clip_min,
            short_clip_max=config.binning_params.short_clip_max,
            use_coverage_bonus=config.binning_params.use_coverage_bonus,
            coverage_bonus_per_10pct=config.binning_params.coverage_bonus_per_10pct,
            max_coverage_bonus=config.binning_params.max_coverage_bonus,
        )
        try:
            model.fit(train_data["feature"], train_data["target"])
        except ValueError:
            return pd.Series(dtype=float)
        signal = model.predict(fold_data["feature"], strategy=config.binning_params.strategy)
        return _normalize_series_datetime_index(signal.mul(fold_data["target"]))

    return evaluate_param_combo


def _expand_params_with_bin_count(
    params: dict[str, object],
    bin_counts: list[int],
) -> list[dict[str, object]]:
    if "bin_count" in params:
        return [dict(params)]
    if not bin_counts:
        return [dict(params)]
    return [{**params, "bin_count": int(bin_count)} for bin_count in bin_counts]


def _build_bin_count_specific_returns(
    feature: pd.Series,
    target: pd.Series,
    bin_count: int,
    config: "ResearchConfig",
) -> pd.Series:
    model = ContinuousBinningModel(
        n_bins=bin_count,
        bin_counts=[bin_count],
        selection_metric=config.binning_params.selection_metric,
        strategy=config.binning_params.strategy,
        metric_threshold=config.binning_params.metric_threshold,
        t_threshold=config.binning_params.t_threshold,
        min_region_width=config.binning_params.min_region_width,
        shrinkage_k=config.binning_params.shrinkage_k,
        long_clip_min=config.binning_params.long_clip_min,
        long_clip_max=config.binning_params.long_clip_max,
        short_clip_min=config.binning_params.short_clip_min,
        short_clip_max=config.binning_params.short_clip_max,
        use_coverage_bonus=config.binning_params.use_coverage_bonus,
        coverage_bonus_per_10pct=config.binning_params.coverage_bonus_per_10pct,
        max_coverage_bonus=config.binning_params.max_coverage_bonus,
    )
    model.fit(feature, target)
    signal = model.predict(feature, strategy=config.binning_params.strategy)
    return _normalize_series_datetime_index(signal.mul(target))


def run_continuous_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full continuous-feature EDA pipeline for every param combo in config.

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run ``run_eda_for_continuous_feature`` to produce a ``ContinuousEDAReport``.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : ResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports.  Created if it does not exist.
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
    combo_feature_target: dict[tuple[tuple[str, object], ...], pd.DataFrame] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None
    reference_target_series: pd.Series | None = None

    print(f"\n{'='*64}")
    print(f"Continuous EDA Pipeline: {config.bias_spec['module_name'].upper()}")
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
        eda_config = EDAConfig(n_bins=15, rolling_window=rolling_window)

        report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        results[label] = saved_path

        expanded_combo_params = _expand_params_with_bin_count(
            params=dict(combo),
            bin_counts=config.binning_params.bin_counts,
        )
        normalized_feature = _normalize_series_datetime_index(feature)
        normalized_target = _normalize_series_datetime_index(target)
        for combo_params in expanded_combo_params:
            combo_bin_count = int(
                cast(int, combo_params.get("bin_count", config.binning_params.bin_counts[0]))
            )
            combo_feature_target[_combo_key(combo_params)] = pd.DataFrame(
                {
                    "feature": normalized_feature,
                    "target": normalized_target,
                }
            )
            successful_param_grid.append(combo_params)
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)
            reference_target_series = normalized_target.reindex(reference_index)

        pearson = report.common_stats.correlation_analysis.pearson
        spread = report.continuous_stats.quintile_spread.spread
        trend = report.continuous_stats.decile_analysis.overall_trend
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings_count = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  pearson={pearson:+.3f}  "
            f"spread={spread:+.3f}  trend={trend}  {viable}  warnings={warnings_count}"
        )

    if config.walkforward.enabled and reference_index is not None and successful_param_grid:
        if reference_target_series is None:
            reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
        else:
            reference_target = reference_target_series.fillna(0.0).rename("walkforward_target")
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_candles_for_config(config)
        walkforward_report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            config=config.walkforward,
            param_grid=successful_param_grid,
            evaluate_param_combo=_build_walkforward_evaluator(combo_feature_target, config),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
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
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            root_dir=config.walkforward.output_root,
            research_context={
                "tickers": [ticker.name for ticker in config.tickers],
                "period_start": str(config.start.date()),
                "period_end": str(config.end.date()),
                "target_col": config.target_col,
                "strategy": config.strategy,
                "binning_bin_counts": config.binning_params.bin_counts,
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


def run_continuous_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    """Run walkforward research only (no EDA) for continuous features.

    Loads feature data for all param combos, builds the return-series evaluator,
    runs walkforward research with optional enhanced selection, and writes artifacts
    to ``config.walkforward.output_root``.

    Parameters
    ----------
    config : ResearchConfig
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
    print(f"Continuous Walkforward Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(
        f"Enhanced selection: {getattr(config.walkforward, 'use_enhanced_selection', False)}"
    )
    print(f"{'='*64}\n")

    combo_feature_target: dict[tuple[tuple[str, object], ...], pd.DataFrame] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None
    reference_target_series: pd.Series | None = None

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
        expanded_combo_params = _expand_params_with_bin_count(
            params=dict(combo),
            bin_counts=config.binning_params.bin_counts,
        )
        normalized_feature = _normalize_series_datetime_index(feature)
        normalized_target = _normalize_series_datetime_index(target)
        for combo_params in expanded_combo_params:
            combo_bin_count = int(
                cast(int, combo_params.get("bin_count", config.binning_params.bin_counts[0]))
            )
            combo_feature_target[_combo_key(combo_params)] = pd.DataFrame(
                {
                    "feature": normalized_feature,
                    "target": normalized_target,
                }
            )
            successful_param_grid.append(combo_params)
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)
            reference_target_series = normalized_target.reindex(reference_index)
        print(f"  [{label}] loaded n={len(feature):,}")

    if not successful_param_grid or reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    if reference_target_series is None:
        reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
    else:
        reference_target = reference_target_series.fillna(0.0).rename("walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
    portfolio_candles = load_candles_for_config(config)

    walkforward_report = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type="continuous",
        module_name=str(config.bias_spec["module_name"]),
        config=config.walkforward,
        param_grid=successful_param_grid,
        evaluate_param_combo=_build_walkforward_evaluator(combo_feature_target, config),
        research_config=config,
        portfolio_candles_df=portfolio_candles,
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
        feature_type="continuous",
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
        research_context={
            "tickers": [ticker.name for ticker in config.tickers],
            "period_start": str(config.start.date()),
            "period_end": str(config.end.date()),
            "target_col": config.target_col,
            "strategy": config.strategy,
            "binning_bin_counts": config.binning_params.bin_counts,
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


def run_continuous_permutation_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "PermutationTestSuite":
    """Run the shared permutation suite using the continuous-research adapter."""
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
        binning_model_factory=lambda _params: ContinuousBinningModel(
            bin_counts=config.binning_params.bin_counts,
            use_coverage_bonus=config.binning_params.use_coverage_bonus,
            coverage_bonus_per_10pct=config.binning_params.coverage_bonus_per_10pct,
            max_coverage_bonus=config.binning_params.max_coverage_bonus,
        ),
        feature_type="continuous",
        feature_name=feature_col,
    )
