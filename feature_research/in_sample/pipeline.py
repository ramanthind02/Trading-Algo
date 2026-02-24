# feature_research/in_sample/pipeline.py
"""Unified EDA pipeline for both continuous and rule-based feature research.

Dispatches on feature_type to handle:
  - CONTINUOUS: runs full EDA + optional Phase 2 binning analysis
  - RULE_BASED: runs EDA with fixed 3-level binning (no Phase 2)

Entry point for tests and scripts — import ``run_eda_pipeline`` or
``run_walkforward_pipeline`` rather than duplicating this logic.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from feature_research.walkforward.runner import WalkforwardRunReport
    from feature_selection.validation.reports import PermutationTestSuite

from feature_research.config import FeatureType
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.config import OutOfSamplePermutationConfig, PermutationTestConfig
from feature_selection.validation.objective_metrics import resolve_objective_metric
from feature_selection.validation.orchestration import run_permutation_test_suite
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_candles_for_config,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.walkforward.io import resolve_walkforward_output_dir, write_walkforward_artifacts
from feature_research.walkforward.runner import run_walkforward_research
from feature_research.walkforward.visualization import plot_fold_timeline, plot_selection_stability
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import (
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    save_eda_report,
)
from utils.core.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    """Extract and normalize timeframe from bias_spec."""
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def _combo_key(params: dict[str, object]) -> tuple[tuple[str, object], ...]:
    """Convert params dict to hashable sorted tuple for use as dict key."""
    return tuple(sorted(params.items(), key=lambda item: item[0]))


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize datetime index to timezone-naive UTC."""
    datetime_index = pd.DatetimeIndex(index)
    return datetime_index.tz_localize(None) if datetime_index.tz is not None else datetime_index


def _normalize_series_datetime_index(series: pd.Series) -> pd.Series:
    """Normalize series datetime index to timezone-naive UTC."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    normalized_series = series.copy()
    normalized_series.index = _normalize_datetime_index(series.index)
    return normalized_series


def _build_continuous_walkforward_evaluator(
    combo_feature_target: dict[tuple[tuple[str, object], ...], pd.DataFrame],
    config: "ResearchConfig",
) -> Callable[..., pd.Series]:
    """Build evaluator for continuous binning walkforward evaluation.

    DISPATCH POINT: Used only for CONTINUOUS feature type.
    """
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> pd.Series:
        combo_data = combo_feature_target[_combo_key(params)]
        fold_index_norm = _normalize_datetime_index(fold_candles.index)
        combo_index_norm = _normalize_datetime_index(combo_data.index)
        # reindex() forbids duplicate target labels (multi-ticker folds); select by mask instead.
        in_fold = combo_index_norm.isin(fold_index_norm)
        fold_data = combo_data.loc[in_fold].dropna()
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


def _build_rule_based_walkforward_evaluator(
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series],
) -> Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]:
    """Build evaluator for rule-based walkforward evaluation.

    DISPATCH POINT: Used only for RULE_BASED feature type.
    Rule-based simply multiplies feature signal by target (pre-computed returns).
    """
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        returns = combo_returns[_combo_key(params)]
        fold_returns = returns.reindex(fold_candles.index).dropna()
        return fold_returns if not fold_returns.empty else pd.Series(dtype=float)

    return evaluate_param_combo


def _expand_params_with_bin_count(
    params: dict[str, object],
    bin_counts: list[int],
) -> list[dict[str, object]]:
    """Expand params by bin_count. Used only for CONTINUOUS."""
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
    """Build returns for a specific bin count. Used only for CONTINUOUS."""
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


def run_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full EDA pipeline for every param combo in config.

    DISPATCHES on feature_type:
      - CONTINUOUS: runs run_eda_for_continuous_feature, builds Phase 2 bin-count grid
      - RULE_BASED: runs run_eda_for_rule_based_feature, skips Phase 2 analysis

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run feature-type-specific EDA reporter.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : ResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags, feature_type).
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
    feature_type_label = config.feature_type.value.upper()

    print(f"\n{'='*64}")
    print(f"EDA Pipeline: {config.bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    # Dispatch on feature_type for EDA pipeline differences
    if config.feature_type == FeatureType.CONTINUOUS:
        # CONTINUOUS: full EDA + Phase 2 binning analysis
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
                evaluate_param_combo=_build_continuous_walkforward_evaluator(combo_feature_target, config),
                research_config=config,
                portfolio_candles_df=portfolio_candles,
                feature_data_by_combo=combo_feature_target,
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
                    "walkforward_selection_method": config.walkforward._effective_selection_method(),
                },
            )
            plt.close(stability_figure)
            plt.close(timeline_figure)

    elif config.feature_type == FeatureType.RULE_BASED:
        # RULE-BASED: EDA only, no Phase 2 binning analysis
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
                evaluate_param_combo=_build_rule_based_walkforward_evaluator(combo_returns),
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
                    "walkforward_selection_method": config.walkforward._effective_selection_method(),
                },
            )
            plt.close(stability_figure)
            plt.close(timeline_figure)

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results


def run_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    """Run walkforward research only (no EDA) for both continuous and rule-based features.

    DISPATCHES on feature_type:
      - CONTINUOUS: uses continuous binning evaluator with bin-count expansion
      - RULE_BASED: uses simple returns multiplier evaluator

    Loads feature data for all param combos, builds the appropriate evaluator,
    runs walkforward research with optional enhanced selection, and writes artifacts
    to ``config.walkforward.output_root``.

    Parameters
    ----------
    config : ResearchConfig
        Research settings. Selection is controlled by ``config.walkforward.selection_method``
        (top_k, enhanced, or stable_region; default stable_region).
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
    feature_type_label = config.feature_type.value.upper()

    print(f"\n{'='*64}")
    print(f"Walkforward Pipeline: {config.bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"Selection method: {config.walkforward._effective_selection_method()}")
    print(f"{'='*64}\n")

    output_dir = resolve_walkforward_output_dir(
        feature_type=config.feature_type.value,
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
    )

    # Dispatch on feature_type
    if config.feature_type == FeatureType.CONTINUOUS:
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
            evaluate_param_combo=_build_continuous_walkforward_evaluator(combo_feature_target, config),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=combo_feature_target,
            output_dir=output_dir,
        )

    elif config.feature_type == FeatureType.RULE_BASED:
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
            evaluate_param_combo=_build_rule_based_walkforward_evaluator(combo_returns),
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown feature_type: {config.feature_type}")

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
        feature_type=config.feature_type.value,
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
            "walkforward_selection_method": config.walkforward._effective_selection_method(),
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
    """Build list of training/testing fold date ranges."""
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


def write_permutation_summary(suite: "PermutationTestSuite", output_dir: Path) -> tuple[Path, Path]:
    """Write permutation test summary for all param combos to the results folder.

    Writes:
      - permutation_summary.csv: one row per param combo (stage1/stage2 p-values, pass, metrics).
      - permutation_summary.md: funnel stats and short human-readable summary.

    Stage 2 is only run for Stage 1 passers (vector shuffle as gate); combos that skipped
    Stage 2 have stage2_pval and stage2_metric empty in CSV.

    Returns
    -------
    tuple[Path, Path]
        Paths to the written CSV and Markdown files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for combo_name, s1 in suite.stage1_reports.items():
        s2 = suite.stage2_reports.get(combo_name)
        row: dict[str, object] = {
            "param_combo": combo_name,
            "stage1_metric": s1.original_metric,
            "stage1_pval": s1.p_value,
            "stage1_passed": s1.passed,
            "stage1_alpha": s1.alpha,
        }
        if s2 is not None:
            row["stage2_metric"] = s2.original_metric
            row["stage2_pval"] = s2.p_value
            row["stage2_passed"] = s2.passed
            row["stage2_mode"] = s2.permutation_mode
        else:
            row["stage2_metric"] = None
            row["stage2_pval"] = None
            row["stage2_passed"] = False
            row["stage2_mode"] = "skipped"
        rows.append(row)

    csv_path = output_dir / "permutation_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    fs = suite.funnel_stats
    md_lines = [
        "# In-Sample Permutation Summary",
        "",
        f"**Feature:** {suite.feature_name}  |  **Type:** {suite.feature_type}",
        "",
        "## Funnel (vector shuffle → pipeline permutation)",
        "",
        "| Stage | Tested | Passed |",
        "|-------|--------|--------|",
        f"| Stage 1 (vector shuffle) | {fs.total_params} | {fs.stage1_pass} |",
        f"| Stage 2 (pipeline/candle) | {fs.stage1_pass} | {fs.stage2_pass} |",
        "",
        f"**Computational savings (Stage 1 gate):** {fs.computational_savings_pct:.1f}%",
        "",
        "## Per-combo metrics and p-values",
        "",
        "| param_combo | S1 metric | S1 p-val | S1 pass | S2 metric | S2 p-val | S2 pass |",
        "|-------------|-----------|----------|--------|-----------|----------|--------|",
    ]
    for row in rows:
        s1_m = row["stage1_metric"]
        s1_p = row["stage1_pval"]
        s1_pass = "✓" if row["stage1_passed"] else "✗"
        s2_m = row.get("stage2_metric")
        s2_p = row.get("stage2_pval")
        s2_pass_cell = "✓" if row.get("stage2_passed") else ("—" if s2_m is None and s2_p is None else "✗")
        s2_m_str = f"{s2_m:.4f}" if s2_m is not None else "—"
        s2_p_str = f"{s2_p:.4f}" if s2_p is not None else "—"
        md_lines.append(
            f"| {row['param_combo']} | {s1_m:.4f} | {s1_p:.4f} | {s1_pass} | {s2_m_str} | {s2_p_str} | {s2_pass_cell} |"
        )
    md_lines.extend(["", "Full data: `permutation_summary.csv`", ""])
    md_path = output_dir / "permutation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return (csv_path, md_path)


def run_permutation_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "PermutationTestSuite":
    """Run the shared permutation suite using the unified research adapter.

    Stage 1 (vector shuffle) runs for all param combos; Stage 2 (pipeline/candle
    permutation) runs only for Stage 1 passers to save compute.

    DISPATCHES on feature_type to determine binning model factory and param grid
    (continuous: expand by bin_count).
    """
    if not config.permutation_suite.enabled:
        raise ValueError("Permutation suite is disabled; set config.permutation_suite.enabled=True.")

    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    if not expanded:
        raise ValueError("No parameter combinations available for permutation suite.")

    candles_df = load_candles_for_config(config)
    seed_spec = expanded[0]
    try:
        seed_feature_data = load_features_for_combo(seed_spec, config, candles_override=candles_df)
    except ValueError as e:
        err_msg = str(e)
        if "Tickers dropped" in err_msg or "Missing tickers" in err_msg:
            # Partial cache: only some tickers have data. Fall back to first ticker so permutation can run.
            single_ticker_config = replace(config, tickers=[config.tickers[0]])
            print(
                f"  [permutation] Multi-ticker alignment failed; using single ticker: {single_ticker_config.tickers[0].name}"
            )
            candles_df = load_candles_for_config(single_ticker_config)
            config = single_ticker_config
            seed_feature_data = load_features_for_combo(seed_spec, config, candles_override=candles_df)
        else:
            raise
    if seed_feature_data is None:
        raise ValueError("Unable to load feature/target data. Ensure cache and candles are available.")

    _, target, feature_col = seed_feature_data
    module_name = config.bias_spec["module_name"]
    timeframe = _normalize_timeframe(config.bias_spec)

    # Param grid: for CONTINUOUS expand by bin_count; for RULE_BASED use bias params only
    if config.feature_type == FeatureType.CONTINUOUS:
        param_grid = [
            combo_params
            for single_spec in expanded
            for combo_params in _expand_params_with_bin_count(
                dict(single_spec["params"]),
                config.binning_params.bin_counts,
            )
        ]
    else:
        param_grid = [single_spec["params"] for single_spec in expanded]

    # Feature extraction uses only bias params (no bin_count) so cache key is correct
    _bias_only_keys = frozenset(config.bias_spec.get("params", {}).keys())

    def extractor_func(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        bias_params = {k: v for k, v in params.items() if k in _bias_only_keys}
        single_spec = {
            "module_name": module_name,
            "timeframes": [timeframe],
            "params": bias_params,
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
        n_jobs_stage2_reps=config.permutation_suite.n_jobs_stage2_reps,
        run_stage1=config.permutation_suite.run_stage1,
        run_stage2=config.permutation_suite.run_stage2,
        run_stage3_walkforward=False,
        out_of_sample=OutOfSamplePermutationConfig(
            objective_metric=config.permutation_suite.objective_metric,
            run_oos_permutation=False,
        ),
    )
    objective_func = resolve_objective_metric(config.permutation_suite.objective_metric)
    fold_structure = _build_fold_structure(
        config.start,
        config.end,
        config.permutation_suite.fold_years,
    )

    # Dispatch on feature_type for binning model factory
    if config.feature_type == FeatureType.CONTINUOUS:
        bp = config.binning_params

        def binning_model_factory(params: dict[str, Any]) -> ContinuousBinningModel:
            bin_count = int(cast(int, params.get("bin_count", bp.bin_counts[0])))
            return ContinuousBinningModel(
                n_bins=bin_count,
                bin_counts=[bin_count],
                selection_metric=bp.selection_metric,
                strategy=bp.strategy,
                metric_threshold=bp.metric_threshold,
                t_threshold=bp.t_threshold,
                min_region_width=bp.min_region_width,
                shrinkage_k=bp.shrinkage_k,
                long_clip_min=bp.long_clip_min,
                long_clip_max=bp.long_clip_max,
                short_clip_min=bp.short_clip_min,
                short_clip_max=bp.short_clip_max,
                use_coverage_bonus=bp.use_coverage_bonus,
                coverage_bonus_per_10pct=bp.coverage_bonus_per_10pct,
                max_coverage_bonus=bp.max_coverage_bonus,
            )
    else:
        def binning_model_factory(_params: dict[str, Any]) -> None:
            return None

    return run_permutation_test_suite(
        candles_df=candles_df,
        feature_spec=config.bias_spec,
        target=target,
        param_grid=param_grid,
        objective_func=objective_func,
        fold_structure=fold_structure,
        config=permutation_config,
        extractor_func=extractor_func,
        binning_model_factory=binning_model_factory,
        feature_type=config.feature_type.value,
        feature_name=feature_col,
    )
