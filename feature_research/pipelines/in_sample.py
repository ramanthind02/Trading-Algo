from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.parameter_analysis import generate_parameter_sensitivity_report
from feature_research.core_helpers import normalize_series_datetime_index, normalize_timeframe_from_bias_spec
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.in_sample.metric_helpers import compute_param_sensitivity_metric
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import (
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    save_eda_report,
)
from utils.core.enums import Ticker

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


matplotlib.use("Agg")


def _default_rolling_window(feature: pd.Series) -> int:
    return max(20, min(252, len(feature) // 4))


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
        bin_index_min=config.binning_params.bin_index_min,
        bin_index_max=config.binning_params.bin_index_max,
    )
    model.fit(feature, target)
    signal = model.predict(feature, strategy=config.binning_params.strategy)
    return normalize_series_datetime_index(signal.mul(target))


def _write_cumsum_plot(
    returns: pd.Series,
    output_path: Path,
    *,
    title: str,
) -> None:
    clean_returns = returns.dropna().sort_index()
    if clean_returns.empty:
        return

    cumulative = clean_returns.cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative.index, cumulative.values, linewidth=1.25)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel("Cum sum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_continuous_in_sample_cumsum_plots(
    feature: pd.Series,
    target: pd.Series,
    combo_output_dir: Path,
    config: "ResearchConfig",
    *,
    label: str,
) -> None:
    plots_dir = combo_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    bin_counts = config.binning_params.bin_counts or [config.binning_params.n_bins]
    for bin_count in sorted({int(bin_count) for bin_count in bin_counts}, reverse=True):
        returns = _build_bin_count_specific_returns(feature, target, bin_count, config)
        _write_cumsum_plot(
            returns=returns,
            output_path=plots_dir / f"in_sample_cumsum_bin_count_{bin_count}.png",
            title=f"In-sample cumulative sum ({label}, bin_count={bin_count})",
        )


def _write_rule_based_in_sample_cumsum_plot(
    feature: pd.Series,
    target: pd.Series,
    combo_output_dir: Path,
    *,
    label: str,
) -> None:
    """Write in-sample cumulative sum plot for rule-based signal (position * target)."""
    plots_dir = combo_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    returns = feature.mul(target).dropna()
    _write_cumsum_plot(
        returns=returns,
        output_path=plots_dir / "in_sample_cumsum.png",
        title=f"In-sample cumulative sum ({label})",
    )


def run_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    timeframe = normalize_timeframe_from_bias_spec(config.bias_spec)
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

    if config.feature_type.value == "continuous":
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
            rolling_window = _default_rolling_window(feature)

            metadata = EDAMetadata(
                feature_name=feature_col,
                param_combo=combo,
                timeframe=timeframe,
                ticker=cast(Ticker, config.tickers[0]),
                timestamp=datetime.now(),
            )
            eda_config = EDAConfig(n_bins=15, rolling_window=rolling_window)
            report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, eda_config)

            combo_output_dir = output_dir / label
            combo_output_dir.mkdir(parents=True, exist_ok=True)
            saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
            _write_continuous_in_sample_cumsum_plots(
                feature=feature,
                target=target,
                combo_output_dir=saved_path,
                config=config,
                label=label,
            )
            results[label] = saved_path

            pearson = report.common_stats.correlation_analysis.pearson
            spread = report.continuous_stats.quintile_spread.spread
            trend = report.continuous_stats.decile_analysis.overall_trend
            viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
            warnings_count = len(report.diagnostics.warnings)
            print(
                f"  [{label}] n={len(feature):,}  pearson={pearson:+.3f}  "
                f"spread={spread:+.3f}  trend={trend}  {viable}  warnings={warnings_count}"
            )
    else:
        combo_store: dict[str, tuple[pd.Series, pd.Series, str]] = {}
        metric_col = config.binning_params.selection_metric
        varying_params = [
            key
            for key, values in config.bias_spec["params"].items()
            if isinstance(values, list) and len(values) > 1
        ]
        metric_rows: list[dict[str, object]] = []

        print(f"Loading data for {len(expanded)} combos...")
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
            combo_store[label] = (feature, target, feature_col)

            if varying_params:
                signals = np.asarray(feature.values == 1).flatten()
                selected_returns = target.values[signals]
                if len(selected_returns) >= 5:
                    try:
                        metric_value = compute_param_sensitivity_metric(selected_returns, metric_col)
                        row: dict[str, object] = {
                            f"param{k + 1}_value": combo[key]
                            for k, key in enumerate(varying_params)
                        }
                        row[metric_col] = metric_value
                        metric_rows.append(row)
                    except Exception:
                        pass

        max_eda_combos = config.param_sensitivity.max_eda_output_combos
        should_preselect = (
            max_eda_combos > 0
            and len(combo_store) > max_eda_combos
            and varying_params
            and metric_rows
        )
        if should_preselect:
            ps_df = pd.DataFrame(metric_rows)
            ps_cfg = config.param_sensitivity
            fixed_params = {
                key: values for key, values in config.bias_spec["params"].items() if key not in varying_params
            }
            print(
                f"\nParam sensitivity pre-selection: {len(combo_store)} combos → "
                f"selecting top {max_eda_combos} for EDA output..."
            )
            try:
                ps_report = generate_parameter_sensitivity_report(
                    results_df=ps_df,
                    param_names=varying_params,
                    metric_col=metric_col,
                    stability_threshold=ps_cfg.stability_threshold,
                    top_k=max_eda_combos,
                    plot_3d_mode="heatmap_slices",
                    smoothing_self_weight=ps_cfg.smoothing_self_weight,
                )
                selected_labels: set[str] = {
                    param_combo_label({**fixed_params, **dict(zip(varying_params, values))})
                    for values in ps_report.top_k_combinations
                }
                print(
                    f"Pre-selection complete: {len(selected_labels)}/{len(combo_store)} combos selected"
                )
            except Exception as exc:
                print(f"Pre-selection failed ({exc}); falling back to top-{max_eda_combos} by raw metric")
                sorted_rows = sorted(
                    metric_rows, key=lambda row: row.get(metric_col, float("-inf")), reverse=True
                )
                selected_labels = {
                    param_combo_label({
                        **fixed_params,
                        **{varying_params[k]: row[f"param{k + 1}_value"] for k in range(len(varying_params))},
                    })
                    for row in sorted_rows[:max_eda_combos]
                }
        else:
            selected_labels = set(combo_store.keys())

        print(
            f"\nRunning EDA for {len(selected_labels)}/{len(combo_store)} combos"
            + (f" (limit={max_eda_combos})" if should_preselect else "")
            + "..."
        )
        for single_spec in expanded:
            combo = single_spec["params"]
            label = param_combo_label(combo)
            if label not in combo_store or label not in selected_labels:
                continue

            feature, target, feature_col = combo_store[label]
            timestamps = pd.DatetimeIndex(feature.index)
            rolling_window = _default_rolling_window(feature)
            metadata = EDAMetadata(
                feature_name=feature_col,
                param_combo=combo,
                timeframe=timeframe,
                ticker=cast(Ticker, config.tickers[0]),
                timestamp=datetime.now(),
            )
            eda_config = EDAConfig(rolling_window=rolling_window, bootstrap_iterations=500)
            report = run_eda_for_rule_based_feature(feature, target, timestamps, metadata, eda_config)

            combo_output_dir = output_dir / label
            combo_output_dir.mkdir(parents=True, exist_ok=True)
            saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
            _write_rule_based_in_sample_cumsum_plot(
                feature=feature,
                target=target,
                combo_output_dir=saved_path,
                label=label,
            )
            results[label] = saved_path

            stats_by_level = report.rule_stats.per_level_stats.stats_by_level
            level_parts = "  ".join(
                f"L[{level}]: sharpe={stats_by_level[level].sharpe:+.2f}"
                if level in stats_by_level
                else f"L[{level}]: n/a"
                for level in [-1, 0, 1]
            )
            viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
            warnings_count = len(report.diagnostics.warnings)
            print(
                f"  [{label}] n={len(feature):,}  {level_parts}  {viable}  warnings={warnings_count}"
            )

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results
