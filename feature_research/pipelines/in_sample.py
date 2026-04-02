from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from feature_research.core_helpers import normalize_timeframe_from_bias_spec
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.in_sample.metric_helpers import compute_param_sensitivity_metric
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import (
    run_eda_for_signed_signal_feature,
    save_eda_report,
)
from utils.core.enums import Ticker

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


SIGNED_SIGNAL_FEATURE_TYPE_LABEL = "SIGNED_SIGNAL"


def _default_rolling_window(feature: pd.Series) -> int:
    return max(20, min(252, len(feature) // 4))


def _write_cumsum_summary(
    returns: pd.Series,
    output_path: Path,
    *,
    title: str,
) -> None:
    clean_returns = returns.dropna().sort_index()
    if clean_returns.empty:
        return

    cumulative = clean_returns.cumsum().rename("cumulative_return")
    cumulative.to_frame().assign(report_title=title).to_csv(output_path, index_label="datetime")


def run_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    timeframe = normalize_timeframe_from_bias_spec(config.bias_spec)
    results: dict[str, Path] = {}

    print(f"\n{'='*64}")
    print(
        f"EDA Pipeline: {config.bias_spec['module_name'].upper()} "
        f"({SIGNED_SIGNAL_FEATURE_TYPE_LABEL})"
    )
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    combo_store: dict[str, tuple[pd.Series, pd.Series, str]] = {}
    varying_params = [
        key for key, values in config.bias_spec["params"].items() if isinstance(values, list) and len(values) > 1
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
        paired = pd.DataFrame({"signal": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        signal = paired["signal"]
        target = paired["target"]
        combo_store[label] = (signal, target, feature_col)

        if varying_params:
            active = signal.fillna(0).to_numpy() != 0
            selected_returns = target.values[active]
            if len(selected_returns) >= 5:
                try:
                    metric_value = compute_param_sensitivity_metric(selected_returns, "t_stat", timeframe)
                    row: dict[str, object] = {
                        f"param{k + 1}_value": combo[key]
                        for k, key in enumerate(varying_params)
                    }
                    row["t_stat"] = metric_value
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
            from eda.parameter_analysis import generate_parameter_sensitivity_report

            ps_report = generate_parameter_sensitivity_report(
                results_df=ps_df,
                param_names=varying_params,
                metric_col="t_stat",
                stability_threshold=ps_cfg.stability_threshold,
                top_k=max_eda_combos,
                plot_3d_mode="heatmap_slices",
                smoothing_self_weight=ps_cfg.smoothing_self_weight,
            )
            selected_labels: set[str] = {
                param_combo_label({**fixed_params, **dict(zip(varying_params, values))})
                for values in ps_report.top_k_combinations
            }
            print(f"Pre-selection complete: {len(selected_labels)}/{len(combo_store)} combos selected")
        except Exception as exc:
            print(f"Pre-selection failed ({exc}); falling back to top-{max_eda_combos} by raw metric")
            sorted_rows = sorted(metric_rows, key=lambda row: row.get("t_stat", float("-inf")), reverse=True)
            selected_labels = {
                param_combo_label(
                    {
                        **fixed_params,
                        **{varying_params[k]: row[f"param{k + 1}_value"] for k in range(len(varying_params))},
                    }
                )
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

        signal, target, feature_col = combo_store[label]
        timestamps = pd.DatetimeIndex(signal.index)
        rolling_window = _default_rolling_window(signal)
        metadata = EDAMetadata(
            feature_name=feature_col,
            param_combo=combo,
            timeframe=timeframe,
            ticker=cast(Ticker, config.tickers[0]),
            timestamp=datetime.now(),
        )
        eda_config = EDAConfig(rolling_window=rolling_window, bootstrap_iterations=500)
        report = run_eda_for_signed_signal_feature(signal, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        _write_cumsum_summary(
            returns=signal.mul(target),
            output_path=saved_path / "in_sample_cumsum.csv",
            title=f"In-sample cumulative sum ({label})",
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
        print(
            f"  [{label}] n={len(signal):,}  {level_parts}  {viable}  "
            f"warnings={len(report.diagnostics.warnings)}"
        )

    return results
