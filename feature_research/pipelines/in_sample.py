from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import cast

import pandas as pd

from feature_research.config import FeatureType, ResearchConfig
from feature_research.core_helpers import normalize_timeframe_from_bias_spec
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    expanded_combo_param_value,
    expanded_spec_combo_label,
    first_bias_spec,
    load_features_for_combo,
    populate_cache_if_needed,
)
from feature_research.in_sample.metric_helpers import compute_param_sensitivity_metric
from feature_research.research_table_exports import (
    write_in_sample_equity_curve_powerbi_csv,
    write_param_sensitivity_powerbi_tables,
)
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import (
    run_eda_for_continuous_feature,
    run_eda_for_signed_signal_feature,
    save_eda_report,
)
from feature_selection.eda.parameter_analysis import generate_parameter_sensitivity_report
from utils.core.enums import Ticker, TimeFrame


def _default_rolling_window(feature: pd.Series) -> int:
    return max(20, min(252, len(feature) // 4))


_PARAM_COMBO_HASH_FOLDER_LEN = 8


def _windows_eda_path_budget() -> int | None:
    """Max full path length for EDA combo output; ``None`` means no shortening."""
    if os.name != "nt":
        return None
    return int(os.environ.get("TRADING_ALGO_EDA_MAX_PATH", "230"))


def _combo_eda_parent_dir(output_dir: Path, label: str, feature_name: str) -> Path:
    """``output_dir / <segment>`` for one combo; shortens *segment* when paths would exceed OS limits."""
    budget = _windows_eda_path_budget()
    if budget is None:
        return output_dir / label

    def path_length_for_segment(segment: str) -> int:
        leaf = output_dir / segment / feature_name / ("x" * _PARAM_COMBO_HASH_FOLDER_LEN)
        return len(str(leaf.resolve()))

    if path_length_for_segment(label) <= budget:
        return output_dir / label

    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    candidates: list[str] = [digest]
    max_keep = min(len(label), 120)
    candidates.extend(f"{digest}__{label[:keep]}" for keep in range(max_keep, 0, -1))

    for seg in candidates:
        if path_length_for_segment(seg) <= budget:
            return output_dir / seg

    for n in range(11, 7, -1):
        seg = hashlib.sha256(label.encode("utf-8")).hexdigest()[:n]
        if path_length_for_segment(seg) <= budget:
            return output_dir / seg

    return output_dir / digest[:8]


def _bias_module_token_from_store_label(label: str) -> str:
    """Leading token before the first ``__`` in :func:`expanded_spec_combo_label` output."""
    i = label.find("__")
    return label[:i] if i != -1 else label


def _param_sensitivity_by_ticker_rows_for_combo(
    label: str,
    feature_col: str,
    signal: pd.Series,
    target: pd.Series,
    ticker_series: pd.Series,
    timeframe: TimeFrame,
) -> list[dict[str, object]]:
    """One sensitivity dict per instrument (same formulas as pooled table)."""
    tkr_s = ticker_series.astype(str)

    def _row_for_ticker(tkr: str) -> dict[str, object]:
        mask = tkr_s.eq(tkr)
        sig_t = signal[mask]
        tgt_t = target[mask]
        strat_t = (sig_t * tgt_t).to_numpy(dtype=float)
        n_obs_t = int(len(strat_t))
        n_nz_t = int((sig_t.fillna(0).to_numpy() != 0).sum())
        sh_t = float("nan")
        ts_t = float("nan")
        so_t = float("nan")
        if n_obs_t >= 5:
            try:
                sh_t = compute_param_sensitivity_metric(strat_t, "sharpe", timeframe)
                ts_t = compute_param_sensitivity_metric(strat_t, "t_stat", timeframe)
                so_t = compute_param_sensitivity_metric(strat_t, "sortino", timeframe)
            except Exception:
                pass
        return {
            "param_sensitivity_by_ticker_key": f"{label}__{tkr}",
            "param_combo_label": label,
            "feature_name": feature_col,
            "ticker": tkr,
            "n_observations": n_obs_t,
            "n_nonzero_signal": n_nz_t,
            "sharpe": sh_t,
            "t_stat": ts_t,
            "sortino": so_t,
        }

    return [_row_for_ticker(tkr) for tkr in sorted(tkr_s.unique())]


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
    config: ResearchConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """In-sample EDA analyzes ``training_window_bounds``; cache is warmed for full ``start``/``end``.

    Central-cache artifacts are built for the configured global OHLC span first so
    validation/OOS phases can read the same parquet without rebuilding. Feature/target
    loads for EDA use the training slice only (after the replace below).
    """
    tr_start, tr_end = config.training_window_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    config = replace(config, start=tr_start, end=tr_end)

    expanded = expand_bias_specs(config.bias_spec)
    meta_spec = first_bias_spec(config.bias_spec)
    timeframe = normalize_timeframe_from_bias_spec(meta_spec)
    results: dict[str, Path] = {}

    print(f"\n{'='*64}")
    feature_type_label = config.feature_type.name
    if isinstance(config.bias_spec, list):
        branch_modules = ", ".join(
            str(s.get("module_name")) for s in config.bias_spec
        )
        eda_module_label = f"MULTI[{branch_modules}]"
    else:
        eda_module_label = str(meta_spec["module_name"]).upper()
    print(f"EDA Pipeline: {eda_module_label} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}  (training_window_bounds)")
    print(
        f"Target  : {config.target_col}  |  Strategy: {config.strategy}"
        + (
            "  |  Continuous → quantile-binned signal (binning_params)"
            if config.feature_type == FeatureType.CONTINUOUS
            else ""
        )
    )
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    combo_store: dict[str, tuple[pd.Series, pd.Series, str, pd.Series, dict[str, object]]] = {}
    varying_params = [
        key
        for key, values in meta_spec.get("params", {}).items()
        if isinstance(values, list) and len(values) > 1
    ]
    metric_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_by_ticker_rows: list[dict[str, object]] = []

    print(f"Loading data for {len(expanded)} combos...")
    for single_spec in expanded:
        combo = single_spec["params"]
        label = expanded_spec_combo_label(single_spec)
        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, feature_col, ticker_s = data
        paired = pd.concat(
            [
                feature.rename("signal"),
                target.rename("target"),
                ticker_s.rename("ticker"),
            ],
            axis=1,
        ).dropna(how="any")
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        signal = paired["signal"]
        target = paired["target"]
        ticker_series = paired["ticker"]
        combo_store[label] = (signal, target, feature_col, ticker_series, dict(combo))

        if config.feature_type == FeatureType.SIGNED_SIGNAL:
            strategy_returns = (signal * target).to_numpy(dtype=float)
            n_observations = int(len(strategy_returns))
            n_nonzero_signal = int((signal.fillna(0).to_numpy() != 0).sum())
            sharpe_v = float("nan")
            t_stat_v = float("nan")
            sortino_v = float("nan")
            if n_observations >= 5:
                try:
                    sharpe_v = compute_param_sensitivity_metric(strategy_returns, "sharpe", timeframe)
                    t_stat_v = compute_param_sensitivity_metric(strategy_returns, "t_stat", timeframe)
                    sortino_v = compute_param_sensitivity_metric(strategy_returns, "sortino", timeframe)
                except Exception:
                    pass
            sensitivity_rows.append(
                {
                    "param_combo_label": label,
                    "feature_name": feature_col,
                    "n_observations": n_observations,
                    "n_nonzero_signal": n_nonzero_signal,
                    "sharpe": sharpe_v,
                    "t_stat": t_stat_v,
                    "sortino": sortino_v,
                }
            )

            sensitivity_by_ticker_rows.extend(
                _param_sensitivity_by_ticker_rows_for_combo(
                    label, feature_col, signal, target, ticker_series, timeframe
                )
            )

            if varying_params and n_observations >= 5 and isfinite(t_stat_v):
                resolved = [
                    expanded_combo_param_value(combo, key) for key in varying_params
                ]
                if all(v is not None for v in resolved):
                    row = {
                        f"param{k + 1}_value": resolved[k]
                        for k in range(len(varying_params))
                    }
                    row["t_stat"] = t_stat_v
                    metric_rows.append(row)

    label_param_pairs = [
        (
            label,
            {
                **dict(params),
                "bias_composite_module": _bias_module_token_from_store_label(label),
            },
        )
        for label, (_, _, _, _, params) in sorted(combo_store.items())
    ]
    if config.feature_type == FeatureType.SIGNED_SIGNAL:
        pbi_paths = write_param_sensitivity_powerbi_tables(
            sensitivity_rows,
            label_param_pairs,
            sensitivity_by_ticker_rows=sensitivity_by_ticker_rows,
        )
        equity_csv = write_in_sample_equity_curve_powerbi_csv(combo_store)
        print(
            "Power BI tables (param sensitivity): "
            f"{pbi_paths.get('param_sensitivity_csv', '')} ({len(sensitivity_rows)} rows)"
        )
        if pbi_paths.get("param_sensitivity_by_ticker_csv"):
            print(
                "Power BI tables (param sensitivity by ticker): "
                f"{pbi_paths['param_sensitivity_by_ticker_csv']} "
                f"({len(sensitivity_by_ticker_rows)} rows)"
            )
        print(f"Power BI equity curve (all combos): {equity_csv}")

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
            key: values
            for key, values in meta_spec.get("params", {}).items()
            if key not in varying_params
        }
        print(
            f"\nParam sensitivity pre-selection: {len(combo_store)} combos → "
            f"selecting top {max_eda_combos} for EDA output..."
        )
        try:
            ps_report = generate_parameter_sensitivity_report(
                results_df=ps_df,
                param_names=varying_params,
                metric_col="t_stat",
                stability_threshold=ps_cfg.stability_threshold,
                top_k=max_eda_combos,
                plot_3d_mode="heatmap_slices",
                smoothing_self_weight=ps_cfg.smoothing_self_weight,
            )
            selected_labels: set[str] = set()
            for values in ps_report.top_k_combinations:
                target_varying = dict(zip(varying_params, values))
                for lbl, (_, _, _, _, pstored) in combo_store.items():
                    if all(
                        expanded_combo_param_value(pstored, vk) == vv
                        for vk, vv in target_varying.items()
                    ):
                        selected_labels.add(lbl)
            print(f"Pre-selection complete: {len(selected_labels)}/{len(combo_store)} combos selected")
        except Exception as exc:
            print(f"Pre-selection failed ({exc}); falling back to top-{max_eda_combos} by raw metric")
            sorted_rows = sorted(metric_rows, key=lambda row: row.get("t_stat", float("-inf")), reverse=True)
            selected_labels = set()
            for row in sorted_rows[:max_eda_combos]:
                target_varying = {
                    varying_params[k]: row[f"param{k + 1}_value"]
                    for k in range(len(varying_params))
                }
                for lbl, (_, _, _, _, pstored) in combo_store.items():
                    if all(
                        expanded_combo_param_value(pstored, vk) == vv
                        for vk, vv in target_varying.items()
                    ):
                        selected_labels.add(lbl)
    else:
        selected_labels = set(combo_store.keys())

    print(
        f"\nRunning EDA for {len(selected_labels)}/{len(combo_store)} combos"
        + (f" (limit={max_eda_combos})" if should_preselect else "")
        + "..."
    )
    for single_spec in expanded:
        combo = single_spec["params"]
        label = expanded_spec_combo_label(single_spec)
        if label not in combo_store or label not in selected_labels:
            continue

        signal, target, feature_col, _ticker_s, _combo_params = combo_store[label]
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
        # CONTINUOUS research uses quantile-binned ±1/0 from load_features_for_combo (same as permutation).
        report = (
            run_eda_for_continuous_feature(signal, target, timestamps, metadata, eda_config)
            if config.feature_type == FeatureType.CONTINUOUS
            else run_eda_for_signed_signal_feature(signal, target, timestamps, metadata, eda_config)
        )

        combo_output_dir = _combo_eda_parent_dir(output_dir, label, feature_col)
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        if combo_output_dir.name != label:
            (combo_output_dir / "eda_param_combo_label.txt").write_text(f"{label}\n", encoding="utf-8")
        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        if config.feature_type == FeatureType.SIGNED_SIGNAL:
            _write_cumsum_summary(
                returns=signal.mul(target),
                output_path=saved_path / "in_sample_cumsum.csv",
                title=f"In-sample cumulative sum ({label})",
            )
        results[label] = saved_path

        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        if config.feature_type == FeatureType.SIGNED_SIGNAL:
            stats_by_level = report.rule_stats.per_level_stats.stats_by_level
            level_keys = sorted(stats_by_level.keys())
            level_parts = "  ".join(
                f"L[{level}]: sharpe={stats_by_level[level].sharpe:+.2f}"
                if level in stats_by_level
                else f"L[{level}]: n/a"
                for level in level_keys
            )
            print(
                f"  [{label}] n={len(signal):,}  {level_parts}  {viable}  "
                f"warnings={len(report.diagnostics.warnings)}"
            )
        else:
            trend = report.continuous_stats.decile_analysis.overall_trend
            print(
                f"  [{label}] n={len(signal):,}  trend={trend}  {viable}  "
                f"warnings={len(report.diagnostics.warnings)}"
            )

    return results
