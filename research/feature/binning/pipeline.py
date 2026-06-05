"""Orchestrate research-only continuous -> quantile bin exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from features.extraction.feature_extractor import extract_features_for_bias_node
from research.feature.binning.config import BinningResearchConfig
from research.feature.binning.transforms import (
    add_cumulative_return_columns,
    add_research_position_column,
    assign_quantile_bins,
    build_param_combo_long_table,
    merge_feature_target_panel,
    quantile_bin_sets_for_strategy,
    per_series_rolling_edges,
    summarize_bins_by_metrics,
)
from research.feature.in_sample.data_loader import expand_bias_specs, param_combo_label, populate_cache_if_needed
from research.feature.shared.visualization_paths import canonical_in_sample_visualization_dir


def _resolve_reports_root(
    config: BinningResearchConfig,
    output_dir: Path | None,
) -> Path:
    base = Path(output_dir) if output_dir is not None else Path(config.reports_dir)
    return base / config.reports_subdir_name


def _attach_investigation_strategy_returns(
    panel: pd.DataFrame,
    config: BinningResearchConfig,
) -> pd.DataFrame:
    """Merge investigation signal and set ``strategy_return`` = signal × target."""
    if config.strategy_bias_spec is None:
        return panel

    expanded = expand_bias_specs(config.strategy_bias_spec)
    if len(expanded) != 1:
        raise ValueError(
            "strategy_bias_spec must expand to exactly one combo for Pass 1 decile analysis, "
            f"got {len(expanded)}."
        )
    single_spec = expanded[0]
    features_df, _targets_df = extract_features_for_bias_node(
        bias_spec=single_spec,
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        target_col=config.target_col,
        use_cache=True,
        populate_on_miss=True,
    )
    signal_cols = [column for column in features_df.columns if column != "ticker"]
    if not signal_cols:
        return panel

    signal_col = signal_cols[0]
    signal_frame = features_df[[signal_col, "ticker"]].rename(columns={signal_col: "strategy_signal"})
    signal_frame = signal_frame.copy()
    signal_frame["strategy_signal"] = pd.to_numeric(signal_frame["strategy_signal"], errors="coerce")

    out = panel.copy()
    left = out.reset_index()
    datetime_col = str(left.columns[0])
    right = signal_frame.reset_index().rename(columns={signal_frame.index.name or "index": datetime_col})
    merged = left.merge(right, on=[datetime_col, "ticker"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["strategy_return"] = (
        merged["strategy_signal"].astype(float) * merged["target"].astype(float)
    )
    restored = merged.set_index(datetime_col)
    restored.index.name = out.index.name
    return restored.sort_index()


def _load_continuous_panel(
    config: BinningResearchConfig,
    single_spec: dict[str, Any],
) -> pd.DataFrame:
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=single_spec,
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        target_col=config.target_col,
        use_cache=True,
        populate_on_miss=True,
    )
    feature_cols = [column for column in features_df.columns if column != "ticker"]
    if config.feature_col_substr:
        matched = [c for c in feature_cols if config.feature_col_substr in c]
        if matched:
            feature_cols = matched
    if not feature_cols:
        return pd.DataFrame()
    merged = merge_feature_target_panel(
        features_df,
        targets_df,
        feature_col=feature_cols[0],
        target_col=config.target_col,
    )
    if merged.empty:
        return pd.DataFrame()
    combo = dict(single_spec.get("params", {}))
    label = param_combo_label(combo)
    merged = merged.copy()
    merged["param_combo_label"] = label
    merged["feature_name"] = feature_cols[0]
    return merged


def run_continuous_binning_phase(
    config: BinningResearchConfig,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """Materialize quantile-binning research tables for continuous bias nodes."""
    if not config.enabled:
        raise ValueError("Binning research is disabled in BinningResearchConfig.")

    reports_root = _resolve_reports_root(config, output_dir)
    reports_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"bar_level_rows": 0, "reports_root": str(reports_root)}

    # Use default LIVE scope so cache population matches extract_features_for_bias_node reads.
    populate_cache_if_needed(config)
    if config.strategy_bias_spec is not None:
        populate_cache_if_needed(config, bias_spec=config.strategy_bias_spec)

    expanded_specs = expand_bias_specs(config.bias_spec)
    panels = [_load_continuous_panel(config, spec) for spec in expanded_specs]
    non_empty = [panel for panel in panels if not panel.empty]
    full_panel = pd.concat(non_empty, axis=0) if non_empty else pd.DataFrame()

    if full_panel.empty:
        empty = pd.DataFrame()
        empty.to_parquet(reports_root / "bar_level_all_tickers.parquet", index=True)
        empty.to_parquet(reports_root / "bin_metrics.parquet", index=False)
        empty.to_parquet(reports_root / "rolling_quantile_edges.parquet", index=False)
        return {"bar_level_rows": 0, "reports_root": str(reports_root)}

    binned = assign_quantile_bins(full_panel, n_bins=config.n_bins)
    metrics_return_col = "target"
    active_signal_only = False
    if config.strategy_bias_spec is not None:
        binned = _attach_investigation_strategy_returns(binned, config)
        if binned.empty:
            empty = pd.DataFrame()
            empty.to_parquet(reports_root / "bar_level_all_tickers.parquet", index=True)
            empty.to_parquet(reports_root / "bin_metrics.parquet", index=False)
            empty.to_parquet(reports_root / "rolling_quantile_edges.parquet", index=False)
            return {"bar_level_rows": 0, "reports_root": str(reports_root)}
        metrics_return_col = "strategy_return"
        active_signal_only = True
    else:
        long_bins, short_bins = quantile_bin_sets_for_strategy(config.strategy, config.n_bins)
        binned = add_research_position_column(
            binned,
            long_bins=long_bins,
            short_bins=short_bins,
        )
    binned = add_cumulative_return_columns(binned)
    bin_metrics = summarize_bins_by_metrics(
        binned,
        timeframe=config.timeframe,
        return_col=metrics_return_col,
        active_signal_only=active_signal_only,
    )
    rolling_edges = per_series_rolling_edges(
        binned,
        window=config.rolling_window,
        min_periods=config.rolling_min_periods,
        n_bins=config.n_bins,
    )

    combo_pairs = [
        (str(panel["param_combo_label"].iloc[0]), dict(spec.get("params", {})))
        for panel, spec in zip(non_empty, [spec for spec, panel in zip(expanded_specs, panels) if not panel.empty], strict=False)
    ]
    param_combo_long = build_param_combo_long_table(combo_pairs)

    binned.to_parquet(reports_root / "bar_level_all_tickers.parquet", index=True)
    bin_metrics.to_parquet(reports_root / "bin_metrics.parquet", index=False)
    rolling_edges.to_parquet(reports_root / "rolling_quantile_edges.parquet", index=False)
    param_combo_long.to_parquet(reports_root / "param_combo_long.parquet", index=False)

    investigation_module = (
        str(config.strategy_bias_spec.get("module_name", ""))
        if config.strategy_bias_spec is not None
        else ""
    )
    manifest = {
        "reports_root": str(reports_root),
        "n_bins": config.n_bins,
        "strategy": config.strategy.value,
        "bar_level_rows": int(len(binned)),
        "n_param_combos": int(len(combo_pairs)),
        "metrics_return_col": metrics_return_col,
        "active_signal_only": active_signal_only,
        "investigation_strategy": investigation_module,
    }
    with open(reports_root / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    _generate_decile_chart(reports_root / "bin_metrics.parquet")

    return {"bar_level_rows": int(len(binned)), "reports_root": str(reports_root)}


def generate_atr_pct_decile_chart(
    bin_metrics_parquet: Path | None = None,
) -> Path | None:
    """Write ``atr_pct_decile_chart.png`` under the canonical in-sample matplotlib dir."""
    from research.feature.binning.config import load_binning_research_config
    from research.feature.visualization.matplotlib_reports import plot_binning_decile_parquet

    parquet_path = (
        bin_metrics_parquet
        if bin_metrics_parquet is not None
        else load_binning_research_config().reports_dir / "binning_phase" / "bin_metrics.parquet"
    )
    out_dir = canonical_in_sample_visualization_dir() / "matplotlib"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "atr_pct_decile_chart.png"
    return plot_binning_decile_parquet(parquet_path, output_path)


def _generate_decile_chart(bin_metrics_parquet: Path) -> None:
    """Write ATR% decile chart PNG to the canonical matplotlib visualization dir."""
    generate_atr_pct_decile_chart(bin_metrics_parquet)
