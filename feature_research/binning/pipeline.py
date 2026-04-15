"""Orchestrate research-only continuous -> quantile bin exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_research.binning.config import BinningResearchConfig
from feature_research.binning.transforms import (
    add_cumulative_return_columns,
    add_research_position_column,
    assign_quantile_bins,
    build_param_combo_long_table,
    merge_feature_target_panel,
    quantile_bin_sets_for_strategy,
    rolling_quantile_edges_long,
    summarize_bins_by_metrics,
)
from feature_research.in_sample.data_loader import expand_bias_specs, param_combo_label, populate_cache_if_needed
from utils.cache import ArtifactScope


def _resolve_reports_root(
    config: BinningResearchConfig,
    output_dir: Path | None,
) -> Path:
    base = Path(output_dir) if output_dir is not None else Path(config.reports_dir)
    return base / config.reports_subdir_name


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
    )
    feature_cols = [column for column in features_df.columns if column != "ticker"]
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

    populate_cache_if_needed(
        config,
        artifact_scope=ArtifactScope.RESEARCH,
    )

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
    long_bins, short_bins = quantile_bin_sets_for_strategy(config.strategy, config.n_bins)
    binned = add_research_position_column(
        binned,
        long_bins=long_bins,
        short_bins=short_bins,
    )
    binned = add_cumulative_return_columns(binned)
    bin_metrics = summarize_bins_by_metrics(binned, timeframe=config.timeframe)
    rolling_edges = rolling_quantile_edges_long(
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

    manifest = {
        "reports_root": str(reports_root),
        "n_bins": config.n_bins,
        "strategy": config.strategy.value,
        "bar_level_rows": int(len(binned)),
        "n_param_combos": int(len(combo_pairs)),
    }
    with open(reports_root / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return {"bar_level_rows": int(len(binned)), "reports_root": str(reports_root)}
