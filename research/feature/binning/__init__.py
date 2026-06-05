"""Research-only continuous-feature binning helpers and orchestration."""

from research.feature.binning.config import BinningResearchConfig, load_binning_research_config
from research.feature.binning.pipeline import run_continuous_binning_phase
from research.feature.binning.transforms import (
    add_cumulative_return_columns,
    add_research_position_column,
    assign_quantile_bins,
    build_param_combo_long_table,
    dedupe_combo_param_pairs,
    discrete_signal_series_from_continuous_panel,
    internal_quantile_levels,
    map_bin_selection_to_position,
    merge_feature_target_panel,
    quantile_bin_sets_for_strategy,
    rolling_quantile_edges_long,
    summarize_bins_by_metrics,
)

__all__ = [
    "BinningResearchConfig",
    "add_cumulative_return_columns",
    "add_research_position_column",
    "assign_quantile_bins",
    "build_param_combo_long_table",
    "dedupe_combo_param_pairs",
    "discrete_signal_series_from_continuous_panel",
    "internal_quantile_levels",
    "load_binning_research_config",
    "map_bin_selection_to_position",
    "merge_feature_target_panel",
    "quantile_bin_sets_for_strategy",
    "rolling_quantile_edges_long",
    "run_continuous_binning_phase",
    "summarize_bins_by_metrics",
]
