"""
Centralized Plotting and Visualization Library

This package provides a centralized location for all plotting and visualization
functions used across the trading algorithm framework. This is the single source
of truth for all visualization utilities.

Author: Trading Research Team
Date: 2025-01-XX
"""

# Decile and binning plots
from metrics.plotting.decile_plots import (
    plot_decile_analysis,
    plot_2bin_analysis,
    plot_uniform_binning,
)

# Distribution and time series plots
from metrics.plotting.distribution import (
    plot_feature_distribution,
    plot_feature_timeseries,
)

# Correlation plots
from metrics.plotting.correlation import (
    plot_feature_correlations,
    plot_feature_deciles,
)

# Cumulative plots
from metrics.plotting.cumulative_plot import (
    plot_cumulative_product,
)

# Rolling decile analysis
from metrics.plotting.rolling_decile import (
    compute_decile_stats,
    plot_rolling_decile_whiskers,
    plot_rolling_decile_heatmap,
)

# Feature explorer plots (pure functions extracted from FeatureExplorer)
from metrics.plotting.feature_explorer_plots import (
    plot_all_feature_deciles,
    plot_feature_2bin,
    plot_all_feature_uniform_bins,
    plot_feature_target_correlations,
    plot_feature_correlation_matrix,
    plot_all_feature_distributions,
    plot_all_feature_timeseries,
    plot_feature_signal_cumsum,
)

# Parameter sensitivity plots
from metrics.plotting.parameter_plots import (
    plot_parameter_sensitivity,
    plot_2d_parameter_surface,
    plot_3d_parameter_interactive,
    plot_4d_parameter_interactive,
    plot_parameter_sensitivity_with_stability,
    plot_2d_stability_heatmap,
    plot_3d_slices,
)

# Graphing utilities (portfolio analytics, quantstats)
from metrics.plotting.graphing import (
    generate_tearsheet,
    compute_baseline_results,
    PortfolioAnalytics,
)

__all__ = [
    # Decile plots
    'plot_decile_analysis',
    'plot_2bin_analysis',
    'plot_uniform_binning',
    # Distribution plots
    'plot_feature_distribution',
    'plot_feature_timeseries',
    # Correlation plots
    'plot_feature_correlations',
    'plot_feature_deciles',
    # Cumulative plots
    'plot_cumulative_product',
    # Rolling decile
    'compute_decile_stats',
    'plot_rolling_decile_whiskers',
    'plot_rolling_decile_heatmap',
    # Feature explorer plots
    'plot_all_feature_deciles',
    'plot_feature_2bin',
    'plot_all_feature_uniform_bins',
    'plot_feature_target_correlations',
    'plot_feature_correlation_matrix',
    'plot_all_feature_distributions',
    'plot_all_feature_timeseries',
    'plot_feature_signal_cumsum',
    # Parameter plots
    'plot_parameter_sensitivity',
    'plot_2d_parameter_surface',
    'plot_3d_parameter_interactive',
    'plot_4d_parameter_interactive',
    'plot_parameter_sensitivity_with_stability',
    'plot_2d_stability_heatmap',
    'plot_3d_slices',
    # Graphing
    'generate_tearsheet',
    'compute_baseline_results',
    'PortfolioAnalytics',
]
