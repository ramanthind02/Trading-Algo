"""Visualization functions for binning diagnostics."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validators.binning.diagnostics import RegionMetadata


def plot_bin_heatmap(
    model: BinningModelBase,
    metric: str = "sharpe",
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "RdYlGn",
) -> Figure:
    """Generate horizontal bar chart heatmap color-coded by metric.

    Args:
        model: Fitted binning model
        metric: Metric to visualize ("sharpe", "t_stat", "sample_count", "mean_return")
        figsize: Figure size in inches
        cmap: Matplotlib colormap name

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def plot_region_boundaries(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Histogram of feature_data with shaded tradeable regions.

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        regions: List of detected regions
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def plot_position_multiplier_curve(
    model: BinningModelBase,
    strategy: str = "long",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Step function showing position multipliers across feature range.

    Args:
        model: Fitted binning model
        strategy: Trading strategy ("long" or "short")
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def create_diagnostic_panel(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    strategy: str = "long",
    figsize: tuple[int, int] = (18, 12),
) -> Figure:
    """Combined 3-panel figure (heatmap, boundaries, multiplier curve).

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        regions: List of detected regions
        strategy: Trading strategy ("long" or "short")
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure with 3 subplots
    """
    raise NotImplementedError("Agent plot-generator will implement")
