"""Visualization functions for binning diagnostics."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validation.binning.diagnostics import RegionMetadata
from utils.core.enums import Direction, DirectionInput, coerce_direction

matplotlib.use("Agg")

_VALID_METRICS = frozenset({"sharpe", "t_stat", "sample_count", "mean_return"})

_METRIC_KEYS: dict[str, str] = {
    "sharpe": "sharpe",
    "t_stat": "t_stat",
    "sample_count": "count",
    "mean_return": "mean_return",
}

_METRIC_LABELS: dict[str, str] = {
    "sharpe": "Sharpe Ratio",
    "t_stat": "t-Statistic",
    "sample_count": "Sample Count",
    "mean_return": "Mean Return",
}


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
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. Choose from: {sorted(_VALID_METRICS)}"
        )

    stats = model.bin_stats_
    ordered_bins = sorted(stats.keys())
    stat_key = _METRIC_KEYS[metric]
    values = [float(stats[b][stat_key]) for b in ordered_bins]

    fig, ax = plt.subplots(figsize=figsize)
    colormap = plt.get_cmap(cmap)

    val_arr = np.array(values, dtype=float)
    val_min, val_max = float(val_arr.min()), float(val_arr.max())
    val_range = val_max - val_min if val_max != val_min else 1.0
    normed = (val_arr - val_min) / val_range
    colors = [colormap(float(n)) for n in normed]

    y_pos = np.arange(len(ordered_bins))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Bin {b}" for b in ordered_bins])
    ax.set_xlabel(_METRIC_LABELS[metric])
    ax.set_title(
        f"Bin Heatmap — {_METRIC_LABELS[metric]}"
        + (f" ({model.feature_column})" if model.feature_column else "")
    )
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(vmin=val_min, vmax=val_max),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=_METRIC_LABELS[metric])

    fig.tight_layout()
    return fig


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
    fig, ax = plt.subplots(figsize=figsize)

    clean_data = feature_data.dropna()
    ax.hist(clean_data, bins=50, color="steelblue", edgecolor="white", alpha=0.7)

    # Draw bin edge lines
    if model.bin_edges_:
        for edge in model.bin_edges_:
            ax.axvline(edge, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Shade tradeable regions
    for region in regions:
        low, high = region.feature_range
        ax.axvspan(
            low,
            high,
            alpha=0.3,
            color="green",
            label=f"Region bins {region.start_bin}-{region.end_bin}",
        )

    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Feature Distribution with Region Boundaries"
        + (f" ({model.feature_column})" if model.feature_column else "")
    )
    if regions:
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_position_multiplier_curve(
    model: BinningModelBase,
    strategy: DirectionInput = Direction.LONG,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Step function showing position multipliers across feature range.

    Args:
        model: Fitted binning model
        strategy: Trading strategy direction
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    direction = coerce_direction(strategy, field_name="strategy")
    strategy_key = direction.value
    stats = model.bin_stats_
    ordered_bins = sorted(stats.keys())
    multipliers = model.position_multipliers_by_strategy_.get(strategy_key, {})
    active_bins = set(model.active_bins_by_strategy_.get(strategy_key, []))

    # Build step edges and values
    edges: list[float] = []
    mults: list[float] = []
    for b in ordered_bins:
        low = float(stats[b]["feature_min"])
        high = float(stats[b]["feature_max"])
        m = float(multipliers.get(b, 0.0))
        edges.extend([low, high])
        mults.extend([m, m])

    ax.plot(edges, mults, drawstyle="steps-mid", color="navy", linewidth=2)

    # Color active bins
    for b in ordered_bins:
        low = float(stats[b]["feature_min"])
        high = float(stats[b]["feature_max"])
        m = float(multipliers.get(b, 0.0))
        color = "green" if b in active_bins and m > 0 else (
            "red" if b in active_bins and m < 0 else "lightgray"
        )
        alpha = 0.4 if b in active_bins else 0.15
        ax.axvspan(low, high, alpha=alpha, color=color)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Position Multiplier")
    ax.set_title(
        f"Position Multiplier Curve — {strategy_key}"
        + (f" ({model.feature_column})" if model.feature_column else "")
    )

    fig.tight_layout()
    return fig


def create_diagnostic_panel(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    strategy: DirectionInput = Direction.LONG,
    figsize: tuple[int, int] = (18, 12),
) -> Figure:
    """Combined 3-panel figure (heatmap, boundaries, multiplier curve).

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        regions: List of detected regions
        strategy: Trading strategy direction
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure with 3 subplots
    """
    direction = coerce_direction(strategy, field_name="strategy")
    strategy_key = direction.value
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # --- Panel 1: Bin heatmap ---
    ax_heatmap = axes[0]
    stats = model.bin_stats_
    ordered_bins = sorted(stats.keys())
    values = [float(stats[b]["sharpe"]) for b in ordered_bins]

    colormap = plt.get_cmap("RdYlGn")
    val_arr = np.array(values, dtype=float)
    val_min, val_max = float(val_arr.min()), float(val_arr.max())
    val_range = val_max - val_min if val_max != val_min else 1.0
    normed = (val_arr - val_min) / val_range
    colors = [colormap(float(n)) for n in normed]

    y_pos = np.arange(len(ordered_bins))
    ax_heatmap.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax_heatmap.set_yticks(y_pos)
    ax_heatmap.set_yticklabels([f"Bin {b}" for b in ordered_bins])
    ax_heatmap.set_xlabel("Sharpe Ratio")
    ax_heatmap.set_title("Bin Heatmap — Sharpe Ratio")
    ax_heatmap.invert_yaxis()

    # --- Panel 2: Region boundaries ---
    ax_regions = axes[1]
    clean_data = feature_data.dropna()
    ax_regions.hist(clean_data, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    if model.bin_edges_:
        for edge in model.bin_edges_:
            ax_regions.axvline(edge, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for region in regions:
        low, high = region.feature_range
        ax_regions.axvspan(low, high, alpha=0.3, color="green")
    ax_regions.set_xlabel("Feature Value")
    ax_regions.set_ylabel("Frequency")
    ax_regions.set_title("Feature Distribution with Region Boundaries")

    # --- Panel 3: Position multiplier curve ---
    ax_mult = axes[2]
    multipliers = model.position_multipliers_by_strategy_.get(strategy_key, {})
    active_bins = set(model.active_bins_by_strategy_.get(strategy_key, []))

    edges: list[float] = []
    mults: list[float] = []
    for b in ordered_bins:
        low = float(stats[b]["feature_min"])
        high = float(stats[b]["feature_max"])
        m = float(multipliers.get(b, 0.0))
        edges.extend([low, high])
        mults.extend([m, m])

    ax_mult.plot(edges, mults, drawstyle="steps-mid", color="navy", linewidth=2)
    for b in ordered_bins:
        low = float(stats[b]["feature_min"])
        high = float(stats[b]["feature_max"])
        m = float(multipliers.get(b, 0.0))
        color = "green" if b in active_bins and m > 0 else (
            "red" if b in active_bins and m < 0 else "lightgray"
        )
        alpha = 0.4 if b in active_bins else 0.15
        ax_mult.axvspan(low, high, alpha=alpha, color=color)
    ax_mult.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax_mult.set_xlabel("Feature Value")
    ax_mult.set_ylabel("Position Multiplier")
    ax_mult.set_title(f"Position Multiplier Curve — {strategy_key}")

    # Shared super-title
    n_bins = getattr(model, "n_bins", "?")
    feature_col = model.feature_column or "unknown"
    fig.suptitle(
        f"Binning Diagnostics: {feature_col} | n_bins={n_bins} | strategy={strategy_key}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig
