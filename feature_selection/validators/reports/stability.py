"""Stability analysis report data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class FoldResult:
    """Results from a single walkforward fold."""

    fold_index: int
    train_period: tuple[datetime, datetime]

    # Raw results (all param combos)
    objective_values: pd.DataFrame  # param combo → objective (Sharpe, t-stat, etc.)

    # Smoothed results (after neighbor averaging)
    smoothed_objectives: pd.Series
    stability_ratios: pd.Series

    # Top-K params for this fold
    top_k_params: list[dict[str, Any]]  # Top K param combos by smoothed objective


@dataclass(frozen=True)
class StabilityReport:
    """Walkforward stability analysis results."""

    # Walkforward folds
    n_folds: int
    fold_dates: list[tuple[datetime, datetime]]  # (train_start, train_end) per fold

    # Parameter grid
    params_grid: dict[str, list]
    n_param_combos: int

    # Per-fold results
    fold_results: list[FoldResult]  # One per fold

    # Aggregated stability metrics
    top_params_consistency: pd.DataFrame  # param combo → % folds in top-K
    stable_neighborhoods: list[dict[str, Any]]  # regions that appear consistently

    # Grid-aware neighbor smoothing
    smoothed_objectives: pd.DataFrame  # param combo → smoothed objective per fold
    stability_ratios: pd.DataFrame  # param combo → stability ratio per fold

    # Temporal consistency
    rank_correlation_across_folds: float  # Spearman correlation of param rankings
    best_region_stability: str  # "Stable" / "Moderate" / "Unstable"

    # Plots
    stability_heatmap: Figure | Path | None
    top_params_bar_chart: Figure | Path | None
    parameter_trajectory_plot: Figure | Path | None

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)
