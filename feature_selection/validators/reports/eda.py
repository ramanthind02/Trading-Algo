"""EDA report data structures."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .base import DescriptiveStats, ADFTestResult, KPSSTestResult, MonotonicityTestResult


@dataclass(frozen=True)
class ContinuousEDAReport:
    """Continuous feature EDA extensions."""

    # Decile analysis
    decile_stats: pd.DataFrame  # decile → mean, std, Sharpe, t-stat
    decile_plot: Figure | Path | None
    monotonicity_test: MonotonicityTestResult

    # Outlier analysis
    outlier_fraction: float
    outlier_impact: float  # change in Sharpe when removing outliers
    outlier_plot: Figure | Path | None

    # Non-linearity
    linear_r2: float
    polynomial_r2: dict[int, float]  # degree → R²
    non_linearity_plot: Figure | Path | None

    # Binning diagnostics (added later in Task 6)
    binning_diagnostics: Any | None = None


@dataclass(frozen=True)
class RuleEDAReport:
    """Rule-based feature EDA extensions."""

    # Level distribution
    level_counts: dict[int, int]  # {-1: count, 0: count, 1: count}
    level_fractions: dict[int, float]
    imbalance_flag: bool  # True if any level < 10%

    # Per-level statistics
    level_stats: pd.DataFrame  # level → mean, std, Sharpe, t-stat, adjusted_Sharpe
    level_confidence_intervals: dict[int, tuple[float, float]]
    level_plot: Figure | Path | None

    # Regime transitions
    transition_matrix: pd.DataFrame  # P(level_t+1 | level_t)
    average_duration: dict[int, float]  # avg time spent in each level
    transition_plot: Figure | Path | None

    # Grid report (added later in Task 7)
    grid_report: Any | None = None


@dataclass(frozen=True)
class EDAReport:
    """Exploratory data analysis results."""

    # Distribution statistics
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats

    # Correlation analysis
    correlations: dict[str, float]  # {'pearson': 0.15, 'spearman': 0.18, ...}
    lagged_correlations: pd.Series  # correlation at different lags

    # Stationarity tests
    adf_test: ADFTestResult
    kpss_test: KPSSTestResult

    # Temporal stability
    rolling_correlation: pd.Series  # time series of rolling correlation
    regime_stats: dict[str, DescriptiveStats]  # per-regime statistics

    # Plots (stored as Figure objects or file paths)
    distribution_plot: Figure | Path | None
    correlation_plot: Figure | Path | None
    time_series_plot: Figure | Path | None
    stationarity_plot: Figure | Path | None

    # Feature-type-specific reports
    continuous_report: ContinuousEDAReport | None = None
    rule_report: RuleEDAReport | None = None

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
