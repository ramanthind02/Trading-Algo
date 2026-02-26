"""Frozen dataclasses for the Feature Validator EDA pipeline (T001–T004)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from utils.core.enums import Ticker, TimeFrame


# ─── T001: Common EDA ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DescriptiveStats:
    """Descriptive statistics for a single series."""
    min_val: float
    max_val: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    nan_count: int
    nan_pct: float
    sample_size: int


@dataclass(frozen=True)
class CorrelationAnalysis:
    """Feature-target correlation at various lags."""
    pearson: float
    spearman: float
    lagged_correlations: dict[int, float]    # lag → correlation (lags 1..max_lag)


@dataclass(frozen=True)
class CommonEDAPlots:
    """Matplotlib Figure objects for common EDA."""
    time_series_fig: Figure      # feature over time (single subplot)


@dataclass(frozen=True)
class CommonEDAStats:
    """Aggregated common EDA statistics."""
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats
    correlation_analysis: CorrelationAnalysis


# ─── T002: Continuous Feature EDA ────────────────────────────────────────────

@dataclass(frozen=True)
class DecileBinStats:
    """Per-bin statistics for decile analysis."""
    bin_edges: np.ndarray         # length n_bins + 1
    mean_return: np.ndarray       # length n_bins
    volatility: np.ndarray        # length n_bins
    sharpe: np.ndarray            # length n_bins  (NaN where vol == 0)
    t_stat: np.ndarray            # length n_bins  (NaN where n < 2)
    sample_count: np.ndarray      # length n_bins (int)


@dataclass(frozen=True)
class DecileAnalysis:
    """Decile binning results with overall trend label."""
    bin_stats: DecileBinStats
    overall_trend: str            # 'monotonic_increasing' | 'monotonic_decreasing' | 'U-shaped' | 'flat'


@dataclass(frozen=True)
class QuintileSpread:
    """Mean return per quintile and Q5-Q1 spread."""
    quintile_means: np.ndarray  # shape (5,), Q1 to Q5
    spread: float               # quintile_means[4] - quintile_means[0]


@dataclass(frozen=True)
class DistributionDiagnostics:
    """Normality diagnostics for feature distribution."""
    skewness: float
    kurtosis: float
    normality_test_stat: float    # Shapiro-Wilk W statistic (or Anderson for n > 5000)
    normality_p_value: float
    is_normal: bool               # p_value > 0.05


@dataclass(frozen=True)
class ContinuousEDAPlots:
    """Matplotlib Figures for continuous feature EDA."""
    decile_plot_fig: Figure       # 3 subplots: mean return, Sharpe, t-stat
    histogram_fig: Figure         # histogram + quantile overlay lines
    quintile_spread_fig: Figure   # mean return per quintile with spread


@dataclass(frozen=True)
class ContinuousEDAStats:
    """Aggregated continuous feature EDA statistics."""
    decile_analysis: DecileAnalysis
    distribution_diagnostics: DistributionDiagnostics
    quintile_spread: QuintileSpread


# ─── T003: Rule-Based Feature EDA ────────────────────────────────────────────

@dataclass(frozen=True)
class LevelStats:
    """Statistics for one discrete level (-1, 0, or 1)."""
    level: int
    mean_return: float
    volatility: float
    sharpe: float                 # NaN if vol == 0
    adjusted_sharpe: float        # NaN if not computable
    sample_count: int
    is_reliable: bool             # False if sample_count < 10


@dataclass(frozen=True)
class PerLevelStats:
    """Per-level statistics for all observed levels."""
    stats_by_level: dict[int, LevelStats]


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for one level."""
    level: int
    mean_return: float
    ci_lower: float
    ci_upper: float
    bootstrap_distribution: np.ndarray    # shape (n_iterations,)


@dataclass(frozen=True)
class BootstrapCIResults:
    """Bootstrap CIs for all levels."""
    ci_by_level: dict[int, BootstrapCI]


@dataclass(frozen=True)
class RuleBasedEDAPlots:
    """Matplotlib Figures for rule-based feature EDA."""
    level_plot_fig: Figure            # bar chart per level with bootstrap CI error bars


@dataclass(frozen=True)
class RuleBasedEDAStats:
    """Aggregated rule-based EDA statistics."""
    per_level_stats: PerLevelStats
    bootstrap_ci_results: BootstrapCIResults


# ─── T004: EDA Report Generation ─────────────────────────────────────────────

@dataclass(frozen=True)
class EDAMetadata:
    """Identity and provenance for an EDA report."""
    feature_name: str
    param_combo: dict[str, Any]
    timeframe: TimeFrame
    ticker: Ticker
    timestamp: datetime


@dataclass(frozen=True)
class EDAConfig:
    """User-specified EDA computation settings."""
    n_bins: int = 15
    rolling_window: int = 252
    objective_fn: Callable[[pd.Series, pd.Series], float] = field(
        default=lambda signals, returns: (returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    )
    max_lag: int = 5
    bootstrap_iterations: int = 1000
    random_seed: int = 42


@dataclass(frozen=True)
class DiagnosticFlags:
    """Warnings and red flags computed from EDA stats."""
    warnings: list[str]
    red_flags: list[str]
    is_viable: bool    # True iff len(red_flags) == 0


@dataclass(frozen=True)
class ContinuousEDAReport:
    """Full EDA report for a continuous feature."""
    metadata: EDAMetadata
    common_stats: CommonEDAStats
    continuous_stats: ContinuousEDAStats
    common_plots: CommonEDAPlots
    continuous_plots: ContinuousEDAPlots
    diagnostics: DiagnosticFlags


@dataclass(frozen=True)
class RuleBasedEDAReport:
    """Full EDA report for a rule-based feature."""
    metadata: EDAMetadata
    common_stats: CommonEDAStats
    rule_stats: RuleBasedEDAStats
    common_plots: CommonEDAPlots
    rule_plots: RuleBasedEDAPlots
    diagnostics: DiagnosticFlags
