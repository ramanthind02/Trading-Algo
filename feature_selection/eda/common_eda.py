"""Common EDA infrastructure for both continuous and rule-based features (T001)."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from feature_selection.eda.eda_dataclasses import (
    CorrelationAnalysis,
    CommonEDAPlots,
    DescriptiveStats,
)


def compute_descriptive_stats(series: pd.Series) -> DescriptiveStats:
    """Compute descriptive statistics for a single series.

    NaN values are counted and reported; statistics are computed on clean data.
    """
    nan_count = int(series.isna().sum())
    nan_pct = nan_count / len(series) if len(series) > 0 else 0.0
    clean = series.dropna()

    return DescriptiveStats(
        min_val=float(clean.min()),
        max_val=float(clean.max()),
        mean=float(clean.mean()),
        median=float(clean.median()),
        std=float(clean.std()),
        skew=float(stats.skew(clean.values)),
        kurtosis=float(stats.kurtosis(clean.values)),
        nan_count=nan_count,
        nan_pct=float(nan_pct),
        sample_size=len(series),
    )


def compute_correlation_analysis(
    feature: pd.Series,
    target: pd.Series,
    max_lag: int = 5,
) -> CorrelationAnalysis:
    """Compute Pearson, Spearman, and lagged correlations.

    Lags 1..max_lag are stored in lagged_correlations dict.
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    f, t = aligned["f"], aligned["t"]

    pearson = float(f.corr(t, method="pearson"))
    spearman = float(f.corr(t, method="spearman"))

    lagged: dict[int, float] = {
        lag: float(f.corr(t.shift(-lag), method="pearson"))
        for lag in range(1, max_lag + 1)
    }

    return CorrelationAnalysis(
        pearson=pearson,
        spearman=spearman,
        lagged_correlations=lagged,
    )


def create_common_eda_plots(
    feature: pd.Series,
    timestamps: pd.DatetimeIndex,
) -> CommonEDAPlots:
    """Create the common EDA time-series figure (feature over time)."""
    fig_ts, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
    ax1.set_title("Feature over time")
    ax1.set_ylabel("Feature value")
    fig_ts.tight_layout()
    plt.close(fig_ts)
    return CommonEDAPlots(time_series_fig=fig_ts)
