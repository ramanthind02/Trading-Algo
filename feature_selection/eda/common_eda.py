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
    TemporalStability,
)


def _validate_aligned_index(feature: pd.Series, target: pd.Series) -> None:
    """Raise ValueError if feature and target do not share the same DatetimeIndex."""
    if not feature.index.equals(target.index):
        raise ValueError(
            f"feature and target index mismatch: "
            f"{feature.index[[0, -1]]} vs {target.index[[0, -1]]}"
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


def compute_temporal_stability(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_window: int = 252,
) -> TemporalStability:
    """Compute rolling feature-target correlation for temporal stability analysis.

    Raises:
        ValueError: If feature and target indices differ, or window > len(feature).
    """
    _validate_aligned_index(feature, target)
    if not feature.index.equals(pd.DatetimeIndex(timestamps)):
        raise ValueError("timestamps must equal feature.index")
    if rolling_window > len(feature):
        raise ValueError(
            f"window ({rolling_window}) exceeds series length ({len(feature)})"
        )

    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    rolling_corr = aligned["f"].rolling(window=rolling_window, min_periods=rolling_window).corr(aligned["t"])
    rolling_corr = rolling_corr.reindex(feature.index)

    delta = rolling_corr.diff().abs()
    break_timestamps = list(rolling_corr.index[delta > 0.3])

    return TemporalStability(
        rolling_correlation=rolling_corr,
        structural_breaks=break_timestamps,
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
    rolling_corr: pd.Series,
) -> CommonEDAPlots:
    """Create the two standard common EDA figures."""
    # 1. Time-series plot (single subplot)
    fig_ts, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
    ax1.set_title("Feature over time")
    ax1.set_ylabel("Feature value")
    fig_ts.tight_layout()
    plt.close(fig_ts)

    # 2. Rolling correlation plot
    fig_rc, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_corr.index, rolling_corr.values, linewidth=0.8, color="purple")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling feature-target correlation")
    ax.set_ylabel("Correlation")
    fig_rc.tight_layout()
    plt.close(fig_rc)

    return CommonEDAPlots(
        time_series_fig=fig_ts,
        rolling_corr_fig=fig_rc,
    )
