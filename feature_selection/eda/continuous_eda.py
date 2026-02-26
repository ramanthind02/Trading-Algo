"""Continuous feature EDA (T002): decile analysis, distribution diagnostics, plots."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from feature_selection.eda.eda_dataclasses import (
    ContinuousEDAPlots,
    DecileAnalysis,
    DecileBinStats,
    DistributionDiagnostics,
    QuintileSpread,
)

_VOL_THRESHOLD = 1e-10  # volatility below this is treated as zero


def compute_decile_analysis(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 15,
) -> DecileAnalysis:
    """Bin feature into n_bins quantile bins and compute per-bin target statistics.

    Args:
        feature: Continuous feature series (DatetimeIndex).
        target: Aligned target return series.
        n_bins: Number of quantile bins (default 15). Must satisfy n_bins <= len/10.

    Raises:
        ValueError: If n_bins > len(feature.dropna()) / 10.
        ValueError: If actual distinct quantile bins formed != n_bins.
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    max_bins = len(aligned) // 10
    if n_bins > max_bins:
        raise ValueError(
            f"n_bins ({n_bins}) exceeds max allowed ({max_bins}) "
            f"for {len(aligned)} samples (need at least 10 per bin)"
        )

    aligned["bin"], bin_edges = pd.qcut(
        aligned["f"], q=n_bins, labels=False, retbins=True, duplicates="drop"
    )

    actual_n_bins = len(bin_edges) - 1
    if actual_n_bins <= 0:
        raise ValueError(
            f"Unable to form any quantile bins from feature with {len(aligned)} non-null samples."
        )

    grp_stats = aligned.groupby("bin")["t"].agg(["mean", "std", "count"])

    n_effective_bins = actual_n_bins
    mean_return = np.full(n_effective_bins, np.nan)
    volatility = np.full(n_effective_bins, np.nan)
    sharpe = np.full(n_effective_bins, np.nan)
    t_stat = np.full(n_effective_bins, np.nan)
    sample_count = np.zeros(n_effective_bins, dtype=int)

    for i in grp_stats.index:
        idx_i = int(i)
        m = float(grp_stats.loc[i, "mean"])
        v = float(grp_stats.loc[i, "std"])
        n = int(grp_stats.loc[i, "count"])
        mean_return[idx_i] = m
        volatility[idx_i] = v
        sample_count[idx_i] = n
        if v > _VOL_THRESHOLD:
            sharpe[idx_i] = m / v
            if n >= 2:
                t_stat[idx_i] = m / (v / np.sqrt(n))

    valid_means = mean_return[~np.isnan(mean_return)]
    tau, p = stats.kendalltau(np.arange(len(valid_means)), valid_means)
    if abs(tau) > 0.5 and p < 0.05:
        trend = "monotonic_increasing" if tau > 0 else "monotonic_decreasing"
    elif abs(tau) < 0.1:
        trend = "flat"
    else:
        trend = "U-shaped"

    bin_stats = DecileBinStats(
        bin_edges=bin_edges,
        mean_return=mean_return,
        volatility=volatility,
        sharpe=sharpe,
        t_stat=t_stat,
        sample_count=sample_count,
    )
    return DecileAnalysis(bin_stats=bin_stats, overall_trend=trend)


def compute_distribution_diagnostics(feature: pd.Series) -> DistributionDiagnostics:
    """Compute normality diagnostics for a feature series.

    Uses Shapiro-Wilk for n <= 5000, D'Agostino K^2 test otherwise.
    """
    clean = feature.dropna().values
    skewness = float(stats.skew(clean))
    kurtosis = float(stats.kurtosis(clean))

    if len(clean) <= 5000:
        stat, p_value = stats.shapiro(clean)
    else:
        stat, p_value = stats.normaltest(clean)

    return DistributionDiagnostics(
        skewness=skewness,
        kurtosis=kurtosis,
        normality_test_stat=float(stat),
        normality_p_value=float(p_value),
        is_normal=bool(p_value > 0.05),
    )


def compute_quintile_spread(
    feature: pd.Series,
    target: pd.Series,
) -> QuintileSpread:
    """Bin feature into 5 quantiles and compute per-quintile mean return.

    spread = mean_return(Q5) - mean_return(Q1).
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    aligned = aligned.copy()
    aligned["quintile"] = pd.qcut(aligned["f"], q=5, labels=False, duplicates="drop")
    quintile_means = (
        aligned.groupby("quintile")["t"]
        .mean()
        .reindex(range(5))
        .to_numpy(dtype=float)
    )
    spread = float(quintile_means[4] - quintile_means[0])
    return QuintileSpread(quintile_means=quintile_means, spread=spread)


def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
    quintile_spread: QuintileSpread,
) -> ContinuousEDAPlots:
    """Create the three standard continuous EDA figures: decile plot, histogram, quintile spread."""
    bs = decile_analysis.bin_stats
    bins = np.arange(len(bs.mean_return))

    # 1. Decile plot — 3 subplots
    fig_d, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1.bar(bins, np.nan_to_num(bs.mean_return), color="steelblue")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Mean return per bin")
    ax2.bar(bins, np.nan_to_num(bs.sharpe), color="green")
    ax2.set_title("Sharpe per bin")
    ax3.bar(bins, np.nan_to_num(bs.t_stat), color="darkorange")
    ax3.set_title("t-statistic per bin")
    ax3.set_xlabel("Bin")
    fig_d.tight_layout()
    plt.close(fig_d)

    # 2. Histogram with bin-edge overlay
    fig_h, ax = plt.subplots(figsize=(10, 4))
    clean = feature.dropna().values
    ax.hist(clean, bins=50, color="steelblue", alpha=0.7, density=True)
    for edge in bs.bin_edges[1:-1]:
        ax.axvline(edge, color="red", alpha=0.4, linewidth=0.8)
    ax.set_title("Feature distribution with quantile bin edges")
    fig_h.tight_layout()
    plt.close(fig_h)

    # 3. Quintile spread figure
    fig_qs, ax = plt.subplots(figsize=(8, 4))
    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    colors = ["#d73027" if v < 0 else "#1a9850" for v in quintile_spread.quintile_means]
    ax.bar(quintile_labels, np.nan_to_num(quintile_spread.quintile_means), color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Mean return by quintile  |  spread = {quintile_spread.spread:.4f}")
    ax.set_xlabel("Quintile")
    ax.set_ylabel("Mean return")
    fig_qs.tight_layout()
    plt.close(fig_qs)

    return ContinuousEDAPlots(
        decile_plot_fig=fig_d,
        histogram_fig=fig_h,
        quintile_spread_fig=fig_qs,
    )
