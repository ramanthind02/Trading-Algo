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
    MonotonicityTest,
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

    actual_bins = sorted(aligned["bin"].dropna().unique())
    mean_return = np.full(n_bins, np.nan)
    volatility = np.full(n_bins, np.nan)
    sharpe = np.full(n_bins, np.nan)
    t_stat = np.full(n_bins, np.nan)
    sample_count = np.zeros(n_bins, dtype=int)

    for b in actual_bins:
        i = int(b)
        grp = aligned.loc[aligned["bin"] == b, "t"]
        n = len(grp)
        sample_count[i] = n
        m = float(grp.mean())
        v = float(grp.std())
        mean_return[i] = m
        volatility[i] = v
        if v > _VOL_THRESHOLD:
            sharpe[i] = m / v
            if n >= 2:
                t_stat[i] = m / (v / np.sqrt(n))

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


def compute_monotonicity_test(bin_means: np.ndarray) -> MonotonicityTest:
    """Kendall's tau monotonicity test over bin mean returns.

    is_monotonic is True iff |tau| > 0.5 and p < 0.05.
    """
    idx = np.arange(len(bin_means))
    tau, p_value = stats.kendalltau(idx, bin_means)
    return MonotonicityTest(
        kendall_tau=float(tau),
        p_value=float(p_value),
        is_monotonic=bool(abs(tau) > 0.5 and p_value < 0.05),
    )


def compute_distribution_diagnostics(feature: pd.Series) -> DistributionDiagnostics:
    """Compute normality diagnostics for a feature series.

    Uses Shapiro-Wilk for n <= 5000, Anderson-Darling otherwise.
    """
    clean = feature.dropna().values
    skewness = float(stats.skew(clean))
    kurtosis = float(stats.kurtosis(clean))

    if len(clean) <= 5000:
        stat, p_value = stats.shapiro(clean)
    else:
        result = stats.anderson(clean, dist="norm")
        stat = float(result.statistic)
        p_value = 0.05 if stat < result.critical_values[2] else 0.01

    return DistributionDiagnostics(
        skewness=skewness,
        kurtosis=kurtosis,
        normality_test_stat=float(stat),
        normality_p_value=float(p_value),
        is_normal=bool(p_value > 0.05),
    )


def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
    dist_diagnostics: DistributionDiagnostics,
) -> ContinuousEDAPlots:
    """Create the four standard continuous EDA figures."""
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

    # 3. Q-Q plot
    fig_qq, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(clean, dist="norm", plot=ax)
    ax.set_title("Q-Q plot vs Normal")
    fig_qq.tight_layout()
    plt.close(fig_qq)

    # 4. KDE plot
    fig_kde, ax = plt.subplots(figsize=(10, 4))
    kde = stats.gaussian_kde(clean)
    x_range = np.linspace(clean.min(), clean.max(), 300)
    ax.plot(x_range, kde(x_range), color="purple", linewidth=1.5)
    ax.set_title("Kernel density estimate")
    fig_kde.tight_layout()
    plt.close(fig_kde)

    return ContinuousEDAPlots(
        decile_plot_fig=fig_d,
        histogram_fig=fig_h,
        qq_plot_fig=fig_qq,
        kde_fig=fig_kde,
    )
