"""Continuous feature EDA (T002): decile analysis and distribution diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from features.eda.common_eda import (
    EDA_VOL_EPS,
    align_feature_target,
    sharpe_from_mean_vol,
)
from features.eda.eda_dataclasses import (
    DecileAnalysis,
    DecileBinStats,
    DistributionDiagnostics,
    QuintileSpread,
)


def compute_decile_analysis(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 15,
) -> DecileAnalysis:
    """Bin feature into n_bins quantile bins and compute per-bin target statistics."""
    aligned = align_feature_target(feature, target)
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
        sharpe[idx_i] = sharpe_from_mean_vol(m, v)
        if v > EDA_VOL_EPS and n >= 2:
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
    """Compute normality diagnostics for a feature series."""
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
    """Bin feature into 5 quantiles and compute per-quintile mean return."""
    aligned = align_feature_target(feature, target).copy()
    aligned["quintile"] = pd.qcut(aligned["f"], q=5, labels=False, duplicates="drop")
    quintile_means = (
        aligned.groupby("quintile")["t"]
        .mean()
        .reindex(range(5))
        .to_numpy(dtype=float)
    )
    spread = float(quintile_means[4] - quintile_means[0])
    return QuintileSpread(quintile_means=quintile_means, spread=spread)
