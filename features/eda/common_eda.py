"""Common EDA infrastructure for both continuous and rule-based features (T001)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from features.eda.eda_dataclasses import (
    CorrelationAnalysis,
    DescriptiveStats,
)

EDA_VOL_EPS = 1e-10


def align_feature_target(feature: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Rows where either feature or target is NaN are dropped."""
    return pd.DataFrame({"f": feature, "t": target}).dropna()


def sharpe_from_mean_vol(
    mean: float,
    volatility: float,
    eps: float = EDA_VOL_EPS,
) -> float:
    """Mean/vol ratio, or NaN when volatility is non-positive or non-finite."""
    if not np.isfinite(volatility) or abs(volatility) <= eps:
        return float("nan")
    if not np.isfinite(mean):
        return float("nan")
    return float(mean / volatility)


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
    aligned = align_feature_target(feature, target)
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
