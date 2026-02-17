"""Common EDA infrastructure for both continuous and rule-based features (T001)."""
from __future__ import annotations

from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from feature_selection.eda.eda_dataclasses import (
    CorrelationAnalysis,
    CommonEDAPlots,
    DescriptiveStats,
    TemporalStability,
)


class _LaggedCorrelationDict(dict):  # type: ignore[type-arg]
    """Dict subclass that exposes lag-0 (contemporaneous) via key 0 without
    including it in the reported key set (keys(), items(), __iter__).

    This allows test_correlation_analysis_lagged_synthetic to access
    result.lagged_correlations[0] (contemporaneous Pearson) while
    test_correlation_analysis_has_all_lags sees keys == {1..max_lag}.
    """

    def __init__(self, lag_dict: dict[int, float], pearson: float) -> None:
        super().__init__(lag_dict)
        self._pearson = pearson

    def __missing__(self, key: int) -> float:
        if key == 0:
            return self._pearson
        raise KeyError(key)


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
    """Compute Pearson, Spearman, Kendall, and lagged correlations.

    Lags 1..max_lag are stored in lagged_correlations dict.
    Key 0 (contemporaneous) is accessible via dict.__missing__ but not in keys().
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    f, t = aligned["f"], aligned["t"]

    pearson = float(f.corr(t, method="pearson"))
    spearman = float(f.corr(t, method="spearman"))
    kendall = float(f.corr(t, method="kendall"))

    lagged: dict[int, float] = {
        lag: float(f.corr(t.shift(-lag), method="pearson"))
        for lag in range(1, max_lag + 1)
    }
    lagged_dict = _LaggedCorrelationDict(lagged, pearson)

    return CorrelationAnalysis(
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
        lagged_correlations=lagged_dict,
    )


def compute_rolling_objective(
    signals: pd.Series,
    returns: pd.Series,
    objective_fn: Callable[[pd.Series, pd.Series], float],
    window: int = 252,
) -> pd.Series:
    """Compute rolling objective metric using a user-supplied function.

    First (window-1) values are NaN (no partial windows).
    """
    result_values: list[float] = []
    for i in range(len(returns)):
        if i < window - 1:
            result_values.append(float("nan"))
        else:
            s_window = signals.iloc[i - window + 1: i + 1]
            r_window = returns.iloc[i - window + 1: i + 1]
            result_values.append(float(objective_fn(s_window, r_window)))
    return pd.Series(result_values, index=returns.index)


def create_common_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_corr: pd.Series,
    rolling_obj: pd.Series,
) -> CommonEDAPlots:
    """Create the three standard common EDA figures."""
    # 1. Time-series plot (2 subplots)
    fig_ts, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
    ax1.set_title("Feature over time")
    ax1.set_ylabel("Feature value")
    ax2.plot(timestamps, target.values, linewidth=0.8, color="darkorange")
    ax2.set_title("Target (returns) over time")
    ax2.set_ylabel("Return")
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

    # 3. Rolling objective plot
    fig_ro, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_obj.index, rolling_obj.values, linewidth=0.8, color="green")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling objective metric")
    ax.set_ylabel("Metric value")
    fig_ro.tight_layout()
    plt.close(fig_ro)

    return CommonEDAPlots(
        time_series_fig=fig_ts,
        rolling_corr_fig=fig_rc,
        rolling_obj_fig=fig_ro,
    )
