"""Unit tests for common_eda.py (T001).

All tests use synthetic data with known properties.
No real market data — no cache required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.common_eda import (
    compute_descriptive_stats,
    compute_temporal_stability,
    compute_correlation_analysis,
    compute_rolling_objective,
    create_common_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis, CommonEDAPlots,
)


def _daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = _daily_index(len(values), start)
    return pd.Series(values, index=idx, dtype=float)


def test_descriptive_stats_known_values() -> None:
    data = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_descriptive_stats(data)
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.min_val == pytest.approx(1.0)
    assert stats.max_val == pytest.approx(5.0)
    assert stats.nan_count == 0
    assert stats.nan_pct == pytest.approx(0.0)
    assert stats.sample_size == 5


def test_nan_handling_explicit() -> None:
    data = _series([1.0, float('nan'), 3.0, float('nan'), 5.0])
    stats = compute_descriptive_stats(data)
    assert stats.nan_count == 2
    assert stats.nan_pct == pytest.approx(0.4)
    assert stats.mean == pytest.approx(3.0)
    assert stats.sample_size == 5


def test_index_misalignment_raises() -> None:
    idx_a = _daily_index(10, "2020-01-01")
    idx_b = _daily_index(10, "2021-01-01")
    feature = pd.Series(np.ones(10), index=idx_a)
    target = pd.Series(np.ones(10), index=idx_b)
    with pytest.raises(ValueError, match="index"):
        compute_temporal_stability(feature, target, idx_a, rolling_window=5)


def test_rolling_window_exceeds_length_raises() -> None:
    n = 10
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    with pytest.raises(ValueError, match="window"):
        compute_temporal_stability(feature, target, idx, rolling_window=n + 1)


def test_temporal_stability_no_lookahead() -> None:
    n = 50
    idx = _daily_index(n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    window = 10
    result = compute_temporal_stability(feature, target, idx, rolling_window=window)
    assert result.rolling_correlation.iloc[:window - 1].isna().all()
    assert pd.notna(result.rolling_correlation.iloc[window - 1])


def test_correlation_analysis_lagged_synthetic() -> None:
    n = 200
    idx = _daily_index(n)
    np.random.seed(42)
    base = pd.Series(np.random.randn(n), index=idx)
    target = base.shift(1).fillna(0)
    result = compute_correlation_analysis(base, target, max_lag=5)
    assert result.lagged_correlations[1] > result.pearson
    assert result.lagged_correlations[1] > result.lagged_correlations[2]


def test_correlation_analysis_has_all_lags() -> None:
    n = 100
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_correlation_analysis(feature, target, max_lag=5)
    assert set(result.lagged_correlations.keys()) == {1, 2, 3, 4, 5}


def test_rolling_objective_sharpe_manual() -> None:
    returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    signals = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])

    def sharpe_fn(s: pd.Series, r: pd.Series) -> float:
        return r.mean() / r.std() if r.std() > 0 else 0.0

    result = compute_rolling_objective(signals, returns, sharpe_fn, window=3)
    assert result.iloc[:2].isna().all()
    expected = np.mean([0.1, 0.2, 0.3]) / np.std([0.1, 0.2, 0.3], ddof=1)
    assert result.iloc[2] == pytest.approx(expected, rel=1e-6)


def test_common_eda_plots_smoke() -> None:
    n = 60
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    rolling_corr = pd.Series(np.random.randn(n), index=idx)
    rolling_obj = pd.Series(np.random.randn(n), index=idx)
    plots = create_common_eda_plots(feature, target, idx, rolling_corr, rolling_obj)
    assert plots.time_series_fig is not None
    assert plots.rolling_corr_fig is not None
    assert plots.rolling_obj_fig is not None
