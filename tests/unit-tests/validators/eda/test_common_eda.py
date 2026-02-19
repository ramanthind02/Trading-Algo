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
    create_common_eda_plots,
    compute_ic_decay,
)
from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis, CommonEDAPlots, ICDecay,
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


def test_common_eda_plots_smoke() -> None:
    n = 60
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    rolling_corr = pd.Series(np.random.randn(n), index=idx)
    plots = create_common_eda_plots(feature, idx, rolling_corr)
    assert plots.time_series_fig is not None
    assert plots.rolling_corr_fig is not None


def test_ic_decay_horizons_present() -> None:
    """Result contains exactly the requested horizons."""
    n = 300
    idx = _daily_index(n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_ic_decay(feature, target, horizons=[1, 5, 10, 21])
    assert set(result.ic_by_horizon.keys()) == {1, 5, 10, 21}
    assert result.horizons == [1, 5, 10, 21]


def test_ic_decay_perfect_lag1_signal() -> None:
    """Feature that perfectly predicts 1-bar returns has IC(1) near 1.0."""
    n = 300
    idx = _daily_index(n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    # target is feature shifted forward by 1 bar
    target = feature.shift(-1).fillna(0)
    result = compute_ic_decay(feature, target, horizons=[1, 5])
    assert result.ic_by_horizon[1] > 0.9


def test_ic_decay_values_bounded() -> None:
    """All IC values must lie in [-1, 1]."""
    n = 200
    idx = _daily_index(n)
    np.random.seed(7)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_ic_decay(feature, target, horizons=[1, 5, 10, 21])
    for h, ic in result.ic_by_horizon.items():
        assert -1.0 <= ic <= 1.0, f"IC at horizon {h} out of bounds: {ic}"


from feature_selection.eda.common_eda import compute_feature_acf
from feature_selection.eda.eda_dataclasses import FeatureACF


def test_feature_acf_shapes() -> None:
    """lags, acf_values, pacf_values all have length max_lag."""
    n = 200
    idx = _daily_index(n)
    np.random.seed(1)
    feature = pd.Series(np.random.randn(n), index=idx)
    result = compute_feature_acf(feature, max_lag=20)
    assert len(result.lags) == 20
    assert len(result.acf_values) == 20
    assert len(result.pacf_values) == 20


def test_feature_acf_lags_values() -> None:
    """lags array is [1, 2, ..., max_lag]."""
    n = 100
    idx = _daily_index(n)
    np.random.seed(2)
    feature = pd.Series(np.random.randn(n), index=idx)
    result = compute_feature_acf(feature, max_lag=5)
    assert list(result.lags) == [1, 2, 3, 4, 5]


def test_feature_acf_persistent_series() -> None:
    """AR(1) series with phi=0.9 -> ACF lag-1 > 0.7."""
    n = 500
    idx = _daily_index(n)
    np.random.seed(0)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.9 * values[i - 1] + np.random.randn() * 0.1
    feature = pd.Series(values, index=idx)
    result = compute_feature_acf(feature, max_lag=5)
    assert result.acf_values[0] > 0.7
