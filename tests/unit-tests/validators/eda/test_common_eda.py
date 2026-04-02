"""Unit tests for common_eda.py (T001).

All tests use synthetic data with known properties.
No real market data — no cache required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.common_eda import (
    compute_descriptive_stats,
    compute_correlation_analysis,
)
from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats,
    CorrelationAnalysis,
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


