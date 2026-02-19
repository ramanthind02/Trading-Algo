"""Unit tests for continuous_eda.py (T002).

All synthetic data — no real market data, no cache required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.continuous_eda import (
    compute_decile_analysis,
    compute_distribution_diagnostics,
    create_continuous_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    DecileAnalysis, DistributionDiagnostics, ContinuousEDAPlots,
)


def _series(n: int, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 10, n), index=idx)
    target = pd.Series(0.5 * np.linspace(0, 10, n) + np.random.randn(n) * 0.5, index=idx)
    return feature, target


def test_decile_analysis_known_values() -> None:
    """15-bin analysis on 150-sample series: each bin has ~10 samples."""
    n = 150
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_decile_analysis(feature, target, n_bins=15)
    assert len(result.bin_stats.sample_count) == 15
    assert result.bin_stats.sample_count.sum() == pytest.approx(n, abs=5)


def test_decile_analysis_bin_edges_length() -> None:
    """bin_edges has length n_bins + 1."""
    feature, target = _series(200)
    result = compute_decile_analysis(feature, target, n_bins=10)
    assert len(result.bin_stats.bin_edges) == 11


def test_distribution_diagnostics_known_skew() -> None:
    """np.random.seed(42) normal -> is_normal=True for small sample."""
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=200)
    feature = pd.Series(np.random.normal(0, 1, 200), index=idx)
    result = compute_distribution_diagnostics(feature)
    assert result.is_normal is True
    assert abs(result.skewness) < 0.5


def test_sharpe_zero_volatility_bin() -> None:
    """Bin with constant returns must produce Sharpe=NaN, no crash."""
    n = 150
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.ones(n) * 0.01, index=idx)
    result = compute_decile_analysis(feature, target, n_bins=15)
    assert np.all(np.isnan(result.bin_stats.sharpe))


def test_n_bins_too_large_raises() -> None:
    """n_bins > len(feature) / 10 -> ValueError."""
    n = 50
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    with pytest.raises(ValueError, match="n_bins"):
        compute_decile_analysis(feature, target, n_bins=10)


def test_n_bins_2_minimum() -> None:
    """n_bins=2 must work without error."""
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_decile_analysis(feature, target, n_bins=2)
    assert len(result.bin_stats.sample_count) == 2


def test_continuous_eda_plots_smoke() -> None:
    """Both Figure objects created without error."""
    feature, target = _series(200)
    da = compute_decile_analysis(feature, target, n_bins=15)
    plots = create_continuous_eda_plots(feature, target, da)
    assert plots.decile_plot_fig is not None
    assert plots.histogram_fig is not None
