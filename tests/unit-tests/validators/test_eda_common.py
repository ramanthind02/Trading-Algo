import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.eda.common import (
    compute_distribution_stats,
    compute_correlations,
    run_stationarity_tests,
)


def test_compute_distribution_stats():
    """Test distribution statistics computation."""
    data = pd.Series(np.random.randn(1000))
    stats = compute_distribution_stats(data)

    assert stats.n_samples == 1000
    assert -0.2 < stats.mean < 0.2  # Approximately zero
    assert 0.8 < stats.std < 1.2  # Approximately one
    assert hasattr(stats, 'skew')
    assert hasattr(stats, 'kurtosis')


def test_compute_correlations():
    """Test correlation computation."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500))
    target = pd.Series(0.3 * feature + np.random.randn(500) * 0.5)

    corrs = compute_correlations(feature, target)

    assert 'pearson' in corrs
    assert 'spearman' in corrs
    assert 'kendall' in corrs
    assert 0.2 < corrs['pearson'] < 0.6  # Positive correlation


def test_run_stationarity_tests():
    """Test ADF and KPSS stationarity tests."""
    # Generate stationary series
    stationary_data = pd.Series(np.random.randn(200))

    adf_result, kpss_result = run_stationarity_tests(stationary_data)

    assert hasattr(adf_result, 'p_value')
    assert hasattr(kpss_result, 'p_value')
    assert hasattr(adf_result, 'is_stationary')
