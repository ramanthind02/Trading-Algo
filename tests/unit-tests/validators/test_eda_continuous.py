import pytest

try:
    from feature_selection.validators.eda.continuous import (
        compute_decile_stats,
        check_monotonicity,
        detect_outliers,
    )
except ModuleNotFoundError:
    pytest.skip(
        "Legacy validators.eda.continuous API removed; use feature_selection.eda.continuous_eda",
        allow_module_level=True,
    )

import numpy as np
import pandas as pd


def test_compute_decile_stats():
    """Test decile-based analysis."""
    np.random.seed(42)

    # Create monotonic feature → target relationship
    feature = pd.Series(np.linspace(0, 10, 1000))
    target = pd.Series(0.5 * feature + np.random.randn(1000) * 0.5)

    decile_stats = compute_decile_stats(feature, target, n_deciles=10)

    assert len(decile_stats) == 10
    assert 'mean_target' in decile_stats.columns
    assert 'std_target' in decile_stats.columns
    assert 'sharpe' in decile_stats.columns

    # Check monotonicity (mean should increase across deciles)
    means = decile_stats['mean_target'].values
    assert means[-1] > means[0]  # Higher decile → higher mean


def test_monotonicity_check():
    """Test monotonicity detection."""
    # Monotonic increasing
    decile_means = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = check_monotonicity(decile_means)

    assert result.is_monotonic is True
    assert result.direction == 'increasing'
    assert result.kendall_tau > 0.8


def test_detect_outliers():
    """Test outlier detection."""
    data = pd.Series(np.random.randn(1000))
    # Add some outliers
    data.iloc[0] = 10.0
    data.iloc[1] = -10.0

    outlier_mask, outlier_fraction = detect_outliers(data, method='iqr')

    assert outlier_fraction > 0
    assert outlier_mask.sum() >= 2  # At least our 2 outliers
