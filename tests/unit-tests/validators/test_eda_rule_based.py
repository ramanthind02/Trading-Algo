import pytest

try:
    from feature_selection.validators.eda.rule_based import (
        compute_level_distribution,
        compute_level_stats,
        compute_transition_matrix,
    )
except ModuleNotFoundError:
    pytest.skip(
        "Legacy validators.eda.rule_based API removed; use feature_selection.eda.rule_based_eda",
        allow_module_level=True,
    )

import numpy as np
import pandas as pd


def test_compute_level_distribution():
    """Test level distribution computation."""
    # Create rule-based feature with -1, 0, 1 levels
    feature = pd.Series([-1] * 50 + [0] * 400 + [1] * 550)

    level_counts, level_fractions, imbalance_flag = compute_level_distribution(feature)

    assert level_counts[-1] == 50
    assert level_counts[0] == 400
    assert level_counts[1] == 550
    assert level_fractions[-1] == 0.05
    assert imbalance_flag is True  # -1 level < 10%


def test_compute_level_stats():
    """Test per-level target statistics."""
    np.random.seed(42)

    feature = pd.Series([-1] * 300 + [0] * 300 + [1] * 300)
    target = pd.Series(
        list(-0.5 + np.random.randn(300) * 0.3) +  # -1 level: negative mean
        list(0.0 + np.random.randn(300) * 0.3) +   # 0 level: zero mean
        list(0.5 + np.random.randn(300) * 0.3)     # 1 level: positive mean
    )

    level_stats = compute_level_stats(feature, target)

    assert len(level_stats) == 3
    assert 'mean_target' in level_stats.columns
    assert 'sharpe' in level_stats.columns

    # Check that level 1 has higher mean than level -1
    mean_neg1 = level_stats.loc[level_stats['level'] == -1, 'mean_target'].values[0]
    mean_pos1 = level_stats.loc[level_stats['level'] == 1, 'mean_target'].values[0]
    assert mean_pos1 > mean_neg1


def test_compute_transition_matrix():
    """Test regime transition matrix computation."""
    # Create feature with transitions
    feature = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 1, 0])

    transition_matrix = compute_transition_matrix(feature)

    assert transition_matrix.shape == (3, 3)
    assert transition_matrix.index.tolist() == [-1, 0, 1]
    assert np.allclose(transition_matrix.sum(axis=1), 1.0)  # Rows sum to 1
