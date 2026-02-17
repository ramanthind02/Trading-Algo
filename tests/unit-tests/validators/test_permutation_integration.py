import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.permutation import run_vector_shuffle_test


def test_run_vector_shuffle_test():
    """Test vector shuffle permutation test."""
    np.random.seed(42)

    # Create feature with predictive power
    feature = pd.Series(np.random.randn(500))
    target = pd.Series(0.3 * feature + np.random.randn(500) * 0.5)

    # Run permutation test
    report = run_vector_shuffle_test(
        feature=feature,
        target=target,
        n_permutations=100,
        confidence_level=0.95,
        random_seed=42,
    )

    assert report.stage == 'stage1_vector_shuffle'
    assert report.n_permutations == 100
    assert report.observed_sharpe != 0
    assert len(report.permuted_sharpes) == 100
    assert hasattr(report, 'p_value')
    assert hasattr(report, 'passed')
