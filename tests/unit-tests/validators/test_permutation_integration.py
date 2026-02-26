import pytest
import numpy as np
import pandas as pd
from feature_selection.validation.permutation_tests import run_vector_shuffle_test
from feature_selection.validation.objective_metrics import metric_sharpe


def test_run_vector_shuffle_test():
    """Test vector shuffle permutation test."""
    np.random.seed(42)

    # Create feature with predictive power (fitted signal: position multipliers)
    feature = pd.Series(np.random.randn(500))
    target = pd.Series(0.3 * feature + np.random.randn(500) * 0.5)

    # Run permutation test (current API: fitted_feature, target, objective_func, nreps, alpha)
    report = run_vector_shuffle_test(
        fitted_feature=feature,
        target=target,
        objective_func=metric_sharpe,
        nreps=100,
        alpha=0.05,
        random_seed=42,
    )

    assert report.param_combo == "default"
    assert report.nreps == 100
    assert report.original_metric is not None
    assert len(report.null_distribution) == 100
    assert hasattr(report, "p_value")
    assert hasattr(report, "passed")
