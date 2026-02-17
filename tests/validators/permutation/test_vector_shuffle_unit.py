"""Unit tests for T013: Vector Shuffle Permutation Test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.permutation_tests import run_vector_shuffle_test
from feature_selection.validation.reports import VectorShuffleReport


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def test_vector_shuffle_report_fields() -> None:
    """VectorShuffleReport contains all required fields."""
    feature = pd.Series([1.0, 0.5, -0.5, 1.0, 0.5] * 20, name='feat')
    target = pd.Series([0.01, -0.01, 0.02, 0.01, -0.01] * 20)
    report = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=42)

    assert isinstance(report, VectorShuffleReport)
    assert hasattr(report, 'param_combo')
    assert hasattr(report, 'original_metric')
    assert hasattr(report, 'null_distribution')
    assert hasattr(report, 'critical_value')
    assert hasattr(report, 'p_value')
    assert hasattr(report, 'passed')
    assert hasattr(report, 'alpha')
    assert hasattr(report, 'nreps')


def test_deterministic_shuffle() -> None:
    """Same random_seed → identical null distribution and p-value."""
    rng = np.random.default_rng(99)
    feature = pd.Series(rng.standard_normal(100), name='feat')
    target = pd.Series(np.random.default_rng(1).standard_normal(100))

    r1 = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=42)
    r2 = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=42)

    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.p_value == r2.p_value


def test_different_seeds_differ() -> None:
    """Different seeds produce different null distributions."""
    feature = pd.Series(np.ones(50), name='feat')
    target = pd.Series(np.random.default_rng(7).standard_normal(50))

    r1 = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=1)
    r2 = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=2)

    # Distributions should differ (all-ones feature always stays the same after shuffle
    # so they'll actually be identical — use a non-uniform feature instead)
    feature2 = pd.Series(np.arange(50, dtype=float), name='feat')
    r3 = run_vector_shuffle_test(feature2, target, _sharpe, nreps=50, random_seed=1)
    r4 = run_vector_shuffle_test(feature2, target, _sharpe, nreps=50, random_seed=2)
    assert not np.array_equal(r3.null_distribution, r4.null_distribution)


def test_null_distribution_length() -> None:
    """null_distribution has exactly nreps entries."""
    feature = pd.Series(np.arange(50, dtype=float), name='feat')
    target = pd.Series(np.random.default_rng(0).standard_normal(50))

    for nreps in (30, 75, 200):
        report = run_vector_shuffle_test(feature, target, _sharpe, nreps=nreps, random_seed=0)
        assert len(report.null_distribution) == nreps
        assert report.nreps == nreps


def test_critical_value_matches_quantile() -> None:
    """critical_value equals (1-alpha) quantile of null_distribution."""
    feature = pd.Series(np.arange(60, dtype=float), name='feat')
    target = pd.Series(np.random.default_rng(3).standard_normal(60))

    for alpha in (0.05, 0.10, 0.20):
        report = run_vector_shuffle_test(
            feature, target, _sharpe, nreps=100, alpha=alpha, random_seed=0
        )
        expected_cv = float(np.percentile(report.null_distribution, (1.0 - alpha) * 100.0))
        assert abs(report.critical_value - expected_cv) < 1e-9


def test_pass_fail_consistency() -> None:
    """passed == (original_metric > critical_value)."""
    feature = pd.Series(np.arange(80, dtype=float), name='feat')
    target = pd.Series(np.random.default_rng(5).standard_normal(80))
    report = run_vector_shuffle_test(feature, target, _sharpe, nreps=100, random_seed=5)

    expected_passed = report.original_metric > report.critical_value
    assert report.passed == expected_passed


def test_p_value_in_unit_interval() -> None:
    """p_value is always in [0, 1]."""
    feature = pd.Series(np.random.default_rng(0).standard_normal(100), name='feat')
    target = pd.Series(np.random.default_rng(1).standard_normal(100))
    report = run_vector_shuffle_test(feature, target, _sharpe, nreps=50, random_seed=0)

    assert 0.0 <= report.p_value <= 1.0


def test_early_stopping_filter() -> None:
    """Filter VectorShuffleReports by passed==True matches alpha criterion."""
    alpha = 0.10
    p_values = [0.05, 0.15, 0.08, 0.20, 0.09]
    reports = [
        VectorShuffleReport(
            param_combo=f'p{i}',
            original_metric=0.5,
            null_distribution=np.array([0.1] * 50),
            critical_value=0.4,
            p_value=p,
            passed=(p <= alpha),
            alpha=alpha,
            nreps=50,
        )
        for i, p in enumerate(p_values)
    ]
    passers = [r for r in reports if r.passed]
    # p=0.05, 0.08, 0.09 pass
    assert len(passers) == 3
    assert all(r.p_value <= alpha for r in passers)


def test_get_fitted_vector() -> None:
    """BinningModelBase.get_fitted_vector() returns pd.Series on training index."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    rng = np.random.default_rng(0)
    n = 100
    feature = pd.Series(rng.standard_normal(n), name='rsi_signal_D_lookback_5')
    target = pd.Series(rng.standard_normal(n))

    model = ContinuousBinningModel(n_bins=5)
    model.fit(feature, target)

    fitted_vec = model.get_fitted_vector(strategy='long')
    assert isinstance(fitted_vec, pd.Series)
    assert len(fitted_vec) == n
    assert fitted_vec.index.equals(feature.index)
