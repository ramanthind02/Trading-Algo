from __future__ import annotations

import numpy as np

from features.validation.permutation_tests import _permutation_tail_stats


def test_permutation_tail_stats_uses_pseudo_count_denominator_nreps_plus_one() -> None:
    original_metric = 5.0
    null_metrics = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    nreps = 6
    null_ge_count = int((null_metrics >= original_metric).sum())
    assert null_ge_count == 2

    p_value, critical_value, passed = _permutation_tail_stats(
        original_metric,
        null_metrics,
        nreps=nreps,
        alpha=0.1,
    )

    assert p_value == (1 + null_ge_count) / (nreps + 1)
    assert p_value == 3 / 7
    assert critical_value == float(np.percentile(null_metrics, 90.0))
    assert passed == (original_metric > critical_value)


def test_permutation_tail_stats_discrete_values_for_nreps_100() -> None:
    """With nreps=100, p-values are k/101 only (k = 1 .. 101)."""
    nreps = 100
    allowed = {k / 101 for k in range(1, 102)}
    for null_ge_count in (0, 1, 6, 50, 99, 100):
        null_metrics = np.full(nreps, 0.0)
        null_metrics[:null_ge_count] = 10.0
        observed = 5.0 if null_ge_count < nreps else 0.0
        p_value, _, _ = _permutation_tail_stats(observed, null_metrics, nreps=nreps, alpha=0.1)
        assert p_value in allowed
        assert p_value == (null_ge_count + 1) / 101
