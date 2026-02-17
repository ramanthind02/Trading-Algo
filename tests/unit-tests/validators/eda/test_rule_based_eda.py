"""Unit tests for rule_based_eda.py (T003).

All synthetic data — no real market data required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.rule_based_eda import (
    compute_per_level_stats,
    compute_bootstrap_ci,
    compute_transition_matrix,
    create_rule_based_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    PerLevelStats, BootstrapCIResults, TransitionMatrix, RuleBasedEDAPlots,
)


def _make_levels(n_per_level: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Synthetic feature with 3 balanced levels and known per-level returns."""
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_per_level * 3)
    feature = pd.Series(
        [-1] * n_per_level + [0] * n_per_level + [1] * n_per_level,
        index=idx, dtype=float,
    )
    target = pd.Series(
        list(-0.3 + np.random.randn(n_per_level) * 0.2) +
        list(0.0 + np.random.randn(n_per_level) * 0.2) +
        list(0.3 + np.random.randn(n_per_level) * 0.2),
        index=idx, dtype=float,
    )
    return feature, target


def test_per_level_stats_known_values() -> None:
    """Level +1 must have higher mean_return than level -1."""
    feature, target = _make_levels()
    result = compute_per_level_stats(feature, target)
    assert result.stats_by_level[1].mean_return > result.stats_by_level[-1].mean_return


def test_per_level_stats_is_reliable_flag() -> None:
    """Level with sample_count < 10 must be flagged is_reliable=False."""
    idx = pd.bdate_range("2020-01-01", periods=15)
    feature = pd.Series([-1] * 5 + [0] * 5 + [1] * 5, index=idx, dtype=float)
    target = pd.Series(np.random.randn(15), index=idx)
    result = compute_per_level_stats(feature, target)
    assert result.stats_by_level[-1].is_reliable is False


def test_per_level_stats_sharpe_zero_vol() -> None:
    """Level with constant returns -> Sharpe=NaN, no crash."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    feature = pd.Series([1] * 30, index=idx, dtype=float)
    target = pd.Series([0.01] * 30, index=idx)
    result = compute_per_level_stats(feature, target)
    assert np.isnan(result.stats_by_level[1].sharpe)


def test_invalid_level_value_raises() -> None:
    """Feature containing value 2 (not in [-1, 0, 1]) raises ValueError."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    feature = pd.Series([1, 0, 2, -1, 0, 1, 0, -1, 1, 0], index=idx, dtype=float)
    target = pd.Series(np.random.randn(10), index=idx)
    with pytest.raises(ValueError, match="level"):
        compute_per_level_stats(feature, target)


def test_bootstrap_ci_deterministic() -> None:
    """Same seed=42 produces identical ci_lower and ci_upper both times."""
    feature, target = _make_levels()
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    result1 = compute_bootstrap_ci(returns_by_level, n_iterations=200, confidence=0.95, seed=42)
    result2 = compute_bootstrap_ci(returns_by_level, n_iterations=200, confidence=0.95, seed=42)
    assert result1.ci_by_level[1].ci_lower == pytest.approx(result2.ci_by_level[1].ci_lower)
    assert result1.ci_by_level[1].ci_upper == pytest.approx(result2.ci_by_level[1].ci_upper)


def test_bootstrap_ci_level1_positive_mean() -> None:
    """CI for level +1 should have positive lower bound given strong synthetic returns."""
    feature, target = _make_levels(n_per_level=300)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    result = compute_bootstrap_ci(returns_by_level, n_iterations=500, confidence=0.95, seed=42)
    assert result.ci_by_level[1].ci_lower > 0.0


def test_transition_matrix_rows_sum_to_one() -> None:
    """Each row of transition_probs sums to 1.0."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    feature = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 1, 0] * 3, index=idx, dtype=float)
    result = compute_transition_matrix(feature)
    row_sums = result.transition_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_transition_matrix_known_counts() -> None:
    """Signal sequence [-1, 0, 1, 0, -1] -> known 4 transitions."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    feature = pd.Series([-1.0, 0.0, 1.0, 0.0, -1.0], index=idx)
    result = compute_transition_matrix(feature)
    assert result.transition_counts.sum() == 4


def test_rule_based_plots_smoke() -> None:
    """Both Figure objects created without error."""
    feature, target = _make_levels()
    per_level = compute_per_level_stats(feature, target)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    bootstrap = compute_bootstrap_ci(returns_by_level, n_iterations=100, confidence=0.95, seed=42)
    plots = create_rule_based_eda_plots(per_level, bootstrap)
    assert plots.level_plot_fig is not None
    assert plots.transition_heatmap_fig is not None
