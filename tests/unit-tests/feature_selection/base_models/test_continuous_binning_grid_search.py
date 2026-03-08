"""Unit tests for ContinuousBinningModel grid search functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_fit_raises_when_bin_counts_empty() -> None:
    """Fit with empty bin_counts raises ValueError (no bin counts to evaluate)."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(100), name="f")
    target = pd.Series(np.random.randn(100) * 0.01, name="t")

    model = ContinuousBinningModel(
        n_bins=5,
        bin_counts=[],  # Explicit empty list
        strategy="long",
    )
    with pytest.raises(ValueError, match="No bin counts to evaluate"):
        model.fit(feature, target)


def test_grid_search_selects_best_bin_count() -> None:
    """Test that grid search over bin_counts selects winning n_bins."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(1000), name="test_feature")
    target = pd.Series(np.random.randn(1000) * 0.01, name="target")

    model = ContinuousBinningModel(
        bin_counts=[10, 8, 5, 3],
        strategy="long",
    )
    model.fit(feature, target)

    assert model.is_fitted_
    assert model.n_bins in [10, 8, 5, 3]  # Should be one of the candidates
    assert "long" in model.selected_bins_
    assert "short" in model.selected_bins_


def test_long_short_mode_selects_two_bins() -> None:
    """Test that long_short strategy selects both long and short bins."""
    np.random.seed(42)
    # Create feature with negative correlation to target
    feature = pd.Series(np.linspace(-1, 1, 1000), name="f")
    target = pd.Series(-feature * 0.1 + np.random.randn(1000) * 0.01, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long_short",
    )
    model.fit(feature, target)

    assert model.selected_bins_["long"] is not None
    assert model.selected_bins_["short"] is not None
    assert model.selected_bins_["long"] != model.selected_bins_["short"]

    pred = model.predict(feature, strategy="long_short")
    assert (pred > 0).any()  # Some 1.0 (long bin)
    assert (pred < 0).any()  # Some -1.0 (short bin)


def test_backwards_compatibility_single_n_bins() -> None:
    """Test that passing n_bins (not bin_counts) still works."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500), name="test_feature")
    target = pd.Series(np.random.randn(500) * 0.01, name="target")

    # n_bins without bin_counts defaults to bin_counts=[n_bins]
    model = ContinuousBinningModel(
        n_bins=10,
        strategy="long",
    )
    model.fit(feature, target)

    assert model.is_fitted_
    assert model.n_bins == 10  # Should use exactly 10 bins
    assert hasattr(model, "selected_bins_")


def test_selected_bins_attributes_exist() -> None:
    """Test that fit() creates selected_bins_ with expected keys."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500), name="test_feature")
    target = pd.Series(np.random.randn(500) * 0.01, name="target")

    model = ContinuousBinningModel(
        bin_counts=[5, 3],
        strategy="long_short",
    )
    model.fit(feature, target)

    assert hasattr(model, "selected_bins_")
    assert "long" in model.selected_bins_
    assert "short" in model.selected_bins_


def test_predict_uses_selected_bins() -> None:
    """Test that predict() returns values only for selected bins."""
    np.random.seed(42)
    # Create continuous feature with clear signal in one tail
    feature = pd.Series(np.linspace(0, 10, 1000), name="f")
    # Strong signal only in high values (top 20%)
    target = pd.Series(np.where(feature > 8, 0.1, 0.0) + np.random.randn(1000) * 0.01, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long",
    )
    model.fit(feature, target)

    pred = model.predict(feature, strategy="long")

    # Binary output: 0 or 1
    assert (pred == 0).sum() > 0, "Some predictions should be 0 (not in selected bin)"
    assert (pred != 0).sum() > 0, "Some predictions should be 1 (in selected bin)"
    assert pred.isin([0.0, 1.0]).all(), "Predictions should be binary 0 or 1"


def test_empty_bin_candidates_handled() -> None:
    """Test that model handles cases where no positive or negative bins exist."""
    np.random.seed(42)
    # All positive target (no short candidates)
    feature = pd.Series(np.random.randn(500), name="f")
    target = pd.Series(np.abs(np.random.randn(500)) * 0.01, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long_short",
    )
    model.fit(feature, target)

    assert model.selected_bins_["long"] is not None


def test_long_only_fit_raises_when_no_positive_t_stat_bin() -> None:
    """When strategy is long and all bins have negative t-stat, fit raises."""
    # Strong negative correlation: every bin has negative mean return (no long bin)
    np.random.seed(42)
    feature = pd.Series(np.linspace(0, 10, 1000), name="f")
    # All bins get negative mean: -0.05 in each bin (no noise so t-stat clearly negative)
    target = pd.Series(-0.05 + np.random.randn(1000) * 0.001, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long",
    )
    with pytest.raises(ValueError, match="No long bin with positive t-stat"):
        model.fit(feature, target)


def test_short_t_stat_raw_not_abs_for_long_short_score() -> None:
    """Score for long_short uses max(long_t_stat, -short_t_stat); short is raw (negative)."""
    np.random.seed(43)
    # Feature with one tail positive returns (long), one tail negative (short)
    feature = pd.Series(np.linspace(-2, 2, 1000), name="f")
    target = pd.Series(
        np.where(feature > 0.5, 0.05, np.where(feature < -0.5, -0.05, 0.0))
        + np.random.randn(1000) * 0.01,
        name="t",
    )
    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long_short",
    )
    model.fit(feature, target)
    assert model.selected_bins_["long"] is not None
    assert model.selected_bins_["short"] is not None


def test_bin_index_max_restricts_selection_to_range() -> None:
    """When bin_index_max=3, selected long bin must be in [0, 3] even if best t-stat is in bin 5."""
    np.random.seed(44)
    # 10 quantile bins; make bin 5 (middle) have the highest positive t-stat
    n = 1000
    feature = pd.Series(np.linspace(0, 1, n), name="f")
    target = pd.Series(np.random.randn(n) * 0.01, name="t")
    # Boost returns in the middle (roughly bin 4 or 5 with 10 bins)
    mid = (feature >= 0.45) & (feature <= 0.55)
    target = target.where(~mid, 0.08)

    model = ContinuousBinningModel(
        bin_counts=[10],
        strategy="long",
        bin_index_min=0,
        bin_index_max=3,
    )
    model.fit(feature, target)
    assert model.is_fitted_
    assert model.selected_bins_["long"] is not None
    assert 0 <= model.selected_bins_["long"] <= 3
    assert model.fit_config_["bin_index_min"] == 0
    assert model.fit_config_["bin_index_max"] == 3
