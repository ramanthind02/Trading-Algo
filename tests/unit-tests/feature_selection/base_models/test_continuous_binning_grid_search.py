"""Unit tests for ContinuousBinningModel grid search functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_grid_search_selects_best_bin_count() -> None:
    """Test that grid search over bin_counts selects winning n_bins."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(1000), name="test_feature")
    target = pd.Series(np.random.randn(1000) * 0.01, name="target")

    model = ContinuousBinningModel(
        bin_counts=[10, 8, 5, 3],
        selection_metric="sharpe",
        strategy="long",
        t_threshold=0.0,  # Accept any bin
    )
    model.fit(feature, target)

    assert model.is_fitted_
    assert model.n_bins in [10, 8, 5, 3]  # Should be one of the candidates
    assert "long" in model.selected_bins_
    assert "short" in model.selected_bins_


def test_coverage_bonus_increases_sharpe() -> None:
    """Test that coverage bonus adds to base Sharpe."""
    # Create synthetic data where one bin covers 60% with positive returns
    feature = pd.Series(np.concatenate([np.ones(600), np.zeros(400)]), name="f")
    target = pd.Series(np.concatenate([np.ones(600) * 0.05, np.zeros(400)]), name="t")

    model_no_bonus = ContinuousBinningModel(
        bin_counts=[2],
        use_coverage_bonus=False,
        strategy="long",
        t_threshold=0.0,
    )
    model_no_bonus.fit(feature, target)

    model_with_bonus = ContinuousBinningModel(
        bin_counts=[2],
        use_coverage_bonus=True,
        coverage_bonus_per_10pct=0.02,
        strategy="long",
        t_threshold=0.0,
    )
    model_with_bonus.fit(feature, target)

    sharpe_no_bonus = model_no_bonus.selected_bins_["long_sharpe"]
    sharpe_with_bonus = model_with_bonus.selected_bins_["long_sharpe"]

    assert sharpe_with_bonus > sharpe_no_bonus  # Bonus should increase


def test_long_short_mode_selects_two_bins() -> None:
    """Test that long_short strategy selects both long and short bins."""
    np.random.seed(42)
    # Create feature with negative correlation to target
    feature = pd.Series(np.linspace(-1, 1, 1000), name="f")
    target = pd.Series(-feature * 0.1 + np.random.randn(1000) * 0.01, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long_short",
        t_threshold=0.0,
    )
    model.fit(feature, target)

    assert model.selected_bins_["long"] is not None
    assert model.selected_bins_["short"] is not None
    assert model.selected_bins_["long"] != model.selected_bins_["short"]

    pred = model.predict(feature, strategy="long_short")
    assert (pred > 0).any()  # Some positive multipliers
    assert (pred < 0).any()  # Some negative multipliers


def test_backwards_compatibility_single_n_bins() -> None:
    """Test that passing n_bins (not bin_counts) still works."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500), name="test_feature")
    target = pd.Series(np.random.randn(500) * 0.01, name="target")

    # Old API: just n_bins
    model = ContinuousBinningModel(
        n_bins=10,  # Should default to bin_counts=[10]
        selection_metric="sharpe",
        strategy="long",
        t_threshold=0.0,
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
        t_threshold=0.0,
    )
    model.fit(feature, target)

    assert hasattr(model, "selected_bins_")
    assert "long" in model.selected_bins_
    assert "short" in model.selected_bins_
    assert "long_sharpe" in model.selected_bins_
    assert "short_sharpe" in model.selected_bins_


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
        t_threshold=0.0,
    )
    model.fit(feature, target)

    pred = model.predict(feature, strategy="long")

    # Most observations should have 0 (not in selected bin)
    # Only selected bin (top bin with high returns) should have non-zero values
    assert (pred == 0).sum() > 0, "Some predictions should be 0 (not in selected bin)"
    assert (pred != 0).sum() > 0, "Some predictions should be non-zero (in selected bin)"
    assert (pred != 0).sum() < len(pred), "Not all predictions should be non-zero"


def test_coverage_bonus_capped_at_max() -> None:
    """Test that coverage bonus is capped at max_coverage_bonus."""
    # 100% coverage bin
    feature = pd.Series(np.ones(1000), name="f")
    target = pd.Series(np.random.randn(1000) * 0.01 + 0.05, name="t")

    model = ContinuousBinningModel(
        bin_counts=[1],  # Single bin = 100% coverage
        use_coverage_bonus=True,
        coverage_bonus_per_10pct=0.1,  # Large bonus
        max_coverage_bonus=0.2,  # But capped
        strategy="long",
        t_threshold=0.0,
    )
    model.fit(feature, target)

    # Calculate expected: 90% above floor, would be 0.9 * 0.1 = 0.9, but capped at 0.2
    base_sharpe = model.bin_stats_[0]["sharpe"]
    adjusted_sharpe = model.selected_bins_["long_sharpe"]
    bonus = adjusted_sharpe - base_sharpe

    assert bonus == pytest.approx(0.2, abs=1e-4)  # Should equal max (with float tolerance)


def test_empty_bin_candidates_handled() -> None:
    """Test that model handles cases where no positive or negative bins exist."""
    np.random.seed(42)
    # All positive target (no short candidates)
    feature = pd.Series(np.random.randn(500), name="f")
    target = pd.Series(np.abs(np.random.randn(500)) * 0.01, name="t")

    model = ContinuousBinningModel(
        bin_counts=[5],
        strategy="long_short",
        t_threshold=0.0,
    )
    model.fit(feature, target)

    # Should have long bin but may not have short bin (all t-stats positive)
    assert model.selected_bins_["long"] is not None
    # selected_bins_["short"] may be None if no negative t-stats exist
