"""Unit tests for BinningModelBase.clone() and ContinuousBinningModel.clone()."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.base_models.base_model import BinningModelBase


def test_continuous_binning_clone_same_args_unfitted() -> None:
    """Cloned model has same constructor args and is unfitted."""
    model = ContinuousBinningModel(
        n_bins=10,
        bin_counts=[10, 8, 5],
        selection_metric="sortino",
        strategy="long",
        use_coverage_bonus=True,
        coverage_bonus_per_10pct=0.02,
        max_coverage_bonus=0.2,
    )
    cloned = model.clone()

    assert isinstance(cloned, ContinuousBinningModel)
    assert cloned.n_bins == model.n_bins
    assert cloned.bin_counts == model.bin_counts
    assert cloned.selection_metric == model.selection_metric
    assert cloned.strategy == model.strategy
    assert cloned.use_coverage_bonus == model.use_coverage_bonus
    assert cloned.coverage_bonus_per_10pct == model.coverage_bonus_per_10pct
    assert cloned.max_coverage_bonus == model.max_coverage_bonus
    assert not cloned.is_fitted_
    assert cloned.bin_edges_ is None


def test_continuous_binning_clone_original_fitted_clone_not() -> None:
    """After fit(original), original is fitted and a pre-fit clone is not."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(200), name="f")
    target = pd.Series(np.random.randn(200) * 0.01, name="t")

    original = ContinuousBinningModel(bin_counts=[5], strategy="long", t_threshold=0.0)
    cloned = original.clone()
    original.fit(feature, target)

    assert original.is_fitted_
    assert not cloned.is_fitted_
    assert cloned.bin_edges_ is None


def test_continuous_binning_clone_predict_matches_deepcopy() -> None:
    """Fit(clone) then predict gives same output as fit(deepcopy) then predict."""
    np.random.seed(43)
    feature = pd.Series(np.random.randn(300), name="f")
    target = pd.Series(np.random.randn(300) * 0.01, name="t")

    template = ContinuousBinningModel(
        bin_counts=[8, 5],
        strategy="long",
        t_threshold=0.0,
    )
    model_via_clone = template.clone()
    model_via_deepcopy = copy.deepcopy(template)

    model_via_clone.fit(feature, target)
    model_via_deepcopy.fit(feature, target)

    pred_clone = model_via_clone.predict(feature, strategy="long")
    pred_deepcopy = model_via_deepcopy.predict(feature, strategy="long")

    pd.testing.assert_series_equal(pred_clone, pred_deepcopy)


def test_base_model_clone_returns_same_type() -> None:
    """BinningModelBase.clone() returns an instance of the same subclass."""
    model = ContinuousBinningModel(bin_counts=[3], strategy="long")
    cloned = model.clone()
    assert type(cloned) is type(model)
    assert cloned is not model
