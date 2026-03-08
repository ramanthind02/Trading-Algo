import numpy as np
import pandas as pd
import pytest

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_fit_builds_v2_state_without_legacy_best_bins() -> None:
    idx = pd.date_range("2020-01-01", periods=220, freq="D")
    feature = pd.Series(np.linspace(0, 1, 220), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.sin(np.linspace(0, 8, 220)) * 0.01, index=idx)

    model = ContinuousBinningModel(n_bins=10)
    model.fit(feature, target)

    assert model.is_fitted_ is True
    assert isinstance(model.bin_stats_, dict)
    assert hasattr(model, "position_multipliers_by_strategy_")
    assert hasattr(model, "selected_bins_")
    assert "long" in model.selected_bins_
    assert "short" in model.selected_bins_
    assert not hasattr(model, "best_long_bin_")
    assert not hasattr(model, "best_short_bin_")


def test_fit_builds_bin_stats_with_selection_metrics() -> None:
    """Fit produces bin_stats with selection_metric_long/short (t_stat used internally)."""
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    feature = pd.Series(np.linspace(-1, 1, 240), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.sin(np.linspace(0, 16, 240)) * 0.01, index=idx)

    model = ContinuousBinningModel(n_bins=8)
    model.fit(feature, target)

    assert model.is_fitted_ is True
    assert model.bin_stats_
    sample_stat = next(iter(model.bin_stats_.values()))
    assert "selection_metric_long" in sample_stat
    assert "selection_metric_short" in sample_stat
