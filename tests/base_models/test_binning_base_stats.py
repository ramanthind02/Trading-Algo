import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_fit_builds_v2_state_without_legacy_best_bins() -> None:
    idx = pd.date_range("2020-01-01", periods=220, freq="D")
    feature = pd.Series(np.linspace(0, 1, 220), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.sin(np.linspace(0, 8, 220)) * 0.01, index=idx)

    model = ContinuousBinningModel(n_bins=10, selection_metric="sharpe")
    model.fit(feature, target)

    assert model.is_fitted_ is True
    assert isinstance(model.bin_stats_, dict)
    assert hasattr(model, "position_multipliers_by_strategy_")
    assert not hasattr(model, "best_long_bin_")
    assert not hasattr(model, "best_short_bin_")
