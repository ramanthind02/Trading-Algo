import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_continuous_predict_returns_binary_multipliers() -> None:
    """Continuous binning outputs binary 0, 1 (long), or -1 (short)."""
    idx = pd.date_range("2021-01-01", periods=320, freq="D")
    feature = pd.Series(np.linspace(-2, 2, 320), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.where(feature > 0.5, 0.02, -0.015), index=idx)

    model = ContinuousBinningModel(n_bins=15, strategy="long_short")
    model.fit(feature, target)
    pred = model.predict(feature, strategy="long_short")

    assert np.isfinite(pred).all()
    assert set(pred.dropna().unique()).issubset({0.0, 1.0, -1.0})
    assert np.any(pred > 0)
    assert np.any(pred < 0)
