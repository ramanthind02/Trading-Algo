import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_continuous_predict_returns_clipped_signed_multipliers() -> None:
    idx = pd.date_range("2021-01-01", periods=320, freq="D")
    feature = pd.Series(np.linspace(-2, 2, 320), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.where(feature > 0.5, 0.02, -0.015), index=idx)

    model = ContinuousBinningModel(
        n_bins=15,
        selection_metric="sharpe",
        metric_threshold=0.0,
        t_threshold=0.0,
        min_region_width=1,
    )
    model.fit(feature, target)
    pred = model.predict(feature, strategy="long_short")

    assert pred.max() <= 2.0
    assert pred.min() >= -2.0
    assert np.any(pred > 0)
    assert np.any(pred < 0)
