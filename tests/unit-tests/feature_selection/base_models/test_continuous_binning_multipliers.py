import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_continuous_predict_returns_signed_multipliers() -> None:
    """Multipliers are raw Sharpe ratios — positive for long bins, negative for short."""
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

    # Multipliers are raw Sharpe values (not clipped) — verify sign correctness
    assert np.isfinite(pred).all(), "All predictions should be finite"
    assert np.any(pred > 0), "Should have positive (long) multipliers"
    assert np.any(pred < 0), "Should have negative (short) multipliers"
