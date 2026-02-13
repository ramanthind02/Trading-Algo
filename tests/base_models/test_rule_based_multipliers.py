import numpy as np
import pandas as pd

from feature_selection.base_models.rule_based import RuleBasedModel


def test_rule_based_maps_levels_to_signed_multipliers() -> None:
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    feature = pd.Series(np.tile([-1, 0, 1, 1], 60), index=idx, name="breakout_signal_D_lookback_20")
    target = pd.Series(np.where(feature == 1, 0.02, np.where(feature == -1, -0.015, 0.0)), index=idx)

    model = RuleBasedModel(selection_metric="sharpe", metric_threshold=0.0)
    model.fit(feature, target)
    pred = model.predict(feature, strategy="long_short")

    assert set(np.sign(pred.unique())) <= {-1.0, 0.0, 1.0}
    assert (pred[feature == 0] == 0).all()
    assert np.any(pred[feature == 1] > 0)
    assert np.any(pred[feature == -1] < 0)
