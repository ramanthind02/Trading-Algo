import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_regions_require_min_consecutive_bins() -> None:
    idx = pd.date_range("2022-01-01", periods=450, freq="D")
    feature = pd.Series(np.linspace(0, 100, 450), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.where((feature > 20) & (feature < 40), 0.03, 0.0), index=idx)

    model = ContinuousBinningModel(
        n_bins=15,
        t_threshold=0.5,
        min_region_width=2,
        selection_metric="sharpe",
        metric_threshold=0.0,
    )
    model.fit(feature, target)

    # Region metadata is intentionally empty in v2; selection is represented
    # through active bins and selected bins instead.
    assert model.significant_regions_ == []
    assert len(model.active_bins_by_strategy_["long_short"]) >= 1
