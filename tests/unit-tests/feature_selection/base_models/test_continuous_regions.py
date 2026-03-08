import numpy as np
import pandas as pd

from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_regions_require_min_consecutive_bins() -> None:
    """Test that model selects bins from a region with clear signal.

    The ContinuousBinningModel no longer uses region detection
    (significant_regions_ is kept empty for backward compat).
    Instead, verify that the model selects at least one active bin
    for the long strategy given a feature with a clear positive-return
    region.
    """
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

    # Model should have selected at least one active bin for the long strategy
    active_long = model.active_bins_by_strategy_.get("long", [])
    assert len(active_long) >= 1, (
        f"Expected at least 1 active long bin, got {len(active_long)}"
    )
