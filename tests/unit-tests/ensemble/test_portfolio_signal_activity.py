"""Unit tests for forecast-to-activity signal conversion in portfolio orchestration."""

from __future__ import annotations

import pandas as pd

from ensemble.portfolio import _forecast_to_activity_signal


def test_forecast_to_activity_signal_treats_short_as_active() -> None:
    """Signed forecasts should map to activity, not long-only direction."""
    forecast = pd.Series([1.0, -0.5, 0.0, -2.0, 0.25])

    signal = _forecast_to_activity_signal(forecast)

    assert signal.tolist() == [1, 1, 0, 1, 1]
