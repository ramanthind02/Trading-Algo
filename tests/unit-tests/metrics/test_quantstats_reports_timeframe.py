from __future__ import annotations

import pandas as pd

from metrics.plotting.graphing.quantstats_reports import _resample_to_daily_if_needed
from utils.core.enums import TimeFrame


def test_resample_to_daily_if_needed_sums_subdaily_returns() -> None:
    idx = pd.date_range("2024-01-01 00:00:00", periods=6, freq="4h")
    returns = pd.Series([0.01, -0.02, 0.03, 0.04, 0.01, -0.01], index=idx)

    daily = _resample_to_daily_if_needed(returns, TimeFrame.H4)

    expected = returns.resample("D").sum().dropna(how="all")
    pd.testing.assert_series_equal(daily, expected)


def test_resample_to_daily_if_needed_keeps_daily_returns_unchanged() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, -0.01, 0.02], index=idx)

    out = _resample_to_daily_if_needed(returns, TimeFrame.D)

    pd.testing.assert_series_equal(out, returns)
