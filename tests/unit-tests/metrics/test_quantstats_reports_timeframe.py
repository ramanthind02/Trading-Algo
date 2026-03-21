from __future__ import annotations

import pandas as pd
import pytest

from metrics.plotting.graphing import quantstats_reports
from metrics.plotting.graphing.quantstats_reports import (
    _resample_to_daily_if_needed,
    generate_tearsheet,
)
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


def test_generate_tearsheet_skips_empty_strategy_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    class _Reports:
        @staticmethod
        def html(*args, **kwargs):  # noqa: ANN002, ANN003
            calls.append((args, kwargs))

    monkeypatch.setattr(quantstats_reports, "HAS_QUANTSTATS", True)
    monkeypatch.setattr(
        quantstats_reports,
        "qs",
        type("_QS", (), {"reports": _Reports})(),
    )

    empty_returns = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    with pytest.warns(UserWarning, match="strategy_returns is empty"):
        generate_tearsheet(
            strategy_returns=empty_returns,
            baseline_returns=empty_returns,
            feature_name="Empty Strategy",
            output_file="ignored.html",
            mode="html",
        )

    assert calls == []
