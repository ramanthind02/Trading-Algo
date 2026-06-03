from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from metrics.plotting.graphing import quantstats_reports
from metrics.plotting.graphing.quantstats_reports import (
    _resample_to_daily_if_needed,
    compute_performance_report,
    generate_tearsheet,
    vol_scale_returns_to_target_annualized_volatility,
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


def test_vol_scale_returns_to_target_annualized_volatility_matches_target() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    daily = pd.Series(rng.normal(0.0, 0.012, size=len(idx)), index=idx)
    scaled = vol_scale_returns_to_target_annualized_volatility(
        daily,
        target_annual_volatility=0.10,
        bars_per_year=252,
    )
    realized = float(scaled.std(ddof=1)) * math.sqrt(252.0)
    assert abs(realized - 0.10) < 0.02


def test_generate_tearsheet_applies_target_annual_volatility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[pd.Series] = []

    class _Reports:
        @staticmethod
        def html(strategy: pd.Series, **_kwargs: object) -> None:
            captured.append(strategy)

    monkeypatch.setattr(quantstats_reports, "HAS_QUANTSTATS", True)
    monkeypatch.setattr(
        quantstats_reports,
        "qs",
        type("_QS", (), {"reports": _Reports})(),
    )

    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    rng = np.random.default_rng(7)
    strategy_returns = pd.Series(rng.normal(0.0, 0.02, size=len(idx)), index=idx)

    generate_tearsheet(
        strategy_returns=strategy_returns,
        feature_name="Scaled",
        output_file="out.html",
        mode="html",
        timeframe=TimeFrame.D,
        target_annual_volatility=0.10,
    )

    assert len(captured) == 1
    out = captured[0]
    realized = float(out.dropna().std(ddof=1)) * math.sqrt(252.0)
    assert abs(realized - 0.10) < 0.03


def test_compute_performance_report_uses_quantfoundry_core_contract() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    strategy = pd.Series([0.01, -0.005, 0.008, -0.002, 0.004, 0.003, -0.001, 0.002], index=idx)
    benchmark = pd.Series([0.006, -0.004, 0.005, -0.001, 0.002, 0.001, -0.002, 0.001], index=idx)

    report = compute_performance_report(
        strategy_returns=strategy,
        baseline_returns=benchmark,
        feature_name="Example",
        timeframe=TimeFrame.D,
    )

    assert report is not None
    assert report.mode.value == "full"
    assert report.column_order == ("Baseline (Always-In)", "Example")
    assert report.to_json_dict()["periods_per_year"] == TimeFrame.D.bars_per_year
