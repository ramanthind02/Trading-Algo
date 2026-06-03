"""Unit tests for calendar ensemble bias nodes."""

from __future__ import annotations

from datetime import datetime

from nodes.seasonal.calendar.fomc_drift import FomcDrift
from nodes.seasonal.calendar.pre_holiday_equity import PreHolidayEquity
from nodes.seasonal.calendar.pre_holiday_gold import PreHolidayGold
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(session: str, close: float = 100.0, ticker: Ticker = Ticker.ES) -> Candle:
    return Candle(
        ticker=ticker,
        tf=TimeFrame.D,
        datetime=datetime.strptime(session, "%Y-%m-%d"),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1.0,
    )


def test_pre_holiday_equity_fires_on_es_not_gc() -> None:
    es_node = PreHolidayEquity(Ticker.ES, TimeFrame.D)
    gc_node = PreHolidayGold(Ticker.GC, TimeFrame.D)
    thanksgiving_window = _candle("2024-11-27")
    assert es_node.add_candle(thanksgiving_window) == [1]
    assert gc_node.add_candle(_candle("2024-11-27", ticker=Ticker.GC)) == [0]


def test_fomc_drift_2024_june_window() -> None:
    node = FomcDrift(Ticker.ES, TimeFrame.D)
    assert node.add_candle(_candle("2024-06-10")) == [1]
    assert node.add_candle(_candle("2024-06-11")) == [1]
    assert node.add_candle(_candle("2024-06-12")) == [1]
    assert node.add_candle(_candle("2024-06-13")) == [0]


def test_fomc_drift_inactive_on_unrelated_ticker() -> None:
    node = FomcDrift(Ticker.CL, TimeFrame.D)
    cl_candle = _candle("2024-06-12", close=80.0, ticker=Ticker.CL)
    assert node.add_candle(cl_candle) == [0]


def test_pre_holiday_gold_christmas_window_includes_d_plus_one() -> None:
    node = PreHolidayGold(Ticker.GC, TimeFrame.D)
    assert node.add_candle(_candle("2024-12-24", ticker=Ticker.GC)) == [1]
    assert node.add_candle(_candle("2024-12-26", ticker=Ticker.GC)) == [1]
    assert node.add_candle(_candle("2024-12-13", ticker=Ticker.GC)) == [0]
