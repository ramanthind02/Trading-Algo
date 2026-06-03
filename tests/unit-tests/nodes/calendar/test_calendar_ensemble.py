"""Unit tests for calendar ensemble composite node."""

from __future__ import annotations

from datetime import datetime

from nodes.seasonal.calendar.calendar_ensemble import CalendarEnsemble
from nodes.seasonal.calendar.fomc_drift import FomcDrift
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(session: str, ticker: Ticker = Ticker.ES) -> Candle:
    return Candle(
        ticker=ticker,
        tf=TimeFrame.D,
        datetime=datetime.strptime(session, "%Y-%m-%d"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )


def test_calendar_ensemble_or_holiday_and_fomc_es() -> None:
    node = CalendarEnsemble(Ticker.ES, TimeFrame.D)
    assert node.add_candle(_candle("2024-11-27")) == [1]
    assert node.add_candle(_candle("2024-06-12")) == [1]
    assert node.add_candle(_candle("2024-06-13")) == [0]


def test_calendar_ensemble_nq_uses_equity_holiday_path() -> None:
    node = CalendarEnsemble(Ticker.NQ, TimeFrame.D)
    assert node.add_candle(_candle("2024-11-27", ticker=Ticker.NQ)) == [1]


def test_calendar_ensemble_gc_uses_gold_holiday_path() -> None:
    node = CalendarEnsemble(Ticker.GC, TimeFrame.D)
    assert node.add_candle(_candle("2024-12-24", ticker=Ticker.GC)) == [1]


def test_fomc_drift_applies_to_nq_not_tlt() -> None:
    nq = FomcDrift(Ticker.NQ, TimeFrame.D)
    tlt = FomcDrift(Ticker.TLT, TimeFrame.D)
    assert nq.add_candle(_candle("2024-06-12", ticker=Ticker.NQ)) == [1]
    assert tlt.add_candle(
        Candle(
            ticker=Ticker.TLT,
            tf=TimeFrame.D,
            datetime=datetime(2024, 6, 12),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    ) == [0]
