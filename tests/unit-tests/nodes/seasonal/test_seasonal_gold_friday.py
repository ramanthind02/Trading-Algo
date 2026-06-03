"""Unit tests for seasonal gold Friday bias node."""

from __future__ import annotations

from datetime import datetime

from nodes.seasonal.weekday.seasonal_gold_friday import SeasonalGoldFriday
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(session: str, ticker: Ticker = Ticker.GC) -> Candle:
    close = 2000.0
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


def test_gc_long_on_friday_only() -> None:
    node = SeasonalGoldFriday(Ticker.GC, TimeFrame.D)
    assert node.add_candle(_candle("2024-05-17")) == [1]
    assert node.add_candle(_candle("2024-05-16")) == [0]


def test_non_gold_ticker_flat_even_on_friday() -> None:
    node = SeasonalGoldFriday(Ticker.ES, TimeFrame.D)
    assert node.add_candle(_candle("2024-05-17", ticker=Ticker.ES)) == [0]
