"""Unit tests for :class:`~nodes.regime.sma.sma_below_filter.SmaBelowFilterNode`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.regime.sma.sma_below_filter import SmaBelowFilterNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _candle(close: float, day: int) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_sma_below_gate_opens_when_close_below_sma() -> None:
    node = SmaBelowFilterNode(Ticker.ES, TimeFrame.D, period=3)
    # Prices: 100, 100, 100 -> SMA 100; next close 99 -> below SMA -> 1.0
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(99.0, 3))
    assert out[0] == pytest.approx(1.0)


def test_sma_below_gate_closed_when_close_above_sma() -> None:
    node = SmaBelowFilterNode(Ticker.ES, TimeFrame.D, period=3)
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(101.0, 3))
    assert out[0] == pytest.approx(0.0)
