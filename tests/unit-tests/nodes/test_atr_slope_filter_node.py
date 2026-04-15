"""Unit tests for AtrSlopeFilterNode (ATR vs prior-bar SMA gate)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from nodes.volatility.atr.atr_slope_filter import AtrSlopeDirection, AtrSlopeFilterNode
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, o: float, h: float, low: float, c: float) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_warmup_first_period_bars_zero() -> None:
    period = 4
    node = AtrSlopeFilterNode(Ticker.ES, TimeFrame.D, period=period, direction=AtrSlopeDirection.RISING)
    outs: list[float] = []
    for i in range(period + 3):
        r = node.add_candle(_candle(i, o=100.0, h=101.0, low=99.0, c=100.0))
        outs.extend(r)
    assert all(v == 0.0 for v in outs[:period])


def test_constant_range_never_rises() -> None:
    period = 2
    node = AtrSlopeFilterNode(Ticker.ES, TimeFrame.D, period=period, direction=AtrSlopeDirection.RISING)
    last: List[float] = []
    for i in range(30):
        last = node.add_candle(_candle(i, o=100.0, h=101.0, low=99.0, c=100.0))
    assert last == [0.0]


def test_period_one_rising_when_range_widens() -> None:
    node = AtrSlopeFilterNode(Ticker.ES, TimeFrame.D, period=1, direction=AtrSlopeDirection.RISING)
    node.add_candle(_candle(0, o=100.0, h=101.0, low=99.0, c=100.0))
    last = node.add_candle(_candle(1, o=100.0, h=102.0, low=98.0, c=100.0))
    assert last == [1.0]


def test_period_one_falling_when_range_narrows() -> None:
    node = AtrSlopeFilterNode(Ticker.ES, TimeFrame.D, period=1, direction=AtrSlopeDirection.FALLING)
    node.add_candle(_candle(0, o=100.0, h=102.0, low=98.0, c=100.0))
    last = node.add_candle(_candle(1, o=100.0, h=101.0, low=99.0, c=100.0))
    assert last == [1.0]


def test_create_fresh_bias_node_string_direction() -> None:
    node = helpers.create_fresh_bias_node(
        "atr_slope_filter",
        Ticker.ES,
        TimeFrame.D,
        {"period": 3, "direction": "down"},
    )
    assert isinstance(node, AtrSlopeFilterNode)
    assert node.direction is AtrSlopeDirection.FALLING
