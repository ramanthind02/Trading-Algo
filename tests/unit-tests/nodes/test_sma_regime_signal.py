"""Unit tests for :class:`~nodes.regime.sma.sma_regime_signal.SmaRegimeSignalNode`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.regime.sma.sma_regime_signal import SmaRegimeSignalNode
from lib.core.enums import PositionMode, Ticker, TimeFrame
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


def test_sma_regime_long_when_close_above_sma() -> None:
    node = SmaRegimeSignalNode(Ticker.ES, TimeFrame.D, period=3)
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(101.0, 3))
    assert out[0] == pytest.approx(1.0)


def test_sma_regime_short_when_close_below_sma() -> None:
    node = SmaRegimeSignalNode(Ticker.ES, TimeFrame.D, period=3)
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(99.0, 3))
    assert out[0] == pytest.approx(-1.0)


def test_sma_regime_long_only_zero_when_close_below_sma() -> None:
    node = SmaRegimeSignalNode(
        Ticker.ES, TimeFrame.D, period=3, mode=PositionMode.LONG_ONLY
    )
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(99.0, 3))
    assert out[0] == pytest.approx(0.0)


def test_sma_regime_warmup_emits_zero() -> None:
    node = SmaRegimeSignalNode(Ticker.ES, TimeFrame.D, period=5)
    out = node.add_candle(_candle(100.0, 0))
    assert out[0] == pytest.approx(0.0)
