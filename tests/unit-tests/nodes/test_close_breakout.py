"""Tests for CloseBreakout long-only close > previous close entry."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from uuid import uuid4

from nodes.close_breakout import CloseBreakout
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def candle(
    dt: datetime,
    close: float,
    *,
    ticker: Ticker = Ticker.ES,
    tf: TimeFrame = TimeFrame.D,
) -> Candle:
    return Candle(
        id=uuid4(),
        datetime=dt,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1_000_000,
        ticker=ticker,
        tf=tf,
    )


class TestCloseBreakout(unittest.TestCase):
    def test_warmup_returns_zero(self) -> None:
        node = CloseBreakout(Ticker.ES, TimeFrame.D)
        out = node.add_candle(candle(datetime(2020, 1, 1), 100.0))
        self.assertEqual(out, [0.0])

    def test_long_when_close_above_previous_close(self) -> None:
        node = CloseBreakout(Ticker.ES, TimeFrame.D)
        base = datetime(2020, 1, 1)
        node.add_candle(candle(base, 100.0))
        out = node.add_candle(candle(base + timedelta(days=1), 101.0))
        self.assertEqual(out, [1.0])

    def test_flat_when_close_below_previous_close(self) -> None:
        node = CloseBreakout(Ticker.ES, TimeFrame.D)
        base = datetime(2020, 1, 1)
        node.add_candle(candle(base, 100.0))
        node.add_candle(candle(base + timedelta(days=1), 101.0))
        out = node.add_candle(candle(base + timedelta(days=2), 99.0))
        self.assertEqual(out, [0.0])

    def test_flat_when_close_equals_previous_close(self) -> None:
        node = CloseBreakout(Ticker.ES, TimeFrame.D)
        base = datetime(2020, 1, 1)
        node.add_candle(candle(base, 100.0))
        node.add_candle(candle(base + timedelta(days=1), 101.0))
        out = node.add_candle(candle(base + timedelta(days=2), 101.0))
        self.assertEqual(out, [0.0])

    def test_standardized_column_name(self) -> None:
        node = CloseBreakout(Ticker.ES, TimeFrame.D)
        self.assertIn("close_breakout_signal_D", node.columns)


if __name__ == "__main__":
    unittest.main()
