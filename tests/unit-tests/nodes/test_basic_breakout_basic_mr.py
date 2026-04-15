"""
Tests for BasicBreakout and BasicMR per docs/bias_nodes/to-do/basic_breakout_and_basic_mr.md.

Verifies: warmup, hold on inside bar, stack up to max_positions, opposite resets,
position mode (LONG_ONLY / SHORT_ONLY), and BasicMR negation.
"""

import unittest
from datetime import datetime, timedelta
from uuid import uuid4

from nodes.basic_breakout import BasicBreakout
from nodes.basic_mr import BasicMR
from utils.core.enums import PositionMode, Ticker, TimeFrame
from utils.core.models import Candle


def candle(
    dt: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    ticker: Ticker = Ticker.ES,
    tf: TimeFrame = TimeFrame.D,
) -> Candle:
    return Candle(
        id=uuid4(),
        datetime=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000_000,
        ticker=ticker,
        tf=tf,
    )


class TestBasicBreakout(unittest.TestCase):
    """Tests for BasicBreakout: hold, stack, opposite reset, position mode."""

    def test_warmup_returns_zero(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D)
        base = datetime(2020, 1, 1)
        # First bar: no prior -> warmup
        c0 = candle(base, 100, 105, 95, 102)
        out0 = node.add_candle(c0)
        self.assertEqual(out0, [0])
        self.assertEqual(node.output[-1], 0)

    def test_break_high_starts_long(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)  # warmup
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # close > prev_high 105
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1, [1])

    def test_inside_bar_holds_previous(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)   # warmup
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # break high -> 1
        c2 = candle(base + timedelta(days=2), 105, 107, 104, 106)  # inside (106 in [104,107], prev 108/99)
        node.add_candle(c0)
        node.add_candle(c1)
        out2 = node.add_candle(c2)
        self.assertEqual(out2, [1], "inside bar should hold long 1")

    def test_stack_same_direction_caps_at_max(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        # Bar0 warmup (prev_high=105, prev_low=95)
        c0 = candle(base, 100, 105, 95, 100)
        node.add_candle(c0)
        prev_high, prev_low = 105.0, 95.0
        for i in range(1, 8):
            # Each bar breaks above prev_high
            hi, lo = prev_high + 2, prev_low + 1
            close = hi + 0.5
            ci = candle(base + timedelta(days=i), 100, hi, lo, close)
            out = node.add_candle(ci)
            expected = min(i, 5)
            self.assertEqual(out[0], expected, f"bar {i} should stack to {expected}")
            prev_high, prev_low = hi, lo

    def test_opposite_breakout_resets_to_one(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # long 1
        c2 = candle(base + timedelta(days=2), 106, 107, 98, 94)    # close < prev_low 99 -> short 1
        node.add_candle(c0)
        node.add_candle(c1)
        out2 = node.add_candle(c2)
        self.assertEqual(out2, [-1])

    def test_strict_inequality_inside_bar_on_equals(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        # close == prev_high -> not above, so inside bar
        c1 = candle(base + timedelta(days=1), 100, 105, 95, 105)
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1[0], 0, "close == prev_high is inside bar (strict)")

    def test_long_only_clamps_negative_to_zero(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, mode=PositionMode.LONG_ONLY, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 94)  # break low -> would be -1
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1, [0])

    def test_short_only_clamps_positive_to_zero(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, mode=PositionMode.SHORT_ONLY, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # break high -> would be 1
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1, [0])

    def test_params_and_column_naming(self) -> None:
        node = BasicBreakout(Ticker.ES, TimeFrame.D, mode=PositionMode.LONG_SHORT, max_positions=5)
        self.assertEqual(node.module_name, "basicbreakout")
        self.assertEqual(node.output_features, ["signal"])
        self.assertEqual(node.params["mode"], PositionMode.LONG_SHORT)
        self.assertEqual(node.params["max_positions"], 5)
        self.assertEqual(node.front_bad, 2)
        names = node.get_column_names()
        self.assertTrue(len(names) == 1)
        self.assertIn("basicbreakout", names[0])
        self.assertIn("signal", names[0])


class TestBasicMR(unittest.TestCase):
    """Tests for BasicMR: negates BasicBreakout, same hold/stack, position mode."""

    def test_negates_breakout_output(self) -> None:
        node = BasicMR(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # breakout long 1 -> MR -1
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1, [-1])

    def test_mr_stacks_negated(self) -> None:
        node = BasicMR(Ticker.ES, TimeFrame.D, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        node.add_candle(c0)
        prev_high, prev_low = 105.0, 95.0
        for i in range(1, 4):
            hi, lo = prev_high + 2, prev_low + 1
            close = hi + 0.5
            ci = candle(base + timedelta(days=i), 100, hi, lo, close)
            out = node.add_candle(ci)
            self.assertEqual(out[0], -min(i, 5), f"MR bar {i} negated stack")
            prev_high, prev_low = hi, lo

    def test_mr_long_only_clamps_negative(self) -> None:
        # Breakout gives long (positive); MR negates to negative; LONG_ONLY clamps to 0
        node = BasicMR(Ticker.ES, TimeFrame.D, mode=PositionMode.LONG_ONLY, max_positions=5)
        base = datetime(2020, 1, 1)
        c0 = candle(base, 100, 105, 95, 100)
        c1 = candle(base + timedelta(days=1), 100, 108, 99, 106)  # breakout 1 -> MR -1 -> clamp 0
        node.add_candle(c0)
        out1 = node.add_candle(c1)
        self.assertEqual(out1, [0])

    def test_mr_params_and_columns(self) -> None:
        node = BasicMR(Ticker.ES, TimeFrame.D, max_positions=5)
        self.assertEqual(node.module_name, "basic_mr")
        self.assertEqual(node.params["max_positions"], 5)
        names = node.get_column_names()
        self.assertIn("basic_mr", names[0])


if __name__ == "__main__":
    unittest.main()
