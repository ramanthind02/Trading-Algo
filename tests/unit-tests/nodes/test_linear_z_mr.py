"""Unit tests for the LinearZMeanReversion bias node (blog AUD/NZD MR port).

Covers: warmup neutrality, sign of the signal vs the z-score, forecast-cap
clipping, zero-variance safety, the optional percent-ROC regime filter, and
taxonomy resolution via the catalog factory.
"""

import math
import unittest
from datetime import datetime, timedelta
from uuid import uuid4

from lib.core.enums import Ticker, TimeFrame
from lib.core.helpers import create_fresh_bias_node
from lib.core.models import Candle

from nodes.mean_reversion.linear_z_mr import LinearZMeanReversion


def _candles(closes, tf: TimeFrame = TimeFrame.D, ticker: Ticker = Ticker.AUDNZD):
    """Build candles from an explicit close-price list (only close is read)."""
    base = datetime(2020, 1, 1)
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        out.append(
            Candle(
                id=uuid4(),
                datetime=base + timedelta(days=i),
                open=c,
                high=c,
                low=c,
                close=c,
                volume=1_000_000,
                ticker=ticker,
                tf=tf,
            )
        )
    return out


def _run(node: LinearZMeanReversion, closes):
    return [node.add_candle(c)[0] for c in _candles(closes)]


class TestLinearZMeanReversion(unittest.TestCase):
    def test_warmup_returns_zero(self):
        node = LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=5)
        self.assertEqual(node.front_bad, 5)  # filter off
        out = _run(node, [98, 99, 100, 101, 102, 103, 104, 105])
        # First front_bad - 1 bars are warmup -> 0.0.
        self.assertEqual(out[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertNotEqual(out[4], 0.0)

    def test_short_when_above_mean(self):
        # Ramp up: last close is the window max -> z > 0 -> negative (short) signal.
        node = LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=5)
        out = _run(node, [98, 99, 100, 101, 102])
        # z = (102 - 100) / sqrt(2) = sqrt(2); signal = -z = -1.41421...
        self.assertLess(out[-1], 0.0)
        self.assertAlmostEqual(out[-1], -math.sqrt(2.0), places=4)

    def test_long_when_below_mean(self):
        # Ramp down: last close is the window min -> z < 0 -> positive (long) signal.
        node = LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=5)
        out = _run(node, [102, 101, 100, 99, 98])
        self.assertGreater(out[-1], 0.0)
        self.assertAlmostEqual(out[-1], math.sqrt(2.0), places=4)

    def test_cap_clipping(self):
        # One outlier among four equal closes gives |z| == 2 exactly; scale=2 -> raw=-4,
        # clipped to -cap.
        node = LinearZMeanReversion(
            Ticker.AUDNZD, TimeFrame.D, lookback=5, scale=2.0, cap=2.0
        )
        out = _run(node, [100, 100, 100, 100, 130])
        self.assertEqual(out[-1], -2.0)

    def test_flat_when_no_variance(self):
        node = LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=5)
        out = _run(node, [100, 100, 100, 100, 100, 100])
        self.assertEqual(out[-1], 0.0)

    def test_roc_filter_gates_signal(self):
        # A monotone ramp moves the SMA every bar, so a tiny roc_threshold flattens
        # the signal; with the filter off (None) the same bar trades.
        closes = list(range(1, 13))  # [1, 2, ..., 12]
        filtered = LinearZMeanReversion(
            Ticker.AUDNZD, TimeFrame.D, lookback=5, roc_threshold=1e-9
        )
        unfiltered = LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=5)

        out_filtered = _run(filtered, closes)
        out_open = _run(unfiltered, closes)

        self.assertEqual(filtered.front_bad, 6)  # +1 warmup bar for the SMA ROC
        self.assertEqual(out_filtered[-1], 0.0)
        self.assertNotEqual(out_open[-1], 0.0)
        self.assertLess(out_open[-1], 0.0)  # rising ramp -> short

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, lookback=1)
        with self.assertRaises(ValueError):
            LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, scale=0.0)
        with self.assertRaises(ValueError):
            LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, cap=-1.0)
        with self.assertRaises(ValueError):
            LinearZMeanReversion(Ticker.AUDNZD, TimeFrame.D, roc_threshold=0.0)

    def test_create_fresh_bias_node_resolves(self):
        node = create_fresh_bias_node(
            "linear_z_mr",
            Ticker.AUDNZD,
            TimeFrame.D,
            {"lookback": 100, "scale": 1.0, "cap": 2.0, "roc_threshold": 0.01},
        )
        self.assertIsInstance(node, LinearZMeanReversion)
        self.assertEqual(node.lookback, 100)
        self.assertEqual(node.roc_threshold, 0.01)


if __name__ == "__main__":
    unittest.main()
