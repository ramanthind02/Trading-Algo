"""
Donchian Breakout Signal — long/short momentum with time exit.

Enters long when bar high breaks above the prior N-bar highest high,
short when bar low breaks below the prior N-bar lowest low.
Exits to flat after ``exit_bars`` bars (time exit only — no opposite
threshold exit, keeping the parameter count minimal).

This is the structural equivalent of CaseyPercentCSignal with
``negate_signal=True``: a breakout entry held for a fixed window,
but with no ATR/EMA/PercentC normalization layer. Fewer parameters,
same momentum regime assumption.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

try:
    from lib.compute.fast_nodes import compute_high_low_channel_fast, CYTHON_NODES_AVAILABLE
except ImportError:
    CYTHON_NODES_AVAILABLE = False


class DonchianBreakoutSignal(BiasNode):
    """
    Donchian channel breakout: long/short entry on N-bar high/low break, time exit.

    Parameters
    ----------
    lookback
        Number of prior bars used for the highest-high / lowest-low channel.
        Current bar is excluded (breakout is vs the *prior* N bars).
    exit_bars
        Maximum bars to hold a position before forcing flat.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 10,
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")

        self.lookback = lookback
        self.exit_bars = exit_bars

        self.module_name = "donchian_breakout_signal"
        self.output_features = ["signal"]
        self.params = {"lookback": lookback, "exit_bars": exit_bars}

        self.front_bad = lookback + 1

        if CYTHON_NODES_AVAILABLE:
            import numpy as np
            self.highs = np.zeros(self.front_bad, dtype=float)
            self.lows = np.zeros(self.front_bad, dtype=float)
            self.buffer_idx = 0
            self.n_filled = 0
        else:
            self._candles: deque[Candle] = deque(maxlen=self.front_bad)

        self.position = 0
        self.bars_in_position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _channel(self) -> tuple[float, float]:
        """Return (highest_high, lowest_low) over the prior ``lookback`` bars."""
        if CYTHON_NODES_AVAILABLE:
            if self.n_filled < self.front_bad:
                return 0.0, 0.0
            prev_idx = (self.buffer_idx - 2 + self.front_bad) % self.front_bad
            return compute_high_low_channel_fast(
                self.highs, self.lows, prev_idx, self.lookback, self.n_filled
            )
        if len(self._candles) < self.front_bad:
            return 0.0, 0.0
        prior = list(self._candles)[-(self.lookback + 1):-1]
        return max(c.high for c in prior), min(c.low for c in prior)

    def _compute_candle(self, candle: Candle) -> List[float]:
        if CYTHON_NODES_AVAILABLE:
            self.highs[self.buffer_idx] = candle.high
            self.lows[self.buffer_idx] = candle.low
            self.buffer_idx = (self.buffer_idx + 1) % self.front_bad
            self.n_filled = min(self.n_filled + 1, self.front_bad)
            ready = self.n_filled >= self.front_bad
        else:
            self._candles.append(candle)
            ready = len(self._candles) >= self.front_bad

        if not ready:
            self.output.append(0.0)
            return [0.0]

        highest_high, lowest_low = self._channel()

        # Time exit: force flat once hold limit reached
        if self.position != 0 and self.bars_in_position >= self.exit_bars:
            self.position = 0
            self.bars_in_position = 0

        if self.position == 0:
            if float(candle.high) > highest_high:
                self.position = 1
                self.bars_in_position = 1
            elif float(candle.low) < lowest_low:
                self.position = -1
                self.bars_in_position = 1
        else:
            self.bars_in_position += 1

        self.output.append(float(self.position))
        return [float(self.position)]
