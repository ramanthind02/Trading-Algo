"""
Algomatic-style momentum signal (long-only).

Implements the article rules: Momentum(N)=Close−Close[N], entry on momentum
crossing above 0 with an RSI ceiling filter, exit when close exceeds the high
from exit_bars ago (article default N=10, RSI(2)<90, exit vs high five bars ago).
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List, Optional

import numpy as np

from nodes import BiasNode
from lib.compute.fast_nodes import compute_momentum_fast
from lib.compute.rsi_helpers import compute_rsi_initial, update_rsi
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class AlgomaticMomentumSignal(BiasNode):
    """
    Discrete long / flat signal matching the Algomatic 10-day momentum article.

    Entry (long): momentum crosses above 0 and RSI(rsi_period) < rsi_max.
    Exit: while long, close > high recorded exit_bars ago.

    Output: 1.0 long, 0.0 flat.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"momentumLookback", "rsiPeriod", "exitBars"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        momentum_lookback: int = 10,
        rsi_period: int = 2,
        rsi_max: float = 90.0,
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if momentum_lookback < 1:
            raise ValueError("momentum_lookback must be >= 1")
        if rsi_period < 2:
            raise ValueError("rsi_period must be >= 2")
        if not (0.0 < rsi_max <= 100.0):
            raise ValueError("rsi_max must be in (0, 100]")
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")

        self.momentum_lookback = momentum_lookback
        self.rsi_period = rsi_period
        self.rsi_max = rsi_max
        self.exit_bars = exit_bars

        self.module_name = "algomatic_momentum_signal"
        self.output_features = ["signal"]
        self.params = {
            "momentumLookback": momentum_lookback,
            "rsiPeriod": rsi_period,
            "rsiMax": rsi_max,
            "exitBars": exit_bars,
        }

        self.front_bad = max(
            momentum_lookback + 2,
            rsi_period + 1,
            exit_bars + 1,
        )

        self.close_buffer: deque[float] = deque(maxlen=momentum_lookback + 1)
        self.high_buffer: deque[float] = deque(maxlen=exit_bars + 1)

        self.n_prices = 0
        self.position = 0
        self.prev_momentum: Optional[float] = None
        self.prev_close = 0.0
        self.prev_rsi: Optional[float] = None

        self.buffer_size = rsi_period + 1
        self.close_buffer_rsi = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        curr_close = float(candle.close)
        curr_high = float(candle.high)

        self.close_buffer.append(curr_close)
        self.high_buffer.append(curr_high)

        self.close_buffer_rsi[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        rsi: Optional[float] = None
        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            self.prev_rsi = None
        elif self.n_prices == self.rsi_period:
            if self.buffer_idx == 0:
                init_prices = self.close_buffer_rsi[: self.rsi_period]
            else:
                init_prices = np.concatenate(
                    [
                        self.close_buffer_rsi[self.buffer_idx :],
                        self.close_buffer_rsi[: self.buffer_idx],
                    ]
                )
            self.upsum, self.dnsum = compute_rsi_initial(init_prices, self.rsi_period)
            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_close = curr_close
        else:
            self.upsum, self.dnsum, rsi = update_rsi(
                self.prev_close,
                curr_close,
                self.upsum,
                self.dnsum,
                self.rsi_period,
            )
            self.prev_close = curr_close

        momentum = 0.0
        if len(self.close_buffer) == self.momentum_lookback + 1:
            past_close = self.close_buffer[0]
            momentum = float(compute_momentum_fast(curr_close, past_close))

        signal = 0.0
        if (
            self.n_prices > self.front_bad
            and rsi is not None
            and self.prev_rsi is not None
        ):
            exit_long = self.position == 1 and curr_close > self.high_buffer[0]
            cross_up = (
                self.prev_momentum is not None
                and self.prev_momentum <= 0.0
                and momentum > 0.0
            )
            entry = self.position == 0 and cross_up and rsi < self.rsi_max

            if exit_long:
                self.position = 0
            elif entry:
                self.position = 1

            signal = float(self.position)

        if len(self.close_buffer) == self.momentum_lookback + 1:
            self.prev_momentum = momentum

        if rsi is not None:
            self.prev_rsi = rsi

        self.output.append(signal)
        return [signal]
