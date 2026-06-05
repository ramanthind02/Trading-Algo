from collections import deque
from typing import ClassVar, Deque, List

import numpy as np

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from lib.core.models import Candle


class LaggedRSI(BiasNode):
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"rsiPeriod", "lagPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        lagPeriod: int = 3,
    ) -> None:
        super().__init__(ticker, tf)
        if rsiPeriod < 2:
            raise ValueError("rsiPeriod must be >= 2")
        if lagPeriod < 1:
            raise ValueError("lagPeriod must be >= 1")

        self.rsi_period = rsiPeriod
        self.lag_period = lagPeriod

        self.module_name = "laggedrsi"
        self.output_features = ["signal"]
        self.params = {"rsiPeriod": rsiPeriod, "lagPeriod": lagPeriod}
        self.front_bad = rsiPeriod + lagPeriod

        self.buffer_size = rsiPeriod + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.n_candles = 0

        self.rsi_history: Deque[float] = deque(maxlen=lagPeriod + 1)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            self.output.append(50.0)
            return [50.0]

        if self.n_prices == self.rsi_period:
            start_idx = (self.buffer_idx - self.rsi_period) % self.buffer_size
            if start_idx < self.buffer_idx:
                init_prices = self.close_buffer[start_idx : self.buffer_idx]
            else:
                init_prices = np.concatenate(
                    [
                        self.close_buffer[start_idx:],
                        self.close_buffer[: self.buffer_idx],
                    ]
                )
            self.upsum, self.dnsum = compute_rsi_initial_fast(init_prices, self.rsi_period)
            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
        else:
            self.upsum, self.dnsum, rsi = update_rsi_fast(
                self.prev_close,
                curr_close,
                self.upsum,
                self.dnsum,
                self.rsi_period,
            )
        self.prev_close = curr_close
        self.rsi_history.append(rsi)

        if self.n_candles < self.front_bad or len(self.rsi_history) <= self.lag_period:
            self.output.append(50.0)
            return [50.0]

        lagged_rsi = self.rsi_history[0]
        self.output.append(lagged_rsi)
        return [lagged_rsi]
