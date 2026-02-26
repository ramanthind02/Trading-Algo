from typing import List

import numpy as np

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from utils.core.models import Candle


class RSILeftTailPressure(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        leftTailLevel: float = 30.0,
    ) -> None:
        super().__init__(ticker, tf)
        if rsiPeriod < 2:
            raise ValueError("rsiPeriod must be >= 2")
        if not 0.0 < leftTailLevel < 100.0:
            raise ValueError("leftTailLevel must be between 0 and 100")

        self.rsi_period = rsiPeriod
        self.left_tail_level = leftTailLevel

        self.module_name = "rsilefttailpressure"
        self.output_features = ["signal"]
        self.params = {"rsiPeriod": rsiPeriod, "leftTailLevel": leftTailLevel}
        self.front_bad = rsiPeriod

        self.buffer_size = rsiPeriod + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.n_candles = 0

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
            self.output.append(0.0)
            return [0.0]

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

        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        pressure = max(0.0, (self.left_tail_level - rsi) / self.left_tail_level)
        pressure = min(1.0, pressure)
        self.output.append(pressure)
        return [pressure]
