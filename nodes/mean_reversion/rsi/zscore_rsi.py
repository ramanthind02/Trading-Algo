"""
Z-Score RSI Bias Node

RSI normalized by standard deviations from its mean. Shows how extreme
the current RSI is relative to its historical distribution.

Output: Continuous ~-3 to +3 (z-score)

The incremental engine :class:`ZScoreRSIEngine` is shared with
:class:`~nodes.mean_reversion.rsi.zscore_rsi_signal.ZScoreRSISignal`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, List

import numpy as np

from nodes import BiasNode
from utils.compute.rsi_helpers import compute_rsi_initial, update_rsi
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


@dataclass
class ZScoreRSIEngine:
    """
    Stateful RSI plus rolling-window z-score of RSI (same math as ZScoreRSI).
    """

    rsi_period: int
    zscore_period: int
    buffer_size: int = field(init=False)
    close_buffer: np.ndarray = field(init=False)
    buffer_idx: int = 0
    n_prices: int = 0
    prev_close: float = 0.0
    upsum: float = 1e-60
    dnsum: float = 1e-60
    rsi_history: deque[float] = field(init=False)
    rsi_sum: float = 0.0
    rsi_sum_sq: float = 0.0
    n_candles: int = 0

    def __post_init__(self) -> None:
        self.buffer_size = self.rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.rsi_history = deque(maxlen=self.zscore_period)

    @property
    def front_bad(self) -> int:
        return self.rsi_period + self.zscore_period

    def step(self, candle: Candle) -> float:
        """Return the current z-score (0.0 during warmup)."""
        self.n_candles += 1
        curr_close = candle.close

        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            rsi = 50.0
        elif self.n_prices == self.rsi_period:
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[: self.rsi_period]
            else:
                init_prices = np.concatenate(
                    [
                        self.close_buffer[self.buffer_idx :],
                        self.close_buffer[: self.buffer_idx],
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

        if len(self.rsi_history) >= self.zscore_period:
            old_rsi = self.rsi_history[0]
            self.rsi_sum -= old_rsi
            self.rsi_sum_sq -= old_rsi * old_rsi

        self.rsi_history.append(rsi)
        self.rsi_sum += rsi
        self.rsi_sum_sq += rsi * rsi

        if self.n_candles < self.front_bad:
            return 0.0

        n = len(self.rsi_history)
        if n > 1:
            mean = self.rsi_sum / n
            variance = (self.rsi_sum_sq / n) - (mean * mean)

            if variance > 0:
                std = float(np.sqrt(variance))
                zscore = (rsi - mean) / std
            else:
                zscore = 0.0
        else:
            zscore = 0.0

        return max(-4.0, min(4.0, zscore))


class ZScoreRSI(BiasNode):
    """
    Z-Score RSI - RSI normalized by standard deviations.

    Calculates standard RSI, then normalizes it using z-score based on
    its historical mean and standard deviation.

    Formula:
        rsi = standard RSI calculation
        zscore_rsi = (rsi - mean(historical_rsi)) / std(historical_rsi)

    Interpretation:
    - Z-Score > 2: RSI is unusually high (potential overbought)
    - Z-Score < -2: RSI is unusually low (potential oversold)
    - Z-Score near 0: RSI is near its historical average

    Output Range: ~-3.0 to +3.0 (continuous, clamped to [-4, 4] internally)
    Neutral Value: 0.0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - zscore_period: Lookback period for z-score calculation (default: 100)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"rsiPeriod", "zscorePeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        zscore_period: int = 100,
    ) -> None:
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.zscore_period = zscore_period

        self.module_name = "zscorersi"
        self.output_features = ["signal"]
        self.params = {
            "rsiPeriod": rsi_period,
            "zscorePeriod": zscore_period,
        }

        self.front_bad = rsi_period + zscore_period

        self._engine = ZScoreRSIEngine(rsi_period=rsi_period, zscore_period=zscore_period)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        zscore = self._engine.step(candle)
        self.output.append(zscore)
        return [zscore]
