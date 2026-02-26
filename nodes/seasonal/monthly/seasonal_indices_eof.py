"""
Seasonal Indices End-of-Month (EOF) bias node.

Rule-based: long from 25th until 1st, with 200-period SMA filter for entry
and exit. Outputs 1 (long), 0 (flat). Exits when price dips below SMA200.
"""

from typing import List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


# Seasonal window: enter on or after 25th, exit on or before 1st
_ENTRY_DAY = 25
_EXIT_DAY = 1
_SMA_PERIOD = 200


class SeasonalIndicesEof(BiasNode):
    """
    Seasonal Indices EOF Bias Node (rule-based).

    Goes long on the 25th of the month (if close > 200-period SMA) and exits
    long on the 1st (on the 2nd the model outputs 0). Exits immediately if
    price dips below the 200-period SMA at any time.

    **Type**: Rule-based (outputs 1 or 0).

    Rules:
    - Enter long: day of month >= 25 and close > SMA(close, 200).
    - Exit long: day of month <= 1 (flat by 2nd) or close < SMA(close, 200).
    """

    def __init__(self, ticker: Ticker, tf: TimeFrame, sma_period: int = 200) -> None:
        super().__init__(ticker, tf)

        self.sma_period = sma_period
        self.module_name = "seasonalindiceseof"
        self.output_features = ["signal"]
        self.params = {"sma_period": sma_period}

        self.front_bad = sma_period

        # Circular buffer for SMA
        self._close_buffer = np.zeros(sma_period, dtype=np.float64)
        self._buffer_idx = 0
        self._n_prices = 0
        # Current position: 0 = flat, 1 = long
        self._position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        close = candle.close
        day = candle.datetime.day

        self._close_buffer[self._buffer_idx] = close
        self._buffer_idx = (self._buffer_idx + 1) % self.sma_period
        self._n_prices += 1

        if self._n_prices < self.front_bad:
            self.output.append(0)
            return [0]

        sma200 = float(np.mean(self._close_buffer))

        # Exit: seasonal (day <= 1) or below SMA
        if self._position == 1 and (day <= _EXIT_DAY or close < sma200):
            self._position = 0
        # Enter: day >= 25 and above SMA
        elif self._position == 0 and day >= _ENTRY_DAY and close > sma200:
            self._position = 1

        self.output.append(self._position)
        return [self._position]
