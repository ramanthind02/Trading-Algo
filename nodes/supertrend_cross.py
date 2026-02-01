"""
SuperTrend Cross Bias Node

Dual SuperTrend trend-following strategy. Generates signals based on
the crossing of two SuperTrend indicators with different parameters.

Output: Rule-based 0 or 1
"""

from typing import List
import numpy as np
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class SuperTrendCross(BiasNode):
    """
    SuperTrend Cross - Dual SuperTrend trend-following strategy.

    Uses two SuperTrend indicators with different multipliers:
    - Fast SuperTrend: lower multiplier (more sensitive)
    - Slow SuperTrend: higher multiplier (smoother)

    Signal Logic:
    - Signal = 1 (Long) when both SuperTrends are bullish (price above both bands)
    - Signal = 0 (Flat) otherwise

    SuperTrend Formula:
        Basic Upper = (High + Low) / 2 + Multiplier * ATR
        Basic Lower = (High + Low) / 2 - Multiplier * ATR

        Final Upper = Min(Basic Upper, Previous Final Upper) if Close > Previous Final Upper
        Final Lower = Max(Basic Lower, Previous Final Lower) if Close < Previous Final Lower

        Trend = Bullish if Close > Final Lower, Bearish if Close < Final Upper

    Output Range: 0 or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - atr_period: ATR period for SuperTrend (default: 10)
    - fast_multiplier: Multiplier for fast SuperTrend (default: 2.0)
    - slow_multiplier: Multiplier for slow SuperTrend (default: 3.0)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        atr_period: int = 10,
        fast_multiplier: float = 2.0,
        slow_multiplier: float = 3.0
    ):
        super().__init__(ticker, tf)

        self.atr_period = atr_period
        self.fast_multiplier = fast_multiplier
        self.slow_multiplier = slow_multiplier

        # Standardized naming metadata
        self.module_name = 'supertrendcross'
        self.output_features = ['signal']
        self.params = {
            'atrPeriod': atr_period,
            'fastMultiplier': fast_multiplier,
            'slowMultiplier': slow_multiplier
        }

        # Warmup needs ATR period
        self.front_bad = atr_period + 1

        # ATR calculation state
        self.tr_buffer = deque(maxlen=atr_period)
        self.prev_close = None

        # SuperTrend state for fast and slow
        self.fast_final_upper = None
        self.fast_final_lower = None
        self.fast_supertrend = None  # 1 = bullish, -1 = bearish

        self.slow_final_upper = None
        self.slow_final_lower = None
        self.slow_supertrend = None  # 1 = bullish, -1 = bearish

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute SuperTrend Cross signal for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the signal (0 or 1)
        """
        self.n_candles += 1
        curr_close = candle.close
        curr_high = candle.high
        curr_low = candle.low

        # Calculate True Range
        if self.prev_close is not None:
            tr = max(
                curr_high - curr_low,
                abs(curr_high - self.prev_close),
                abs(curr_low - self.prev_close)
            )
        else:
            tr = curr_high - curr_low

        self.tr_buffer.append(tr)
        self.prev_close = curr_close

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate ATR
        atr = np.mean(self.tr_buffer)

        # Calculate mid price
        mid = (curr_high + curr_low) / 2

        # Calculate basic bands
        fast_basic_upper = mid + self.fast_multiplier * atr
        fast_basic_lower = mid - self.fast_multiplier * atr
        slow_basic_upper = mid + self.slow_multiplier * atr
        slow_basic_lower = mid - self.slow_multiplier * atr

        # Update Fast SuperTrend
        if self.fast_final_upper is None:
            self.fast_final_upper = fast_basic_upper
            self.fast_final_lower = fast_basic_lower
            self.fast_supertrend = 1 if curr_close > mid else -1
        else:
            # Final Upper Band
            if fast_basic_upper < self.fast_final_upper or self.prev_close > self.fast_final_upper:
                self.fast_final_upper = fast_basic_upper
            # Final Lower Band
            if fast_basic_lower > self.fast_final_lower or self.prev_close < self.fast_final_lower:
                self.fast_final_lower = fast_basic_lower

            # Determine trend
            if self.fast_supertrend == 1:
                if curr_close < self.fast_final_lower:
                    self.fast_supertrend = -1
            else:
                if curr_close > self.fast_final_upper:
                    self.fast_supertrend = 1

        # Update Slow SuperTrend
        if self.slow_final_upper is None:
            self.slow_final_upper = slow_basic_upper
            self.slow_final_lower = slow_basic_lower
            self.slow_supertrend = 1 if curr_close > mid else -1
        else:
            # Final Upper Band
            if slow_basic_upper < self.slow_final_upper or self.prev_close > self.slow_final_upper:
                self.slow_final_upper = slow_basic_upper
            # Final Lower Band
            if slow_basic_lower > self.slow_final_lower or self.prev_close < self.slow_final_lower:
                self.slow_final_lower = slow_basic_lower

            # Determine trend
            if self.slow_supertrend == 1:
                if curr_close < self.slow_final_lower:
                    self.slow_supertrend = -1
            else:
                if curr_close > self.slow_final_upper:
                    self.slow_supertrend = 1

        # Generate signal: both SuperTrends must be bullish
        if self.fast_supertrend == 1 and self.slow_supertrend == 1:
            signal = 1
        else:
            signal = 0

        self.output.append(float(signal))
        return [float(signal)]
