"""
DeMark REI (Range Expansion Index) Bias Node

Tom DeMark's Range Expansion Index measures the relationship between
today's highs/lows and the highs/lows from two days ago.

Output: Continuous -100 to +100
"""

from typing import List
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class DemarkREI(BiasNode):
    """
    DeMark REI - Range Expansion Index.

    Tom DeMark's Range Expansion Index (REI) is a momentum oscillator
    that measures the relationship between current price extremes and
    those from two days ago, using conditional calculations.

    The REI identifies potential reversal points by measuring how
    the current day's range compares to the range from two days ago.

    Formula:
        Conditions for numerator:
        - If high >= low_2_days_ago and high >= high_2_days_ago:
            high_diff = high - high_2_days_ago
        - Else: high_diff = 0
        - If low <= high_2_days_ago and low <= low_2_days_ago:
            low_diff = low - low_2_days_ago
        - Else: low_diff = 0

        numerator = sum(high_diff + low_diff) over period
        denominator = sum(abs(high_diff) + abs(low_diff)) over period

        REI = 100 * (numerator / denominator) if denominator != 0 else 0

    Output Range: -100.0 to +100.0 (continuous)
    Neutral Value: 0.0 (during warmup)

    Parameters:
    - period: Lookback period for REI calculation (default: 5)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 5
    ):
        super().__init__(ticker, tf)

        self.period = period

        # Standardized naming metadata
        self.module_name = 'rei'
        self.output_features = ['signal']
        self.params = {'period': period}

        # Warmup: need at least 3 candles for 2-day lookback + period for averaging
        self.front_bad = period + 2

        # Candle buffer (need 3 candles: current, 1 day ago, 2 days ago)
        self.high_buffer = deque(maxlen=3)
        self.low_buffer = deque(maxlen=3)

        # REI component buffers
        self.num_buffer = deque(maxlen=period)
        self.denom_buffer = deque(maxlen=period)

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute DeMark REI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the REI value (-100.0 to +100.0)
        """
        self.n_candles += 1

        # Store current high/low
        self.high_buffer.append(candle.high)
        self.low_buffer.append(candle.low)

        # Need at least 3 candles for calculation
        if len(self.high_buffer) < 3:
            self.output.append(0.0)
            return [0.0]

        # Get values
        high = self.high_buffer[-1]          # Current high
        low = self.low_buffer[-1]            # Current low
        high_2 = self.high_buffer[-3]        # High from 2 days ago
        low_2 = self.low_buffer[-3]          # Low from 2 days ago

        # Calculate high difference (conditional)
        if high >= low_2 and high >= high_2:
            high_diff = high - high_2
        else:
            high_diff = 0.0

        # Calculate low difference (conditional)
        if low <= high_2 and low <= low_2:
            low_diff = low - low_2
        else:
            low_diff = 0.0

        # Store components for period averaging
        self.num_buffer.append(high_diff + low_diff)
        self.denom_buffer.append(abs(high_diff) + abs(low_diff))

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate REI
        numerator = sum(self.num_buffer)
        denominator = sum(self.denom_buffer)

        if denominator != 0:
            rei = 100.0 * (numerator / denominator)
        else:
            rei = 0.0

        # Ensure within bounds
        rei = max(-100.0, min(100.0, rei))

        self.output.append(rei)
        return [rei]
