"""
%B (Percent B) Bias Node

Bollinger Band mean-reversion strategy. Generates binary signals based on
price position relative to Bollinger Bands.

Output: Rule-based 0 or 1 (long signal when oversold, flat otherwise)
"""

from typing import List
import numpy as np
from collections import deque
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class PercentB(BiasNode):
    """
    %B (Percent B) - Bollinger Band mean-reversion strategy.

    %B measures where the current price is relative to the Bollinger Bands:
        %B = (Price - Lower Band) / (Upper Band - Lower Band)

    - %B = 0: Price is at the lower band
    - %B = 1: Price is at the upper band
    - %B < 0: Price is below the lower band (oversold)
    - %B > 1: Price is above the upper band (overbought)

    Trading Logic (mean-reversion):
    - Signal = 1 (Long) when %B < lower_threshold (oversold)
    - Signal = 0 (Flat) otherwise

    Output Range: 0 or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - period: Bollinger Band period (default: 20)
    - std_dev: Number of standard deviations for bands (default: 2.0)
    - lower_threshold: %B threshold for oversold (default: 0.0)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 20,
        std_dev: float = 2.0,
        lower_threshold: float = 0.0
    ):
        super().__init__(ticker, tf)

        self.period = period
        self.std_dev = std_dev
        self.lower_threshold = lower_threshold

        # Standardized naming metadata
        self.module_name = 'percentb'
        self.output_features = ['signal']
        self.params = {
            'period': period,
            'stdDev': std_dev,
            'lowerThreshold': lower_threshold
        }

        # Warmup period
        self.front_bad = period

        # Price buffer for Bollinger Band calculation
        self.close_buffer = deque(maxlen=period)
        self.n_candles = 0

        # Current position state
        self.current_signal = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute %B signal for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the signal (0 or 1)
        """
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate Bollinger Bands
        closes = np.array(self.close_buffer)
        sma = np.mean(closes)
        std = np.std(closes, ddof=1)  # Sample standard deviation

        upper_band = sma + self.std_dev * std
        lower_band = sma - self.std_dev * std

        # Calculate %B
        band_width = upper_band - lower_band
        if band_width > 0:
            percent_b = (curr_close - lower_band) / band_width
        else:
            percent_b = 0.5  # Neutral if no band width

        # Generate signal (mean-reversion logic)
        if percent_b < self.lower_threshold:
            self.current_signal = 1  # Oversold, go long
        else:
            self.current_signal = 0  # Not oversold, stay flat

        signal = float(self.current_signal)
        self.output.append(signal)
        return [signal]
