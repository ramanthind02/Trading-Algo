"""
Detrended RSI Bias Node

RSI calculated on detrended prices (removes trend bias). Uses a moving
average to remove the trend component, then applies RSI to the detrended prices.

Output: Continuous 0-100
"""

from typing import List
import numpy as np
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class DetrendedRSI(BiasNode):
    """
    Detrended RSI - RSI on detrended prices.

    Removes the trend component by subtracting a moving average from price,
    then calculates RSI on the detrended values. This isolates the oscillating
    component of price, making RSI more effective at identifying overbought/
    oversold conditions without trend interference.

    Formula:
        detrended_price = close - SMA(close, detrend_period)
        detrended_rsi = RSI(detrended_price, rsi_period)

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - detrend_period: Period for trend removal SMA (default: 50)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        detrend_period: int = 50
    ):
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.detrend_period = detrend_period

        # Standardized naming metadata
        self.module_name = 'detrendedrsi'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'detrendPeriod': detrend_period
        }

        # Warmup needs detrend period + RSI period
        self.front_bad = detrend_period + rsi_period

        # Price buffer for detrending
        self.close_buffer = deque(maxlen=detrend_period)

        # Detrended values buffer for RSI
        self.detrended_buffer = deque(maxlen=rsi_period + 1)

        # RSI state
        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.prev_detrended = None
        self.rsi_initialized = False

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Detrended RSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Detrended RSI value (0.0-100.0)
        """
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        # Calculate detrended price
        if len(self.close_buffer) >= self.detrend_period:
            sma = np.mean(self.close_buffer)
            detrended = curr_close - sma
            self.detrended_buffer.append(detrended)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            if len(self.detrended_buffer) > 0:
                self.prev_detrended = self.detrended_buffer[-1]
            self.output.append(50.0)
            return [50.0]

        # Initialize RSI on first valid computation
        if not self.rsi_initialized and len(self.detrended_buffer) >= self.rsi_period:
            detrended_array = np.array(list(self.detrended_buffer)[:self.rsi_period])
            self.upsum, self.dnsum = self._compute_rsi_initial(detrended_array)
            self.rsi_initialized = True

        # Calculate RSI on detrended values
        if self.rsi_initialized and self.prev_detrended is not None:
            curr_detrended = self.detrended_buffer[-1]
            diff = curr_detrended - self.prev_detrended

            if diff > 0:
                self.upsum = ((self.rsi_period - 1) * self.upsum + diff) / self.rsi_period
                self.dnsum *= (self.rsi_period - 1.0) / self.rsi_period
            else:
                self.dnsum = ((self.rsi_period - 1) * self.dnsum - diff) / self.rsi_period
                self.upsum *= (self.rsi_period - 1.0) / self.rsi_period

            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_detrended = curr_detrended
        else:
            rsi = 50.0
            if len(self.detrended_buffer) > 0:
                self.prev_detrended = self.detrended_buffer[-1]

        # Ensure RSI is within bounds
        rsi = max(0.0, min(100.0, rsi))

        self.output.append(rsi)
        return [rsi]

    def _compute_rsi_initial(self, values: np.ndarray) -> tuple:
        """
        Initialize RSI computation for the first valid period.

        Parameters:
        - values: Array of detrended values

        Returns:
        - tuple: (upsum, dnsum) as averages
        """
        upsum = 1e-60
        dnsum = 1e-60

        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if diff > 0:
                upsum += diff
            else:
                dnsum -= diff

        # Convert to average
        upsum /= (len(values) - 1)
        dnsum /= (len(values) - 1)

        return upsum, dnsum
