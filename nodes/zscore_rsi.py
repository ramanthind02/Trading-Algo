"""
Z-Score RSI Bias Node

RSI normalized by standard deviations from its mean. Shows how extreme
the current RSI is relative to its historical distribution.

Output: Continuous ~-3 to +3 (z-score)
"""

from typing import List
import numpy as np
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from utils.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


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

    Output Range: ~-3.0 to +3.0 (continuous, not clamped)
    Neutral Value: 0.0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - zscore_period: Lookback period for z-score calculation (default: 100)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        zscore_period: int = 100
    ):
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.zscore_period = zscore_period

        # Standardized naming metadata
        self.module_name = 'zscorersi'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'zscorePeriod': zscore_period
        }

        # Warmup needs RSI period + z-score period
        self.front_bad = rsi_period + zscore_period

        # RSI calculation state
        self.buffer_size = rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        # RSI history for z-score calculation
        self.rsi_history = deque(maxlen=zscore_period)

        # Running statistics for efficient z-score calculation
        self.rsi_sum = 0.0
        self.rsi_sum_sq = 0.0

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Z-Score RSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Z-Score RSI value (~-3.0 to +3.0)
        """
        self.n_candles += 1
        curr_close = candle.close

        # Store current close in circular buffer
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        # Calculate RSI
        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            rsi = 50.0
        elif self.n_prices == self.rsi_period:
            # Initialize RSI
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[:self.rsi_period]
            else:
                init_prices = np.concatenate([
                    self.close_buffer[self.buffer_idx:],
                    self.close_buffer[:self.buffer_idx]
                ])
            self.upsum, self.dnsum = compute_rsi_initial(init_prices, self.rsi_period)
            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_close = curr_close
        else:
            # Update RSI
            self.upsum, self.dnsum, rsi = update_rsi(
                self.prev_close,
                curr_close,
                self.upsum,
                self.dnsum,
                self.rsi_period
            )
            self.prev_close = curr_close

        # Update running statistics
        if len(self.rsi_history) >= self.zscore_period:
            old_rsi = self.rsi_history[0]
            self.rsi_sum -= old_rsi
            self.rsi_sum_sq -= old_rsi * old_rsi

        self.rsi_history.append(rsi)
        self.rsi_sum += rsi
        self.rsi_sum_sq += rsi * rsi

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate z-score using running statistics
        n = len(self.rsi_history)
        if n > 1:
            mean = self.rsi_sum / n
            variance = (self.rsi_sum_sq / n) - (mean * mean)

            if variance > 0:
                std = np.sqrt(variance)
                zscore = (rsi - mean) / std
            else:
                zscore = 0.0
        else:
            zscore = 0.0

        # Clamp to reasonable range (optional, for display purposes)
        zscore = max(-4.0, min(4.0, zscore))

        self.output.append(zscore)
        return [zscore]
