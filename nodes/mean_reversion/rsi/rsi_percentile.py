"""
RSI Percentile Bias Node

RSI ranked within its historical distribution. Provides context on whether
the current RSI reading is historically high or low.

Output: Continuous 0-100 (percentile rank of RSI)
"""

from typing import List
import numpy as np
from collections import deque
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from utils.compute.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


class RSIPercentile(BiasNode):
    """
    RSI Percentile - RSI ranked within historical distribution.

    Calculates standard RSI, then ranks the current RSI value against
    its historical distribution over a lookback period.

    This helps identify:
    - RSI = 30 that is at the 10th percentile (very oversold relative to history)
    - RSI = 30 that is at the 50th percentile (normal for this market)

    Formula:
        rsi = standard RSI calculation
        rsi_percentile = percentile_rank(rsi, historical_rsi_values) * 100

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - percentile_period: Lookback period for percentile ranking (default: 252)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        percentile_period: int = 252
    ):
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.percentile_period = percentile_period

        # Standardized naming metadata
        self.module_name = 'rsipercentile'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'percentilePeriod': percentile_period
        }

        # Warmup needs RSI period + percentile period
        self.front_bad = rsi_period + percentile_period

        # RSI calculation state
        self.buffer_size = rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        # RSI history for percentile calculation
        self.rsi_history = deque(maxlen=percentile_period)

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI Percentile for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the RSI Percentile value (0.0-100.0)
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

        # Store RSI in history
        self.rsi_history.append(rsi)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(50.0)
            return [50.0]

        # Calculate percentile rank
        if len(self.rsi_history) > 0:
            current_rsi = rsi
            historical = np.array(self.rsi_history)

            # Percentile rank: what percentage of values are less than current
            count_below = np.sum(historical < current_rsi)
            count_equal = np.sum(historical == current_rsi)

            # Use midpoint method for ties
            percentile = (count_below + 0.5 * count_equal) / len(historical) * 100
        else:
            percentile = 50.0

        # Ensure within bounds
        percentile = max(0.0, min(100.0, percentile))

        self.output.append(percentile)
        return [percentile]
