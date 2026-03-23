"""
Cyclical RSI Bias Node

DSP-based cycle analysis RSI. Uses a bandpass filter to extract the
dominant cycle from price data, then applies RSI to the cyclical component.

Output: Continuous 0-100 (RSI of cyclical component)
"""

from typing import ClassVar, List
import numpy as np
from collections import deque
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class CyclicalRSI(BiasNode):
    """
    Cyclical RSI - DSP-based cycle analysis RSI.

    Applies a bandpass filter to extract the cyclical component of price,
    then calculates RSI on this filtered data. This helps identify overbought/
    oversold conditions within the dominant market cycle.

    The bandpass filter is implemented as a simple moving average difference
    (approximation of a bandpass filter):
        cycle = short_sma - long_sma

    Then RSI is calculated on the cycle values.

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - short_period: Short SMA period for bandpass (default: 5)
    - long_period: Long SMA period for bandpass (default: 20)
    - rsi_period: RSI calculation period (default: 14)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"shortPeriod", "longPeriod", "rsiPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        short_period: int = 5,
        long_period: int = 20,
        rsi_period: int = 14
    ):
        super().__init__(ticker, tf)

        self.short_period = short_period
        self.long_period = long_period
        self.rsi_period = rsi_period

        # Standardized naming metadata
        self.module_name = 'cyclicalrsi'
        self.output_features = ['signal']
        self.params = {
            'shortPeriod': short_period,
            'longPeriod': long_period,
            'rsiPeriod': rsi_period
        }

        # Warmup needs enough data for long SMA + RSI
        self.front_bad = long_period + rsi_period

        # Price buffer for SMA calculations
        self.close_buffer = deque(maxlen=long_period)

        # Cycle buffer for RSI calculation
        self.cycle_buffer = deque(maxlen=rsi_period + 1)

        # RSI state
        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.prev_cycle = None
        self.rsi_initialized = False

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Cyclical RSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Cyclical RSI value (0.0-100.0)
        """
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        # Calculate cycle component (bandpass filter approximation)
        if len(self.close_buffer) >= self.long_period:
            closes = list(self.close_buffer)
            short_sma = np.mean(closes[-self.short_period:])
            long_sma = np.mean(closes)
            cycle = short_sma - long_sma
            self.cycle_buffer.append(cycle)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            if len(self.cycle_buffer) > 0:
                self.prev_cycle = self.cycle_buffer[-1]
            self.output.append(50.0)
            return [50.0]

        # Initialize RSI on first valid computation
        if not self.rsi_initialized and len(self.cycle_buffer) >= self.rsi_period:
            cycle_array = np.array(list(self.cycle_buffer)[:self.rsi_period])
            self.upsum, self.dnsum = self._compute_rsi_initial(cycle_array)
            self.rsi_initialized = True

        # Calculate RSI on cycle values
        if self.rsi_initialized and self.prev_cycle is not None:
            curr_cycle = self.cycle_buffer[-1]
            diff = curr_cycle - self.prev_cycle

            if diff > 0:
                self.upsum = ((self.rsi_period - 1) * self.upsum + diff) / self.rsi_period
                self.dnsum *= (self.rsi_period - 1.0) / self.rsi_period
            else:
                self.dnsum = ((self.rsi_period - 1) * self.dnsum - diff) / self.rsi_period
                self.upsum *= (self.rsi_period - 1.0) / self.rsi_period

            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_cycle = curr_cycle
        else:
            rsi = 50.0
            if len(self.cycle_buffer) > 0:
                self.prev_cycle = self.cycle_buffer[-1]

        # Ensure RSI is within bounds
        rsi = max(0.0, min(100.0, rsi))

        self.output.append(rsi)
        return [rsi]

    def _compute_rsi_initial(self, values: np.ndarray) -> tuple:
        """
        Initialize RSI computation for the first valid period.

        Parameters:
        - values: Array of values (cycle components)

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
