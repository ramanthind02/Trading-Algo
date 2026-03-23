"""
Stochastic RSI Bias Node

RSI normalized to its min/max range over a lookback period.
Combines RSI with Stochastic oscillator logic.

Output: Continuous 0-100
"""

from typing import ClassVar, List
import numpy as np
from collections import deque
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from utils.compute.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


class StochasticRSI(BiasNode):
    """
    Stochastic RSI - RSI normalized to min/max range.

    Applies the Stochastic formula to RSI values instead of price:
        StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low) * 100

    This creates an oscillator that:
    - Reaches 0 when RSI is at its lowest in the lookback period
    - Reaches 100 when RSI is at its highest in the lookback period

    The Stochastic RSI is more sensitive than regular RSI and generates
    more signals, especially in ranging markets.

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - stoch_period: Stochastic lookback period (default: 14)
    - smooth_k: Smoothing for %K line (default: 3)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"rsiPeriod", "stochPeriod", "smoothK"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3
    ):
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.smooth_k = smooth_k

        # Standardized naming metadata
        self.module_name = 'stochrsi'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'stochPeriod': stoch_period,
            'smoothK': smooth_k
        }

        # Warmup needs RSI period + stochastic period + smoothing
        self.front_bad = rsi_period + stoch_period + smooth_k

        # RSI calculation state
        self.buffer_size = rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        # RSI history for stochastic calculation
        self.rsi_history = deque(maxlen=stoch_period)

        # Stochastic RSI history for smoothing
        self.stoch_rsi_history = deque(maxlen=smooth_k)

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Stochastic RSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Stochastic RSI value (0.0-100.0)
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

        # Calculate Stochastic RSI if we have enough RSI history
        if len(self.rsi_history) >= self.stoch_period:
            rsi_array = np.array(self.rsi_history)
            rsi_low = np.min(rsi_array)
            rsi_high = np.max(rsi_array)

            # Calculate raw Stochastic RSI
            if rsi_high - rsi_low > 0:
                raw_stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
            else:
                raw_stoch_rsi = 50.0

            self.stoch_rsi_history.append(raw_stoch_rsi)
        else:
            self.stoch_rsi_history.append(50.0)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(50.0)
            return [50.0]

        # Smooth the Stochastic RSI
        if len(self.stoch_rsi_history) >= self.smooth_k:
            stoch_rsi = np.mean(list(self.stoch_rsi_history)[-self.smooth_k:])
        else:
            stoch_rsi = 50.0

        # Ensure within bounds
        stoch_rsi = max(0.0, min(100.0, stoch_rsi))

        self.output.append(stoch_rsi)
        return [stoch_rsi]
