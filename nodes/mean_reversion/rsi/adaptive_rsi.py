"""
Adaptive RSI Bias Node

RSI with volatility-adaptive lookback period. When volatility is high,
uses a shorter lookback period for faster response. When volatility is low,
uses a longer lookback period for smoother signals.

Output: Continuous 0-100 (same as standard RSI)
"""

from typing import List
import numpy as np
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from utils.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


class AdaptiveRSI(BiasNode):
    """
    Adaptive RSI - RSI with volatility-adaptive lookback period.

    The lookback period dynamically adjusts based on recent volatility:
    - High volatility → shorter period (faster response)
    - Low volatility → longer period (smoother signals)

    Uses ATR-based volatility measurement to determine the adaptive period.

    Formula:
        volatility_ratio = current_atr / average_atr
        adaptive_period = min_period + (max_period - min_period) * (1 - volatility_ratio)
        RSI = standard RSI calculation with adaptive_period

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - min_period: Minimum RSI lookback period (default: 2)
    - max_period: Maximum RSI lookback period (default: 14)
    - volatility_period: Period for ATR calculation (default: 20)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        min_period: int = 2,
        max_period: int = 14,
        volatility_period: int = 20
    ):
        super().__init__(ticker, tf)

        self.min_period = min_period
        self.max_period = max_period
        self.volatility_period = volatility_period

        # Standardized naming metadata
        self.module_name = 'adaptiversi'
        self.output_features = ['signal']
        self.params = {
            'minPeriod': min_period,
            'maxPeriod': max_period,
            'volatilityPeriod': volatility_period
        }

        # Warmup needs enough data for volatility calculation + max RSI period
        self.front_bad = volatility_period + max_period

        # Price and ATR buffers
        self.close_buffer = deque(maxlen=max_period + 1)
        self.tr_buffer = deque(maxlen=volatility_period)
        self.atr_history = deque(maxlen=volatility_period)

        # State tracking
        self.prev_close = None
        self.n_candles = 0

        # RSI state (will be recomputed each time due to adaptive period)
        self.current_period = max_period

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Adaptive RSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Adaptive RSI value (0.0-100.0)
        """
        self.n_candles += 1
        curr_close = candle.close

        # Calculate True Range
        if self.prev_close is not None:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self.prev_close),
                abs(candle.low - self.prev_close)
            )
        else:
            tr = candle.high - candle.low

        self.tr_buffer.append(tr)
        self.close_buffer.append(curr_close)

        # Calculate current ATR
        if len(self.tr_buffer) >= self.volatility_period:
            current_atr = np.mean(self.tr_buffer)
            self.atr_history.append(current_atr)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.prev_close = curr_close
            self.output.append(50.0)
            return [50.0]

        # Calculate adaptive period based on volatility
        if len(self.atr_history) >= 2:
            avg_atr = np.mean(self.atr_history)
            current_atr = self.atr_history[-1]

            if avg_atr > 0:
                # Volatility ratio: >1 means high vol, <1 means low vol
                vol_ratio = min(current_atr / avg_atr, 2.0)  # Cap at 2x

                # High volatility → shorter period, low volatility → longer period
                # Invert the ratio so high vol gives low period
                period_range = self.max_period - self.min_period
                adaptive_period = int(
                    self.max_period - (vol_ratio - 0.5) * period_range
                )
                adaptive_period = max(self.min_period, min(self.max_period, adaptive_period))
            else:
                adaptive_period = self.max_period
        else:
            adaptive_period = self.max_period

        self.current_period = adaptive_period

        # Calculate RSI with adaptive period
        if len(self.close_buffer) > adaptive_period:
            # Get the closes needed for RSI calculation
            closes = list(self.close_buffer)
            recent_closes = closes[-(adaptive_period + 1):]

            # Initialize RSI calculation
            prices_array = np.array(recent_closes[:-1], dtype=np.float64)
            upsum, dnsum = compute_rsi_initial(prices_array, adaptive_period)

            # Update with current close
            upsum, dnsum, rsi = update_rsi(
                recent_closes[-2],
                recent_closes[-1],
                upsum,
                dnsum,
                adaptive_period
            )
        else:
            rsi = 50.0

        # Ensure RSI is within bounds
        rsi = max(0.0, min(100.0, rsi))

        self.prev_close = curr_close
        self.output.append(rsi)
        return [rsi]
