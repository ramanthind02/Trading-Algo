"""
Double 7s Bias Node

Larry Connors' mean-reversion strategy. Generates binary long signals
when price is at 7-day lows and above the 200-day moving average.

Output: Rule-based 0 or 1
"""

from typing import List
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class Double7s(BiasNode):
    """
    Double 7s - Larry Connors mean-reversion strategy.

    A classic mean-reversion strategy that goes long when:
    1. Price closes at a 7-day low (below all closes in last 7 days)
    2. Price is above the 200-day moving average (uptrend filter)

    Exit when price closes at a 7-day high.

    This strategy is designed for equity indices and works on the principle
    that short-term pullbacks in an uptrend present buying opportunities.

    Output Range: 0 or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - short_period: Period for high/low lookback (default: 7)
    - ma_period: Period for trend filter MA (default: 200)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        short_period: int = 7,
        ma_period: int = 200
    ):
        super().__init__(ticker, tf)

        self.short_period = short_period
        self.ma_period = ma_period

        # Standardized naming metadata
        self.module_name = 'double7s'
        self.output_features = ['signal']
        self.params = {
            'shortPeriod': short_period,
            'maPeriod': ma_period
        }

        # Warmup needs enough data for MA
        self.front_bad = ma_period

        # Price buffers
        self.close_buffer = deque(maxlen=short_period + 1)
        self.ma_buffer = deque(maxlen=ma_period)
        self.ma_sum = 0.0

        self.n_candles = 0

        # Current position state (0 = flat, 1 = long)
        self.current_position = 0

        # Define standardized columns
        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Double 7s signal for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the signal (0 or 1)
        """
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        # Update MA using running sum for efficiency
        if len(self.ma_buffer) >= self.ma_period:
            old_price = self.ma_buffer[0]
            self.ma_sum = self.ma_sum - old_price + curr_close
        else:
            self.ma_sum += curr_close
        self.ma_buffer.append(curr_close)

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate 200-day MA
        ma_200 = self.ma_sum / self.ma_period

        # Check if we have enough data for short-term analysis
        if len(self.close_buffer) <= self.short_period:
            self.output.append(float(self.current_position))
            return [float(self.current_position)]

        # Get historical closes for comparison
        closes = list(self.close_buffer)
        recent_closes = closes[:-1]  # Exclude current close for comparison

        # Check for 7-day low (current close below all recent closes)
        is_7day_low = curr_close < min(recent_closes)

        # Check for 7-day high (current close above all recent closes)
        is_7day_high = curr_close > max(recent_closes)

        # Above 200-day MA filter
        above_ma = curr_close > ma_200

        # Entry logic: 7-day low AND above MA
        if self.current_position == 0 and is_7day_low and above_ma:
            self.current_position = 1

        # Exit logic: 7-day high
        if self.current_position == 1 and is_7day_high:
            self.current_position = 0

        signal = float(self.current_position)
        self.output.append(signal)
        return [signal]
