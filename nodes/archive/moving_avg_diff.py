from typing import List
from utils.core.enums import TimeFrame, Ticker
from nodes.bias_nodes import BiasNode
from utils.core.models import Candle
import collections
import math


class MovingAverageDifference(BiasNode):
    """
    Computes the Quantifying Trend: Moving Average Difference indicator.
    This indicator normalizes the difference between a short-term and long-term
    moving average, accounting for market age, volatility, and price scales.
    """
    def __init__(self, ticker: Ticker, tf: TimeFrame, short_lookback: int, long_lookback: int):
        """
        Initializes the MovingAverageDifference indicator.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - short_lookback (int): The number of bars for the short-term moving average.
        - long_lookback (int): The number of bars for the long-term moving average.
        - lag (int): The number of bars to lag the long-term moving average.

        Returns: None
        """
        super().__init__(ticker, tf)

        if not (short_lookback > 0 and long_lookback > 0):
            raise ValueError("Lookback periods must be positive integers.")
        if short_lookback >= long_lookback:
            # While the formula uses fabs(diff) for the square root,
            # for a meaningful trend indicator where short MA typically reacts faster,
            # short_lookback is usually less than long_lookback.
            print("Warning: Short lookback period is not less than long lookback period. This may lead to unconventional indicator behavior.")

        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.lag = short_lookback 

        # History needs to accommodate the longest lookback + lag + 1 (for ATR's prev_close)
        # The earliest candle needed for ATR calculation is `current_idx - (long_lookback + lag)`.
        # So, we need `(long_lookback + lag) + 1` candles in total for that window,
        # plus one more for the previous close of the very first candle in that window.
        # This makes the minimum required history `long_lookback + lag + 2`.
        self.required_history_len = self.long_lookback + self.lag + 2
        self.candles_history = collections.deque(maxlen=self.required_history_len)

        self.columns = [f'ma_difference_{short_lookback}_{long_lookback}_{self.lag}']

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """
        Computes the cumulative distribution function (CDF) for the standard normal distribution.
        CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        This is used for compressing the indicator's distribution.
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the MA Difference indicator.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed MA Difference indicator value.
                       Returns [0.0] if there's not enough historical data.
        """
        self.candles_history.append(candle)

        if len(self.candles_history) < self.required_history_len:
            return [0.0] # Not enough data yet to perform all calculations

        # Create a list snapshot for indexed access for clarity in calculations
        current_candles_list = list(self.candles_history)
        
        # --- 1. Calculate SMA (Short-term Moving Average) ---
        sma_sum = 0.0
        # SMA uses the most recent `short_lookback` candles
        for i in range(1, self.short_lookback + 1):
            sma_sum += current_candles_list[-i].close
        sma = sma_sum / self.short_lookback

        # --- 2. Calculate LMA (Long-term Moving Average) ---
        lma_sum = 0.0
        # LMA is lagged: it uses `long_lookback` candles ending `self.lag` bars ago
        lma_start_index_in_list = len(current_candles_list) - self.lag - self.long_lookback
        lma_end_index_in_list = len(current_candles_list) - self.lag
        
        # These checks should be redundant if `required_history_len` is correctly set,
        # but added for robustness.
        if lma_start_index_in_list < 0 or lma_end_index_in_list > len(current_candles_list):
            return [0.0] 

        for i in range(lma_start_index_in_list, lma_end_index_in_list):
            lma_sum += current_candles_list[i].close
        lma = lma_sum / self.long_lookback

        # --- 3. Calculate ATR (Average True Range) ---
        # ATR is computed over a window of `long_lookback + lag` bars, ending at the current candle.
        atr_period = self.long_lookback + self.lag
        
        if atr_period == 0: # Should not happen if long_lookback > 0, but as a safeguard
            atr = 1.0 # Default or small positive value to avoid division by zero
        else:
            true_ranges = []
            # Loop from the start of the ATR window up to the current candle
            # The window starts at `len(current_candles_list) - atr_period`
            # The first candle in the loop needs `current_candles_list[i-1]` for `prev_close`
            # which means index `len(current_candles_list) - atr_period - 1` must be valid.
            # This is guaranteed by `required_history_len`.
            for i in range(len(current_candles_list) - atr_period, len(current_candles_list)):
                current_candle_for_tr = current_candles_list[i]
                previous_candle_for_tr = current_candles_list[i-1] 

                high_minus_low = current_candle_for_tr.high - current_candle_for_tr.low
                high_minus_prev_close = abs(current_candle_for_tr.high - previous_candle_for_tr.close)
                low_minus_prev_close = abs(current_candle_for_tr.low - previous_candle_for_tr.close)
                
                true_ranges.append(max(high_minus_low, high_minus_prev_close, low_minus_prev_close))
            
            atr = sum(true_ranges) / atr_period
            
            if atr == 0.0: # Prevent division by zero if ATR is exactly zero
                atr = 1.e-60 # Small epsilon

        # --- 4. Calculate 'diff_val' (center difference) ---
        diff_val = 0.5 * (self.long_lookback - 1.0) + self.lag
        diff_val -= 0.5 * (self.short_lookback - 1.0)
        
        # --- 5. Calculate 'denom_raw' (denominator for NormDiff) ---
        # As per C++ code, it's sqrt(fabs(diff)) * atr. Epsilon is added just before division.
        denom_raw = math.sqrt(abs(diff_val)) * atr
        
        # --- 6. Calculate NormDiff (Normalized MA Difference) ---
        # Add a small epsilon to the denominator to prevent division by zero
        norm_diff = (sma - lma) / (denom_raw + 1.e-60)

        # --- 7. Calculate final MA DIFFERENCE indicator ---
        # Apply the compressing transform as per Equation (4.8)
        ma_difference = 100.0 * self._normal_cdf(1.5 * norm_diff) - 50.0

        return [ma_difference]