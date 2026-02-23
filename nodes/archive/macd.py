import math
from typing import List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class MACD(BiasNode):
    """
    MACD (Moving Average Convergence Divergence) Bias Node
    
    Computes the difference between short-term and long-term exponential moving averages,
    normalized by ATR (Average True Range) and transformed through a normal CDF.
    Optionally applies smoothing and computes the difference from the smoothed signal.
    
    The calculation:
    1. Compute short-term and long-term exponential moving averages of close prices
    2. Calculate the difference (short - long)
    3. Normalize by ATR * sqrt(|center_long - center_short|)
    4. Transform: 100 * normal_cdf(normalized_value) - 50
    5. If n_to_smooth > 1, smooth the result and compute difference from smoothed signal
    
    Parameters:
    - short_length: Period for short-term EMA (default: 12)
    - long_length: Period for long-term EMA (default: 26)
    - n_to_smooth: Period for signal line smoothing (default: 9, set to 1 for no smoothing)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, short_length: int = 12, 
                 long_length: int = 26, n_to_smooth: int = 9):
        """
        Initialize MACD node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - short_length: Period for short-term EMA (default: 12)
        - long_length: Period for long-term EMA (default: 26)
        - n_to_smooth: Period for signal line smoothing (default: 9)
        """
        super().__init__(ticker, tf)
        
        self.short_length = short_length
        self.long_length = long_length
        self.n_to_smooth = n_to_smooth
        
        # Number of candles needed before we can compute valid output
        self.front_bad = long_length + n_to_smooth
        
        # EMA computation state
        self.long_alpha = 2.0 / (long_length + 1.0)
        self.short_alpha = 2.0 / (short_length + 1.0)
        self.long_sum = None
        self.short_sum = None
        
        # Smoothing state (for signal line)
        self.smooth_alpha = 2.0 / (n_to_smooth + 1.0) if n_to_smooth > 1 else 0.0
        self.smoothed = None
        
        # Price history for computation
        self.open_prices = []
        self.high_prices = []
        self.low_prices = []
        self.close_prices = []
        
        # Raw MACD values before smoothing/differencing
        self.raw_macd = []
        
        # Column name for output
        self.columns = [f'macd_{short_length}_{long_length}_{n_to_smooth}']
    
    def _compute_atr(self, index: int, length: int) -> float:
        """
        Compute Average True Range at the given index
        
        Parameters:
        - index: The index to compute ATR for
        - length: The lookback period for ATR
        
        Returns:
        - ATR value
        """
        if index < length:
            length = index  # Use what we have if not enough data
        
        if length <= 0:
            return 0.0
        
        tr_sum = 0.0
        for i in range(index - length + 1, index + 1):
            if i == 0:
                # First candle: TR = high - low
                tr = self.high_prices[i] - self.low_prices[i]
            else:
                # TR = max(high - low, |high - prev_close|, |low - prev_close|)
                hl = self.high_prices[i] - self.low_prices[i]
                hc = abs(self.high_prices[i] - self.close_prices[i - 1])
                lc = abs(self.low_prices[i] - self.close_prices[i - 1])
                tr = max(hl, hc, lc)
            tr_sum += tr
        
        return tr_sum / length
    
    def _normal_cdf(self, x: float) -> float:
        """
        Compute the cumulative distribution function of the standard normal distribution
        
        Parameters:
        - x: Input value
        
        Returns:
        - CDF value between 0 and 1
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute MACD for the given candle
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the MACD value
        """
        self.open_prices.append(candle.open)
        self.high_prices.append(candle.high)
        self.low_prices.append(candle.low)
        self.close_prices.append(candle.close)
        
        n = len(self.close_prices)
        icase = n - 1  # Current index
        
        # Initialize on first candle
        if n == 1:
            self.long_sum = self.close_prices[0]
            self.short_sum = self.close_prices[0]
            self.raw_macd.append(0.0)
            self.output.append(0.0)
            return [0.0]
        
        # Compute long-term and short-term exponential smoothing
        self.long_sum = self.long_alpha * candle.close + (1.0 - self.long_alpha) * self.long_sum
        self.short_sum = self.short_alpha * candle.close + (1.0 - self.short_alpha) * self.short_sum
        
        # Compute the normalizing factor
        # Center of long block minus center of short block for random walk variance
        diff = 0.5 * (self.long_length - 1.0)
        diff -= 0.5 * (self.short_length - 1.0)
        denom = math.sqrt(abs(diff))
        
        # Determine ATR length (limited by available data)
        k = self.long_length + self.n_to_smooth
        if k > icase:
            k = icase
        
        # Multiply normalizing factor by ATR to get scaling factor
        denom *= self._compute_atr(icase, k)
        
        # Compute normalized MACD
        if denom > 1e-15:
            raw_value = (self.short_sum - self.long_sum) / denom
            # Transform through normal CDF and scale to [-50, 50]
            raw_value = 100.0 * self._normal_cdf(1.0 * raw_value) - 50.0
        else:
            raw_value = 0.0
        
        self.raw_macd.append(raw_value)
        
        # Apply smoothing and compute difference if requested
        if self.n_to_smooth > 1:
            if self.smoothed is None:
                # Initialize smoothed value
                self.smoothed = self.raw_macd[0]
            
            # Update smoothed signal line
            self.smoothed = self.smooth_alpha * raw_value + (1.0 - self.smooth_alpha) * self.smoothed
            
            # MACD histogram: difference between MACD and signal line
            result = raw_value - self.smoothed
        else:
            # No smoothing, just use raw MACD
            result = raw_value
        
        self.output.append(result)
        return [result]