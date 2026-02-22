from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque
from scipy.stats import norm

# Try to import Cython optimized version
try:
    from utils.fast_nodes import compute_high_low_channel_fast, CYTHON_NODES_AVAILABLE
except ImportError:
    CYTHON_NODES_AVAILABLE = False


class WilliamsRNode(BiasNode):
    """
    WilliamsRNode computes the Williams %R indicator, which measures where the current
    price is relative to the high-low range over a lookback period.
    
    Williams %R is a momentum indicator that identifies overbought and oversold conditions.
    It is similar to the Stochastic Oscillator but inverted and typically ranges from -100 to 0.
    
    This implementation normalizes the output and centers it around 0 for consistency with
    other mean-reversion indicators.
    
    Formula:
    1. Find highest high and lowest low over lookback period
    2. Compute: %R = (highest_high - close) / (highest_high - lowest_low) × -100
    3. Normalize by ATR for volatility adjustment
    4. Transform through normal CDF and scale to [-50, 50] range
    
    Traditional interpretation:
    - %R < -80: Oversold (potential buy signal)
    - %R > -20: Overbought (potential sell signal)
    
    Our normalized output:
    - Negative values: Oversold conditions (price near bottom of range)
    - Positive values: Overbought conditions (price near top of range)
    - Near 0: Neutral (price in middle of range)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14, 
                 atr_length: int = 14, compression: float = 1.0):
        """
        Initializes the WilliamsRNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lookback (int): The lookback period for high-low range (default: 14).
        - atr_length (int): The period for ATR calculation for normalization (default: 14).
        - compression (float): Compression factor for the output (default: 1.0).
                              Increase for more compression, decrease for less.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.lookback = lookback
        self.atr_length = atr_length
        self.compression = compression
        
        # Standardized naming metadata
        self.module_name = 'williamsr'
        self.output_features = ['signal']
        self.params = {'lookback': lookback, 'atr_length': atr_length, 'compression': compression}
        
        # Storage for high and low prices
        if CYTHON_NODES_AVAILABLE:
            # Cython path: use numpy arrays for 5-10x speedup
            self.high_prices_arr = np.zeros(lookback, dtype=np.float64)
            self.low_prices_arr = np.zeros(lookback, dtype=np.float64)
            self.high_low_idx = 0
            self.high_low_n_filled = 0
        else:
            # Fallback: use deques
            self.high_prices = deque(maxlen=lookback)
            self.low_prices = deque(maxlen=lookback)
        
        # Storage for ATR calculation
        self.prev_close: Optional[float] = None
        self.true_ranges = deque(maxlen=atr_length)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = max(lookback, atr_length)
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the Williams %R feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the normalized Williams %R value centered around 0
        """
        self.candle_count += 1
        
        # Calculate True Range for ATR
        if self.prev_close is not None:
            hl = candle.high - candle.low
            hc = abs(candle.high - self.prev_close)
            lc = abs(candle.low - self.prev_close)
            true_range = max(hl, hc, lc)
        else:
            true_range = candle.high - candle.low
        
        self.true_ranges.append(true_range)
        
        # Add current high and low to history
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: store in numpy arrays
            self.high_prices_arr[self.high_low_idx] = candle.high
            self.low_prices_arr[self.high_low_idx] = candle.low
            self.high_low_idx = (self.high_low_idx + 1) % self.lookback
            self.high_low_n_filled = min(self.high_low_n_filled + 1, self.lookback)
        else:
            # Fallback: use deques
            self.high_prices.append(candle.high)
            self.low_prices.append(candle.low)
        
        # Update previous close
        self.prev_close = candle.close
        
        # If we don't have enough data, return neutral value (0.0)
        if self.candle_count < self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Find highest high and lowest low over lookback period
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: use compute_high_low_channel_fast
            # After writing, high_low_idx points to the next write position
            # The current candle is at (high_low_idx - 1) % lookback
            # We want to include current candle, so start from there
            curr_idx = (self.high_low_idx - 1 + self.lookback) % self.lookback
            highest_high, lowest_low = compute_high_low_channel_fast(
                self.high_prices_arr,
                self.low_prices_arr,
                curr_idx,
                self.lookback,
                self.high_low_n_filled
            )
        else:
            # Fallback: use Python max/min
            highest_high = max(self.high_prices)
            lowest_low = min(self.low_prices)
        
        # Compute range
        price_range = highest_high - lowest_low
        
        # Compute ATR for normalization
        atr = np.mean(self.true_ranges) if len(self.true_ranges) > 0 else 0.0
        
        # Compute Williams %R
        if price_range > 0.0:
            # Traditional Williams %R: (highest_high - close) / range × -100
            # This gives values from -100 (at lowest_low) to 0 (at highest_high)
            williams_r = ((highest_high - candle.close) / price_range) * -100.0
            
            # Normalize: Convert to position in range (0 to 1), then center around 0
            # Position: 0 = at lowest_low, 1 = at highest_high
            position = (candle.close - lowest_low) / price_range
            
            # Center around 0.5 and scale: -0.5 to +0.5
            centered_position = position - 0.5
            
            # Normalize by ATR if available for volatility adjustment
            if atr > 0.0:
                # Scale by range/ATR to account for volatility
                volatility_factor = price_range / atr
                normalized_value = centered_position * volatility_factor
            else:
                normalized_value = centered_position * 10.0  # Default scaling
            
            # Transform through normal CDF and scale to [-50, 50]
            output_value = 100.0 * norm.cdf(self.compression * normalized_value) - 50.0
        else:
            # No range, return neutral
            output_value = 0.0
        
        # Update bias based on output value
        if output_value > 0:
            self.bias = Bias.BULLISH  # Price near top of range (overbought)
        elif output_value < 0:
            self.bias = Bias.BEARISH  # Price near bottom of range (oversold)
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output.append(output_value)
        
        return [output_value]