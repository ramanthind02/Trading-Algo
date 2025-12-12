from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque

# Try to import Cython optimizations, fall back to pure Python if not available
try:
    from utils.cython_optimized import compute_ma_diff_fast
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    from scipy.stats import norm

class MADiffNode(BiasNode):
    """
    MADiffNode computes the difference between the current close price and a moving average,
    normalized by ATR and scaled to be centered around 0.
    
    This feature measures how far the current price is from its moving average, adjusted for
    volatility. The output is transformed through a normal CDF and scaled to [-50, 50] range.
    
    Formula:
    1. Compute log MA over lookback period
    2. Compute difference: (log(close) - log_ma)
    3. Normalize by ATR * sqrt(lookback + 1)
    4. Transform through normal CDF and scale: 100 * norm_cdf(diff) - 50
    
    This centers the output around 0 with typical range of [-50, 50].
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 20, atr_length: int = 252, compression: float = 1.0):
        """
        Initializes the MADiffNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lookback (int): The lookback period for moving average (default: 20).
        - atr_length (int): The period for ATR calculation (default: 252 for daily data).
        - compression (float): Compression factor for the output (default: 1.0).
                              Increase for more compression, decrease for less.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.lookback = lookback
        self.atr_length = atr_length
        self.compression = compression
        
        # Standardized naming metadata
        self.module_name = 'ma_diff'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Storage for historical data
        self.log_closes = deque(maxlen=lookback)
        self.prev_close: Optional[float] = None
        self.true_ranges = deque(maxlen=atr_length)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = max(lookback, atr_length)
        
        # Define standardized columns (will be: ma_diff_signal_D_lookback_50)
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the MA difference feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the normalized MA difference value centered around 0
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
        
        # Add current log close to history
        log_close = np.log(candle.close)
        self.log_closes.append(log_close)
        
        # Update previous close
        self.prev_close = candle.close
        
        # If we don't have enough data, return neutral value (0.0)
        if self.candle_count <= self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Use Cython-optimized computation if available (15-25x faster)
        if CYTHON_AVAILABLE:
            output_value = compute_ma_diff_fast(
                log_close,
                self.log_closes,
                self.true_ranges,
                self.lookback,
                self.compression
            )
        else:
            # Fallback to pure Python implementation
            # Compute moving average of log closes (excluding current candle)
            log_ma = np.mean(list(self.log_closes)[:-1])  # Use all but the last (current) value
            
            # Compute ATR
            atr = np.mean(self.true_ranges) if len(self.true_ranges) > 0 else 0.0
            
            # Compute normalized difference
            if atr > 0.0:
                denom = atr * np.sqrt(self.lookback + 1.0)
                diff = (log_close - log_ma) / denom
                
                # Transform through normal CDF and scale to [-50, 50]
                output_value = 100.0 * norm.cdf(self.compression * diff) - 50.0
            else:
                output_value = 0.0
        
        # Update bias based on output value
        if output_value > 0:
            self.bias = Bias.BULLISH
        elif output_value < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [output_value, bool(output_value > 0)]
        
        return self.output
