from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from utils.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from nodes import BiasNode
from collections import deque


class RSIRegime(BiasNode):
    """
    RSI with Regime Filter Bias Node - Cython-accelerated
    
    Computes the RSI indicator centered from -100 to 100 (instead of 0-100)
    with an optional regime filter using a 200-period moving average.
    
    RSI Transformation:
    - Standard RSI: 0-100 range
    - Centered RSI: (RSI - 50) * 2, resulting in -100 to 100 range
    
    Regime Filter:
    - "off": Always output centered RSI
    - "bullish": Only output RSI when price > 200-period MA (output NaN otherwise)
    - "bearish": Only output RSI when price < 200-period MA (output NaN otherwise)
    
    This allows the RSI to only be active during specific market regimes,
    helping to filter out signals during unfavorable conditions.
    
    Performance: Uses Cython-backed fast kernels for RSI computation.
    Regime filter logic remains in Python (O(1) per candle).
    
    Parameters:
    - lookback: Period for RSI calculation (default: 14)
    - ma_period: Period for moving average regime filter (default: 200)
    - regime_filter: Regime filter mode as string - "off", "bullish", or "bearish" (default: "off")
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame, 
        lookback: int = 14,
        ma_period: int = 200,
        regime_filter: str = "off"
    ):
        """
        Initialize RSI with Regime Filter node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: RSI period (default: 14)
        - ma_period: Moving average period for regime filter (default: 200)
        - regime_filter: Regime filter mode as string - "off", "bullish", or "bearish" (default: "off")
        """
        super().__init__(ticker, tf)
        
        if not (lookback > 0):
            raise ValueError("RSI lookback period must be a positive integer.")
        if not (ma_period > 0):
            raise ValueError("Moving average period must be a positive integer.")
        
        # Normalize regime filter to lowercase for case-insensitive comparison
        regime_filter_lower = regime_filter.lower() if isinstance(regime_filter, str) else "off"
        if regime_filter_lower not in ["off", "bullish", "bearish"]:
            raise ValueError(f'Invalid regime_filter: "{regime_filter}". Must be "off", "bullish", or "bearish".')
        
        self.lookback = lookback
        self.ma_period = ma_period
        self.regime_filter = regime_filter_lower
        
        # Standardized naming metadata
        self.module_name = 'rsi_regime'
        self.output_features = ['signal']
        self.params = {
            'lookback': lookback,
            'ma_period': ma_period,
            'regime_filter': self.regime_filter
        }
        
        # Number of candles needed before we can compute valid output
        # Need max of RSI lookback and MA period
        self.front_bad = max(lookback, ma_period)
        
        # RSI computation state
        self.upsum = 1e-60  # Small value to avoid division by zero
        self.dnsum = 1e-60
        
        # OPTIMIZATION: Use circular buffer instead of growing array
        # We only need lookback+1 prices max (for initialization + current)
        self.buffer_size = lookback + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0  # Current position in circular buffer
        self.n_prices = 0    # Total number of prices seen
        
        # Store previous close for RSI update (more efficient than buffer access)
        self.prev_close = 0.0
        
        # Moving average calculation - use deque for efficient circular buffer
        self.ma_buffer = deque(maxlen=ma_period)
        self.ma_sum = 0.0  # Running sum for efficient MA calculation
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _update_moving_average(self, price: float) -> float:
        """
        Update the moving average efficiently using a running sum.
        
        Parameters:
        - price: Current closing price
        
        Returns:
        - Current moving average value
        """
        # If buffer is not full, just add to sum
        if len(self.ma_buffer) < self.ma_period:
            self.ma_buffer.append(price)
            self.ma_sum += price
            # Return average of what we have so far
            return self.ma_sum / len(self.ma_buffer) if len(self.ma_buffer) > 0 else price
        
        # Buffer is full, remove oldest value and add new one
        old_price = self.ma_buffer[0]
        self.ma_buffer.append(price)
        self.ma_sum = self.ma_sum - old_price + price
        
        return self.ma_sum / self.ma_period
    
    def _check_regime(self, current_price: float, ma_value: float) -> bool:
        """
        Check if current price is in the correct regime based on filter settings.
        
        Parameters:
        - current_price: Current closing price
        - ma_value: Current moving average value
        
        Returns:
        - True if regime condition is met (or filter is "off"), False otherwise
        """
        if self.regime_filter == "off":
            return True
        elif self.regime_filter == "bullish":
            return current_price > ma_value
        elif self.regime_filter == "bearish":
            return current_price < ma_value
        else:
            return True  # Default to allowing signal
    
    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Compute RSI with regime filter for the given candle.
        
        Returns centered RSI (-100 to 100) if regime condition is met,
        otherwise returns 0.0.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the centered RSI value or 0.0
        """
        # Store current close in circular buffer
        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1
        
        # Update moving average
        ma_value = self._update_moving_average(curr_close)
        
        # Return NaN until we have enough data for both RSI and MA
        # This ensures these warmup values are excluded from binning
        if self.n_prices < self.front_bad:
            self.prev_close = curr_close
            self.output.append(float('nan'))
            return [float('nan')]
        
        # Initialize RSI on the first valid computation
        if self.n_prices == self.front_bad:
            # Extract the lookback period from circular buffer for initialization
            # Buffer contains exactly lookback prices at this point
            if self.buffer_idx == 0:
                # Buffer filled exactly once, data is contiguous
                init_prices = self.close_buffer[:self.lookback]
            else:
                # Data wraps around (shouldn't happen on first fill, but handle it)
                init_prices = np.concatenate([
                    self.close_buffer[self.buffer_idx:],
                    self.close_buffer[:self.buffer_idx]
                ])
            
            # Call Cython-backed initialization
            self.upsum, self.dnsum = compute_rsi_initial_fast(init_prices, self.lookback)
        
        # Update RSI using Cython-backed function
        # Only need prev_close and curr_close (no array access needed!)
        self.upsum, self.dnsum, rsi = update_rsi_fast(
            self.prev_close,
            curr_close,
            self.upsum,
            self.dnsum,
            self.lookback
        )
        
        # Update prev_close for next iteration
        self.prev_close = curr_close
        
        # Center RSI from 0-100 to -100 to 100: (rsi - 50) * 2
        centered_rsi = (rsi - 50.0) * 2.0
        
        # Apply regime filter
        if not self._check_regime(curr_close, ma_value):
            # Regime condition not met, output NaN
            # This allows binning models to exclude these values during fitting
            # (they will be dropped via .dropna() in the binning model's fit method)
            output_value = float('nan')
        else:
            # Regime condition met, output centered RSI
            output_value = centered_rsi
        
        self.output.append(output_value)
        return [output_value]
