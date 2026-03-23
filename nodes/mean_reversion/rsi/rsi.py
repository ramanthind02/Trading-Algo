from typing import ClassVar, List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from utils.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from nodes import BiasNode


# ============================================================================
# PYTHON CLASS (Maintains state, delegates computation to Numba)
# ============================================================================

class RSI(BiasNode):
    """
    RSI (Relative Strength Index) Bias Node - Cython-accelerated
    
    Computes the standard RSI indicator using exponential moving average
    of up and down price movements.
    
    RSI = 100 * (average_gain) / (average_gain + average_loss)
    
    The RSI oscillates between 0 and 100, with values above 70 typically
    considered overbought and values below 30 considered oversold.
    
    Performance: Uses Cython-backed fast kernels for optimal performance.
    
    Parameters:
    - lookback: Period for RSI calculation (default: 14)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14):
        """
        Initialize RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: RSI period (default: 14)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        # Standardized naming metadata
        self.module_name = 'rsi'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Number of candles needed before we can compute valid output
        self.front_bad = lookback
        
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
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI for the given candle.
        
        Delegates heavy computation to Cython-backed fast kernels for optimal performance.
        Uses circular buffer to avoid expensive array append operations.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the RSI value
        """
        # Store current close in circular buffer
        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1
        
        # Return neutral value (50.0) until we have enough data
        if self.n_prices < self.front_bad:
            self.prev_close = curr_close
            self.output.append(50.0)
            return [50.0]
        
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
        
        self.output.append(rsi)
        return [rsi]
