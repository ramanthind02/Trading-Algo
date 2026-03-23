from typing import ClassVar, List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from utils.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from nodes import BiasNode


# ============================================================================
# PURE PYTHON AVERAGING FUNCTION (Simple O(1) operation, no need for Numba)
# ============================================================================

def compute_avg_rsi(rsi_buffer: np.ndarray, buffer_size: int) -> float:
    """
    Compute average of RSI values in the buffer.
    
    This is a simple averaging operation that doesn't require Numba acceleration.
    The performance-critical RSI computation is handled by Cython-backed fast kernels.
    
    Parameters:
    - rsi_buffer: Circular buffer containing RSI values
    - buffer_size: Number of valid RSI values in buffer
    
    Returns:
    - float: Average RSI value
    """
    if buffer_size == 0:
        return 50.0  # Neutral RSI value
    
    total = 0.0
    for i in range(buffer_size):
        total += rsi_buffer[i]
    
    return total / buffer_size


# ============================================================================
# PYTHON CLASS (Maintains state, delegates computation to Numba)
# ============================================================================

class CumulativeRSI(BiasNode):
    """
    Cumulative RSI Bias Node - Cython-accelerated
    
    Computes RSI for each candle and then takes the average of the last 
    'avg_period' RSI values to create a smoother, less noisy signal.
    
    This combines the standard RSI calculation with a rolling average
    to reduce false signals while maintaining the core RSI characteristics.
    
    The node first calculates individual RSI values using the same logic
    as the standard RSI node, then maintains a rolling buffer of these
    RSI values and returns their average.
    
    Performance: Uses Cython-backed fast kernels for RSI computation.
    
    Parameters:
    - lookback: Period for individual RSI calculation (default: 14)
    - avg_period: Number of RSI values to average (default: 5)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback", "avg_period"})
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14, avg_period: int = 5):
        """
        Initialize Cumulative RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: RSI period for individual calculations (default: 14)
        - avg_period: Number of RSI values to average (default: 5)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.avg_period = avg_period
        
        # Standardized naming metadata
        self.module_name = 'cumulative_rsi'
        self.output_features = ['signal']
        self.params = {'lookback': lookback, 'avg_period': avg_period}
        
        # Number of candles needed before we can compute valid output
        # Need lookback candles for RSI + avg_period-1 additional for averaging
        self.front_bad = lookback + avg_period - 1
        
        # RSI computation state (same as regular RSI)
        self.upsum = 1e-60  # Small value to avoid division by zero
        self.dnsum = 1e-60
        
        # RSI averaging state
        self.rsi_buffer = np.zeros(avg_period, dtype=np.float64)
        self.rsi_buffer_idx = 0  # Current position in RSI circular buffer
        self.n_rsi_values = 0    # Number of RSI values computed so far
        
        # OPTIMIZATION: Use circular buffer for price data (same as regular RSI)
        self.buffer_size = lookback + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0  # Current position in price circular buffer
        self.n_prices = 0    # Total number of prices seen
        
        # Store previous close for RSI update
        self.prev_close = 0.0
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Cumulative RSI for the given candle.
        
        This method:
        1. Calculates the RSI for the current candle using Cython-backed fast kernels
        2. Stores the RSI value in a rolling buffer
        3. Returns the average of the RSI values in the buffer
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Cumulative RSI value
        """
        # Store current close in circular buffer (same as regular RSI)
        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1
        
        # Return neutral value until we have enough data for RSI calculation
        if self.n_prices < self.lookback:
            self.prev_close = curr_close
            # Store neutral RSI in buffer for consistency
            if self.n_rsi_values < self.avg_period:
                self.rsi_buffer[self.rsi_buffer_idx] = 50.0
                self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.avg_period
                self.n_rsi_values += 1
            
            self.output.append(50.0)
            return [50.0]
        
        # Initialize RSI on the first valid computation (same as regular RSI)
        if self.n_prices == self.lookback:
            # Extract the lookback period from circular buffer for initialization
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
        
        # Update RSI using Cython-backed function (same as regular RSI)
        self.upsum, self.dnsum, current_rsi = update_rsi_fast(
            self.prev_close,
            curr_close,
            self.upsum,
            self.dnsum,
            self.lookback
        )
        
        # Store RSI value in averaging buffer
        self.rsi_buffer[self.rsi_buffer_idx] = current_rsi
        self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.avg_period
        if self.n_rsi_values < self.avg_period:
            self.n_rsi_values += 1
        
        # Compute average RSI using pure Python function (simple O(1) operation)
        avg_rsi = compute_avg_rsi(self.rsi_buffer, min(self.n_rsi_values, self.avg_period))
        
        # Update prev_close for next iteration
        self.prev_close = curr_close
        
        self.output.append(avg_rsi)
        return [avg_rsi]
    
