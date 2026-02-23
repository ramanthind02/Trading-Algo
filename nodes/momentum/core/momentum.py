from typing import List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode
from collections import deque
from utils.compute.fast_nodes import compute_momentum_fast


class Momentum(BiasNode):
    """
    Momentum Bias Node
    
    Computes the raw Momentum indicator, which is the direct price difference
    over a given lookback period.
    
    Momentum(t) = Close(t) - Close(t-x)
    
    where x is the lookback period.
    
    This is the most direct and raw measure of price momentum - just the
    difference in closing price over a given interval.
    
    Unlike ROC (Rate of Change), Momentum is not normalized and returns
    the absolute price difference rather than a percentage.
    
    Parameters:
    - lookback: Period for Momentum calculation (default: 10)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 10):
        """
        Initialize Momentum node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Momentum period (default: 10)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        # Standardized naming metadata
        self.module_name = 'momentum'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Number of candles needed before we can compute valid output
        # We need lookback+1 candles: current + lookback historical
        self.front_bad = lookback
        
        # Store closing prices in a deque (circular buffer)
        # We need to keep lookback+1 prices to compare current with lookback periods ago
        # When buffer is full, it will have: [close(t-lookback), ..., close(t-1), close(t)]
        self.close_buffer = deque(maxlen=lookback + 1)
        self.n_prices = 0
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Momentum for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Momentum value (raw price difference)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Return neutral value (0.0) until we have enough data
        if self.n_prices <= self.front_bad:
            # Still need to add to buffer for tracking
            self.close_buffer.append(curr_close)
            self.output.append(0.0)
            return [0.0]
        
        # Add current close to buffer (will automatically remove oldest if full)
        # When n_prices == lookback + 1, the deque will have lookback+1 items
        # and the oldest (index 0) will be the close from lookback periods ago
        self.close_buffer.append(curr_close)
        
        # Calculate Momentum: Close(t) - Close(t-x)
        # The deque automatically maintains the lookback period
        # The oldest value is at index 0, which is lookback periods ago
        past_close = self.close_buffer[0]
        
        # Use Cython-backed fast kernel for momentum computation
        momentum = compute_momentum_fast(curr_close, past_close)
        
        self.output.append(momentum)
        return [momentum]

