from typing import List
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode
from collections import deque
from utils.fast_nodes import compute_momentum_fast


class SimpleMomentum(BiasNode):
    """
    Simple Momentum Bias Node
    
    A simple momentum strategy that generates directional signals based on
    price comparison with a previous period.
    
    Signal Logic:
    - Long (1): When current close > close N periods ago
    - Short (-1): When current close < close N periods ago
    
    This is a simple but effective momentum strategy that can be applied
    to many futures and instruments. Variations like comparing to 2 or 3
    periods ago also produce good results.
    
    Parameters:
    - lookback: Number of periods to look back for comparison (default: 1)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 1):
        """
        Initialize Simple Momentum node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Number of periods to look back (default: 1, i.e., yesterday)
        """
        super().__init__(ticker, tf)
        
        if not (lookback > 0):
            raise ValueError("Lookback period must be a positive integer.")
        
        self.lookback = lookback
        # Standardized naming metadata
        self.module_name = 'simple_momentum'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Number of candles needed before we can compute valid output
        # We need lookback+1 candles: current + lookback historical
        self.front_bad = lookback + 1
        
        # Store closing prices in a deque (circular buffer)
        # We need to keep lookback+1 prices to compare current with lookback periods ago
        self.close_buffer = deque(maxlen=self.front_bad)
        self.n_prices = 0
        
        # Define standardized columns
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Compute Simple Momentum signal for the given candle.
        
        Returns 1 if current close > close N periods ago (long signal)
        Returns -1 if current close < close N periods ago (short signal)
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal value (1.0 for long, -1.0 for short)
        """
        curr_close = candle.close
        self.close_buffer.append(curr_close)
        self.n_prices += 1
        
        # Return neutral value (0.0) until we have enough data
        if self.n_prices < self.front_bad:
            signal = 0.0
            self.output.append(signal)
            return [signal]
        
        # Get the close price from lookback periods ago
        # The deque maintains lookback+1 items, with index 0 being the oldest
        past_close = self.close_buffer[0]
        
        # Use Cython-backed fast kernel to compute momentum (price difference)
        momentum = compute_momentum_fast(curr_close, past_close)
        
        # Generate signal: 1 if price is higher, -1 if lower
        if momentum > 0:
            signal = 1.0  # Long signal
        elif momentum < 0:
            signal = -1.0  # Short signal
        else:
            signal = 0.0  # Neutral (price unchanged)
        
        self.output.append(signal)
        return [signal]
