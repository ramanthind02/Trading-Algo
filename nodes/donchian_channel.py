from typing import List, Tuple, Optional
from utils.enums import TimeFrame, Ticker
from nodes import BiasNode
from utils.models import Candle
import numpy as np

# Try to import Cython optimized version
try:
    from utils.fast_nodes import compute_high_low_channel_fast, CYTHON_NODES_AVAILABLE
except ImportError:
    CYTHON_NODES_AVAILABLE = False


class DonchianChannel(BiasNode):
    """
    Implements the Donchian Channel strategy as a bias node.
    
    The Donchian Channel strategy is always in the market and uses a single channel:
    - Long when price breaks above the highest high of the lookback period
    - Short when price breaks below the lowest low of the lookback period
    
    Unlike the Turtle Trading strategy, this implementation:
    1. Is always in the market (no flat state)
    2. Uses a single lookback period for both entry and exit
    3. Does not filter trades based on previous results
    """
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int):
        """
        Initializes the DonchianChannel bias node.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lookback (int): Number of periods to look back for channel calculation.

        Returns: None
        """
        super().__init__(ticker, tf)

        if not (lookback > 0):
            raise ValueError("Lookback period must be a positive integer.")

        self.lookback = lookback
        
        # Standardized naming metadata
        self.module_name = 'donchian'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Number of candles needed before we can compute valid output
        # Need lookback + 1 candles (lookback for calculation + current candle)
        self.front_bad = lookback + 1
        
        if CYTHON_NODES_AVAILABLE:
            # Cython path: use numpy arrays for 5-10x speedup
            # Buffer size is lookback+1 to match the semantics of recent_candles
            self.highs = np.zeros(self.front_bad, dtype=np.float64)
            self.lows = np.zeros(self.front_bad, dtype=np.float64)
            self.buffer_idx = 0
            self.n_filled = 0
        else:
            # Fallback: use deque
            import collections
            self.candles_history = collections.deque(maxlen=self.front_bad)
        
        # For tracking current position
        self.current_position = 1  # Start with long position (1=long, -1=short)
        
        # Define standardized columns
        self.ensure_standardized_columns()

    def calculate_donchian_channel(self, lookback: int) -> Tuple[float, float]:
        """
        Calculates the Donchian channel (highest high and lowest low) over the specified lookback period.

        Parameters:
        - lookback (int): Number of periods to look back.

        Returns:
        - Tuple[float, float]: (highest_high, lowest_low) over the lookback period.
        """
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: use compute_high_low_channel_fast
            # We need lookback elements, excluding the current one
            # After writing, buffer_idx points to the next write position
            # The current candle is at (buffer_idx - 1) % front_bad
            # We want to exclude current, so look back from (buffer_idx - 2) % front_bad
            if self.n_filled < lookback + 1:  # Need lookback+1 total (including current)
                return 0.0, 0.0
            
            # The most recent element before current (where we want to start looking back)
            # buffer_idx was just incremented, so previous is at (buffer_idx - 2) % front_bad
            prev_idx = (self.buffer_idx - 2 + self.front_bad) % self.front_bad
            
            return compute_high_low_channel_fast(
                self.highs,
                self.lows,
                prev_idx,
                lookback,
                self.n_filled
            )
        else:
            # Fallback: use Python implementation
            if len(self.candles_history) < lookback:
                return 0.0, 0.0
                
            # Get the most recent candles excluding the current one
            recent_candles = list(self.candles_history)[-(lookback+1):-1]
            
            if not recent_candles:
                return 0.0, 0.0
                
            highest_high = max(candle.high for candle in recent_candles)
            lowest_low = min(candle.low for candle in recent_candles)
            
            return highest_high, lowest_low

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the Donchian Channel bias.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed bias value (1=long, -1=short).
        """
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path: store in numpy arrays
            self.highs[self.buffer_idx] = candle.high
            self.lows[self.buffer_idx] = candle.low
            self.buffer_idx = (self.buffer_idx + 1) % self.front_bad
            self.n_filled = min(self.n_filled + 1, self.front_bad)
        else:
            # Fallback: use deque
            self.candles_history.append(candle)

        # Return current position if not enough data
        if (CYTHON_NODES_AVAILABLE and self.n_filled < self.front_bad) or \
           (not CYTHON_NODES_AVAILABLE and len(self.candles_history) < self.front_bad):
            position = float(self.current_position)
            self.output.append(position)
            return [position]

        # Calculate Donchian channel
        highest_high, lowest_low = self.calculate_donchian_channel(self.lookback)
        
        # Current price
        current_price = candle.close
        
        # Check for position changes
        if self.current_position == 1:  # Currently long
            # Switch to short if price breaks below the channel low
            if current_price < lowest_low:
                self.current_position = -1
        
        elif self.current_position == -1:  # Currently short
            # Switch to long if price breaks above the channel high
            if current_price > highest_high:
                self.current_position = 1
        
        # Return the numeric position
        position = float(self.current_position)
        self.output.append(position)
        return [position]
