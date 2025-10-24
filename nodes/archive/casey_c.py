"""
CaseyC% Momentum Oscillator Node

Based on the CaseyC indicator by Ali Casey / StatOasis.com (c) 2022
This is a momentum oscillator that takes the percent change on closes
and ranks the total over a lookback period.

Similar performance to RSI but uses a different calculation method.
"""

from typing import List
import numpy as np
from numba import njit
from collections import deque
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from utils.cython_nodes import compute_ma_from_deque


# ============================================================================
# NUMBA-COMPILED COMPUTATION FUNCTIONS (Fast!)
# ============================================================================

@njit
def compute_casey_c(pct_changes: np.ndarray, rank_len: int, smooth_len: int) -> float:
    """
    Compute CaseyC% value from percent changes.
    
    The algorithm:
    1. Get the current percent change
    2. Rank it within the lookback period (normalize to 0-100)
    3. Smooth the ranked values
    
    Parameters:
    - pct_changes: Array of percent changes (most recent last)
    - rank_len: Lookback period for ranking
    - smooth_len: Smoothing period
    
    Returns:
    - casey_c: The CaseyC% value (0-100)
    """
    n = len(pct_changes)
    
    if n < rank_len:
        return 50.0  # Neutral value
    
    # Get the most recent rank_len percent changes
    recent_changes = pct_changes[n - rank_len:]
    
    # Current percent change
    current_change = recent_changes[-1]
    
    # Find highest and lowest in the rank period
    highest = np.max(recent_changes)
    lowest = np.min(recent_changes)
    
    # Rank the current change (normalize to 0-100)
    if highest - lowest == 0.0:
        ranked = 50.0  # Neutral if no range
    else:
        ranked = ((current_change - lowest) / (highest - lowest)) * 100.0
    
    return ranked


@njit
def smooth_values(values: np.ndarray, smooth_len: int) -> float:
    """
    Compute smoothed average of the most recent values.
    
    Uses simple moving average for smoothing.
    
    Parameters:
    - values: Array of values to smooth
    - smooth_len: Smoothing period
    
    Returns:
    - smoothed: The smoothed value
    """
    n = len(values)
    
    if n < smooth_len:
        # Not enough data, return the most recent value
        return values[-1] if n > 0 else 50.0
    
    # Simple moving average of the most recent smooth_len values
    recent = values[n - smooth_len:]
    return np.mean(recent)


# ============================================================================
# PYTHON CLASS (Maintains state, delegates computation to Numba)
# ============================================================================

class CaseyCNode(BiasNode):
    """
    CaseyC% Momentum Oscillator - Numba-accelerated
    
    This is a momentum oscillator that ranks percent changes in closes
    over a lookback period, similar to RSI but with different calculation.
    
    The indicator:
    1. Computes percent change: (Close - Close[1]) / Close[1]
    2. Ranks current change within lookback period (0-100)
    3. Smooths the ranked values
    
    Typical values:
    - Above 70: Overbought (bullish momentum)
    - Below 30-35: Oversold (bearish momentum, potential buy)
    - 50: Neutral
    
    Parameters:
    - rank_len: Lookback period for ranking (default: 7)
    - smooth_len: Smoothing period (default: 3)
    - oversold: Oversold threshold for long signal (default: 35)
    - overbought: Overbought threshold for exit signal (default: 70)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame, 
        rank_len: int = 7,
        smooth_len: int = 3,
        oversold: float = 35.0,
        overbought: float = 70.0,
        ma_period: int = 200
    ):
        """
        Initialize CaseyC node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - rank_len: Lookback period for ranking (default: 7)
        - smooth_len: Smoothing period (default: 3)
        - oversold: Oversold threshold (default: 35)
        - overbought: Overbought threshold (default: 70)
        - ma_period: Moving average period for trend filter (default: 200)
        """
        super().__init__(ticker, tf)
        
        self.rank_len = rank_len
        self.smooth_len = smooth_len
        self.oversold = oversold
        self.overbought = overbought
        self.ma_period = ma_period
        
        # Number of candles needed before we can compute valid output
        # Need max of rank_len + smooth_len and ma_period
        self.front_bad = max(rank_len + smooth_len, ma_period)
        
        # Storage for historical data
        self.prev_close = None
        self.pct_changes = []  # Store percent changes
        self.ranked_values = []  # Store ranked values for smoothing
        
        # Price history for MA computation (trend filter)
        self.close_prices = deque(maxlen=ma_period)
        
        # Position tracking (1 = long, 0 = flat)
        self.position = 0
        
        # Track candle count
        self.candle_count = 0
        
        # Column names for output: raw value and position
        self.columns = [
            f'casey_c_{rank_len}_{smooth_len}',
            f'casey_c_{rank_len}_{smooth_len}_position'
        ]
    
    def _compute_ma(self) -> float:
        """
        Compute moving average using Cython-optimized function
        
        Returns:
        - float: Moving average value
        """
        return compute_ma_from_deque(self.close_prices, self.ma_period)
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute CaseyC% for the given candle.
        
        Delegates heavy computation to Numba-compiled functions for speedup.
        Filters signals based on 200-period MA trend filter.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing [casey_c_value, position]
        """
        self.candle_count += 1
        curr_close = candle.close
        
        # Add to price history for MA computation
        self.close_prices.append(curr_close)
        
        # Compute percent change if we have previous close
        # Add division by zero protection
        if self.prev_close is not None and self.prev_close > 1e-10:
            pct_change = (curr_close - self.prev_close) / self.prev_close
            self.pct_changes.append(pct_change)
        elif self.prev_close is not None:
            # If prev_close is 0 or very small, skip this percent change
            # This prevents division by zero errors
            pass
        
        # Update previous close
        self.prev_close = curr_close
        
        # Return neutral values until we have enough data
        if self.candle_count <= self.front_bad or len(self.pct_changes) < self.rank_len:
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0]  # Changed from 50.0 to 0.0 for consistency
            return self.output
        
        # Compute ranked value using Numba
        pct_array = np.array(self.pct_changes, dtype=np.float64)
        ranked = compute_casey_c(pct_array, self.rank_len, self.smooth_len)
        
        # Store ranked value for smoothing
        self.ranked_values.append(ranked)
        
        # Smooth the ranked values
        if len(self.ranked_values) < self.smooth_len:
            casey_c = ranked  # Not enough for smoothing yet
        else:
            ranked_array = np.array(self.ranked_values, dtype=np.float64)
            casey_c = smooth_values(ranked_array, self.smooth_len)
        
        # Compute 200-period MA for trend filter
        ma_200 = self._compute_ma()
        
        # Check if price is above 200-day MA (trend filter)
        above_ma = curr_close > ma_200
        
        # CRITICAL: If below MA, output 0 for both raw value and position
        if not above_ma:
            self.position = 0
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0]
            return self.output
        
        # Update position based on strategy rules (only when above MA)
        # Entry: if CC < OS then buy (and above MA)
        # Exit: if CC crosses above OB then sell (or drops below MA)
        
        if casey_c < self.oversold and self.position == 0:
            # Enter long position
            self.position = 1
            self.bias = Bias.BULLISH
        elif casey_c > self.overbought and self.position == 1:
            # Exit long position
            self.position = 0
            self.bias = Bias.NEUTRAL
        
        # Update bias based on current value (even if not changing position)
        if self.position == 0:
            if casey_c < self.oversold:
                self.bias = Bias.BULLISH  # Potential buy signal
            elif casey_c > self.overbought:
                self.bias = Bias.BEARISH  # Overbought
            else:
                self.bias = Bias.NEUTRAL
        else:
            # Already in position
            self.bias = Bias.BULLISH if self.position == 1 else Bias.BEARISH
        
        # Limit storage to prevent unbounded growth
        # Keep only what we need for computation
        max_history = max(self.rank_len, self.smooth_len) + 100
        if len(self.pct_changes) > max_history:
            self.pct_changes = self.pct_changes[-max_history:]
        if len(self.ranked_values) > max_history:
            self.ranked_values = self.ranked_values[-max_history:]
        
        # Output: [raw CaseyC value, position (1=long, 0=flat)]
        self.output = [casey_c, self.position]
        
        return self.output
