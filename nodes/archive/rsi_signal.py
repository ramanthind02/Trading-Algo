from typing import List, Optional
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from nodes.rsi import RSI
from utils.compute.cython.cython_nodes import compute_ma_from_deque
from collections import deque
import numpy as np


class RSISignalNode(BiasNode):
    """
    RSI Signal Node with Moving Average Filter
    
    This node generates a binary trading signal (1 = long, 0 = flat) based on RSI levels
    and a 200-period moving average filter. It only goes long (no shorting).
    
    Trading Logic:
    - Enter long (output = 1) when RSI < lower_level AND price > 200-day MA
    - Exit long (output = 0) when RSI > upper_level OR price < 200-day MA
    - Otherwise maintain current position
    
    The 200-day MA acts as a trend filter to only take long positions in uptrends.
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - lower_level: RSI level to enter long position (default: 25)
    - upper_level: RSI level to exit long position (default: 75)
    - ma_period: Moving average period for trend filter (default: 200)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, 
                 rsi_lookback: int = 14,
                 lower_level: float = 25.0,
                 upper_level: float = 75.0,
                 ma_period: int = 200):
        """
        Initialize RSI Signal node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - rsi_lookback: RSI period (default: 14)
        - lower_level: RSI level to enter long (default: 25)
        - upper_level: RSI level to exit long (default: 75)
        - ma_period: MA period for trend filter (default: 200)
        """
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.lower_level = lower_level
        self.upper_level = upper_level
        self.ma_period = ma_period
        
        # Number of candles needed before we can compute valid output
        self.front_bad = max(rsi_lookback, ma_period)
        
        # Use the numba-optimized RSI node for RSI computation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        # Price history for MA computation
        self.close_prices = deque(maxlen=ma_period)
        
        # Current position state (0 = flat, 1 = long)
        self.position = 0
        
        # Track candle count
        self.candle_count = 0
        
        # Column name for output
        self.columns = [f'rsi_signal_{rsi_lookback}_{int(lower_level)}_{int(upper_level)}_{ma_period}']
    

    
    def _compute_ma(self) -> float:
        """
        Compute moving average using Cython-optimized function
        
        Returns:
        - float: Moving average value
        """
        return compute_ma_from_deque(self.close_prices, self.ma_period)
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI signal for the given candle
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal value (0 or 1)
        """
        self.candle_count += 1
        self.close_prices.append(candle.close)
        
        # Return flat (0) until we have enough data
        if self.candle_count <= self.front_bad:
            self.position = 0
            self.bias = Bias.NEUTRAL
            self.output = [0]
            return [0]
        
        # Compute RSI using the numba-optimized RSI node
        rsi = self.rsi_node._compute_candle(candle)[0]
        
        # Compute MA
        ma_200 = self._compute_ma()
        
        # Check if price is above 200-day MA (trend filter)
        above_ma = candle.close > ma_200
        
        # Trading logic
        if self.position == 0:
            # Currently flat - check for entry signal
            if rsi < self.lower_level and above_ma:
                # Enter long position
                self.position = 1
                self.bias = Bias.BULLISH
        else:
            # Currently long - check for exit signal
            if rsi > self.upper_level or not above_ma:
                # Exit long position
                self.position = 0
                self.bias = Bias.NEUTRAL
            else:
                # Maintain long position
                self.bias = Bias.BULLISH
        
        self.output = [self.position]
        return [self.position]
