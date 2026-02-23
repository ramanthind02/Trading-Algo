from typing import List, Tuple
from utils.core.enums import TimeFrame, Ticker
from nodes.bias_nodes import BiasNode
from utils.core.models import Candle
import collections
import math
import numpy as np


class BollingerBand(BiasNode):
    """
    Implements the Bollinger Band trend following strategy as a bias node.
    
    The Bollinger Band strategy uses:
    - Entry: Close above upper band (go long), close below lower band (go short)
    - Exit: Close below middle band (exit long), close above middle band (exit short)
    
    The strategy uses a moving average and standard deviation bands to identify
    trend changes and volatility breakouts.
    """
    def __init__(self, ticker: Ticker, tf: TimeFrame, ma_length: int = 80, std_mult: float = 1.0):
        """
        Initializes the BollingerBand bias node.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - ma_length (int): Length of the moving average period (default: 80).
        - std_mult (float): Standard deviation multiplier for bands (default: 1.0).

        Returns: None
        """
        super().__init__(ticker, tf)

        if not (ma_length > 0):
            raise ValueError("Moving average length must be a positive integer.")
        if not (std_mult > 0):
            raise ValueError("Standard deviation multiplier must be a positive number.")

        self.ma_length = ma_length
        self.std_mult = std_mult
        
        # We need enough history to calculate the moving average and standard deviation
        self.required_history_len = ma_length + 1
        self.candles_history = collections.deque(maxlen=self.required_history_len)
        
        # For tracking current position
        self.current_position = 0  # 0=flat, 1=long, -1=short
        self.previous_close = None  # To track the previous close for entry/exit decisions
        
        self.columns = [f'bollinger_{ma_length}_{std_mult:.1f}']

    def _calculate_bands(self) -> Tuple[float, float, float]:
        """
        Calculates the Bollinger Bands (middle, upper, lower).

        Returns:
        - Tuple[float, float, float]: (middle_band, upper_band, lower_band)
        """
        if len(self.candles_history) < self.ma_length:
            return 0.0, 0.0, 0.0
            
        # Get the closing prices for the MA period
        closes = [candle.close for candle in list(self.candles_history)[-self.ma_length:]]
        
        # Calculate middle band (simple moving average)
        middle_band = sum(closes) / self.ma_length
        
        # Calculate standard deviation
        squared_diff = [(close - middle_band) ** 2 for close in closes]
        variance = sum(squared_diff) / self.ma_length
        std_dev = math.sqrt(variance)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (self.std_mult * std_dev)
        lower_band = middle_band - (self.std_mult * std_dev)
        
        return middle_band, upper_band, lower_band

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the Bollinger Band bias.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed bias value (1=long, 0=flat, -1=short).
        """
        # Store the previous close before adding the new candle
        if len(self.candles_history) > 0:
            self.previous_close = self.candles_history[-1].close
        
        # Add the new candle to history
        self.candles_history.append(candle)

        if len(self.candles_history) < self.ma_length:
            return [0.0]  # Not enough data yet to perform calculations
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self._calculate_bands()
        
        # Current close
        current_close = candle.close
        
        # Check for position changes based on the previous close
        if self.previous_close is not None:
            if self.current_position == 0:  # Currently flat
                # Entry conditions
                if self.previous_close > upper_band:
                    # Buy signal: previous close above upper band
                    self.current_position = 1
                elif self.previous_close < lower_band:
                    # Sell signal: previous close below lower band
                    self.current_position = -1
            
            elif self.current_position == 1:  # Currently long
                # Exit long condition
                if self.previous_close < middle_band:
                    # Exit long: previous close below middle band
                    self.current_position = 0
            
            elif self.current_position == -1:  # Currently short
                # Exit short condition
                if self.previous_close > middle_band:
                    # Exit short: previous close above middle band
                    self.current_position = 0
        
        # Return the current position as the bias
        return [float(self.current_position)]
