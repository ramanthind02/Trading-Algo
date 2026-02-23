from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque


class StochasticRSINode(BiasNode):
    """
    StochasticRSINode computes the Stochastic RSI indicator, which applies the Stochastic
    oscillator formula to RSI values instead of price.
    
    This creates a more sensitive momentum indicator that oscillates between 0 and 100,
    identifying overbought and oversold conditions in the RSI itself.
    
    The calculation involves three steps:
    1. Compute RSI over the first lookback period
    2. Apply Stochastic formula to RSI values over the second lookback period
    3. Optionally smooth the result with exponential moving average
    
    Formula:
    1. RSI = 100 * (average_gain) / (average_gain + average_loss)
    2. Stoch RSI = 100 * (RSI - min_RSI) / (max_RSI - min_RSI)
    3. If n_to_smooth > 1: Apply EMA smoothing
    
    Output is centered around 50 with range [0, 100]:
    - Values > 80: Overbought
    - Values < 20: Oversold
    - Around 50: Neutral
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, rsi_lookback: int = 14, 
                 stoch_lookback: int = 14, n_to_smooth: int = 3):
        """
        Initializes the StochasticRSINode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - rsi_lookback (int): The lookback period for RSI calculation (default: 14).
        - stoch_lookback (int): The lookback period for Stochastic calculation (default: 14).
        - n_to_smooth (int): The period for exponential smoothing (default: 3, set to 1 for no smoothing).
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.rsi_lookback = rsi_lookback
        self.stoch_lookback = stoch_lookback
        self.n_to_smooth = n_to_smooth
        
        # RSI computation state
        self.upsum = 1e-60  # Small value to avoid division by zero
        self.dnsum = 1e-60
        
        # Price history for RSI computation
        self.close_prices = []
        
        # RSI values for Stochastic computation
        self.rsi_values = []
        
        # Smoothing state
        self.smooth_alpha = 2.0 / (n_to_smooth + 1.0) if n_to_smooth > 1 else 0.0
        self.smoothed: Optional[float] = None
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = rsi_lookback + stoch_lookback - 1
        
        # Define the columns attribute required by the MLManager
        self.columns = [f'stoch_rsi_{rsi_lookback}_{stoch_lookback}_{n_to_smooth}']
    
    def _compute_rsi(self) -> float:
        """
        Computes RSI for the current state.
        
        Returns:
        - float: RSI value between 0 and 100
        """
        n = len(self.close_prices)
        
        # Initialize RSI on the first valid computation
        if n == self.rsi_lookback:
            # Initialize with simple average of gains and losses
            self.upsum = 1e-60
            self.dnsum = 1e-60
            for i in range(1, self.rsi_lookback):
                diff = self.close_prices[i] - self.close_prices[i-1]
                if diff > 0.0:
                    self.upsum += diff
                else:
                    self.dnsum -= diff
            
            # Convert to average
            self.upsum /= (self.rsi_lookback - 1)
            self.dnsum /= (self.rsi_lookback - 1)
        
        # Update RSI using exponential moving average
        if n > self.rsi_lookback:
            diff = self.close_prices[-1] - self.close_prices[-2]
            
            if diff > 0.0:
                # Price went up
                self.upsum = ((self.rsi_lookback - 1) * self.upsum + diff) / self.rsi_lookback
                self.dnsum *= (self.rsi_lookback - 1.0) / self.rsi_lookback
            else:
                # Price went down
                self.dnsum = ((self.rsi_lookback - 1) * self.dnsum - diff) / self.rsi_lookback
                self.upsum *= (self.rsi_lookback - 1.0) / self.rsi_lookback
        
        # Compute RSI
        rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
        return rsi
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the Stochastic RSI feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the Stochastic RSI value (0-100, centered around 50)
        """
        self.candle_count += 1
        
        # Add current close to history
        self.close_prices.append(candle.close)
        
        # If we don't have enough data for RSI, return neutral value (50.0)
        if self.candle_count < self.rsi_lookback:
            self.bias = Bias.NEUTRAL
            self.output = [50.0]
            return self.output
        
        # Compute RSI
        rsi = self._compute_rsi()
        self.rsi_values.append(rsi)
        
        # If we don't have enough RSI values for Stochastic, return neutral value (50.0)
        if self.candle_count < self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [50.0]
            return self.output
        
        # Compute Stochastic on RSI values
        # Get the last stoch_lookback RSI values
        recent_rsi = self.rsi_values[-self.stoch_lookback:]
        
        min_rsi = min(recent_rsi)
        max_rsi = max(recent_rsi)
        
        # Compute Stochastic RSI
        if (max_rsi - min_rsi) > 1e-60:
            stoch_rsi = 100.0 * (rsi - min_rsi) / (max_rsi - min_rsi)
        else:
            stoch_rsi = 50.0  # Neutral if no range
        
        # Apply smoothing if requested
        if self.n_to_smooth > 1:
            if self.smoothed is None:
                # Initialize smoothed value
                self.smoothed = stoch_rsi
            else:
                # Update smoothed value with EMA
                self.smoothed = self.smooth_alpha * stoch_rsi + (1.0 - self.smooth_alpha) * self.smoothed
            
            output_value = self.smoothed
        else:
            output_value = stoch_rsi
        
        # Update bias based on output value
        # Traditional levels: >80 overbought, <20 oversold
        if output_value > 65:
            self.bias = Bias.BULLISH  # Overbought (mean reversion opportunity)
        elif output_value < 35:
            self.bias = Bias.BEARISH  # Oversold (mean reversion opportunity)
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [output_value]
        
        return self.output