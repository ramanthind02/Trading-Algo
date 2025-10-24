from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from collections import deque
from scipy.stats import norm


class ReturnZScoreNode(BiasNode):
    """
    ReturnZScoreNode computes the z-score of recent log returns relative to their
    historical distribution. This captures TRUE mean reversion since returns are stationary.
    
    Unlike price-based indicators, this works with log returns which are:
    1. Stationary (no unit root)
    2. Mean-reverting (tend to revert to ~0)
    3. Properly normalized across different volatility regimes
    
    The z-score measures how many standard deviations the recent return is from
    the historical mean. Extreme z-scores predict mean reversion.
    
    Formula:
    1. Compute N-period log return: log(close[t]) - log(close[t-N])
    2. Compute historical mean and std of N-period returns over lookback window
    3. Z-score = (current_return - mean_return) / std_return
    4. Transform through normal CDF and scale to [-50, 50] range
    
    Output interpretation:
    - Large positive values: Recent return is extremely positive → expect reversion down
    - Large negative values: Recent return is extremely negative → expect reversion up
    - Near 0: Return is typical, no strong mean reversion signal
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, return_period: int = 5, 
                 lookback: int = 252, compression: float = 1.0):
        """
        Initializes the ReturnZScoreNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - return_period (int): Period for computing returns (default: 5 days).
        - lookback (int): Lookback window for computing mean/std of returns (default: 252 days).
        - compression (float): Compression factor for the output (default: 1.0).
                              Increase for more compression, decrease for less.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.return_period = return_period
        self.lookback = lookback
        self.compression = compression
        
        # Storage for log prices
        self.log_prices = []
        
        # Storage for computed returns (for calculating mean/std)
        self.returns = deque(maxlen=lookback)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = lookback + return_period
        
        # Define the columns attribute required by the MLManager
        self.columns = [f'return_zscore_{return_period}_{lookback}']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the return z-score feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the return z-score value centered around 0
        """
        self.candle_count += 1
        
        # Safety check: ensure close price is positive
        if candle.close <= 0:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Add current log price to history
        log_price = np.log(candle.close)
        self.log_prices.append(log_price)
        
        # If we don't have enough data to compute returns, return neutral value (0.0)
        if self.candle_count <= self.return_period:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Compute N-period log return
        current_return = self.log_prices[-1] - self.log_prices[-self.return_period - 1]
        self.returns.append(current_return)
        
        # If we don't have enough returns for statistics, return neutral value (0.0)
        if len(self.returns) < self.lookback:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Compute mean and std of historical returns
        returns_array = np.array(self.returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # Compute z-score
        if std_return > 1e-10:
            z_score = (current_return - mean_return) / std_return
            
            # Transform through normal CDF and scale to [-50, 50]
            # This gives us a bounded output that's easier to work with
            output_value = 100.0 * norm.cdf(self.compression * z_score) - 50.0
        else:
            # If no volatility, return neutral
            output_value = 0.0
        
        # Update bias based on output value
        # Positive z-score (positive output) = recent returns too high → expect reversion down (bearish)
        # Negative z-score (negative output) = recent returns too low → expect reversion up (bullish)
        if output_value > 10:
            self.bias = Bias.BEARISH  # Extreme positive return → expect reversion down
        elif output_value < -10:
            self.bias = Bias.BULLISH  # Extreme negative return → expect reversion up
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [output_value]
        
        return self.output
