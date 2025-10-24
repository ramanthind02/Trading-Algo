from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from collections import deque


class ReturnAutocorrNode(BiasNode):
    """
    ReturnAutocorrNode computes the autocorrelation of log returns at various lags.
    This DIRECTLY measures mean reversion strength.
    
    Autocorrelation measures the correlation between returns and their lagged values:
    - Negative autocorrelation = Mean reversion (today's positive return predicts tomorrow's negative return)
    - Zero autocorrelation = Random walk (no predictability)
    - Positive autocorrelation = Momentum/trending (today's positive return predicts tomorrow's positive return)
    
    For mean reversion trading, we want to identify when autocorrelation is strongly negative,
    indicating that extreme moves are likely to reverse.
    
    This node computes multiple autocorrelation features:
    1. Lag-1 autocorrelation (most important for mean reversion)
    2. Recent autocorrelation strength (rolling window)
    3. Interaction with current return (autocorr * current_return)
    
    Formula:
    1. Compute 1-period log returns: log(close[t]) - log(close[t-1])
    2. Compute autocorrelation: corr(returns[t], returns[t-lag])
    3. Output scaled autocorrelation and interaction terms
    
    Output interpretation:
    - Negative values: Mean reversion detected (negative autocorr)
    - Positive values: Momentum detected (positive autocorr)
    - Magnitude indicates strength of the pattern
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lag: int = 1, 
                 lookback: int = 60, min_periods: int = 30):
        """
        Initializes the ReturnAutocorrNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lag (int): Lag for autocorrelation (default: 1 for lag-1 autocorr).
        - lookback (int): Rolling window for computing autocorrelation (default: 60).
        - min_periods (int): Minimum periods needed before computing (default: 30).
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.lag = lag
        self.lookback = lookback
        self.min_periods = min_periods
        
        # Storage for log prices
        self.log_prices = []
        
        # Storage for returns
        self.returns = deque(maxlen=lookback + lag)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = min_periods + lag + 1
        
        # Define the columns attribute required by the MLManager
        # Output 3 features: autocorr, autocorr*return, autocorr_strength
        self.columns = [
            f'return_autocorr_{lag}_{lookback}',
            f'autocorr_return_interaction_{lag}_{lookback}',
            f'autocorr_strength_{lag}_{lookback}'
        ]
    
    def _compute_autocorr(self, returns_array: np.ndarray) -> float:
        """
        Computes autocorrelation at the specified lag.
        
        Parameters:
        - returns_array: Array of returns
        
        Returns:
        - float: Autocorrelation coefficient
        """
        if len(returns_array) < self.lag + self.min_periods:
            return 0.0
        
        # Split into current and lagged returns
        current_returns = returns_array[self.lag:]
        lagged_returns = returns_array[:-self.lag]
        
        # Compute correlation
        if len(current_returns) > 0 and np.std(current_returns) > 1e-10 and np.std(lagged_returns) > 1e-10:
            autocorr = np.corrcoef(current_returns, lagged_returns)[0, 1]
            return autocorr if not np.isnan(autocorr) else 0.0
        else:
            return 0.0
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the return autocorrelation features for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing [autocorr, autocorr*return, autocorr_strength]
        """
        self.candle_count += 1
        
        # Safety check: ensure close price is positive
        if candle.close <= 0:
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0.0, 0.0]
            return self.output
        
        # Add current log price to history
        log_price = np.log(candle.close)
        self.log_prices.append(log_price)
        
        # If we don't have enough data to compute returns, return neutral values
        if self.candle_count <= 1:
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0.0, 0.0]
            return self.output
        
        # Compute 1-period log return
        current_return = self.log_prices[-1] - self.log_prices[-2]
        self.returns.append(current_return)
        
        # If we don't have enough returns, return neutral values
        if len(self.returns) < self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0.0, 0.0]
            return self.output
        
        # Compute autocorrelation
        returns_array = np.array(self.returns)
        autocorr = self._compute_autocorr(returns_array)
        
        # Feature 1: Raw autocorrelation (scaled to [-50, 50])
        autocorr_scaled = autocorr * 50.0
        
        # Feature 2: Interaction term (autocorr * current_return)
        # This captures: "Given current return and autocorr, what's the expected next return?"
        # If autocorr is negative and current return is positive, expect negative next return
        interaction = autocorr * current_return * 1000.0  # Scale up for visibility
        
        # Feature 3: Autocorrelation strength (absolute value)
        # Measures how strong the pattern is, regardless of direction
        autocorr_strength = abs(autocorr) * 50.0
        
        # Update bias based on autocorrelation and current return
        # Negative autocorr + positive return = expect reversion down (bearish)
        # Negative autocorr + negative return = expect reversion up (bullish)
        if autocorr < -0.1:  # Mean reversion detected
            if current_return > 0:
                self.bias = Bias.BEARISH  # Positive return likely to reverse
            elif current_return < 0:
                self.bias = Bias.BULLISH  # Negative return likely to reverse
            else:
                self.bias = Bias.NEUTRAL
        elif autocorr > 0.1:  # Momentum detected
            if current_return > 0:
                self.bias = Bias.BULLISH  # Positive return likely to continue
            elif current_return < 0:
                self.bias = Bias.BEARISH  # Negative return likely to continue
            else:
                self.bias = Bias.NEUTRAL
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [autocorr_scaled, interaction, autocorr_strength]
        
        return self.output
