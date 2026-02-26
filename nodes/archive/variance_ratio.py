from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from collections import deque


class VarianceRatioNode(BiasNode):
    """
    VarianceRatioNode computes the variance ratio test statistic, which directly tests
    for mean reversion vs random walk vs momentum.
    
    The variance ratio compares the variance of multi-period returns to what would be
    expected under a random walk:
    
    VR(k) = Var(k-period returns) / (k * Var(1-period returns))
    
    Under a random walk (no predictability):
    - VR(k) ≈ 1.0
    
    Under mean reversion:
    - VR(k) < 1.0 (multi-period variance grows slower than linearly)
    - Lower values indicate stronger mean reversion
    
    Under momentum/trending:
    - VR(k) > 1.0 (multi-period variance grows faster than linearly)
    - Higher values indicate stronger momentum
    
    This is a POWERFUL test for mean reversion because it directly measures whether
    returns exhibit the statistical properties of mean reversion.
    
    Formula:
    1. Compute 1-period returns: r1[t] = log(close[t]) - log(close[t-1])
    2. Compute k-period returns: rk[t] = log(close[t]) - log(close[t-k])
    3. VR(k) = Var(rk) / (k * Var(r1))
    4. Output: (VR - 1) scaled to indicate strength and direction
    
    Output interpretation:
    - Negative values: Mean reversion detected (VR < 1)
    - Zero: Random walk (VR ≈ 1)
    - Positive values: Momentum detected (VR > 1)
    - Magnitude indicates strength of the pattern
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, ratio_period: int = 5, 
                 lookback: int = 252, min_periods: int = 60):
        """
        Initializes the VarianceRatioNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - ratio_period (int): Period k for computing k-period returns (default: 5).
        - lookback (int): Rolling window for computing variances (default: 252).
        - min_periods (int): Minimum periods needed before computing (default: 60).
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.ratio_period = ratio_period
        self.lookback = lookback
        self.min_periods = min_periods
        
        # Storage for log prices
        self.log_prices = []
        
        # Storage for 1-period returns
        self.returns_1 = deque(maxlen=lookback)
        
        # Storage for k-period returns
        self.returns_k = deque(maxlen=lookback)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = max(min_periods, ratio_period) + 1
        
        # Define the columns attribute required by the MLManager
        # Output 3 features: variance_ratio, vr_deviation, vr_zscore
        self.columns = [
            f'variance_ratio_{ratio_period}_{lookback}',
            f'vr_deviation_{ratio_period}_{lookback}',
            f'vr_strength_{ratio_period}_{lookback}'
        ]
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the variance ratio features for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing [variance_ratio, vr_deviation, vr_strength]
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
        return_1 = self.log_prices[-1] - self.log_prices[-2]
        self.returns_1.append(return_1)
        
        # Compute k-period log return if we have enough data
        if self.candle_count > self.ratio_period:
            return_k = self.log_prices[-1] - self.log_prices[-self.ratio_period - 1]
            self.returns_k.append(return_k)
        
        # If we don't have enough returns, return neutral values
        if len(self.returns_1) < self.min_periods or len(self.returns_k) < self.min_periods:
            self.bias = Bias.NEUTRAL
            self.output = [0.0, 0.0, 0.0]
            return self.output
        
        # Compute variances
        var_1 = np.var(self.returns_1, ddof=1)
        var_k = np.var(self.returns_k, ddof=1)
        
        # Compute variance ratio
        if var_1 > 1e-10:
            variance_ratio = var_k / (self.ratio_period * var_1)
            
            # Feature 1: Variance ratio scaled to [-50, 50]
            # VR = 1 → 0, VR < 1 → negative (mean reversion), VR > 1 → positive (momentum)
            vr_scaled = (variance_ratio - 1.0) * 100.0
            
            # Feature 2: Deviation from random walk (VR - 1)
            # Measures how far from random walk we are
            vr_deviation = (variance_ratio - 1.0) * 50.0
            
            # Feature 3: Strength of pattern (absolute deviation)
            # How strong is the deviation from random walk, regardless of direction
            vr_strength = abs(variance_ratio - 1.0) * 50.0
            
        else:
            # If no variance, return neutral
            vr_scaled = 0.0
            vr_deviation = 0.0
            vr_strength = 0.0
            variance_ratio = 1.0
        
        # Update bias based on variance ratio
        if variance_ratio < 0.85:  # Strong mean reversion
            self.bias = Bias.NEUTRAL  # Mean reversion is good for our strategy
        elif variance_ratio > 1.15:  # Strong momentum
            self.bias = Bias.NEUTRAL  # Momentum might not be good for mean reversion
        else:
            self.bias = Bias.NEUTRAL  # Close to random walk
        
        # Store the output
        self.output = [vr_scaled, vr_deviation, vr_strength]
        
        return self.output
