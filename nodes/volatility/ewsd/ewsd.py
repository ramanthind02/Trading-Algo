"""
Exponentially Weighted Standard Deviation (EWSD) Bias Node

This module implements Robert Carver's EWSD methodology for volatility estimation,
which is used for position sizing by estimating variable risk.

The EWSD calculation involves:
1. Short-run volatility estimate using EWMA on squared returns
2. Long-run historical volatility estimate
3. Blending the two estimates (70% short-run, 30% long-run)

References:
- Carver, R. "Systematic Trading" (2015)
- Carver, R. "Leveraged Trading" (2019)

Author: Trading Research Team
Date: 2025-10-23
"""

from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque

try:
    from utils.compute.fast_nodes import CYTHON_NODES_AVAILABLE, compute_stddev_sample_fast
except ImportError:
    CYTHON_NODES_AVAILABLE = False
    compute_stddev_sample_fast = None  # type: ignore[assignment]


class EWSDNode(BiasNode):
    """
    EWSD (Exponentially Weighted Standard Deviation) Node.
    
    Computes volatility using Robert Carver's preferred methodology:
    - Short-run EWMA estimate with lambda=0.06061 (32-day span)
    - Long-run historical standard deviation
    - Blended estimate: 70% short-run + 30% long-run
    
    This provides a robust measure of risk that:
    - Responds quickly to volatility changes (short-run component)
    - Remains stable and mean-reverting (long-run component)
    - Prevents oversized positions before market crises
    
    Outputs:
    - ewsd_daily_pct: Daily standard deviation (percentage)
    - ewsd_annual_pct: Annualized standard deviation (percentage)
    
    Cython: When built, uses compute_stddev_sample_fast from fast_nodes for
    the long-run standard deviation (sample ddof=1); EWMA variance remains O(1).
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        lambda_short: float = 0.06061,  # 32-day span
        long_run_window: int = 2520,     # 10 years for long-run estimate
        blend_short_weight: float = 0.7,
        blend_long_weight: float = 0.3
    ):
        """
        Initialize the EWSD bias node.
        
        Parameters
        ----------
        ticker : Ticker
            The ticker symbol
        tf : TimeFrame
            The timeframe of the candles
        lambda_short : float, default=0.06061
            EWMA smoothing parameter for short-run estimate.
            0.06061 corresponds to a 32-day span, Carver's preferred value.
            Span = 2/(lambda) - 1, so lambda = 2/(span+1)
        long_run_window : int, default=2520
            Lookback window for long-run historical volatility (10 years)
        blend_short_weight : float, default=0.7
            Weight for short-run estimate in blend
        blend_long_weight : float, default=0.3
            Weight for long-run estimate in blend
        """
        super().__init__(ticker, tf)
        
        # EWMA parameters
        self.lambda_short = lambda_short
        self.long_run_window = long_run_window
        self.blend_short_weight = blend_short_weight
        self.blend_long_weight = blend_long_weight
        
        # State variables
        self.prev_close: Optional[float] = None
        self.prev_variance_sq: Optional[float] = None
        self.returns_history = deque(maxlen=long_run_window)
        
        # Initial estimates (only used before any meaningful return history)
        self.sigma_long: float = 0.01  # 1% daily prior, replaced as soon as returns arrive
        self.initial_variance_sq: float = self.sigma_long ** 2
        
        # Define output columns
        self.columns = ['ewsd_daily_pct', 'ewsd_annual_pct']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute EWSD for the current candle.
        
        This follows Robert Carver's methodology:
        1. Calculate daily return
        2. Update EWMA variance estimate
        3. Calculate short-run standard deviation
        4. Blend with long-run standard deviation
        5. Annualize the result
        
        Parameters
        ----------
        candle : Candle
            The candle to compute EWSD for
            
        Returns
        -------
        List
            [ewsd_daily_pct, ewsd_annual_pct]
            - ewsd_daily_pct: Daily volatility as percentage
            - ewsd_annual_pct: Annualized volatility as percentage (daily * 16)
        """
        if self.prev_close is None:
            # First candle: initialize with default estimate
            ewsd_daily_pct = self.sigma_long
            self.prev_close = candle.close
        else:
            # 1. Calculate daily return (percentage)
            daily_return = (candle.close - self.prev_close) / self.prev_close
            
            # Store return for long-run calculation
            self.returns_history.append(daily_return)
            
            # 2. Calculate EWMA variance
            r_t_sq = daily_return ** 2
            
            if self.prev_variance_sq is None:
                # Initialize variance with squared return
                current_variance_sq = r_t_sq
            else:
                # EWMA: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2
                current_variance_sq = (
                    self.lambda_short * self.prev_variance_sq + 
                    (1 - self.lambda_short) * r_t_sq
                )
            
            # 3. Calculate short-run standard deviation
            sigma_short = np.sqrt(current_variance_sq)
            
            # 4. Update long-run standard deviation (expanding window, Cython when available)
            n_obs = len(self.returns_history)
            if n_obs >= 2:
                arr = np.array(self.returns_history, dtype=np.float64)
                if CYTHON_NODES_AVAILABLE and compute_stddev_sample_fast is not None:
                    self.sigma_long = compute_stddev_sample_fast(arr, n_obs)
                else:
                    self.sigma_long = float(np.std(arr, ddof=1))
            elif n_obs == 1:
                self.sigma_long = abs(float(self.returns_history[0]))
            
            # 5. Blend short-run and long-run estimates
            # Carver's preferred blend: 70% short-run + 30% long-run
            ewsd_daily_pct = (
                self.blend_short_weight * sigma_short + 
                self.blend_long_weight * self.sigma_long
            )
            
            # Update state
            self.prev_variance_sq = current_variance_sq
            self.prev_close = candle.close
        
        # 6. Annualize the result
        # Annualize with Carver's rounded factor: 16 ~= sqrt(252)
        ewsd_annual_pct = ewsd_daily_pct * 16
        
        # Convert to percentage (multiply by 100)
        ewsd_daily_pct_output = ewsd_daily_pct * 100
        ewsd_annual_pct_output = ewsd_annual_pct * 100
        
        # Set bias to NEUTRAL (volatility doesn't indicate direction)
        self.bias = Bias.NEUTRAL
        
        # Store and return output
        self.output = [ewsd_daily_pct_output, ewsd_annual_pct_output]
        return self.output
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"EWSDNode(ticker={self.ticker}, tf={self.tf}, "
                f"lambda={self.lambda_short:.5f}, long_window={self.long_run_window})")
