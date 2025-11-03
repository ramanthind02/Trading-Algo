"""
RSI (Relative Strength Index) computation helpers with Numba acceleration.

This module contains the core RSI calculation functions that can be shared
between different RSI-based nodes (RSI, Cumulative RSI, etc.).

Functions:
- compute_rsi_initial: Initialize RSI computation for the first valid period
- update_rsi: Update RSI using exponential moving average
"""

import numpy as np
from numba import njit


@njit
def compute_rsi_initial(close_prices: np.ndarray, lookback: int) -> tuple:
    """
    Initialize RSI computation for the first valid period.
    
    Parameters:
    - close_prices: Array of close prices
    - lookback: RSI period
    
    Returns:
    - tuple: (upsum, dnsum) as averages
    """
    upsum = 1e-60
    dnsum = 1e-60
    
    for i in range(1, lookback):
        diff = close_prices[i] - close_prices[i-1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff
    
    # Convert to average
    upsum /= (lookback - 1)
    dnsum /= (lookback - 1)
    
    return upsum, dnsum


@njit
def update_rsi(prev_close: float, curr_close: float, 
               upsum: float, dnsum: float, lookback: int) -> tuple:
    """
    Update RSI using exponential moving average.
    
    Parameters:
    - prev_close: Previous close price
    - curr_close: Current close price
    - upsum: Current average gain
    - dnsum: Current average loss
    - lookback: RSI period
    
    Returns:
    - tuple: (new_upsum, new_dnsum, rsi_value)
    """
    diff = curr_close - prev_close
    
    if diff > 0.0:
        # Price went up
        upsum = ((lookback - 1) * upsum + diff) / lookback
        dnsum *= (lookback - 1.0) / lookback
    else:
        # Price went down
        dnsum = ((lookback - 1) * dnsum - diff) / lookback
        upsum *= (lookback - 1.0) / lookback
    
    # Compute RSI
    rsi = 100.0 * upsum / (upsum + dnsum)
    
    return upsum, dnsum, rsi