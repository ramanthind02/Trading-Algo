"""
Equity Tracking Utilities

This module provides functions for tracking equity peaks, drawdowns, and
other equity-related calculations.

Author: Trading Research Team
Date: 2025-01-XX
"""

import pandas as pd


def equity_peak(equity: pd.Series) -> pd.Series:
    """
    Compute running equity peak (highest equity seen so far).
    
    Parameters
    ----------
    equity : pd.Series
        Equity curve values
        
    Returns
    -------
    pd.Series
        Running peak equity with same index as input
        
    Examples
    --------
    >>> equity = pd.Series([1.0, 1.1, 1.05, 1.15, 1.12])
    >>> peak = equity_peak(equity)
    >>> print(peak)  # [1.0, 1.1, 1.1, 1.15, 1.15]
    """
    return equity.expanding().max()

