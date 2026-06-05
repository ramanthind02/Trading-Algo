"""
Cumulative Returns and Equity Curves

This module provides functions for computing cumulative returns and equity curves
from return series.

Author: Trading Research Team
Date: 2025-01-XX
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def cumulative_returns(
    returns: pd.Series,
    initial_value: float = 1.0
) -> pd.Series:
    """
    Compute cumulative returns from a return series.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns (decimal form, e.g., 0.01 for 1%)
    initial_value : float, default=1.0
        Initial value for cumulative calculation
        
    Returns
    -------
    pd.Series
        Cumulative returns (equity curve) with same index as input
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> cum_returns = cumulative_returns(returns)
    >>> print(cum_returns.iloc[-1])  # Final cumulative value
    """
    if len(returns) == 0:
        return pd.Series([initial_value], index=returns.index)
    
    # Compute cumulative product: (1 + r1) * (1 + r2) * ...
    cumulative = (1 + returns).cumprod() * initial_value

    return cumulative

