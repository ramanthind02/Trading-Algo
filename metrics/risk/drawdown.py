"""
Drawdown Calculations

This module provides functions for computing drawdown metrics from equity curves
or return series.

Author: Trading Research Team
Date: 2025-01-XX
"""

import pandas as pd
from typing import Union
from metrics.equity import cumulative_returns, equity_peak


def drawdown_series(
    returns: Union[pd.Series, None] = None,
    equity: Union[pd.Series, None] = None
) -> pd.Series:
    """
    Compute drawdown series from returns or equity curve.
    
    Drawdown is calculated as: (equity - peak) / peak
    Returns negative values (drawdown is negative).
    
    Parameters
    ----------
    returns : pd.Series, optional
        Series of returns. If provided, equity curve is computed first.
    equity : pd.Series, optional
        Equity curve values. If provided, used directly.
        Either returns or equity must be provided.
        
    Returns
    -------
    pd.Series
        Drawdown series (negative values) with same index as input
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    >>> dd = drawdown_series(returns=returns)
    >>> print(dd.min())  # Maximum drawdown (most negative)
    
    >>> equity = pd.Series([1.0, 1.1, 1.05, 1.15])
    >>> dd = drawdown_series(equity=equity)
    """
    if equity is None:
        if returns is None:
            raise ValueError("Either returns or equity must be provided")
        equity = cumulative_returns(returns)
    elif returns is not None:
        raise ValueError("Provide either returns or equity, not both")
    
    peak = equity_peak(equity)
    drawdown = (equity - peak) / peak
    
    return drawdown


def max_drawdown(
    returns: Union[pd.Series, None] = None,
    equity: Union[pd.Series, None] = None
) -> float:
    """
    Compute maximum drawdown from returns or equity curve.
    
    Maximum drawdown is the most negative drawdown value (largest peak-to-trough decline).
    
    Parameters
    ----------
    returns : pd.Series, optional
        Series of returns. If provided, equity curve is computed first.
    equity : pd.Series, optional
        Equity curve values. If provided, used directly.
        Either returns or equity must be provided.
        
    Returns
    -------
    float
        Maximum drawdown (negative value, e.g., -0.10 for 10% drawdown)
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    >>> md = max_drawdown(returns=returns)
    >>> print(f"Max Drawdown: {md:.2%}")
    """
    drawdown = drawdown_series(returns=returns, equity=equity)
    return drawdown.min()
