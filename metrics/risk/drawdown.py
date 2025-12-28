"""
Drawdown Calculations

This module provides functions for computing drawdown metrics from equity curves
or return series.

Author: Trading Research Team
Date: 2025-01-XX
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
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


def trailing_drawdown_threshold(
    equity: pd.Series,
    trailing_drawdown_pct: float
) -> pd.Series:
    """
    Compute trailing drawdown threshold series.

    The trailing threshold at each point is: peak * (1 - trailing_drawdown_pct)
    This represents the minimum equity level before a trailing drawdown breach.

    Parameters
    ----------
    equity : pd.Series
        Equity curve values (can start at 0 for percentage-based tracking)
    trailing_drawdown_pct : float
        Trailing drawdown percentage as decimal (e.g., 0.05 for 5%)

    Returns
    -------
    pd.Series
        Trailing threshold series with same index as equity

    Examples
    --------
    >>> equity = pd.Series([0.0, 0.02, 0.05, 0.03, 0.08])
    >>> threshold = trailing_drawdown_threshold(equity, 0.05)
    >>> # threshold follows: peak - trailing_drawdown_pct
    """
    peak = equity_peak(equity)
    threshold = peak - trailing_drawdown_pct
    return threshold


def daily_drawdown(returns: pd.Series) -> pd.Series:
    """
    Get daily drawdown values (single-day losses).

    This simply returns the negative returns, which represent single-day losses.
    Used for daily drawdown limit checks in prop firm challenges.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns

    Returns
    -------
    pd.Series
        Daily drawdown (same as returns, negative values are losses)

    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.05])
    >>> daily_dd = daily_drawdown(returns)
    >>> # daily_dd is same as returns: [0.01, -0.02, 0.03, -0.05]
    """
    return returns


def check_drawdown_breach(
    equity: pd.Series,
    max_drawdown_pct: float,
    returns: Optional[pd.Series] = None,
    trailing_drawdown_pct: Optional[float] = None,
    max_daily_drawdown_pct: Optional[float] = None
) -> Tuple[bool, int, str]:
    """
    Check if any drawdown rule has been breached.

    Checks in order: max drawdown, trailing drawdown, daily drawdown.
    Returns on first breach found.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (additive, starting from 0.0)
    max_drawdown_pct : float
        Maximum allowed drawdown from initial equity (e.g., 0.10 for 10%)
    returns : pd.Series, optional
        Daily returns series (required if max_daily_drawdown_pct is set)
    trailing_drawdown_pct : float, optional
        Maximum allowed drawdown from peak (e.g., 0.05 for 5%)
    max_daily_drawdown_pct : float, optional
        Maximum allowed single-day loss (e.g., 0.05 for 5%)

    Returns
    -------
    Tuple[bool, int, str]
        - breached: Whether any rule was breached
        - index: Integer position where breach occurred (-1 if no breach)
        - reason: String describing breach type ("max_drawdown",
          "trailing_drawdown", "daily_drawdown", or "none")

    Examples
    --------
    >>> equity = pd.Series([0.0, 0.02, -0.05, -0.12])
    >>> breached, idx, reason = check_drawdown_breach(
    ...     equity, max_drawdown_pct=0.10
    ... )
    >>> print(f"Breached: {breached}, at index {idx}, reason: {reason}")
    """
    # Check max drawdown (from initial 0.0)
    # Breach if equity falls below -max_drawdown_pct
    for i in range(len(equity)):
        if equity.iloc[i] < -max_drawdown_pct:
            return True, i, "max_drawdown"

    # Check trailing drawdown
    if trailing_drawdown_pct is not None:
        threshold = trailing_drawdown_threshold(equity, trailing_drawdown_pct)
        for i in range(len(equity)):
            if equity.iloc[i] < threshold.iloc[i]:
                return True, i, "trailing_drawdown"

    # Check daily drawdown
    if max_daily_drawdown_pct is not None:
        if returns is None:
            raise ValueError("returns must be provided to check daily drawdown")
        for i in range(len(returns)):
            if returns.iloc[i] < -max_daily_drawdown_pct:
                return True, i, "daily_drawdown"

    return False, -1, "none"
