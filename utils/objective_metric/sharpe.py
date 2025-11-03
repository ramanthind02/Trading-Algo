"""
Sharpe Ratio Metric

This module implements the Sharpe ratio, a classic risk-adjusted performance metric
that measures excess return per unit of total volatility.

Author: Trading Research Team
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
from typing import Union

from utils.objective_metric.base_metric import ObjectiveMetric


class SharpeRatio(ObjectiveMetric):
    """
    Sharpe ratio metric.
    
    The Sharpe ratio measures the excess return per unit of total volatility.
    It's one of the most widely used risk-adjusted performance metrics.
    
    Formula:
        Sharpe = (Mean Return - Risk-Free Rate) / Standard Deviation * sqrt(annualization_factor)
    
    Parameters
    ----------
    annualization_factor : float, default=252
        Factor to annualize the metric (252 for daily returns)
    min_std : float, default=1e-6
        Minimum standard deviation to prevent division by zero
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized). For daily returns, this is typically very small.
        
    Examples
    --------
    >>> sharpe = SharpeRatio(annualization_factor=252)
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> ratio = sharpe.compute(returns)
    >>> print(f"Sharpe Ratio: {ratio:.2f}")
    
    >>> # Use as callable
    >>> ratio = sharpe(returns)
    
    >>> # With risk-free rate
    >>> sharpe = SharpeRatio(annualization_factor=252, risk_free_rate=0.02)
    >>> ratio = sharpe(returns)
    
    Notes
    -----
    The Sharpe ratio penalizes both upside and downside volatility equally.
    For strategies where upside volatility is desirable, consider using
    Sortino ratio instead.
    
    The Sharpe ratio assumes:
    - Returns are normally distributed (often violated in practice)
    - Volatility is a good measure of risk
    - Upside and downside volatility are equally undesirable
    """
    
    def __init__(
        self, 
        annualization_factor: float = 252, 
        min_std: float = 1e-6,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize Sharpe ratio metric.
        
        Parameters
        ----------
        annualization_factor : float, default=252
            Annualization factor
        min_std : float, default=1e-6
            Minimum standard deviation
        risk_free_rate : float, default=0.0
            Risk-free rate (annualized)
        """
        super().__init__(annualization_factor=annualization_factor, min_std=min_std)
        self.risk_free_rate = risk_free_rate
        # Convert annualized risk-free rate to per-period rate
        self.risk_free_rate_per_period = risk_free_rate / annualization_factor
    
    def compute(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute Sharpe ratio from returns.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        # Validate and convert to numpy array
        returns_array = self._validate_returns(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        # Compute mean return
        mean_return = self._compute_mean(returns_array)
        
        # Compute standard deviation
        std_return = self._compute_std(returns_array)
        
        # Compute excess return
        excess_return = mean_return - self.risk_free_rate_per_period
        
        # Compute Sharpe ratio
        sharpe = excess_return / std_return
        
        # Annualize
        return self._annualize(sharpe)
    
    def compute_volatility(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute annualized volatility (for diagnostic purposes).
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Annualized standard deviation
        """
        returns_array = self._validate_returns(returns)
        std_return = self._compute_std(returns_array)
        return self._annualize(std_return)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SharpeRatio(annualization_factor={self.annualization_factor}, "
                f"risk_free_rate={self.risk_free_rate})")
