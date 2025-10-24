"""
Sortino Ratio Metric

This module implements the Sortino ratio, a risk-adjusted performance metric
that only penalizes downside volatility (negative returns).

Author: Trading Research Team
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
from typing import Union

from feature_selection.objective_metric.base_metric import ObjectiveMetric


class SortinoRatio(ObjectiveMetric):
    """
    Sortino ratio metric.
    
    The Sortino ratio is a variation of the Sharpe ratio that only penalizes
    downside volatility (returns below a target return, typically 0). This makes
    it more suitable for strategies where upside volatility is desirable.
    
    Formula:
        Sortino = (Mean Return - Target Return) / Downside Deviation * sqrt(annualization_factor)
    
    Where downside deviation is the standard deviation of returns below the target.
    
    Parameters
    ----------
    annualization_factor : float, default=252
        Factor to annualize the metric (252 for daily returns)
    min_std : float, default=1e-6
        Minimum downside deviation to prevent division by zero
    target_return : float, default=0.0
        Target return threshold (returns below this are considered downside)
        
    Examples
    --------
    >>> sortino = SortinoRatio(annualization_factor=252)
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> ratio = sortino.compute(returns)
    >>> print(f"Sortino Ratio: {ratio:.2f}")
    
    >>> # Use as callable
    >>> ratio = sortino(returns)
    
    Notes
    -----
    The Sortino ratio is preferred over Sharpe ratio when:
    - Return distribution is asymmetric
    - Upside volatility is desirable (not a risk)
    - Strategy has positive skew
    - Downside protection is more important than overall volatility
    """
    
    def __init__(
        self, 
        annualization_factor: float = 252, 
        min_std: float = 1e-6,
        target_return: float = 0.0
    ):
        """
        Initialize Sortino ratio metric.
        
        Parameters
        ----------
        annualization_factor : float, default=252
            Annualization factor
        min_std : float, default=1e-6
            Minimum downside deviation
        target_return : float, default=0.0
            Target return threshold
        """
        super().__init__(annualization_factor=annualization_factor, min_std=min_std)
        self.target_return = target_return
    
    def compute(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute Sortino ratio from returns.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Annualized Sortino ratio
        """
        # Validate and convert to numpy array
        returns_array = self._validate_returns(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        # Compute mean return
        mean_return = self._compute_mean(returns_array)
        
        # Compute downside deviation (only negative returns)
        downside_returns = returns_array[returns_array < self.target_return]
        
        if len(downside_returns) == 0:
            # No downside volatility
            if mean_return > self.target_return:
                # Positive mean with no downside - return scaled mean
                return self._annualize(mean_return)
            else:
                return 0.0
        
        # Calculate downside standard deviation
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else self.min_std
        downside_std = max(downside_std, self.min_std)
        
        # Compute Sortino ratio
        sortino = (mean_return - self.target_return) / downside_std
        
        # Annualize
        return self._annualize(sortino)
    
    def compute_downside_deviation(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute downside deviation (for diagnostic purposes).
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Downside standard deviation
        """
        returns_array = self._validate_returns(returns)
        downside_returns = returns_array[returns_array < self.target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else self.min_std
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SortinoRatio(annualization_factor={self.annualization_factor}, "
                f"target_return={self.target_return})")
