"""
Base Objective Metric Class

This module provides an abstract base class for objective metrics used in
feature selection and bin evaluation. Objective metrics compute risk-adjusted
performance measures from return distributions.

Author: Trading Research Team
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional


class ObjectiveMetric(ABC):
    """
    Abstract base class for objective metrics.
    
    An objective metric computes a risk-adjusted performance measure from
    a series of returns. Common examples include Sharpe ratio, Sortino ratio,
    Calmar ratio, etc.
    
    All metrics are annualized by default using a configurable annualization factor.
    
    Parameters
    ----------
    annualization_factor : float, default=252
        Factor to annualize the metric (252 for daily returns, 52 for weekly, 12 for monthly)
    min_std : float, default=1e-6
        Minimum standard deviation to prevent division by zero
        
    Attributes
    ----------
    annualization_factor : float
        Annualization factor
    min_std : float
        Minimum standard deviation threshold
    """
    
    def __init__(self, annualization_factor: float = 252, min_std: float = 1e-6):
        """Initialize objective metric."""
        self.annualization_factor = annualization_factor
        self.min_std = min_std
    
    @abstractmethod
    def compute(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute the objective metric from returns.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Computed metric value (annualized)
        """
        pass
    
    def __call__(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Allow the metric to be called as a function.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        float
            Computed metric value
        """
        return self.compute(returns)
    
    def _validate_returns(self, returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Validate and convert returns to numpy array.
        
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns
        -------
        np.ndarray
            Returns as numpy array with NaN values removed
        """
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = np.asarray(returns)
        
        # Remove NaN values
        returns_array = returns_array[~np.isnan(returns_array)]
        
        return returns_array
    
    def _compute_mean(self, returns: np.ndarray) -> float:
        """
        Compute mean return.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns
            
        Returns
        -------
        float
            Mean return
        """
        if len(returns) == 0:
            return 0.0
        return np.mean(returns)
    
    def _compute_std(self, returns: np.ndarray) -> float:
        """
        Compute standard deviation of returns.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns
            
        Returns
        -------
        float
            Standard deviation
        """
        if len(returns) <= 1:
            return self.min_std
        return max(np.std(returns, ddof=1), self.min_std)
    
    def _annualize(self, value: float) -> float:
        """
        Annualize a metric value.
        
        Parameters
        ----------
        value : float
            Value to annualize
            
        Returns
        -------
        float
            Annualized value
        """
        return value * np.sqrt(self.annualization_factor)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(annualization_factor={self.annualization_factor})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__class__.__name__
