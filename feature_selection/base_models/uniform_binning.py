"""
Uniform Binning Model

This module implements a uniform (equal-width) binning model that follows the sklearn API.
It creates bins using equal-width intervals (unlike quantile binning which uses equal-frequency).

Author: Trading Research Team
Date: 2025-01-25
"""

import pandas as pd
import numpy as np
from typing import Optional

from feature_selection.base_models.base_model import BinningModelBase


class UniformBinningModel(BinningModelBase):
    """
    Uniform (equal-width) binning model with sklearn-style API.
    
    Creates bins using equal-width intervals, where each bin spans the same range
    of feature values. This is useful when you want to understand behavior at
    specific feature value ranges, regardless of sample distribution.
    
    Unlike ContinuousBinningModel which uses equal-frequency bins (pd.qcut),
    UniformBinningModel uses equal-width bins (pd.cut).
    
    This model follows the sklearn estimator API:
    - fit(X, y): Train the model on feature and target data
    - predict(X, strategy='long'): Generate binary signals for new data
    - get_params(): Get model hyperparameters
    - set_params(**params): Set model hyperparameters
    - score(X, y): Evaluate model performance (optional)
    
    Parameters
    ----------
    n_bins : int, default=3
        Number of bins to create
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std (risk-adjusted)
        - 'mean': mean return only
    strategy : str, default='long'
        Strategy type: 'long' or 'short'
        
    Attributes
    ----------
    thresholds_ : np.ndarray
        Threshold values defining bin boundaries (set after fit)
    best_long_bin_ : int
        Index of the bin selected for long strategy (set after fit)
    best_short_bin_ : int
        Index of the bin selected for short strategy (set after fit)
    bin_stats_ : dict
        Statistics for each bin (set after fit)
    is_fitted_ : bool
        Whether the model has been fitted
        
    Examples
    --------
    >>> from feature_selection.base_models import UniformBinningModel
    >>> 
    >>> # Create model
    >>> model = UniformBinningModel(n_bins=5, selection_metric='sortino')
    >>> 
    >>> # Fit on training data
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Generate predictions
    >>> signals_long = model.predict(X_test, strategy='long')
    >>> signals_short = model.predict(X_test, strategy='short')
    >>> 
    >>> # Get bin statistics
    >>> stats = model.get_bin_stats()
    >>> print(f"Best long bin: {model.best_long_bin_}")
    >>> print(f"Best short bin: {model.best_short_bin_}")
    """
    
    def __init__(self, n_bins: int = 3, selection_metric: str = 'sortino', strategy: str = 'long'):
        """
        Initialize uniform binning model.
        
        Parameters
        ----------
        n_bins : int, default=3
            Number of bins to create
        selection_metric : str, default='sortino'
            Metric to use for bin selection ('sortino' or 'mean')
        strategy : str, default='long'
            Strategy type: 'long' or 'short'
        """
        super().__init__(n_bins=n_bins, selection_metric=selection_metric, strategy=strategy)
    
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """
        Create bins using equal-width intervals (uniform binning).
        
        This method implements the abstract _create_bins method from BinningModelBase.
        It uses pandas cut to create bins with equal width.
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (not used for uniform binning, but required by interface)
            
        Returns
        -------
        pd.Series
            Bin assignments for each sample
        """
        try:
            bins = pd.cut(feature_data, bins=self.n_bins, labels=False, duplicates='drop', include_lowest=True)
        except ValueError:
            # If cut fails, try with fewer bins
            for n in range(self.n_bins - 1, 1, -1):
                try:
                    bins = pd.cut(feature_data, bins=n, labels=False, duplicates='drop', include_lowest=True)
                    break
                except ValueError:
                    continue
            else:
                # Last resort: use unique values
                unique_vals = feature_data.unique()
                n_unique = len(unique_vals)
                if n_unique <= self.n_bins:
                    sorted_unique = np.sort(unique_vals)
                    bin_map = {val: idx for idx, val in enumerate(sorted_unique)}
                    bins = feature_data.map(bin_map).astype(int)
                else:
                    raise ValueError(f"Could not create {self.n_bins} uniform bins")
        
        return bins
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        This method is part of the sklearn API and enables grid search and
        other hyperparameter tuning methods.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {
            'n_bins': self.n_bins,
            'selection_metric': self.selection_metric,
            'normalize_by': self.normalize_by
        }
    
    def set_params(self, **params) -> 'UniformBinningModel':
        """
        Set the parameters of this estimator.
        
        This method is part of the sklearn API and enables grid search and
        other hyperparameter tuning methods.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        
        # Reset fitted state when parameters change
        self._reset_fitted_state()
        
        return self
    
    def score(self, X: pd.Series, y: pd.Series, strategy: str = 'long') -> float:
        """
        Return the mean return for the selected strategy.
        
        This method is part of the sklearn API and provides a quick way to
        evaluate model performance.
        
        Parameters
        ----------
        X : pd.Series
            Feature values
        y : pd.Series
            Target values (returns)
        strategy : str, default='long'
            Strategy to evaluate: 'long' or 'short'
            
        Returns
        -------
        float
            Mean return for the selected strategy
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling score()")
        
        signals = self.predict(X, strategy=strategy)
        selected_returns = (y * signals)[signals != 0]
        if len(selected_returns) == 0:
            return 0.0
        return selected_returns.mean()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"UniformBinningModel(n_bins={self.n_bins}, selection_metric='{self.selection_metric}')"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"UniformBinningModel(n_bins={self.n_bins}, {fitted_str})"
    model_type = "uniform_binning"
