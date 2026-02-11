"""
Quantile Binning Model

This module implements a quantile-based binning model that follows the sklearn API.
It creates bins using quantile-based splitting (equal number of samples per bin).

Author: Trading Research Team
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
from typing import Optional

from feature_selection.base_models.base_model import BinningModelBase


class QuantileBinningModel(BinningModelBase):
    """
    Quantile-based binning model with sklearn-style API.
    
    Creates bins using quantile-based splitting (equal number of samples per bin).
    This is a simple, unsupervised approach that doesn't use the target variable
    for binning.
    
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
    >>> from feature_selection.base_models import QuantileBinningModel
    >>> 
    >>> # Create model
    >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
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
    >>> 
    >>> # Get/set hyperparameters
    >>> params = model.get_params()
    >>> model.set_params(n_bins=4)
    """
    
    def __init__(self, n_bins: int = 3, selection_metric: str = 'sortino', strategy: str = 'long'):
        """
        Initialize quantile binning model.
        
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
        Create bins using quantile-based splitting, ensuring n_bins are created.
        
        This method implements the abstract _create_bins method from BaseModel.
        It uses pandas qcut to create bins with equal number of samples.
        
        If duplicates cause qcut to collapse bins, uses a hybrid approach:
        1. Try qcut first
        2. If fewer bins created, use unique value boundaries to ensure n_bins
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (not used for quantile binning, but required by interface)
            
        Returns
        -------
        pd.Series
            Bin assignments for each sample
        """
        try:
            bins = pd.qcut(feature_data, self.n_bins, labels=False, duplicates='drop')
            n_created_bins = bins.nunique()
            
            # If we got fewer bins than requested, use a different approach
            if n_created_bins < self.n_bins:
                # Get unique values and create bins based on unique value boundaries
                unique_vals = feature_data.unique()
                n_unique = len(unique_vals)
                
                if n_unique <= self.n_bins:
                    # Not enough unique values: assign each unique value to its own bin
                    # Map unique values to bin indices
                    sorted_unique = np.sort(unique_vals)
                    bin_map = {val: idx for idx, val in enumerate(sorted_unique)}
                    bins = feature_data.map(bin_map).astype(int)
                else:
                    # Enough unique values: try to create bins using quantiles on unique values
                    # This helps when many values are duplicates but we have enough unique values
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    unique_quantiles = np.quantile(unique_vals, quantiles)
                    # Remove duplicates from quantiles
                    unique_quantiles = np.unique(unique_quantiles)
                    
                    if len(unique_quantiles) - 1 >= self.n_bins:
                        # Use quantile-based thresholds on unique values
                        bins = pd.cut(feature_data, bins=unique_quantiles, labels=False, include_lowest=True, duplicates='drop')
                        n_created_bins = bins.nunique()
                    
                    # If still not enough bins, use equal-width binning on the full range
                    # This ensures we get the requested number of bins even if distribution is skewed
                    if n_created_bins < self.n_bins:
                        bins = pd.cut(feature_data, bins=self.n_bins, labels=False, duplicates='drop', include_lowest=True)
                        n_created_bins = bins.nunique()
                        
                        # If equal-width still doesn't work (very few unique values), 
                        # split the data range into n_bins equal-width intervals
                        if n_created_bins < self.n_bins:
                            min_val = feature_data.min()
                            max_val = feature_data.max()
                            # Create n_bins equal-width intervals
                            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
                            # Ensure first and last edges include all values
                            bin_edges[0] = min_val - 1e-10
                            bin_edges[-1] = max_val + 1e-10
                            bins = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
        except (ValueError, TypeError):
            # Fallback: use equal-width bins on the full range
            min_val = feature_data.min()
            max_val = feature_data.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_edges[0] = min_val - 1e-10
            bin_edges[-1] = max_val + 1e-10
            bins = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
        
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
    
    def set_params(self, **params) -> 'QuantileBinningModel':
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
        self.is_fitted_ = False
        self.thresholds_ = None
        self.best_long_bin_ = None
        self.best_short_bin_ = None
        self.bin_stats_ = None
        
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
        return f"QuantileBinningModel(n_bins={self.n_bins}, selection_metric='{self.selection_metric}')"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"QuantileBinningModel(n_bins={self.n_bins}, {fitted_str})"
