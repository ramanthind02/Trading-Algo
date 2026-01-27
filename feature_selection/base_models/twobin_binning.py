"""
Two-Bin Binning Model

This module implements a simple two-bin binning model that splits features
based on positive and negative values. This is specifically designed for
momentum models where the sign of the feature is the primary signal.

Author: Trading Research Team
Date: 2025-01-24
"""

import pandas as pd
import numpy as np
from typing import Optional

from feature_selection.base_models.base_model import BinningModelBase


class TwoBinBinningModel(BinningModelBase):
    """
    Two-bin binning model that splits features based on positive/negative values.
    
    This model creates exactly 2 bins:
    - Bin 0: Negative values (feature < 0)
    - Bin 1: Positive values (feature >= 0)
    
    This is specifically designed for momentum models where the sign of the
    feature is the primary signal.
    
    This model follows the sklearn estimator API:
    - fit(X, y): Train the model on feature and target data
    - predict(X, strategy='long'): Generate binary signals for new data
    - get_params(): Get model hyperparameters
    - set_params(**params): Set model hyperparameters
    
    Parameters
    ----------
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
        Always contains [0.0] to split negative/positive
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
    >>> from feature_selection.base_models import TwoBinBinningModel
    >>> 
    >>> # Create model (always 2 bins)
    >>> model = TwoBinBinningModel(selection_metric='sortino')
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
    
    def __init__(self, selection_metric: str = 'sortino', strategy: str = 'long'):
        """
        Initialize two-bin binning model.
        
        Parameters
        ----------
        selection_metric : str, default='sortino'
            Metric to use for bin selection ('sortino' or 'mean')
        strategy : str, default='long'
            Strategy type: 'long' or 'short'
        """
        # Always use 2 bins for this model
        super().__init__(n_bins=2, selection_metric=selection_metric, strategy=strategy)
    
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """
        Create bins based on positive/negative values.
        
        This method implements the abstract _create_bins method from BinningModelBase.
        It splits features into:
        - Bin 0: Negative values (feature < 0)
        - Bin 1: Positive values (feature >= 0)
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (not used for two-bin splitting, but required by interface)
            
        Returns
        -------
        pd.Series
            Bin assignments for each sample (0 for negative, 1 for positive)
        """
        # Split based on sign: negative (< 0) -> bin 0, positive (>= 0) -> bin 1
        bins = (feature_data >= 0).astype(int)
        
        return bins
    
    def _extract_thresholds(self, df: pd.DataFrame, n_bins: int) -> np.ndarray:
        """
        Extract threshold values for bin boundaries.
        
        For TwoBinBinningModel, the threshold is always 0.0 (splitting negative/positive).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'feature' and 'bin' columns
        n_bins : int
            Number of bins (always 2 for this model)
            
        Returns
        -------
        np.ndarray
            Threshold array containing [0.0]
        """
        # Always split at 0.0 for positive/negative
        return np.array([0.0])
    
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
            'n_bins': self.n_bins,  # Always 2
            'selection_metric': self.selection_metric,
            'normalize_by': self.normalize_by
        }
    
    def set_params(self, **params) -> 'TwoBinBinningModel':
        """
        Set the parameters of this estimator.
        
        This method is part of the sklearn API and enables grid search and
        other hyperparameter tuning methods.
        
        Note: n_bins is always 2 for this model and cannot be changed.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters (n_bins will be ignored if provided)
            
        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            if key == 'n_bins':
                # Ignore n_bins - always 2 for this model
                continue
            elif hasattr(self, key):
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
        
        # Get predictions
        signals = self.predict(X, strategy=strategy)
        
        # Calculate mean return for selected signals
        selected_returns = y[signals == 1]
        
        if len(selected_returns) == 0:
            return 0.0
        
        return selected_returns.mean()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"TwoBinBinningModel(selection_metric='{self.selection_metric}')"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"TwoBinBinningModel(n_bins=2, {fitted_str})"
