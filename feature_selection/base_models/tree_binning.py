"""
Decision Tree Binning Model

This module implements a decision tree-based binning model that follows the sklearn API.
It uses DecisionTreeRegressor to find optimal split points based on the target variable.

Author: Trading Research Team
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings

from sklearn.tree import DecisionTreeRegressor
from feature_selection.base_models.base_model import BinningModelBase


class DecisionTreeBinningModel(BinningModelBase):
    """
    Decision tree-based binning model with sklearn-style API.
    
    Uses DecisionTreeRegressor to find optimal split points based on the target
    variable. This is a supervised approach that can capture non-linear relationships
    and natural breakpoints in the feature-target relationship.
    
    This model follows the sklearn estimator API:
    - fit(X, y): Train the model on feature and target data
    - predict(X, strategy='long'): Generate binary signals for new data
    - get_params(): Get model hyperparameters
    - set_params(**params): Set model hyperparameters
    - score(X, y): Evaluate model performance (optional)
    
    Parameters
    ----------
    n_bins : int, default=3
        Number of bins to create (max_leaf_nodes for the tree)
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std (risk-adjusted)
        - 'mean': mean return only
    min_samples_leaf_pct : float, default=0.05
        Minimum percentage of samples required in each leaf (prevents overfitting)
        Default is 5% of training data per leaf
        
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
    tree_ : DecisionTreeRegressor
        Fitted decision tree (set after fit)
    is_fitted_ : bool
        Whether the model has been fitted
        
    Examples
    --------
    >>> from feature_selection.base_models import DecisionTreeBinningModel
    >>> 
    >>> # Create model
    >>> model = DecisionTreeBinningModel(n_bins=3, min_samples_leaf_pct=0.05)
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
    >>> print(f"Tree created {len(model.thresholds_)} splits")
    >>> 
    >>> # Get/set hyperparameters
    >>> params = model.get_params()
    >>> model.set_params(min_samples_leaf_pct=0.10)
    
    Notes
    -----
    Decision tree binning differs from quantile binning:
    - Supervised: Uses target variable to find optimal splits
    - Adaptive: Finds natural breakpoints in feature-target relationship
    - May be more stable: Captures true non-linear patterns
    - Regularized: min_samples_leaf prevents overfitting
    
    If the tree cannot create enough splits (e.g., feature has limited unique values),
    it will fall back to quantile binning with a warning.
    """
    
    def __init__(
        self, 
        n_bins: int = 3, 
        selection_metric: str = 'sortino', 
        min_samples_leaf_pct: float = 0.05,
        strategy: str = 'long'
    ):
        """
        Initialize decision tree binning model.
        
        Parameters
        ----------
        n_bins : int, default=3
            Number of bins to create (max_leaf_nodes for the tree)
        selection_metric : str, default='sortino'
            Metric to use for bin selection ('sortino' or 'mean')
        min_samples_leaf_pct : float, default=0.05
            Minimum percentage of samples required in each leaf
        strategy : str, default='long'
            Strategy type: 'long' or 'short'
        """
        super().__init__(n_bins=n_bins, selection_metric=selection_metric, strategy=strategy)
        self.min_samples_leaf_pct = min_samples_leaf_pct
        self.tree_ = None
    
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """
        Create bins using decision tree splitting.
        
        This method implements the abstract _create_bins method from BaseModel.
        It uses DecisionTreeRegressor to find optimal split points based on
        the target variable.
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (used to find optimal splits)
            
        Returns
        -------
        pd.Series
            Bin assignments for each sample
        """
        min_samples_leaf = max(int(len(feature_data) * self.min_samples_leaf_pct), 10)
        
        # Fit decision tree regressor
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.n_bins,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=0.00001,
            random_state=42
        )
        
        X = feature_data.values.reshape(-1, 1)
        y = target_data.values
        self.tree_.fit(X, y)
        
        # Extract thresholds from tree structure
        thresholds = []
        tree_ = self.tree_.tree_
        
        def extract_thresholds(node=0):
            """Recursively extract split thresholds from tree."""
            if tree_.feature[node] != -2:  # Not a leaf node
                thresholds.append(tree_.threshold[node])
                # Recurse on children
                if tree_.children_left[node] != -1:
                    extract_thresholds(tree_.children_left[node])
                if tree_.children_right[node] != -1:
                    extract_thresholds(tree_.children_right[node])
        
        extract_thresholds()
        thresholds_array = np.array(sorted(set(thresholds)))
        
        # Fallback to quantile binning if tree didn't create enough splits
        if len(thresholds_array) < self.n_bins - 1:
            warnings.warn(
                f"Decision tree only created {len(thresholds_array)} splits (expected {self.n_bins-1}). "
                f"Falling back to quantile binning. Try lowering min_samples_leaf_pct or reducing n_bins."
            )
            # Use quantile binning as fallback
            try:
                bins = pd.qcut(feature_data, self.n_bins, labels=False, duplicates='drop')
            except ValueError:
                bins = pd.cut(feature_data, self.n_bins, labels=False, duplicates='drop')
        else:
            # Assign bins based on tree thresholds
            bins = pd.Series(np.digitize(feature_data.values, thresholds_array), index=feature_data.index)
        
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
            'min_samples_leaf_pct': self.min_samples_leaf_pct,
            'normalize_by': self.normalize_by
        }
    
    def set_params(self, **params) -> 'decision_tree_binning':
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
        self.tree_ = None
        
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
    
    def get_feature_importance(self) -> Optional[float]:
        """
        Get feature importance from the fitted tree.
        
        Returns
        -------
        float or None
            Feature importance score, or None if model not fitted
        """
        if not self.is_fitted_ or self.tree_ is None:
            return None
        
        return self.tree_.feature_importances_[0] if len(self.tree_.feature_importances_) > 0 else None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"DecisionTreeBinningModel(n_bins={self.n_bins}, "
                f"selection_metric='{self.selection_metric}', "
                f"min_samples_leaf_pct={self.min_samples_leaf_pct})")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"DecisionTreeBinningModel(n_bins={self.n_bins}, {fitted_str})"
    model_type = "decision_tree_binning"
