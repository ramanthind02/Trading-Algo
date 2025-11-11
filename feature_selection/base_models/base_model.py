"""
Base Model Classes for Walk-Forward Analysis

This module provides a sklearn-style interface for different binning strategies
used in walk-forward analysis. All models follow the fit/predict pattern and
use Sortino ratio for bin/leaf selection.

Author: Trading Research Team
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
from sklearn.tree import DecisionTreeRegressor


class BaseModel(ABC):
    """
    Abstract base class for walk-forward models.
    
    Follows sklearn-style interface with fit() and predict() methods.
    Subclasses must implement the binning/splitting logic and prediction.
    
    Parameters
    ----------
    n_bins : int, default=3
        Number of bins/leaves to create
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std (risk-adjusted)
        - 'mean': mean return only
    """
    
    def __init__(self, n_bins: int = 3, selection_metric: str = 'sortino', normalize_by: Optional[str] = 'ewsd'):
        self.n_bins = n_bins
        self.selection_metric = selection_metric
        self.normalize_by = normalize_by  # 'ewsd', 'atr', or None
        
        # Fitted parameters (set during fit())
        self.thresholds_ = None
        self.best_long_bin_ = None
        self.best_short_bin_ = None
        self.bin_stats_ = None
        self.is_fitted_ = False
        self.normalization_data_ = None  # Store normalization series
    
    def _normalize_feature(
        self,
        feature_data: pd.Series,
        normalization_data: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Normalize feature by volatility (EWSD or ATR).
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values to normalize
        normalization_data : Optional[pd.Series]
            Volatility data (EWSD or ATR) aligned with feature_data.
            If None and normalize_by is set, will raise an error.
            
        Returns
        -------
        pd.Series
            Normalized feature values
        """
        if self.normalize_by is None:
            return feature_data
        
        if normalization_data is None:
            raise ValueError(
                f"normalize_by='{self.normalize_by}' but no normalization_data provided. "
                f"Pass the {self.normalize_by.upper()} column when calling fit()."
            )
        
        # Align indices
        aligned_feature = feature_data.reindex(normalization_data.index)
        
        # Safe division with minimum threshold
        MIN_VOL = 1e-8 if self.normalize_by == 'ewsd' else 1.0
        safe_vol = normalization_data.clip(lower=MIN_VOL)
        
        # Normalize: feature / volatility
        normalized = aligned_feature / safe_vol
        
        return normalized
    
    @abstractmethod
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """
        Create bins from feature data.
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (used for supervised binning)
            
        Returns
        -------
        pd.Series
            Bin assignments for each sample
        """
        pass
    
    def _calculate_bin_stats(self, df: pd.DataFrame, n_bins: int) -> Dict:
        """
        Calculate statistics for each bin.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'feature', 'target', and 'bin' columns
        n_bins : int
            Number of bins
            
        Returns
        -------
        Dict
            Dictionary with statistics for each bin
        """
        bin_stats = {}
        MIN_STD = 1e-6
        
        for bin_idx in range(n_bins):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                mean_ret = bin_data['target'].mean()
                std_ret = bin_data['target'].std()
                
                # Calculate downside deviation (only negative returns)
                downside_returns = bin_data['target'][bin_data['target'] < 0]
                downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
                
                # Calculate Sortino ratio (annualized) for long strategy
                sortino_metric = (mean_ret / max(downside_std, MIN_STD)) * np.sqrt(252) if downside_std > 0 else (mean_ret * np.sqrt(252) if mean_ret > 0 else 0)
                
                # Calculate Sortino ratio on negated returns for short strategy
                # When shorting, we profit from negative returns (so we get -return)
                # Negated mean = -mean_ret
                # For negated returns, downside is when negated return < 0, i.e., when original return > 0
                upside_returns = bin_data['target'][bin_data['target'] > 0]
                upside_std = upside_returns.std() if len(upside_returns) > 0 else 0
                negated_mean = -mean_ret
                sortino_metric_short = (negated_mean / max(upside_std, MIN_STD)) * np.sqrt(252) if upside_std > 0 else (negated_mean * np.sqrt(252) if negated_mean > 0 else 0)
                
                bin_stats[bin_idx] = {
                    'mean_return': mean_ret,
                    'std_return': std_ret,
                    'downside_std': downside_std,
                    'sortino_metric': sortino_metric,
                    'sortino_metric_short': sortino_metric_short,
                    'count': len(bin_data),
                    'feature_min': bin_data['feature'].min(),
                    'feature_max': bin_data['feature'].max()
                }
        
        return bin_stats
    
    def _select_best_bins(self, bin_stats: Dict, n_unique: int) -> Tuple[int, int]:
        """
        Select best bins for long and short strategies.
        
        Parameters
        ----------
        bin_stats : Dict
            Statistics for each bin
        n_unique : int
            Number of unique feature values
            
        Returns
        -------
        Tuple[int, int]
            (best_long_bin, best_short_bin)
        """
        if n_unique == 2:
            # For binary features:
            # Long: select bin 1 (the "signal active" bin)
            # Short: select bin 0 (the "signal inactive" bin)
            best_long_bin = 1
            best_short_bin = 0
        elif self.selection_metric == 'sortino':
            # Long: Find bin with highest Sortino ratio
            best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['sortino_metric'])
            
            # Short: Find bin with highest Sortino ratio on negated returns
            # (shorting profits from negative returns, so we compute Sortino on -returns)
            best_short_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['sortino_metric_short'])
            
        elif self.selection_metric == 'mean':
            # Long: Find bin with highest mean return
            best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
            
            # Short: Find bin with lowest mean return (most negative)
            # When shorting, we profit from negative returns, so we want the most negative mean
            best_short_bin = min(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
        else:
            raise ValueError(f"Unknown selection_metric: {self.selection_metric}. Use 'sortino' or 'mean'")
        
        return best_long_bin, best_short_bin
    
    def _extract_thresholds(self, df: pd.DataFrame, n_bins: int) -> np.ndarray:
        """
        Extract threshold values from bin assignments.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'feature' and 'bin' columns
        n_bins : int
            Number of bins
            
        Returns
        -------
        np.ndarray
            Array of threshold values
        """
        thresholds = []
        for bin_idx in range(n_bins - 1):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                thresholds.append(bin_data['feature'].max())
        
        return np.array(sorted(thresholds))
    
    def fit(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        normalization_data: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Binning is ALWAYS done on raw features, not normalized features.
        Normalization is only applied to signals after binning (via predict_scaled).
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values (training data) - RAW, not normalized
        target_data : pd.Series
            Target values (training data)
        normalization_data : Optional[pd.Series]
            Volatility data for normalization (EWSD or ATR).
            Stored for later use in predict_scaled().
            
        Returns
        -------
        self
            Fitted model
        """
        # Store normalization data for later use in predict_scaled()
        self.normalization_data_ = normalization_data
        
        # Create clean dataset (NO normalization - bin on raw features)
        df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
        
        if len(df) < self.n_bins * 10:
            raise ValueError(f"Insufficient data: need at least {self.n_bins * 10} samples")
        
        # Check if feature is binary
        unique_values = df['feature'].unique()
        n_unique = len(unique_values)
        
        if n_unique == 2:
            # Binary feature: create exactly 2 bins
            sorted_values = np.sort(unique_values)
            df['bin'] = (df['feature'] == sorted_values[1]).astype(int)
            n_bins = 2
            self.thresholds_ = np.array([sorted_values[0]])
        else:
            # Use subclass-specific binning logic
            df['bin'] = self._create_bins(df['feature'], df['target'])
            n_bins = self.n_bins
            self.thresholds_ = self._extract_thresholds(df, n_bins)
        
        # Calculate bin statistics
        self.bin_stats_ = self._calculate_bin_stats(df, n_bins)
        
        # Select best bins for long and short
        self.best_long_bin_, self.best_short_bin_ = self._select_best_bins(self.bin_stats_, n_unique)
        
        self.is_fitted_ = True
        return self
    
    def predict(
        self,
        feature_data: pd.Series,
        strategy: str = 'long',
        normalization_data: Optional[pd.Series] = None,
        scaled: bool = False
    ) -> pd.Series:
        """
        Predict signals for new data.
        
        Binning is ALWAYS done on RAW features. If scaled=True and normalize_by is set,
        volatility scaling is applied AFTER binning to convert binary signals to position sizes.
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values to classify - RAW, not normalized
        strategy : str, default='long'
            Strategy to use: 'long' or 'short'
        normalization_data : Optional[pd.Series]
            Volatility data (EWSD or ATR) for scaling.
            Required if scaled=True and normalize_by is set.
        scaled : bool, default=False
            If True, apply volatility scaling to signals (position sizes).
            If False, return binary signals (1 or 0).
            
        Returns
        -------
        pd.Series
            If scaled=False: Binary signals (1 if in best bin, 0 otherwise)
            If scaled=True: Volatility-scaled position sizes (0 or 1/volatility)
            
        Examples
        --------
        >>> # Get binary signals (1 or 0)
        >>> signals = model.predict(X_test, strategy='long', scaled=False)
        >>> 
        >>> # Get volatility-scaled position sizes
        >>> positions = model.predict(
        ...     X_test,
        ...     strategy='long',
        ...     normalization_data=ewsd_test,
        ...     scaled=True
        ... )
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict()")
        
        # Select appropriate bin based on strategy
        if strategy == 'long':
            best_bin = self.best_long_bin_
        elif strategy == 'short':
            best_bin = self.best_short_bin_
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'long' or 'short'")
        
        # Assign bins based on thresholds (always on RAW features)
        bins = np.digitize(feature_data.values, self.thresholds_)
        
        # Create binary signal: 1 if in best bin, 0 otherwise
        signal = pd.Series((bins == best_bin).astype(int), index=feature_data.index)
        
        # If not scaled or no normalization, return binary signals
        if not scaled or self.normalize_by is None:
            return signal
        
        # Apply volatility scaling to signals
        if normalization_data is None:
            raise ValueError(
                f"scaled=True and normalize_by='{self.normalize_by}' but no normalization_data provided. "
                f"Pass the {self.normalize_by.upper()} column when calling predict()."
            )
        
        # Align indices
        aligned_signal = signal.reindex(normalization_data.index, fill_value=0)
        
        # Safe division with minimum threshold
        MIN_VOL = 1e-8 if self.normalize_by == 'ewsd' else 1.0
        safe_vol = normalization_data.clip(lower=MIN_VOL)
        
        # Scale signals: signal / volatility
        # If signal=0, result=0. If signal=1, result=1/vol (position size)
        scaled_signal = aligned_signal / safe_vol
        
        return scaled_signal
    
    def get_bin_stats(self) -> Dict:
        """
        Get statistics for each bin.
        
        Returns
        -------
        Dict
            Dictionary with statistics for each bin
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling get_bin_stats()")
        
        return self.bin_stats_
    
    def compute_objective_metric(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        objective_metric: 'ObjectiveMetric',
        strategy: str = 'long',
        normalization_data: Optional[pd.Series] = None
    ) -> float:
        """
        Fit model and compute objective metric for the best bin.
        
        This is the main method used by permutation tests. It encapsulates:
        1. Fitting the model on the data
        2. Getting predictions for the specified strategy
        3. Computing the objective metric on selected returns
        
        This follows the Single Responsibility Principle: the model is responsible
        for its own performance evaluation.
        
        Parameters
        ----------
        feature_data : pd.Series
            Feature values
        target_data : pd.Series
            Target values (returns)
        objective_metric : ObjectiveMetric
            Metric instance to compute (e.g., SortinoRatio, SharpeRatio)
        strategy : str, default='long'
            Strategy to evaluate: 'long' or 'short'
        normalization_data : Optional[pd.Series]
            Volatility data for normalization (EWSD or ATR).
            Required if normalize_by is set.
            
        Returns
        -------
        float
            Objective metric value for the best bin
            
        Examples
        --------
        >>> from feature_selection.base_models import QuantileBinningModel
        >>> from utils.objective_metric import SortinoRatio
        >>> 
        >>> model = QuantileBinningModel(n_bins=3)
        >>> metric = SortinoRatio(annualization_factor=252)
        >>> 
        >>> # Compute metric (fits model internally)
        >>> sortino = model.compute_objective_metric(
        ...     feature_data=X_train,
        ...     target_data=y_train,
        ...     objective_metric=metric,
        ...     strategy='long'
        ... )
        """
        # Fit the model
        self.fit(feature_data, target_data, normalization_data=normalization_data)
        
        # Get predictions for the specified strategy
        signals = self.predict(feature_data, strategy=strategy, normalization_data=normalization_data)
        
        # Get selected returns
        selected_returns = target_data[signals == 1]
        
        if len(selected_returns) == 0:
            return 0.0
        
        # For short strategy, negate returns (shorting profits from negative returns)
        if strategy == 'short':
            selected_returns = -selected_returns
        
        # Compute objective metric
        metric_value = objective_metric.compute(selected_returns)
        
        return metric_value



