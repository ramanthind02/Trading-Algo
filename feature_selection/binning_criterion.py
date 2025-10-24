"""
Binning-Based Criterion Functions for Permutation Tests

This module provides criterion functions based on bin selection strategies
for use in permutation tests. Instead of using profit factor with threshold
optimization, these functions evaluate features based on their ability to
create bins with high Sortino ratios.

Author: Trading Research Team
Date: 2025-10-19
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_bin_sortino_criterion(
    feature_series: pd.Series,
    target_series: pd.Series,
    n_bins: int = 3,
    min_samples_per_bin_pct: float = 0.10
) -> Dict[str, float]:
    """
    Compute criterion based on best bin's Sortino ratio.
    
    This function:
    1. Creates n_bins using quantile binning
    2. Computes Sortino ratio for each bin
    3. Returns the maximum Sortino ratio as the criterion
    
    This is suitable for in-sample permutation tests where we want to test
    if a feature can create bins with significantly better risk-adjusted returns
    than random permutations.
    
    Parameters
    ----------
    feature_series : pd.Series
        Feature values
    target_series : pd.Series
        Target values (returns)
    n_bins : int, default=3
        Number of bins to create
    min_samples_per_bin_pct : float, default=0.10
        Minimum percentage of samples required per bin
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'best_sortino': Maximum Sortino ratio across all bins
        - 'best_bin': Index of bin with highest Sortino
        - 'n_bins_created': Actual number of bins created
        - 'mean_return': Mean return of best bin
        - 'downside_std': Downside std of best bin
        
    Examples
    --------
    >>> result = compute_bin_sortino_criterion(feature, target, n_bins=3)
    >>> print(f"Best Sortino: {result['best_sortino']:.2f}")
    """
    # Create DataFrame and drop NaN
    df = pd.DataFrame({'feature': feature_series, 'target': target_series}).dropna()
    
    if len(df) < n_bins * 10:
        # Not enough data
        return {
            'best_sortino': -1e60,
            'best_bin': -1,
            'n_bins_created': 0,
            'mean_return': 0.0,
            'downside_std': 0.0
        }
    
    # Check if feature is binary
    unique_values = df['feature'].unique()
    n_unique = len(unique_values)
    
    if n_unique == 2:
        # Binary feature: create exactly 2 bins
        sorted_values = np.sort(unique_values)
        df['bin'] = (df['feature'] == sorted_values[1]).astype(int)
        n_bins_actual = 2
    elif n_unique < n_bins:
        # Fewer unique values than requested bins
        n_bins_actual = n_unique
        try:
            df['bin'] = pd.qcut(df['feature'], n_bins_actual, labels=False, duplicates='drop')
        except ValueError:
            df['bin'] = pd.cut(df['feature'], n_bins_actual, labels=False, duplicates='drop')
    else:
        # Normal case: use quantile binning
        n_bins_actual = n_bins
        try:
            df['bin'] = pd.qcut(df['feature'], n_bins, labels=False, duplicates='drop')
        except ValueError:
            # If qcut fails, use equal-width bins
            df['bin'] = pd.cut(df['feature'], n_bins, labels=False, duplicates='drop')
    
    # Calculate Sortino ratio for each bin
    bin_sortinos = []
    bin_stats = {}
    
    for bin_idx in sorted(df['bin'].unique()):
        bin_data = df[df['bin'] == bin_idx]
        
        # Check minimum samples
        min_samples = max(int(len(df) * min_samples_per_bin_pct), 10)
        if len(bin_data) < min_samples:
            continue
        
        mean_ret = bin_data['target'].mean()
        
        # Calculate downside deviation (only negative returns)
        downside_returns = bin_data['target'][bin_data['target'] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Calculate annualized Sortino ratio
        MIN_STD = 1e-6
        if downside_std > 0:
            sortino = (mean_ret / max(downside_std, MIN_STD)) * np.sqrt(252)
        else:
            # No downside volatility - use mean return if positive
            sortino = mean_ret * np.sqrt(252) if mean_ret > 0 else 0
        
        bin_sortinos.append(sortino)
        bin_stats[bin_idx] = {
            'sortino': sortino,
            'mean_return': mean_ret,
            'downside_std': downside_std,
            'count': len(bin_data)
        }
    
    if len(bin_sortinos) == 0:
        # No valid bins
        return {
            'best_sortino': -1e60,
            'best_bin': -1,
            'n_bins_created': 0,
            'mean_return': 0.0,
            'downside_std': 0.0
        }
    
    # Find best bin
    best_sortino = max(bin_sortinos)
    best_bin_idx = max(bin_stats.keys(), key=lambda k: bin_stats[k]['sortino'])
    best_stats = bin_stats[best_bin_idx]
    
    return {
        'best_sortino': best_sortino,
        'best_bin': best_bin_idx,
        'n_bins_created': len(bin_stats),
        'mean_return': best_stats['mean_return'],
        'downside_std': best_stats['downside_std'],
        'bin_count': best_stats['count']
    }


class BinningSortinoCriterion:
    """
    Picklable criterion function for binning-based permutation tests.
    
    This class wraps the binning Sortino computation in a callable object
    that can be pickled for multiprocessing.
    
    Examples
    --------
    >>> criterion = BinningSortinoCriterion(n_bins=3)
    >>> result = criterion(data, 'my_feature')
    >>> print(f"Best Sortino: {result:.2f}")
    """
    
    def __init__(
        self,
        n_bins: int = 3,
        min_samples_per_bin_pct: float = 0.10,
        target_col: str = 'target'
    ):
        """
        Initialize criterion function.
        
        Parameters
        ----------
        n_bins : int, default=3
            Number of bins to create
        min_samples_per_bin_pct : float, default=0.10
            Minimum percentage of samples per bin
        target_col : str, default='target'
            Name of target column in DataFrame
        """
        self.n_bins = n_bins
        self.min_samples_per_bin_pct = min_samples_per_bin_pct
        self.target_col = target_col
    
    def __call__(self, data: pd.DataFrame, feature_col: str) -> float:
        """
        Compute best Sortino ratio for a feature.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing feature and target
        feature_col : str
            Name of feature column
            
        Returns
        -------
        float
            Best Sortino ratio across all bins
        """
        result = compute_bin_sortino_criterion(
            feature_series=data[feature_col],
            target_series=data[self.target_col],
            n_bins=self.n_bins,
            min_samples_per_bin_pct=self.min_samples_per_bin_pct
        )
        return result['best_sortino']
    
    def compute_full_result(
        self,
        feature_series: pd.Series,
        target_series: pd.Series
    ) -> Dict[str, float]:
        """
        Compute full result dictionary (for original criterion computation).
        
        Parameters
        ----------
        feature_series : pd.Series
            Feature values
        target_series : pd.Series
            Target values
            
        Returns
        -------
        Dict[str, float]
            Full result dictionary with all statistics
        """
        return compute_bin_sortino_criterion(
            feature_series=feature_series,
            target_series=target_series,
            n_bins=self.n_bins,
            min_samples_per_bin_pct=self.min_samples_per_bin_pct
        )
