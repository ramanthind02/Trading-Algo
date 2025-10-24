"""
Walk-Forward Bin Selection Using Decision Trees

This module implements adaptive bin selection for trading features using
decision trees. For each walk-forward step:
1. Train decision tree on training data to create 3-4 bins
2. Select the bin with highest mean return on training data
3. Apply the decision rule to test data
4. Track performance and selected bins over time

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Tuple, List, Dict, Optional
from datetime import datetime

from utils.walkforward import apply_function_to_walkforward


def train_bin_selector(
    feature_data: pd.Series,
    target_data: pd.Series,
    n_bins: int = 3,
    selection_metric: str = 'sortino'
) -> Tuple[np.ndarray, int, int, Dict]:
    """
    Train a decision tree to create bins and select the best ones for long and short.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values (training data)
    target_data : pd.Series
        Target values (training data)
    n_bins : int, default=3
        Number of bins to create
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std for long, -mean / upside_std for short
        - 'mean': mean return for long, -mean for short
        
    Returns
    -------
    Tuple[np.ndarray, int, int, Dict]
        - thresholds: Array of threshold values defining bins
        - best_long_bin: Index of the bin with best long return (highest)
        - best_short_bin: Index of the bin with best short return (lowest/most negative)
        - bin_stats: Dictionary with statistics for each bin
    """
    # Create bins using quantiles as initial split points
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if len(df) < n_bins * 10:
        raise ValueError(f"Insufficient data: need at least {n_bins * 10} samples")
    
    # Check if feature is binary (only 2 unique values)
    unique_values = df['feature'].unique()
    n_unique = len(unique_values)
    
    if n_unique == 2:
        # Binary feature: create exactly 2 bins using the exact values
        # Sort the unique values to ensure consistent bin assignment
        sorted_values = np.sort(unique_values)
        
        # Assign bins: bin 0 for first value, bin 1 for second value
        df['bin'] = (df['feature'] == sorted_values[1]).astype(int)
        
        # Override n_bins to 2 for binary features
        n_bins = 2
    else:
        # Non-binary feature: use quantiles to create bins
        try:
            df['bin'] = pd.qcut(df['feature'], n_bins, labels=False, duplicates='drop')
        except ValueError:
            # If qcut fails, use cut with equal-width bins
            df['bin'] = pd.cut(df['feature'], n_bins, labels=False, duplicates='drop')
    
    # Calculate statistics for each bin
    bin_stats = {}
    for bin_idx in range(n_bins):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) > 0:
            mean_ret = bin_data['target'].mean()
            std_ret = bin_data['target'].std()
            
            # Calculate downside deviation (only negative returns)
            downside_returns = bin_data['target'][bin_data['target'] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calculate Sortino ratio (annualized)
            # Use a minimum downside std to avoid division by zero
            MIN_STD = 1e-6
            sortino_metric = (mean_ret / max(downside_std, MIN_STD)) * np.sqrt(252) if downside_std > 0 else (mean_ret * np.sqrt(252) if mean_ret > 0 else 0)
            
            bin_stats[bin_idx] = {
                'mean_return': mean_ret,
                'std_return': std_ret,
                'downside_std': downside_std,
                'sortino_metric': sortino_metric,
                'count': len(bin_data),
                'feature_min': bin_data['feature'].min(),
                'feature_max': bin_data['feature'].max()
            }
    
    # Select bins for both long and short strategies
    if n_unique == 2:
        # For binary features:
        # Long: select bin 1 (the "signal active" bin)
        # Short: select bin 0 (the "signal inactive" bin)
        best_long_bin = 1
        best_short_bin = 0
    elif selection_metric == 'sortino':
        # Long: Find bin with highest Sortino ratio (mean/downside_std)
        best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['sortino_metric'])
        
        # Short: Find bin with lowest mean return (most negative)
        # For shorts, we want the bin where returns are most negative
        # We profit when we short and the asset goes down
        best_short_bin = min(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
        
    elif selection_metric == 'mean':
        # Long: Find bin with highest mean return
        best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
        
        # Short: Find bin with lowest mean return (most negative)
        best_short_bin = min(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}. Use 'sortino' or 'mean'")
    
    # Extract thresholds (bin edges)
    thresholds = []
    for bin_idx in range(n_bins - 1):
        # Threshold is the max of current bin
        if bin_idx in bin_stats:
            thresholds.append(bin_stats[bin_idx]['feature_max'])
    
    thresholds = np.array(sorted(thresholds))
    
    return thresholds, best_long_bin, best_short_bin, bin_stats


def train_bin_selector_tree(
    feature_data: pd.Series,
    target_data: pd.Series,
    n_bins: int = 3,
    selection_metric: str = 'sortino',
    min_samples_leaf_pct: float = 0.05
) -> Tuple[np.ndarray, int, int, Dict]:
    """
    Train a decision tree to create bins and select the best ones for long and short.
    
    Uses supervised learning (DecisionTreeRegressor) to find optimal split points
    based on the target variable. This can capture non-linear relationships and
    natural breakpoints that quantile binning might miss.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values (training data)
    target_data : pd.Series
        Target values (training data)
    n_bins : int, default=3
        Number of bins to create (max_leaf_nodes for the tree)
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std for long, -mean / upside_std for short
        - 'mean': mean return for long, -mean for short
    min_samples_leaf_pct : float, default=0.05
        Minimum percentage of samples required in each leaf (prevents overfitting)
        
    Returns
    -------
    Tuple[np.ndarray, int, int, Dict]
        - thresholds: Array of threshold values defining bins
        - best_long_bin: Index of the bin with best long return (highest)
        - best_short_bin: Index of the bin with best short return (lowest/most negative)
        - bin_stats: Dictionary with statistics for each bin
    """
    # Create clean dataset
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if len(df) < n_bins * 10:
        raise ValueError(f"Insufficient data: need at least {n_bins * 10} samples")
    
    # Check if feature is binary (only 2 unique values)
    unique_values = df['feature'].unique()
    n_unique = len(unique_values)
    
    if n_unique == 2:
        # Binary feature: create exactly 2 bins using the exact values
        sorted_values = np.sort(unique_values)
        df['bin'] = (df['feature'] == sorted_values[1]).astype(int)
        n_bins = 2
        thresholds = np.array([sorted_values[0]])  # Single threshold between the two values
    else:
        # Non-binary feature: use decision tree to find optimal splits
        min_samples_leaf = max(int(len(df) * min_samples_leaf_pct), 10)
        
        # Fit decision tree regressor
        tree = DecisionTreeRegressor(
            max_leaf_nodes=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=0.00001,  # Very small threshold to allow more splits
            random_state=42
        )
        
        X = df['feature'].values.reshape(-1, 1)
        y = df['target'].values
        tree.fit(X, y)
        
        # Extract thresholds from tree structure
        thresholds = []
        tree_ = tree.tree_
        
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
        thresholds = np.array(sorted(set(thresholds)))  # Remove duplicates and sort
        
        # Fallback to quantile binning if tree didn't create enough splits
        if len(thresholds) < n_bins - 1:
            import warnings
            warnings.warn(
                f"Decision tree only created {len(thresholds)} splits (expected {n_bins-1}). "
                f"Falling back to quantile binning. Try lowering min_samples_leaf_pct or reducing n_bins."
            )
            # Use quantile binning as fallback
            try:
                df['bin'] = pd.qcut(df['feature'], n_bins, labels=False, duplicates='drop')
            except ValueError:
                df['bin'] = pd.cut(df['feature'], n_bins, labels=False, duplicates='drop')
            
            # Recalculate thresholds from quantile bins
            thresholds = []
            for bin_idx in range(n_bins - 1):
                bin_data = df[df['bin'] == bin_idx]
                if len(bin_data) > 0:
                    thresholds.append(bin_data['feature'].max())
            thresholds = np.array(sorted(thresholds))
        else:
            # Assign bins based on tree thresholds
            df['bin'] = np.digitize(df['feature'].values, thresholds)
    
    # Calculate statistics for each bin
    bin_stats = {}
    for bin_idx in sorted(df['bin'].unique()):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) > 0:
            mean_ret = bin_data['target'].mean()
            std_ret = bin_data['target'].std()
            
            # Calculate downside deviation (only negative returns)
            downside_returns = bin_data['target'][bin_data['target'] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calculate Sortino ratio (annualized)
            MIN_STD = 1e-6
            sortino_metric = (mean_ret / max(downside_std, MIN_STD)) * np.sqrt(252) if downside_std > 0 else (mean_ret * np.sqrt(252) if mean_ret > 0 else 0)
            
            bin_stats[bin_idx] = {
                'mean_return': mean_ret,
                'std_return': std_ret,
                'downside_std': downside_std,
                'sortino_metric': sortino_metric,
                'count': len(bin_data),
                'feature_min': bin_data['feature'].min(),
                'feature_max': bin_data['feature'].max()
            }
    
    # Select bins for both long and short strategies
    if n_unique == 2:
        # For binary features:
        # Long: select bin 1 (the "signal active" bin)
        # Short: select bin 0 (the "signal inactive" bin)
        best_long_bin = 1
        best_short_bin = 0
    elif selection_metric == 'sortino':
        # Long: Find bin with highest Sortino ratio
        best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['sortino_metric'])
        
        # Short: Find bin with lowest mean return (most negative)
        best_short_bin = min(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
        
    elif selection_metric == 'mean':
        # Long: Find bin with highest mean return
        best_long_bin = max(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
        
        # Short: Find bin with lowest mean return (most negative)
        best_short_bin = min(bin_stats.keys(), key=lambda k: bin_stats[k]['mean_return'])
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}. Use 'sortino' or 'mean'")
    
    return thresholds, best_long_bin, best_short_bin, bin_stats


def apply_bin_selector(
    feature_data: pd.Series,
    thresholds: np.ndarray,
    best_bin: int
) -> pd.Series:
    """
    Apply bin selection rule to new data.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values to classify
    thresholds : np.ndarray
        Threshold values defining bins
    best_bin : int
        Index of the bin to select (0, 1, 2, etc.)
        
    Returns
    -------
    pd.Series
        Binary signal: 1 if in best bin, 0 otherwise
    """
    # Assign bins based on thresholds
    bins = np.digitize(feature_data.values, thresholds)
    
    # Create signal: 1 if in best bin, 0 otherwise
    signal = pd.Series((bins == best_bin).astype(int), index=feature_data.index)
    
    return signal


def walkforward_bin_selection(
    feature_data: pd.Series,
    target_data: pd.Series,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    n_bins: int = 3,
    selection_metric: str = 'sortino',
    verbose: bool = True,
    raw_return: pd.Series = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Perform walk-forward bin selection analysis for both long and short strategies.
    
    For each step:
    1. Train on historical window to find optimal bins (long and short)
    2. Test on next period using those bins
    3. Track performance and selected bins for both strategies
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values with DatetimeIndex
    target_data : pd.Series
        Target values with DatetimeIndex (can be ATR-normalized)
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int, default=252
        Number of days for test period (~1 year)
    num_steps : int, default=10
        Number of walk-forward steps
    n_bins : int, default=3
        Number of bins to create
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std (risk-adjusted, penalizes only downside)
        - 'mean': mean return only (ignores variance)
    verbose : bool, default=True
        Print progress
    raw_return : pd.Series, optional
        Raw returns (close/open - 1) for equity curve calculation.
        If None, uses target_data for equity curve.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[Dict]]
        - results_df: DataFrame with test period results (includes both long and short)
        - step_info: List of dictionaries with detailed info per step
    """
    # Combine into DataFrame
    df_dict = {'feature': feature_data, 'target': target_data}
    if raw_return is not None:
        df_dict['raw_return'] = raw_return
    df = pd.DataFrame(df_dict)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    # Define function for each walk-forward step
    def process_step(train_data, test_data, split_info):
        step = split_info['step']
        
        if verbose:
            print(f"\n=== Step {step + 1}/{num_steps} ===")
            print(f"  Train: {split_info['train_start'].date()} to {split_info['train_end'].date()} (n={split_info['train_size']})")
            print(f"  Test:  {split_info['test_start'].date()} to {split_info['test_end'].date()} (n={split_info['test_size']})")
        
        # Train bin selector for both long and short
        try:
            thresholds, best_long_bin, best_short_bin, bin_stats = train_bin_selector(
                train_data['feature'],
                train_data['target'],
                n_bins=n_bins,
                selection_metric=selection_metric
            )
            
            if verbose:
                # Show selection criterion for both long and short
                print(f"  Selected LONG bin: {best_long_bin} (mean: {bin_stats[best_long_bin]['mean_return']:.6f})")
                print(f"  Selected SHORT bin: {best_short_bin} (mean: {bin_stats[best_short_bin]['mean_return']:.6f})")
                
                print(f"  Bin statistics:")
                for bin_idx, stats in bin_stats.items():
                    long_marker = " <-- LONG" if bin_idx == best_long_bin else ""
                    short_marker = " <-- SHORT" if bin_idx == best_short_bin else ""
                    marker = long_marker + short_marker
                    sortino_str = f"sortino={stats['sortino_metric']:.2f}, " if 'sortino_metric' in stats else ""
                    print(f"    Bin {bin_idx}: {sortino_str}mean={stats['mean_return']:.6f}, "
                          f"downside_std={stats['downside_std']:.6f}, count={stats['count']}, "
                          f"range=[{stats['feature_min']:.3f}, {stats['feature_max']:.3f}]{marker}")
            
            # Apply to test data - LONG strategy
            test_signal_long = apply_bin_selector(
                test_data['feature'],
                thresholds,
                best_long_bin
            )
            
            # Apply to test data - SHORT strategy
            test_signal_short = apply_bin_selector(
                test_data['feature'],
                thresholds,
                best_short_bin
            )
            
            # Calculate LONG test performance
            test_returns_long = test_data['target'][test_signal_long == 1]
            
            if 'raw_return' in test_data.columns:
                test_raw_returns_long = test_data['raw_return'][test_signal_long == 1]
                raw_mean_long = test_raw_returns_long.mean() if len(test_raw_returns_long) > 0 else 0
                raw_sum_long = test_raw_returns_long.sum() if len(test_raw_returns_long) > 0 else 0
            else:
                raw_mean_long = None
                raw_sum_long = None
            
            if len(test_returns_long) > 0:
                test_mean_long = test_returns_long.mean()
                test_std_long = test_returns_long.std()
                test_downside_returns_long = test_returns_long[test_returns_long < 0]
                test_downside_std_long = test_downside_returns_long.std() if len(test_downside_returns_long) > 0 else 0
                test_sortino_long = (test_mean_long / test_downside_std_long * np.sqrt(252)) if test_downside_std_long > 0 else (test_mean_long * np.sqrt(252) if test_mean_long > 0 else 0)
                n_trades_long = len(test_returns_long)
            else:
                test_mean_long = 0
                test_std_long = 0
                test_downside_std_long = 0
                test_sortino_long = 0
                n_trades_long = 0
            
            # Calculate SHORT test performance (invert returns for shorting)
            test_returns_short_raw = test_data['target'][test_signal_short == 1]
            test_returns_short = -test_returns_short_raw  # Invert for short positions
            
            if 'raw_return' in test_data.columns:
                test_raw_returns_short_raw = test_data['raw_return'][test_signal_short == 1]
                test_raw_returns_short = -test_raw_returns_short_raw  # Invert for short positions
                raw_mean_short = test_raw_returns_short.mean() if len(test_raw_returns_short) > 0 else 0
                raw_sum_short = test_raw_returns_short.sum() if len(test_raw_returns_short) > 0 else 0
            else:
                raw_mean_short = None
                raw_sum_short = None
            
            if len(test_returns_short) > 0:
                test_mean_short = test_returns_short.mean()
                test_std_short = test_returns_short.std()
                test_downside_returns_short = test_returns_short[test_returns_short < 0]
                test_downside_std_short = test_downside_returns_short.std() if len(test_downside_returns_short) > 0 else 0
                test_sortino_short = (test_mean_short / test_downside_std_short * np.sqrt(252)) if test_downside_std_short > 0 else (test_mean_short * np.sqrt(252) if test_mean_short > 0 else 0)
                n_trades_short = len(test_returns_short)
            else:
                test_mean_short = 0
                test_std_short = 0
                test_downside_std_short = 0
                test_sortino_short = 0
                n_trades_short = 0
            
            if verbose:
                print(f"  LONG Test performance: mean={test_mean_long:.6f}, sortino={test_sortino_long:.2f}, n_trades={n_trades_long}")
                if raw_mean_long is not None:
                    print(f"  LONG Raw return mean: {raw_mean_long:.6f}")
                print(f"  SHORT Test performance: mean={test_mean_short:.6f}, sortino={test_sortino_short:.2f}, n_trades={n_trades_short}")
                if raw_mean_short is not None:
                    print(f"  SHORT Raw return mean: {raw_mean_short:.6f}")
            
            return {
                'step': step,
                'thresholds': thresholds,
                'best_long_bin': best_long_bin,
                'best_short_bin': best_short_bin,
                'bin_stats': bin_stats,
                # Long metrics
                'test_mean_return_long': test_mean_long,
                'test_std_return_long': test_std_long,
                'test_downside_std_long': test_downside_std_long,
                'test_sortino_long': test_sortino_long,
                'n_trades_long': n_trades_long,
                'test_signal_long': test_signal_long,
                'test_raw_mean_long': raw_mean_long,
                'test_raw_sum_long': raw_sum_long,
                # Short metrics
                'test_mean_return_short': test_mean_short,
                'test_std_return_short': test_std_short,
                'test_downside_std_short': test_downside_std_short,
                'test_sortino_short': test_sortino_short,
                'n_trades_short': n_trades_short,
                'test_signal_short': test_signal_short,
                'test_raw_mean_short': raw_mean_short,
                'test_raw_sum_short': raw_sum_short
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return {
                'step': step,
                'error': str(e)
            }
    
    # Run walk-forward analysis
    results = apply_function_to_walkforward(
        df=df,
        func=process_step,
        train_start=train_start,
        train_end=train_end,
        test_step=test_step,
        num_steps=num_steps,
        verbose=False  # We handle verbosity in process_step
    )
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'step': r['step'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            # Long metrics
            'best_long_bin': r.get('best_long_bin', np.nan),
            'test_mean_return_long': r.get('test_mean_return_long', np.nan),
            'test_sortino_long': r.get('test_sortino_long', np.nan),
            'n_trades_long': r.get('n_trades_long', 0),
            'test_raw_mean_long': r.get('test_raw_mean_long', np.nan),
            'test_raw_sum_long': r.get('test_raw_sum_long', np.nan),
            # Short metrics
            'best_short_bin': r.get('best_short_bin', np.nan),
            'test_mean_return_short': r.get('test_mean_return_short', np.nan),
            'test_sortino_short': r.get('test_sortino_short', np.nan),
            'n_trades_short': r.get('n_trades_short', 0),
            'test_raw_mean_short': r.get('test_raw_mean_short', np.nan),
            'test_raw_sum_short': r.get('test_raw_sum_short', np.nan)
        }
        for r in results
    ])
    
    return results_df, results


def walkforward_bin_selection_tree(
    feature_data: pd.Series,
    target_data: pd.Series,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    n_bins: int = 3,
    selection_metric: str = 'sortino',
    min_samples_leaf_pct: float = 0.05,
    verbose: bool = True,
    raw_return: pd.Series = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Perform walk-forward bin selection analysis using decision tree binning.
    
    This variant uses DecisionTreeRegressor to find optimal split points based on
    the target variable, rather than using quantile-based binning. The tree learns
    natural breakpoints in the feature-target relationship.
    
    For each step:
    1. Train decision tree on historical window to find optimal bins
    2. Select the bin with highest Sortino ratio
    3. Test on next period using that bin
    4. Track performance and selected bins
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values with DatetimeIndex
    target_data : pd.Series
        Target values with DatetimeIndex (can be ATR-normalized)
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int, default=252
        Number of days for test period (~1 year)
    num_steps : int, default=10
        Number of walk-forward steps
    n_bins : int, default=3
        Number of bins to create (max_leaf_nodes for decision tree)
    selection_metric : str, default='sortino'
        Metric to use for bin selection:
        - 'sortino': mean / downside_std (risk-adjusted, penalizes only downside)
        - 'mean': mean return only (ignores variance)
    min_samples_leaf_pct : float, default=0.05
        Minimum percentage of samples required in each leaf (prevents overfitting)
        Default is 5% of training data per leaf
    verbose : bool, default=True
        Print progress
    raw_return : pd.Series, optional
        Raw returns (close/open - 1) for equity curve calculation.
        If None, uses target_data for equity curve.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[Dict]]
        - results_df: DataFrame with test period results
        - step_info: List of dictionaries with detailed info per step
        
    Notes
    -----
    Decision tree binning differs from quantile binning:
    - Supervised: Uses target variable to find optimal splits
    - Adaptive: Finds natural breakpoints in feature-target relationship
    - May be more stable: Captures true non-linear patterns
    - Regularized: min_samples_leaf prevents overfitting
    
    Examples
    --------
    >>> results_df, step_info = walkforward_bin_selection_tree(
    ...     feature_data=feature_series,
    ...     target_data=target_series,
    ...     train_start=datetime(2000, 1, 1),
    ...     train_end=datetime(2010, 1, 1),
    ...     test_step=252,
    ...     num_steps=15,
    ...     n_bins=3,
    ...     min_samples_leaf_pct=0.05  # Each bin must have at least 5% of data
    ... )
    """
    # Combine into DataFrame
    df_dict = {'feature': feature_data, 'target': target_data}
    if raw_return is not None:
        df_dict['raw_return'] = raw_return
    df = pd.DataFrame(df_dict)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    # Define function for each walk-forward step
    def process_step(train_data, test_data, split_info):
        step = split_info['step']
        
        if verbose:
            print(f"\n=== Step {step + 1}/{num_steps} ===")
            print(f"  Train: {split_info['train_start'].date()} to {split_info['train_end'].date()} (n={split_info['train_size']})")
            print(f"  Test:  {split_info['test_start'].date()} to {split_info['test_end'].date()} (n={split_info['test_size']})")
        
        # Train bin selector using DECISION TREE for both long and short
        try:
            thresholds, best_long_bin, best_short_bin, bin_stats = train_bin_selector_tree(
                train_data['feature'],
                train_data['target'],
                n_bins=n_bins,
                selection_metric=selection_metric,
                min_samples_leaf_pct=min_samples_leaf_pct
            )
            
            if verbose:
                print(f"  Decision tree found {len(thresholds)} thresholds: {thresholds}")
                
                # Show selection criterion for both long and short
                print(f"  Selected LONG bin: {best_long_bin} (mean: {bin_stats[best_long_bin]['mean_return']:.6f})")
                print(f"  Selected SHORT bin: {best_short_bin} (mean: {bin_stats[best_short_bin]['mean_return']:.6f})")
                
                print(f"  Bin statistics:")
                for bin_idx, stats in bin_stats.items():
                    long_marker = " <-- LONG" if bin_idx == best_long_bin else ""
                    short_marker = " <-- SHORT" if bin_idx == best_short_bin else ""
                    marker = long_marker + short_marker
                    sortino_str = f"sortino={stats['sortino_metric']:.2f}, " if 'sortino_metric' in stats else ""
                    print(f"    Bin {bin_idx}: {sortino_str}mean={stats['mean_return']:.6f}, "
                          f"downside_std={stats['downside_std']:.6f}, count={stats['count']}, "
                          f"range=[{stats['feature_min']:.3f}, {stats['feature_max']:.3f}]{marker}")
            
            # Apply to test data - LONG strategy
            test_signal_long = apply_bin_selector(
                test_data['feature'],
                thresholds,
                best_long_bin
            )
            
            # Apply to test data - SHORT strategy
            test_signal_short = apply_bin_selector(
                test_data['feature'],
                thresholds,
                best_short_bin
            )
            
            # Calculate LONG test performance
            test_returns_long = test_data['target'][test_signal_long == 1]
            
            if 'raw_return' in test_data.columns:
                test_raw_returns_long = test_data['raw_return'][test_signal_long == 1]
                raw_mean_long = test_raw_returns_long.mean() if len(test_raw_returns_long) > 0 else 0
                raw_sum_long = test_raw_returns_long.sum() if len(test_raw_returns_long) > 0 else 0
            else:
                raw_mean_long = None
                raw_sum_long = None
            
            if len(test_returns_long) > 0:
                test_mean_long = test_returns_long.mean()
                test_std_long = test_returns_long.std()
                test_downside_returns_long = test_returns_long[test_returns_long < 0]
                test_downside_std_long = test_downside_returns_long.std() if len(test_downside_returns_long) > 0 else 0
                test_sortino_long = (test_mean_long / test_downside_std_long * np.sqrt(252)) if test_downside_std_long > 0 else (test_mean_long * np.sqrt(252) if test_mean_long > 0 else 0)
                n_trades_long = len(test_returns_long)
            else:
                test_mean_long = 0
                test_std_long = 0
                test_downside_std_long = 0
                test_sortino_long = 0
                n_trades_long = 0
            
            # Calculate SHORT test performance (invert returns for shorting)
            test_returns_short_raw = test_data['target'][test_signal_short == 1]
            test_returns_short = -test_returns_short_raw  # Invert for short positions
            
            if 'raw_return' in test_data.columns:
                test_raw_returns_short_raw = test_data['raw_return'][test_signal_short == 1]
                test_raw_returns_short = -test_raw_returns_short_raw  # Invert for short positions
                raw_mean_short = test_raw_returns_short.mean() if len(test_raw_returns_short) > 0 else 0
                raw_sum_short = test_raw_returns_short.sum() if len(test_raw_returns_short) > 0 else 0
            else:
                raw_mean_short = None
                raw_sum_short = None
            
            if len(test_returns_short) > 0:
                test_mean_short = test_returns_short.mean()
                test_std_short = test_returns_short.std()
                test_downside_returns_short = test_returns_short[test_returns_short < 0]
                test_downside_std_short = test_downside_returns_short.std() if len(test_downside_returns_short) > 0 else 0
                test_sortino_short = (test_mean_short / test_downside_std_short * np.sqrt(252)) if test_downside_std_short > 0 else (test_mean_short * np.sqrt(252) if test_mean_short > 0 else 0)
                n_trades_short = len(test_returns_short)
            else:
                test_mean_short = 0
                test_std_short = 0
                test_downside_std_short = 0
                test_sortino_short = 0
                n_trades_short = 0
            
            if verbose:
                print(f"  LONG Test performance: mean={test_mean_long:.6f}, sortino={test_sortino_long:.2f}, n_trades={n_trades_long}")
                if raw_mean_long is not None:
                    print(f"  LONG Raw return mean: {raw_mean_long:.6f}")
                print(f"  SHORT Test performance: mean={test_mean_short:.6f}, sortino={test_sortino_short:.2f}, n_trades={n_trades_short}")
                if raw_mean_short is not None:
                    print(f"  SHORT Raw return mean: {raw_mean_short:.6f}")
            
            return {
                'step': step,
                'thresholds': thresholds,
                'best_long_bin': best_long_bin,
                'best_short_bin': best_short_bin,
                'bin_stats': bin_stats,
                # Long metrics
                'test_mean_return_long': test_mean_long,
                'test_std_return_long': test_std_long,
                'test_downside_std_long': test_downside_std_long,
                'test_sortino_long': test_sortino_long,
                'n_trades_long': n_trades_long,
                'test_signal_long': test_signal_long,
                'test_raw_mean_long': raw_mean_long,
                'test_raw_sum_long': raw_sum_long,
                # Short metrics
                'test_mean_return_short': test_mean_short,
                'test_std_return_short': test_std_short,
                'test_downside_std_short': test_downside_std_short,
                'test_sortino_short': test_sortino_short,
                'n_trades_short': n_trades_short,
                'test_signal_short': test_signal_short,
                'test_raw_mean_short': raw_mean_short,
                'test_raw_sum_short': raw_sum_short
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return {
                'step': step,
                'error': str(e)
            }
    
    # Run walk-forward analysis
    results = apply_function_to_walkforward(
        df=df,
        func=process_step,
        train_start=train_start,
        train_end=train_end,
        test_step=test_step,
        num_steps=num_steps,
        verbose=False  # We handle verbosity in process_step
    )
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'step': r['step'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            # Long metrics
            'best_long_bin': r.get('best_long_bin', np.nan),
            'test_mean_return_long': r.get('test_mean_return_long', np.nan),
            'test_sortino_long': r.get('test_sortino_long', np.nan),
            'n_trades_long': r.get('n_trades_long', 0),
            'test_raw_mean_long': r.get('test_raw_mean_long', np.nan),
            'test_raw_sum_long': r.get('test_raw_sum_long', np.nan),
            # Short metrics
            'best_short_bin': r.get('best_short_bin', np.nan),
            'test_mean_return_short': r.get('test_mean_return_short', np.nan),
            'test_sortino_short': r.get('test_sortino_short', np.nan),
            'n_trades_short': r.get('n_trades_short', 0),
            'test_raw_mean_short': r.get('test_raw_mean_short', np.nan),
            'test_raw_sum_short': r.get('test_raw_sum_short', np.nan)
        }
        for r in results
    ])
    
    return results_df, results


def plot_walkforward_results(
    results_df: pd.DataFrame,
    step_info: List[Dict],
    feature_name: str = "Feature",
    figsize: Tuple[int, int] = (18, 12),
    equity_figsize: Tuple[int, int] = (16, 6)
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot walk-forward bin selection results for both LONG and SHORT strategies.
    
    Creates two separate figures:
    Figure 1: Bin selection, mean return, Sortino ratio, and number of trades (Long vs Short)
    Figure 2: Cumulative equity curves (Long vs Short comparison)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from walkforward_bin_selection (must include both long and short columns)
    step_info : List[Dict]
        Detailed step information
    feature_name : str, default="Feature"
        Name of the feature for plot title
    figsize : Tuple[int, int], default=(18, 12)
        Figure size for main metrics plot
    equity_figsize : Tuple[int, int], default=(16, 6)
        Figure size for equity curve plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Figure]
        (metrics_fig, equity_fig) - Two figure objects
    """
    # Figure 1: Main metrics (4 rows x 2 columns: Long vs Short)
    fig_metrics, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
    fig_metrics.suptitle(f'{feature_name} Walk-Forward Results: LONG vs SHORT', 
                         fontsize=16, fontweight='bold', y=0.995)
    
    steps = results_df['step'].values
    
    # Row 1: Selected bins (Long vs Short)
    # Long bins
    ax = axes[0, 0]
    if 'best_long_bin' in results_df.columns:
        ax.plot(steps, results_df['best_long_bin'], 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_ylabel('LONG - Selected Bin', fontsize=10, fontweight='bold')
        ax.set_title('Long Strategy', fontsize=12, fontweight='bold', color='#2ecc71')
        ax.grid(True, alpha=0.3)
        ax.set_yticks(sorted(results_df['best_long_bin'].dropna().unique()))
    
    # Short bins
    ax = axes[0, 1]
    if 'best_short_bin' in results_df.columns:
        ax.plot(steps, results_df['best_short_bin'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_ylabel('SHORT - Selected Bin', fontsize=10, fontweight='bold')
        ax.set_title('Short Strategy', fontsize=12, fontweight='bold', color='#e74c3c')
        ax.grid(True, alpha=0.3)
        ax.set_yticks(sorted(results_df['best_short_bin'].dropna().unique()))
    
    # Row 2: Test mean returns
    # Long returns
    ax = axes[1, 0]
    if 'test_mean_return_long' in results_df.columns:
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in results_df['test_mean_return_long']]
        ax.bar(steps, results_df['test_mean_return_long'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('LONG - Mean Return', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Short returns
    ax = axes[1, 1]
    if 'test_mean_return_short' in results_df.columns:
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in results_df['test_mean_return_short']]
        ax.bar(steps, results_df['test_mean_return_short'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('SHORT - Mean Return', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Test Sortino ratios
    # Long Sortino
    ax = axes[2, 0]
    if 'test_sortino_long' in results_df.columns:
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in results_df['test_sortino_long']]
        ax.bar(steps, results_df['test_sortino_long'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('LONG - Sortino', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Short Sortino
    ax = axes[2, 1]
    if 'test_sortino_short' in results_df.columns:
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in results_df['test_sortino_short']]
        ax.bar(steps, results_df['test_sortino_short'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('SHORT - Sortino', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 4: Number of trades
    # Long trades
    ax = axes[3, 0]
    if 'n_trades_long' in results_df.columns:
        ax.bar(steps, results_df['n_trades_long'], color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.set_ylabel('LONG - # Trades', fontsize=10, fontweight='bold')
        ax.set_xlabel('Walk-Forward Step', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Short trades
    ax = axes[3, 1]
    if 'n_trades_short' in results_df.columns:
        ax.bar(steps, results_df['n_trades_short'], color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_ylabel('SHORT - # Trades', fontsize=10, fontweight='bold')
        ax.set_xlabel('Walk-Forward Step', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig_metrics.tight_layout()
    
    # Figure 2: Equity curves - separate subplots (Long | Short | Combined)
    fig_equity, axes_equity = plt.subplots(1, 3, figsize=equity_figsize, sharey=False)
    fig_equity.suptitle(f'{feature_name} - Equity Curves', fontsize=14, fontweight='bold')
    
    # Prepend 0 at the start so equity curves start at 0
    steps_with_start = np.concatenate([[0], steps])
    
    # Calculate LONG cumulative returns
    if 'test_raw_sum_long' in results_df.columns and not results_df['test_raw_sum_long'].isna().all():
        cumulative_pnl_long = results_df['test_raw_sum_long'].fillna(0).cumsum()
        equity_label = 'Raw Returns'
    else:
        cumulative_pnl_long = (results_df['test_mean_return_long'] * results_df['n_trades_long']).cumsum()
        equity_label = 'Normalized Returns'
    
    cumulative_pnl_long_with_start = np.concatenate([[0], cumulative_pnl_long.values])
    
    # Calculate SHORT cumulative returns
    if 'test_raw_sum_short' in results_df.columns and not results_df['test_raw_sum_short'].isna().all():
        cumulative_pnl_short = results_df['test_raw_sum_short'].fillna(0).cumsum()
    else:
        cumulative_pnl_short = (results_df['test_mean_return_short'] * results_df['n_trades_short']).cumsum()
    
    cumulative_pnl_short_with_start = np.concatenate([[0], cumulative_pnl_short.values])
    
    # Calculate COMBINED
    cumulative_pnl_combined = cumulative_pnl_long_with_start + cumulative_pnl_short_with_start
    
    # Subplot 1: LONG equity curve
    ax = axes_equity[0]
    ax.plot(steps_with_start, cumulative_pnl_long_with_start, linewidth=3, 
            color='#2ecc71', marker='o', markersize=6, alpha=0.8)
    ax.fill_between(steps_with_start, 0, cumulative_pnl_long_with_start, alpha=0.3, color='#2ecc71')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title('LONG Strategy', fontsize=12, fontweight='bold', color='#2ecc71')
    ax.set_ylabel(f'Cumulative P&L ({equity_label})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Walk-Forward Step', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    final_pnl_long = cumulative_pnl_long.iloc[-1] if len(cumulative_pnl_long) > 0 else 0
    ax.text(0.05, 0.95, f'Final: {final_pnl_long:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Subplot 2: SHORT equity curve
    ax = axes_equity[1]
    ax.plot(steps_with_start, cumulative_pnl_short_with_start, linewidth=3, 
            color='#e74c3c', marker='s', markersize=6, alpha=0.8)
    ax.fill_between(steps_with_start, 0, cumulative_pnl_short_with_start, alpha=0.3, color='#e74c3c')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title('SHORT Strategy', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.set_ylabel(f'Cumulative P&L ({equity_label})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Walk-Forward Step', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    final_pnl_short = cumulative_pnl_short.iloc[-1] if len(cumulative_pnl_short) > 0 else 0
    ax.text(0.05, 0.95, f'Final: {final_pnl_short:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Subplot 3: COMBINED equity curve
    ax = axes_equity[2]
    ax.plot(steps_with_start, cumulative_pnl_combined, linewidth=3, 
            color='#3498db', marker='^', markersize=6, alpha=0.8)
    ax.fill_between(steps_with_start, 0, cumulative_pnl_combined, alpha=0.3, color='#3498db')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title('COMBINED (L+S)', fontsize=12, fontweight='bold', color='#3498db')
    ax.set_ylabel(f'Cumulative P&L ({equity_label})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Walk-Forward Step', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    final_pnl_combined = final_pnl_long + final_pnl_short
    ax.text(0.05, 0.95, f'Final: {final_pnl_combined:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig_equity.tight_layout()
    
    return fig_metrics, fig_equity


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    from utils.enums import Ticker
    from feature_selection.feature_explorer import FeatureExplorer
    
    # Extract feature and target
    explorer = FeatureExplorer.from_bias_node(
        module_name='cmma',
        ticker=[Ticker.ES, Ticker.NQ, Ticker.YM],
        params={'lookback': 250, 'atr_length': 252},
        start=datetime(2000, 1, 1),
        end=datetime(2024, 12, 31)
    )
    
    # Get feature and target data
    feature_name = 'cmma_250_252_D_cmma_250_252'
    feature_data = explorer[feature_name].feature_data
    target_data = explorer[feature_name].target_data
    
    # Run walk-forward bin selection
    results_df, step_info = walkforward_bin_selection(
        feature_data=feature_data,
        target_data=target_data,
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2010, 1, 1),
        test_step=252,  # 1 year test periods
        num_steps=15,   # 15 years of testing
        n_bins=3,
        verbose=True
    )
    
    # Plot results (shows both LONG and SHORT strategies by default)
    fig_metrics, fig_equity = plot_walkforward_results(results_df, step_info, feature_name)
    plt.show()
    
    # Print summary for both LONG and SHORT strategies
    print("\n" + "="*70)
    print("WALK-FORWARD BIN SELECTION SUMMARY (LONG & SHORT)")
    print("="*70)
    print(f"Total steps: {len(results_df)}")
    print("\n--- LONG STRATEGY ---")
    print(f"  Average test return: {results_df['test_mean_return_long'].mean():.6f}")
    print(f"  Average test Sortino: {results_df['test_sortino_long'].mean():.2f}")
    print(f"  Total trades: {results_df['n_trades_long'].sum()}")
    print(f"  Bin selection frequency:")
    print("  ", results_df['best_long_bin'].value_counts().sort_index().to_dict())
    
    print("\n--- SHORT STRATEGY ---")
    print(f"  Average test return: {results_df['test_mean_return_short'].mean():.6f}")
    print(f"  Average test Sortino: {results_df['test_sortino_short'].mean():.2f}")
    print(f"  Total trades: {results_df['n_trades_short'].sum()}")
    print(f"  Bin selection frequency:")
    print("  ", results_df['best_short_bin'].value_counts().sort_index().to_dict())
    
    print("\n--- COMBINED (LONG + SHORT) ---")
    combined_return = (results_df['test_mean_return_long'].mean() + results_df['test_mean_return_short'].mean()) / 2
    print(f"  Average combined return: {combined_return:.6f}")
    print(f"  Total trades: {results_df['n_trades_long'].sum() + results_df['n_trades_short'].sum()}")
