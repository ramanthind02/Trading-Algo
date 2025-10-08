import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path to import spearman_rho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import spearman_rho


def optimize_threshold(
    feature: pd.Series,
    target: pd.Series,
    min_kept: int = 100
) -> Dict:
    """
    Find optimal thresholds to maximize long/short profit factors.
    
    This function optimizes buying (high) and selling (low) thresholds separately
    to maximize profit factors for long and short trades. It automatically computes
    the Spearman Rho correlation and flips the feature sign if the correlation is
    negative, ensuring the feature is positively correlated with returns.
    
    The algorithm:
    1. Computes Spearman Rho correlation between feature and target
    2. Flips feature sign if correlation is negative (to make it positive)
    3. Sorts feature values and finds thresholds that maximize:
       - High threshold: Profit factor for values >= threshold (long trades)
       - Low threshold: Profit factor for values < threshold (short trades)
    4. Returns thresholds, profit factors, and filtered data
    
    Args:
        feature: Feature values (indicator/signal)
        target: Target values (log returns)
        min_kept: Minimum number of samples to keep for each threshold (default: 100)
        
    Returns:
        Dict containing:
            - 'spearman_rho': Spearman correlation coefficient
            - 'sign_flipped': Boolean indicating if feature sign was flipped
            - 'pf_all': Profit factor of entire dataset
            - 'high_thresh': Upper threshold for long trades (buy signal)
            - 'pf_high': Profit factor for feature >= high_thresh
            - 'n_high': Number of samples >= high_thresh
            - 'high_feature': Feature values >= high_thresh
            - 'high_target': Target values for high_feature
            - 'low_thresh': Lower threshold for short trades (sell signal)
            - 'pf_low': Profit factor for feature < low_thresh
            - 'n_low': Number of samples < low_thresh
            - 'low_feature': Feature values < low_thresh
            - 'low_target': Target values for low_feature
            
    Example:
        >>> feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> target = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01])
        >>> result = optimize_threshold(feature, target, min_kept=2)
        >>> print(f"High threshold: {result['high_thresh']}, PF: {result['pf_high']}")
        >>> print(f"Low threshold: {result['low_thresh']}, PF: {result['pf_low']}")
    """

    clean_feature = feature
    clean_target = target
    
    n = len(clean_feature)
    
    if n < max(2, min_kept):
        raise ValueError(f"Need at least {max(2, min_kept)} valid data points")
    
    if min_kept < 1:
        min_kept = 1
    
    # Compute Spearman Rho correlation
    rho = spearman_rho(clean_feature, clean_target)
    
    # Flip sign if correlation is negative
    sign_flipped = rho < 0
    work_signal = -clean_feature.values if sign_flipped else clean_feature.values.copy()
    work_return = clean_target.values.copy()
    
    # Sort signal and simultaneously move returns
    sorted_indices = np.argsort(work_signal)
    work_signal = work_signal[sorted_indices]
    work_return = work_return[sorted_indices]
    
    # Vectorized computation using cumulative sums for performance
    # Separate positive and negative returns
    wins = np.where(work_return > 0, work_return, 0.0)
    losses = np.where(work_return <= 0, -work_return, 0.0)
    
    # Compute cumulative wins and losses from the start (for 'below' set)
    cumsum_wins_below = np.cumsum(wins)
    cumsum_losses_below = np.cumsum(losses)
    
    # Total wins and losses
    total_wins = cumsum_wins_below[-1]
    total_losses = cumsum_losses_below[-1]
    
    # Compute wins and losses for 'above' set (everything after index i)
    # above = total - below
    cumsum_wins_above = total_wins - cumsum_wins_below
    cumsum_losses_above = total_losses - cumsum_losses_below
    
    # Compute profit factors for all possible thresholds
    # Add small epsilon to avoid division by zero
    pf_above_all = cumsum_wins_above / (cumsum_losses_above + 1.0e-30)
    pf_below_all = cumsum_wins_below / (cumsum_losses_below + 1.0e-30)
    
    # Find unique threshold positions (where signal value changes)
    # We need to check i+1 positions, so compare consecutive values
    unique_thresh_mask = np.ones(n - 1, dtype=bool)
    unique_thresh_mask[:-1] = work_signal[1:-1] != work_signal[:-2]
    
    # Profit factor for entire dataset
    pf_all = total_wins / (total_losses + 1.0e-30)
    
    # Find best high threshold (for long trades)
    # Valid indices: must have at least min_kept samples above threshold
    # Index i means threshold is at work_signal[i+1], so n-i-1 samples above
    valid_high_mask = np.arange(n - 1) <= (n - min_kept)
    valid_high_mask &= unique_thresh_mask
    
    if valid_high_mask.any():
        # Prepend the "all data" case (index -1 means threshold at smallest value)
        pf_high_candidates = np.concatenate([[pf_all], pf_above_all[:-1]])
        pf_high_candidates[1:][~valid_high_mask] = -np.inf
        best_high_index = np.argmax(pf_high_candidates)
        best_high_pf = pf_high_candidates[best_high_index]
        # Adjust index: 0 means "all data", 1+ means actual index in work_signal
    else:
        best_high_index = 0
        best_high_pf = pf_all
    
    # Find best low threshold (for short trades)
    # Valid indices: must have at least min_kept samples below threshold
    # Index i means threshold is at work_signal[i+1], so i+1 samples below
    valid_low_mask = np.arange(n - 1) >= (min_kept - 1)
    valid_low_mask &= unique_thresh_mask
    
    if valid_low_mask.any():
        pf_low_candidates = pf_below_all[:-1].copy()
        pf_low_candidates[~valid_low_mask] = -np.inf
        best_low_index = np.argmax(pf_low_candidates) + 1  # +1 because we use i+1 as threshold
        best_low_pf = pf_low_candidates[best_low_index - 1]
    else:
        best_low_index = n - 1
        best_low_pf = -1.0
    
    # Get the optimal thresholds
    high_thresh = work_signal[best_high_index]
    low_thresh = work_signal[best_low_index]
    
    # If sign was flipped, flip thresholds back to original feature space
    if sign_flipped:
        high_thresh = -high_thresh
        low_thresh = -low_thresh
        # Create masks in original feature space
        high_mask = clean_feature <= high_thresh  # Note: reversed due to sign flip
        low_mask = clean_feature > low_thresh      # Note: reversed due to sign flip
    else:
        # Create masks in original feature space
        high_mask = clean_feature >= high_thresh
        low_mask = clean_feature < low_thresh

    max_pf = max(best_high_pf, best_low_pf)

    # Return results
    return {
        'spearman_rho': rho,
        'sign_flipped': sign_flipped,
        'pf_all': pf_all,
        'high_thresh': high_thresh,
        'pf_high': best_high_pf,
        'n_high': high_mask.sum(),
        'high_feature': clean_feature[high_mask],
        'high_target': clean_target[high_mask],
        'low_thresh': low_thresh,
        'pf_low': best_low_pf,
        'n_low': low_mask.sum(),
        'low_feature': clean_feature[low_mask],
        'low_target': clean_target[low_mask],
        'max_pf': max_pf,
    }


def optimize_all_features(
    df: pd.DataFrame,
    target_col: str,
    min_kept: int = 100,
    exclude_cols: list = None
) -> pd.DataFrame:
    """
    Optimize thresholds for all features in a DataFrame and return a summary table.
    
    This function loops through all columns in the DataFrame (except the target and
    any excluded columns), optimizes thresholds for each feature, and compiles the
    results into a summary DataFrame for easy inspection and comparison.
    
    Args:
        df: DataFrame containing features and target
        target_col: Name of the target column (e.g., 'log_returns')
        min_kept: Minimum number of samples to keep for each threshold (default: 100)
        exclude_cols: List of column names to exclude from optimization (default: None)
        
    Returns:
        pd.DataFrame: Summary table with columns:
            - 'feature': Feature name
            - 'spearman_rho': Spearman correlation with target
            - 'sign_flipped': Whether feature sign was flipped
            - 'pf_all': Profit factor of entire dataset
            - 'high_thresh': Optimal high threshold for long trades
            - 'pf_high': Profit factor for high threshold
            - 'n_high': Number of samples >= high threshold
            - 'low_thresh': Optimal low threshold for short trades
            - 'pf_low': Profit factor for low threshold
            - 'n_low': Number of samples < low threshold
            - 'max_pf': Maximum profit factor (max of pf_high and pf_low)
            
        Sorted by 'max_pf' in descending order.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'returns': [0.01, -0.02, 0.03, 0.04, -0.01]
        ... })
        >>> summary = optimize_all_features(df, 'returns', min_kept=2)
        >>> print(summary[['feature', 'spearman_rho', 'max_pf']].head())
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Get target series
    target = df[target_col]
    
    # Determine which columns to process
    if exclude_cols is None:
        exclude_cols = []
    
    # Exclude target and any specified columns
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns to optimize")
    
    # Store results for each feature
    results = []
    
    # Loop through each feature column
    for col in feature_cols:
        try:
            # Get feature series
            feature = df[col]
            
            # Optimize threshold for this feature
            result = optimize_threshold(feature, target, min_kept=min_kept)
            
            # Extract summary statistics (exclude the filtered data series)
            summary = {
                'feature': col,
                'spearman_rho': result['spearman_rho'],
                'sign_flipped': result['sign_flipped'],
                'pf_all': result['pf_all'],
                'high_thresh': result['high_thresh'],
                'pf_high': result['pf_high'],
                'n_high': result['n_high'],
                'low_thresh': result['low_thresh'],
                'pf_low': result['pf_low'],
                'n_low': result['n_low'],
                'max_pf': result['max_pf'],
            }
            
            results.append(summary)
            
        except Exception as e:
            # Log error but continue processing other features
            print(f"Warning: Failed to optimize feature '{col}': {str(e)}")
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Sort by max_pf in descending order (best features first)
    summary_df = summary_df.sort_values('max_pf', ascending=False).reset_index(drop=True)
    
    return summary_df
