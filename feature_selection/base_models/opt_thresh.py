import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path to import spearman_rho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Use Cython-optimized version (16x faster) if available, otherwise falls back to pure Python
from utils.fast_stats import spearman_rho


def optimize_threshold(
    feature: pd.Series,
    target: pd.Series,
    min_kept: int = 100,
    verbose: bool = False
) -> Dict:
    """
    Find optimal thresholds to maximize long/short profit factors.
    
    This function tests all 4 possible strategies independently without assuming
    correlation direction:
    1. Long above threshold (buy when feature > thresh)
    2. Long below threshold (buy when feature < thresh)
    3. Short above threshold (sell when feature > thresh)
    4. Short below threshold (sell when feature < thresh)
    
    For long trades, it picks the best between strategies 1 and 2.
    For short trades, it picks the best between strategies 3 and 4.
    
    This approach captures asymmetric patterns like RSI where extreme low values
    may predict mean reversion even if overall correlation is positive.
    
    The algorithm:
    1. Computes Spearman Rho correlation (for reference only, not used for optimization)
    2. Sorts feature values and tests all possible thresholds
    3. For each threshold, computes profit factors for all 4 strategies
    4. Selects best long strategy (above or below) and best short strategy independently
    5. Returns thresholds, profit factors, and filtered data
    
    Args:
        feature: Feature values (indicator/signal)
        target: Target values (log returns)
        min_kept: Minimum number of samples to keep for each threshold (default: 100)
        verbose: Whether to print optimization results (default: False)
        
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

    # Remove NaN values from both feature and target
    valid_mask = ~(feature.isna() | target.isna() | np.isinf(feature) | np.isinf(target))
    clean_feature = feature[valid_mask]
    clean_target = target[valid_mask]
    
    n = len(clean_feature)
    
    if n < max(2, min_kept):
        raise ValueError(f"Need at least {max(2, min_kept)} valid data points")
    
    if min_kept < 1:
        min_kept = 1
    
    # Compute Spearman Rho correlation (for reference only)
    rho = spearman_rho(clean_feature, clean_target)
    
    # NO SIGN FLIPPING - we test all strategies independently
    work_signal = clean_feature.values.copy()
    work_return = clean_target.values.copy()
    
    # Sort signal and simultaneously move returns
    sorted_indices = np.argsort(work_signal)
    work_signal = work_signal[sorted_indices]
    work_return = work_return[sorted_indices]
    
    # Calculate wins and losses for ALL 4 strategies:
    # 1. Long above threshold: wins when return > 0
    # 2. Long below threshold: wins when return > 0
    # 3. Short above threshold: wins when return < 0
    # 4. Short below threshold: wins when return < 0
    
    # For LONG positions (both above and below)
    wins_long = np.where(work_return > 0, work_return, 0.0)
    losses_long = np.where(work_return <= 0, -work_return, 0.0)
    
    # For SHORT positions (both above and below)
    wins_short = np.where(work_return < 0, -work_return, 0.0)
    losses_short = np.where(work_return >= 0, work_return, 0.0)
    
    # Compute cumulative sums
    cumsum_wins_long = np.cumsum(wins_long)
    cumsum_losses_long = np.cumsum(losses_long)
    cumsum_wins_short = np.cumsum(wins_short)
    cumsum_losses_short = np.cumsum(losses_short)
    
    # Total wins and losses
    total_wins_long = np.sum(wins_long)
    total_losses_long = np.sum(losses_long)
    total_wins_short = np.sum(wins_short)
    total_losses_short = np.sum(losses_short)
    
    # Compute wins/losses for ABOVE set (everything after index i)
    cumsum_wins_long_above = total_wins_long - cumsum_wins_long
    cumsum_losses_long_above = total_losses_long - cumsum_losses_long
    cumsum_wins_short_above = total_wins_short - cumsum_wins_short
    cumsum_losses_short_above = total_losses_short - cumsum_losses_short
    
    # Compute profit factors for all 4 strategies at each threshold
    eps = 1.0e-30
    pf_long_above = cumsum_wins_long_above / (cumsum_losses_long_above + eps)
    pf_long_below = cumsum_wins_long / (cumsum_losses_long + eps)
    pf_short_above = cumsum_wins_short_above / (cumsum_losses_short_above + eps)
    pf_short_below = cumsum_wins_short / (cumsum_losses_short + eps)
    
    # Find unique threshold positions (where signal value changes)
    unique_thresh_mask = np.ones(n - 1, dtype=bool)
    unique_thresh_mask[:-1] = work_signal[1:-1] != work_signal[:-2]
    
    # Profit factor for entire dataset (assuming long position)
    pf_all = total_wins_long / (total_losses_long + eps)
    
    # ========================================================================
    # FIND BEST LONG STRATEGY (choose between above or below threshold)
    # ========================================================================
    
    # Valid indices for above threshold: must have at least min_kept samples above
    valid_above_mask = np.arange(n - 1) <= (n - min_kept)
    valid_above_mask &= unique_thresh_mask
    
    # Valid indices for below threshold: must have at least min_kept samples below
    valid_below_mask = np.arange(n - 1) >= (min_kept - 1)
    valid_below_mask &= unique_thresh_mask
    
    # Find best LONG ABOVE threshold
    best_long_above_pf = -np.inf
    best_long_above_idx = 0
    if valid_above_mask.any():
        pf_candidates = pf_long_above[:-1].copy()
        pf_candidates[~valid_above_mask] = -np.inf
        best_long_above_idx = np.argmax(pf_candidates)
        best_long_above_pf = pf_candidates[best_long_above_idx]
    
    # Find best LONG BELOW threshold
    best_long_below_pf = -np.inf
    best_long_below_idx = n - 1
    if valid_below_mask.any():
        pf_candidates = pf_long_below[:-1].copy()
        pf_candidates[~valid_below_mask] = -np.inf
        best_long_below_idx = np.argmax(pf_candidates) + 1
        best_long_below_pf = pf_candidates[best_long_below_idx - 1]
    
    # Pick the best long strategy
    if best_long_above_pf >= best_long_below_pf:
        # Use LONG ABOVE strategy
        best_long_pf = best_long_above_pf
        best_long_idx = best_long_above_idx
        long_is_above = True
    else:
        # Use LONG BELOW strategy
        best_long_pf = best_long_below_pf
        best_long_idx = best_long_below_idx
        long_is_above = False
    
    # ========================================================================
    # FIND BEST SHORT STRATEGY (choose between above or below threshold)
    # ========================================================================
    
    # Find best SHORT ABOVE threshold
    best_short_above_pf = -np.inf
    best_short_above_idx = 0
    if valid_above_mask.any():
        pf_candidates = pf_short_above[:-1].copy()
        pf_candidates[~valid_above_mask] = -np.inf
        best_short_above_idx = np.argmax(pf_candidates)
        best_short_above_pf = pf_candidates[best_short_above_idx]
    
    # Find best SHORT BELOW threshold
    best_short_below_pf = -np.inf
    best_short_below_idx = n - 1
    if valid_below_mask.any():
        pf_candidates = pf_short_below[:-1].copy()
        pf_candidates[~valid_below_mask] = -np.inf
        best_short_below_idx = np.argmax(pf_candidates) + 1
        best_short_below_pf = pf_candidates[best_short_below_idx - 1]
    
    # Pick the best short strategy
    if best_short_above_pf >= best_short_below_pf:
        # Use SHORT ABOVE strategy
        best_short_pf = best_short_above_pf
        best_short_idx = best_short_above_idx
        short_is_above = True
    else:
        # Use SHORT BELOW strategy
        best_short_pf = best_short_below_pf
        best_short_idx = best_short_below_idx
        short_is_above = False
    
    # ========================================================================
    # CREATE OUTPUT (maintain backward compatibility)
    # ========================================================================
    
    # Get thresholds in original feature space
    high_thresh = work_signal[best_long_idx]
    low_thresh = work_signal[best_short_idx]
    
    # Create masks based on which strategy was chosen
    if long_is_above:
        high_mask = clean_feature >= high_thresh
    else:
        high_mask = clean_feature < high_thresh
    
    if short_is_above:
        low_mask = clean_feature >= low_thresh
    else:
        low_mask = clean_feature < low_thresh
    
    max_pf = max(best_long_pf, best_short_pf)
    
    # For backward compatibility, set sign_flipped based on whether strategies match correlation
    # This is just for reporting - it doesn't affect the optimization
    sign_flipped = False  # No longer used for optimization

    # Print results if verbose
    if verbose:
        feature_name = feature.name if hasattr(feature, 'name') else 'Feature'
        print(f"\n  Threshold Optimization Results for {feature_name}:")
        print(f"    Spearman Rho: {rho:.4f}")
        print(f"    Long Strategy:  {'ABOVE' if long_is_above else 'BELOW'} threshold {high_thresh:.4f}, PF={best_long_pf:.2f}, n={high_mask.sum()}")
        print(f"    Short Strategy: {'ABOVE' if short_is_above else 'BELOW'} threshold {low_thresh:.4f}, PF={best_short_pf:.2f}, n={low_mask.sum()}")
        print(f"    Best Strategy:  {'LONG' if best_long_pf >= best_short_pf else 'SHORT'} with PF={max_pf:.2f}")

    # Return results (same format for backward compatibility + new strategy flags)
    return {
        'spearman_rho': rho,
        'sign_flipped': sign_flipped,  # Always False now, kept for compatibility
        'pf_all': pf_all,
        'high_thresh': high_thresh,
        'pf_high': best_long_pf,  # Best long strategy (above or below)
        'n_high': high_mask.sum(),
        'high_feature': clean_feature[high_mask],
        'high_target': clean_target[high_mask],
        'low_thresh': low_thresh,
        'pf_low': best_short_pf,  # Best short strategy (above or below)
        'n_low': low_mask.sum(),
        'low_feature': clean_feature[low_mask],
        'low_target': clean_target[low_mask],
        'max_pf': max_pf,
        # NEW: Strategy direction flags
        'long_is_above': long_is_above,  # True = long when feature >= thresh, False = long when feature < thresh
        'short_is_above': short_is_above,  # True = short when feature >= thresh, False = short when feature < thresh
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
