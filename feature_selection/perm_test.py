import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os
from datetime import datetime

# Add parent directory to path to import helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import spearman_rho
from feature_selection.opt_thresh import optimize_threshold


def permutation_test(
    df: pd.DataFrame,
    target_col: str,
    floor: float = 0.02,
    alpha: float = 0.1,
    nreps: int = 100,
    exclude_cols: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform permutation test with solo and unbiased p-values.
    
    This function computes two types of p-values for each feature:
    - Solo p-value: How often does this feature's permuted criterion beat its original?
    - Unbiased p-value: How often does the MAX criterion across all features beat this feature's original?
    
    The unbiased p-value accounts for multiple comparisons and is more conservative.
    
    Algorithm:
    1. Compute original profit factors for all features (long, short, and best)
    2. For each permutation:
       - Shuffle target values
       - Compute profit factors for all features
       - For each feature: if permuted >= original, increment solo count
       - Track max profit factor across all features
       - For each feature: if max >= original, increment unbiased count
    3. Calculate p-values as counts / nreps
    
    Args:
        df: DataFrame containing features and target
        target_col: Name of the target column (e.g., 'log_returns')
        floor: Minimum fraction (0-0.5) of cases that must trade (default: 0.1)
        alpha: Familywise alpha level (0-1, generally small) (default: 0.05)
        nreps: Number of permutation replications (default: 1000)
        exclude_cols: List of column names to exclude from testing (default: None)
        random_seed: Random seed for reproducibility (default: None)
        verbose: Whether to print progress messages (default: True)
        
    Returns:
        pd.DataFrame: Detailed results with columns:
            - 'feature': Feature name
            - 'spearman_rho': Spearman correlation with target
            - 'sign_flipped': Whether feature sign was flipped
            - 'long_thresh': Optimal high threshold for long trades
            - 'long_pf': Profit factor for long trades
            - 'long_solo_pval': Solo p-value for long trades
            - 'long_unbiased_pval': Unbiased p-value for long trades
            - 'short_thresh': Optimal low threshold for short trades
            - 'short_pf': Profit factor for short trades
            - 'short_solo_pval': Solo p-value for short trades
            - 'short_unbiased_pval': Unbiased p-value for short trades
            - 'best_pf': Best profit factor (max of long_pf and short_pf)
            - 'best_solo_pval': Solo p-value for best strategy
            - 'best_unbiased_pval': Unbiased p-value for best strategy
            - 'preferred_strategy': 'long' or 'short' based on which has better profit factor
            - 'significant': Whether feature passes test (preferred strategy's unbiased_pval <= alpha)
            
        Sorted by best_pf in descending order (best to worst).
        
    Example:
        >>> df = pd.DataFrame({
        ...     'feature1': np.random.randn(1000),
        ...     'feature2': np.random.randn(1000),
        ...     'returns': np.random.randn(1000) * 0.01
        ... })
        >>> results = permutation_test(df, 'returns', nreps=100)
        >>> selected = results[results['significant']]
        >>> print(f"Selected {len(selected)} features")
    """
    
    if verbose:
        print("="*80)
        print("PERMUTATION TEST")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target column: {target_col}")
        print(f"Floor (min fraction to trade): {floor}")
        print(f"Alpha level: {alpha}")
        print(f"Number of replications: {nreps}")
        if random_seed is not None:
            print(f"Random seed: {random_seed}")
        print("="*80)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    if not (0 <= floor <= 0.5):
        raise ValueError(f"Floor must be between 0 and 0.5, got {floor}")
    
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    if nreps < 1:
        raise ValueError(f"nreps must be at least 1, got {nreps}")
    
    # Get target series
    target = df[target_col].values.copy()
    n_cases = len(target)
    min_kept = int(floor * n_cases + 0.5)
    
    if verbose:
        print(f"\nNumber of cases: {n_cases}")
        print(f"Minimum cases to keep per threshold: {min_kept}")
    
    # Determine which columns to process
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    n_features = len(feature_cols)
    
    if n_features == 0:
        raise ValueError("No feature columns to test")
    
    if verbose:
        print(f"Number of features to test: {n_features}")
        print("\n" + "="*80)
        print("STEP 1: Computing original (unpermuted) criteria")
        print("="*80)
    
    # Step 1: Compute original criteria for all features
    original_results = []
    
    for i, col in enumerate(feature_cols):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing feature {i+1}/{n_features}: {col}")
        
        try:
            feature = df[col]
            result = optimize_threshold(feature, df[target_col], min_kept=min_kept)
            
            original_results.append({
                'feature': col,
                'spearman_rho': result['spearman_rho'],
                'sign_flipped': result['sign_flipped'],
                'long_thresh': result['high_thresh'],
                'long_pf': result['pf_high'],
                'short_thresh': result['low_thresh'],
                'short_pf': result['pf_low'],
                'best_pf': result['max_pf'],
            })
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to optimize feature '{col}': {str(e)}")
            continue
    
    # Create DataFrame and sort by criterion
    results_df = pd.DataFrame(original_results)
    n_valid_features = len(results_df)
    
    if n_valid_features == 0:
        raise ValueError("No features could be successfully optimized")
    
    if verbose:
        print(f"\nSuccessfully optimized {n_valid_features} features")
        print(f"\nTop 5 features by best profit factor:")
        top5 = results_df.nlargest(5, 'best_pf')[['feature', 'best_pf', 'spearman_rho']]
        print(top5.to_string(index=False))
    
    # Initialize counters (includes the original unpermuted run)
    results_df['long_solo_count'] = 1
    results_df['short_solo_count'] = 1
    results_df['best_solo_count'] = 1
    results_df['long_unbiased_count'] = 1
    results_df['short_unbiased_count'] = 1
    results_df['best_unbiased_count'] = 1
    
    if verbose:
        print("\n" + "="*80)
        print(f"STEP 2: Running {nreps-1} permutation replications")
        print("="*80)
    
    # Step 2: Permutation testing
    for irep in range(nreps - 1):
        if verbose and (irep + 1) % 100 == 0:
            print(f"Replication {irep+1}/{nreps-1}...")
        
        # Shuffle target once per replication
        permuted_target_series = pd.Series(np.random.permutation(target), index=df.index)
        
        # Track maximum profit factors across all features for unbiased p-values
        max_long_pf = -1e60
        max_short_pf = -1e60
        max_best_pf = -1e60
        
        # Compute profit factors for all features with permuted target
        for idx, row in results_df.iterrows():
            feature_name = row['feature']
            
            # Compute profit factors for this feature with permuted target
            try:
                result = optimize_threshold(df[feature_name], permuted_target_series, min_kept=min_kept)
                perm_long_pf = result['pf_high']
                perm_short_pf = result['pf_low']
                perm_best_pf = result['max_pf']
            except:
                perm_long_pf = perm_short_pf = perm_best_pf = -1e60
            
            # Solo counts: compare permuted to original for this feature
            if perm_long_pf >= row['long_pf']:
                results_df.loc[idx, 'long_solo_count'] += 1
            if perm_short_pf >= row['short_pf']:
                results_df.loc[idx, 'short_solo_count'] += 1
            if perm_best_pf >= row['best_pf']:
                results_df.loc[idx, 'best_solo_count'] += 1
            
            # Track maximum across all features
            max_long_pf = max(max_long_pf, perm_long_pf)
            max_short_pf = max(max_short_pf, perm_short_pf)
            max_best_pf = max(max_best_pf, perm_best_pf)
        
        # Unbiased counts: compare max across all features to each feature's original
        for idx, row in results_df.iterrows():
            if max_long_pf >= row['long_pf']:
                results_df.loc[idx, 'long_unbiased_count'] += 1
            if max_short_pf >= row['short_pf']:
                results_df.loc[idx, 'short_unbiased_count'] += 1
            if max_best_pf >= row['best_pf']:
                results_df.loc[idx, 'best_unbiased_count'] += 1
    
    if verbose:
        print(f"\nCompleted {nreps-1} permutations")
        print("\n" + "="*80)
        print("STEP 3: Computing p-values")
        print("="*80)
    
    # Step 3: Compute p-values
    results_df['long_solo_pval'] = results_df['long_solo_count'] / nreps
    results_df['short_solo_pval'] = results_df['short_solo_count'] / nreps
    results_df['long_unbiased_pval'] = results_df['long_unbiased_count'] / nreps
    results_df['short_unbiased_pval'] = results_df['short_unbiased_count'] / nreps
    
    # Best p-values should be the minimum of long and short
    # This makes sense because if you're choosing the best strategy,
    # you want the p-value for whichever strategy you actually chose
    results_df['best_solo_pval'] = results_df[['long_solo_pval', 'short_solo_pval']].min(axis=1)
    results_df['best_unbiased_pval'] = results_df[['long_unbiased_pval', 'short_unbiased_pval']].min(axis=1)
    
    # Drop count columns
    results_df.drop(columns=[
        'long_solo_count', 'short_solo_count', 'best_solo_count',
        'long_unbiased_count', 'short_unbiased_count', 'best_unbiased_count'
    ], inplace=True)
    
    # Determine preferred strategy for each feature (long or short)
    results_df['preferred_strategy'] = results_df.apply(
        lambda row: 'long' if row['long_pf'] >= row['short_pf'] else 'short',
        axis=1
    )
    
    # Mark significant features based on their preferred strategy's unbiased p-value
    # If long is preferred, use long_unbiased_pval; if short is preferred, use short_unbiased_pval
    results_df['significant'] = results_df.apply(
        lambda row: row['long_unbiased_pval'] <= alpha if row['preferred_strategy'] == 'long' 
                    else row['short_unbiased_pval'] <= alpha,
        axis=1
    )
    
    # Sort by best_pf descending (best to worst) for final output
    results_df = results_df.sort_values('best_pf', ascending=False).reset_index(drop=True)
    
    # Count significant features
    n_significant = results_df['significant'].sum()
    
    if verbose:
        print(f"\nNumber of significant features (p <= {alpha}): {n_significant}")
        
        if n_significant > 0:
            print(f"\nSignificant features:")
            sig_features = results_df[results_df['significant']][[
                'feature', 'best_pf', 'preferred_strategy', 'long_unbiased_pval', 'short_unbiased_pval'
            ]]
            print(sig_features.to_string(index=False))
        else:
            print("\nNo features passed the test.")
            print(f"Best long unbiased p-value: {results_df['long_unbiased_pval'].min():.4f}")
            print(f"Best short unbiased p-value: {results_df['short_unbiased_pval'].min():.4f}")
        
        print("\n" + "="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    return results_df
