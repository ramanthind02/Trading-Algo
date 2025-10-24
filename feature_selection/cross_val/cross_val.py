import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_selection.opt_thresh import optimize_threshold


def compute_profit_factor(returns: np.ndarray) -> float:
    """
    Compute profit factor from returns array.
    
    Args:
        returns: Array of returns
        
    Returns:
        Profit factor (wins / losses)
    """
    wins = returns[returns > 0].sum()
    losses = -returns[returns <= 0].sum()
    return wins / (losses + 1e-30)


def compute_sortino_ratio(returns: pd.Series, annualize: bool = True) -> float:
    """
    Compute Sortino ratio from returns series.
    
    Sortino ratio is like Sharpe ratio but only penalizes downside volatility,
    making it a better measure of risk-adjusted returns for strategies that
    have asymmetric return distributions.
    
    Formula: (mean_return / downside_std) * sqrt(252) if annualized
    
    Args:
        returns: Series of returns (log returns)
        annualize: Whether to annualize the ratio (default: True)
        
    Returns:
        Sortino ratio (annualized if annualize=True)
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    
    # Downside deviation: std of negative returns only
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        # No downside volatility - return a high value
        return 100.0 if mean_return > 0 else 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 100.0 if mean_return > 0 else 0.0
    
    sortino = mean_return / downside_std
    
    if annualize:
        sortino *= np.sqrt(252)  # Annualize assuming 252 trading days
    
    return sortino


def apply_threshold_strategy(
    feature: pd.Series,
    target: pd.Series,
    high_thresh: float,
    low_thresh: float,
    sign_flipped: bool = False,
    long_is_above: bool = None,
    short_is_above: bool = None
) -> Dict[str, float]:
    """
    Apply learned thresholds to compute out-of-sample profit factors.
    
    Reuses the threshold logic from optimize_threshold but applies
    pre-determined thresholds instead of optimizing them.
    
    Args:
        feature: Feature values
        target: Target returns
        high_thresh: Threshold for long trades
        low_thresh: Threshold for short trades
        sign_flipped: (DEPRECATED) Whether feature sign was flipped during training
        long_is_above: If True, long when feature >= high_thresh; if False, long when feature < high_thresh
        short_is_above: If True, short when feature >= low_thresh; if False, short when feature < low_thresh
        
    Returns:
        Dict with profit factors for long, short, and best strategies
    """
    # NEW LOGIC: Use long_is_above and short_is_above flags if provided
    if long_is_above is not None:
        # Use the new strategy flags
        if long_is_above:
            high_mask = feature >= high_thresh
        else:
            high_mask = feature < high_thresh
    else:
        # BACKWARD COMPATIBILITY: Fall back to old sign_flipped logic
        if sign_flipped:
            high_mask = feature <= high_thresh
        else:
            high_mask = feature >= high_thresh
    
    if short_is_above is not None:
        # Use the new strategy flags
        if short_is_above:
            low_mask = feature >= low_thresh
        else:
            low_mask = feature < low_thresh
    else:
        # BACKWARD COMPATIBILITY: Fall back to old sign_flipped logic
        if sign_flipped:
            low_mask = feature > low_thresh
        else:
            low_mask = feature < low_thresh
    
    # Compute profit factors
    pf_high = compute_profit_factor(target[high_mask].values) if high_mask.sum() > 0 else 0.0
    pf_low = compute_profit_factor(target[low_mask].values) if low_mask.sum() > 0 else 0.0
    pf_best = max(pf_high, pf_low)
    
    # Get returns for each strategy (for cumulative plotting)
    high_returns = target[high_mask] if high_mask.sum() > 0 else pd.Series([], dtype=float)
    low_returns = target[low_mask] if low_mask.sum() > 0 else pd.Series([], dtype=float)
    
    return {
        'pf_high': pf_high,
        'pf_low': pf_low,
        'pf_best': pf_best,
        'n_high': high_mask.sum(),
        'n_low': low_mask.sum(),
        'high_returns': high_returns,
        'low_returns': low_returns
    }


def cross_validate_features(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    floor: float = 0.02,
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform k-fold cross-validation on features using optimize_threshold.
    
    For each fold:
    1. Use training data to find optimal thresholds via optimize_threshold
    2. Apply those thresholds to test data to compute out-of-sample profit factors
    3. Aggregate results across all folds
    
    Args:
        df: DataFrame containing features and target
        target_col: Name of the target column
        n_splits: Number of CV folds (default: 5)
        floor: Minimum fraction of cases to trade (default: 0.02)
        exclude_cols: Columns to exclude from testing (default: None)
        verbose: Whether to print progress (default: True)
        
    Returns:
        DataFrame with cross-validation results for each feature:
            - 'feature': Feature name
            - 'mean_pf_high': Mean out-of-sample profit factor for long strategy
            - 'std_pf_high': Std dev of out-of-sample PF for long
            - 'mean_pf_low': Mean out-of-sample profit factor for short strategy
            - 'std_pf_low': Std dev of out-of-sample PF for short
            - 'mean_pf_best': Mean out-of-sample profit factor for best strategy
            - 'std_pf_best': Std dev of out-of-sample PF for best
            - 'mean_n_high': Average number of samples in long trades
            - 'mean_n_low': Average number of samples in short trades
            - 'in_sample_pf_high': In-sample PF for long (from full data)
            - 'in_sample_pf_low': In-sample PF for short (from full data)
            - 'in_sample_pf_best': In-sample PF for best (from full data)
    """
    
    if verbose:
        print("="*80)
        print("CROSS-VALIDATION WITH OPTIMIZE_THRESHOLD")
        print("="*80)
        print(f"Target column: {target_col}")
        print(f"Number of folds: {n_splits}")
        print(f"Floor (min fraction to trade): {floor}")
        print("="*80)
    
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Determine feature columns
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    n_features = len(feature_cols)
    
    if n_features == 0:
        raise ValueError("No feature columns to test")
    
    if verbose:
        print(f"\nNumber of features: {n_features}")
        print(f"Number of samples: {len(df)}")
    
    # Setup cross-validation (no shuffle to preserve time-series order)
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    # Store results for each feature
    results = []
    
    # Process each feature
    for feat_idx, feature_col in enumerate(feature_cols):
        if verbose and (feat_idx + 1) % 10 == 0:
            print(f"Processing feature {feat_idx+1}/{n_features}: {feature_col}")
        
        # Collect all OOS returns across folds for aggregation
        all_high_returns = []
        all_low_returns = []
        
        try:
            # Perform k-fold cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
                # Split data
                train_feature = df[feature_col].iloc[train_idx]
                train_target = df[target_col].iloc[train_idx]
                test_feature = df[feature_col].iloc[test_idx]
                test_target = df[target_col].iloc[test_idx]
                
                # Compute min_kept based on training set size
                min_kept = int(floor * len(train_idx) + 0.5)
                
                # Optimize thresholds on training data
                train_result = optimize_threshold(
                    train_feature,
                    train_target,
                    min_kept=min_kept,
                    verbose=verbose
                )
                
                # Apply thresholds to test data
                test_result = apply_threshold_strategy(
                    test_feature,
                    test_target,
                    train_result['high_thresh'],
                    train_result['low_thresh'],
                    train_result['sign_flipped'],
                    long_is_above=train_result.get('long_is_above'),
                    short_is_above=train_result.get('short_is_above')
                )
                
                # Skip this fold if sample sizes are too small for reliable estimates
                # Require at least 10 samples for each strategy to avoid extreme profit factors
                min_test_samples = 10
                
                if test_result['n_high'] < min_test_samples or test_result['n_low'] < min_test_samples:
                    if verbose:
                        print(f"  Skipping fold {fold_idx} for '{feature_col}': insufficient samples (n_high={test_result['n_high']}, n_low={test_result['n_low']})")
                    continue
                
                # Collect OOS returns from this fold
                all_high_returns.extend(test_result['high_returns'].values)
                all_low_returns.extend(test_result['low_returns'].values)
            
            # Compute in-sample performance on full dataset
            min_kept_full = int(floor * len(df) + 0.5)
            full_result = optimize_threshold(
                df[feature_col],
                df[target_col],
                min_kept=min_kept_full
            )
            
            # Only include features that have at least one valid fold
            if len(all_high_returns) == 0 or len(all_low_returns) == 0:
                if verbose:
                    print(f"  Skipping '{feature_col}': no valid folds (all had insufficient samples)")
                continue
            
            # Compute aggregated profit factors from all OOS returns
            all_high_returns = np.array(all_high_returns)
            all_low_returns = np.array(all_low_returns)
            
            high_wins = all_high_returns[all_high_returns > 0].sum()
            high_losses = -all_high_returns[all_high_returns <= 0].sum()
            oos_pf_high = high_wins / (high_losses + 1e-30)
            
            low_wins = all_low_returns[all_low_returns > 0].sum()
            low_losses = -all_low_returns[all_low_returns <= 0].sum()
            oos_pf_low = low_wins / (low_losses + 1e-30)
            
            oos_pf_best = max(oos_pf_high, oos_pf_low)
            
            # Aggregate results
            results.append({
                'feature': feature_col,
                'oos_pf_high': oos_pf_high,
                'oos_pf_low': oos_pf_low,
                'oos_pf_best': oos_pf_best,
                'total_n_high': len(all_high_returns),
                'total_n_low': len(all_low_returns),
                'in_sample_pf_high': full_result['pf_high'],
                'in_sample_pf_low': full_result['pf_low'],
                'in_sample_pf_best': full_result['max_pf']
            })
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to cross-validate feature '{feature_col}': {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by aggregated out-of-sample best profit factor
    results_df = results_df.sort_values('oos_pf_best', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nSuccessfully validated {len(results_df)} features")
        print(f"\nTop 5 features by aggregated out-of-sample profit factor:")
        top5 = results_df.head(5)[['feature', 'oos_pf_best', 'in_sample_pf_best']]
        print(top5.to_string(index=False))
    
    return results_df


def walkforward_validate_features(
    df: pd.DataFrame,
    target_col: str,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    floor: float = 0.1,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    plot_cumulative: bool = True,
    plot_top_n: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Perform walk-forward validation on features using optimize_threshold.
    
    This function uses a rolling time-based split where:
    1. Train on a fixed window of historical data
    2. Test on the next period immediately after training
    3. Roll forward and repeat
    
    Args:
        df: DataFrame containing features and target (must have datetime index)
        target_col: Name of the target column
        train_start: Start date for initial training period
        train_end: End date for initial training period
        test_step: Number of days for test period (also the roll-forward step size)
        num_steps: Number of walk-forward steps to perform
        floor: Minimum fraction of cases to trade (default: 0.02)
        feature_cols: List of specific features to test (None = all features)
        exclude_cols: Columns to exclude from testing (default: None)
        plot_cumulative: Whether to generate cumulative return plots (default: False)
        plot_top_n: Number of top features to plot (default: 5)
        verbose: Whether to print progress (default: True)
        
    Returns:
        Tuple containing:
        - DataFrame with walk-forward validation results for each feature:
            - 'feature': Feature name
            - 'n_valid_steps': Number of steps with sufficient samples
            - 'mean_pf_high': Mean out-of-sample profit factor for long strategy
            - 'std_pf_high': Std dev of out-of-sample PF for long
            - 'mean_pf_low': Mean out-of-sample profit factor for short strategy
            - 'std_pf_low': Std dev of out-of-sample PF for short
            - 'mean_pf_best': Mean out-of-sample profit factor for best strategy
            - 'std_pf_best': Std dev of out-of-sample PF for best
            - 'mean_n_high': Average number of samples in long trades
            - 'mean_n_low': Average number of samples in short trades
        - Dict of matplotlib figures (empty if plot_cumulative=False)
    """
    
    if verbose:
        print("="*80)
        print("WALK-FORWARD VALIDATION WITH OPTIMIZE_THRESHOLD")
        print("="*80)
        print(f"Target column: {target_col}")
        print(f"Training period: {train_start} to {train_end}")
        print(f"Test step: {test_step} days")
        print(f"Number of steps: {num_steps}")
        print(f"Floor (min fraction to trade): {floor}")
        print("="*80)
    
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for walk-forward validation")
    
    # Ensure datetime parameters are timezone-aware if DataFrame index is timezone-aware
    if df.index.tz is not None:
        if train_start.tzinfo is None:
            train_start = train_start.replace(tzinfo=df.index.tz)
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=df.index.tz)
    
    # Determine feature columns
    if exclude_cols is None:
        exclude_cols = []
    
    if feature_cols is None:
        # Use all columns except target and excluded
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    else:
        # Use regex matching to find columns that match any of the requested feature patterns
        import re
        available_features = [col for col in df.columns if col != target_col and col not in exclude_cols]
        matched_features = []
        for pattern in feature_cols:
            # Match columns that contain the pattern (case-insensitive)
            regex = re.compile(pattern, re.IGNORECASE)
            matched = [col for col in available_features if regex.search(col)]
            matched_features.extend(matched)
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(matched_features))
    
    n_features = len(feature_cols)
    
    if n_features == 0:
        raise ValueError("No feature columns to test")
    
    if verbose:
        print(f"\nNumber of features: {n_features}")
        print(f"Total samples: {len(df)}")
        print("\n" + "="*80)
    
    # Store results for each feature
    results = []
    
    # Process each feature
    for feat_idx, feature_col in enumerate(feature_cols):
        if verbose:
            print(f"\nProcessing feature {feat_idx+1}/{n_features}: {feature_col}")
        
        # Store step results
        step_results = {
            'pf_high': [],
            'pf_low': [],
            'pf_best': [],
            'n_high': [],
            'n_low': [],
            'high_returns': [],  # For cumulative plotting - long strategy
            'low_returns': []    # For cumulative plotting - short strategy
        }
        
        try:
            # Perform walk-forward validation
            for step in range(num_steps):
                # Calculate date ranges for this step
                current_train_start = train_start + timedelta(days=step * test_step)
                current_train_end = train_end + timedelta(days=step * test_step)
                current_test_start = current_train_end
                current_test_end = current_test_start + timedelta(days=test_step)
                
                # Filter data for each period
                train_mask = (df.index >= current_train_start) & (df.index < current_train_end)
                test_mask = (df.index >= current_test_start) & (df.index < current_test_end)
                
                # Get train and test data
                train_feature = df[feature_col][train_mask]
                train_target = df[target_col][train_mask]
                test_feature = df[feature_col][test_mask]
                test_target = df[target_col][test_mask]
                
                if verbose and step == 0:
                    print(f"  Step {step}: Train={len(train_feature)}, Test={len(test_feature)}")
                
                # Skip if insufficient data
                if len(train_feature) < 100 or len(test_feature) < 10:
                    if verbose:
                        print(f"  Skipping step {step}: insufficient data")
                    continue
                
                # Compute min_kept based on training set size
                min_kept = int(floor * len(train_feature) + 0.5)
                
                # Optimize thresholds on training data
                train_result = optimize_threshold(
                    train_feature,
                    train_target,
                    min_kept=min_kept,
                    verbose=verbose
                )
                
                # Apply thresholds to test data
                test_result = apply_threshold_strategy(
                    test_feature,
                    test_target,
                    train_result['high_thresh'],
                    train_result['low_thresh'],
                    train_result['sign_flipped'],
                    long_is_above=train_result.get('long_is_above'),
                    short_is_above=train_result.get('short_is_above')
                )
                
                # Track long and short strategies separately
                # Only require minimum samples for the strategy we're using
                min_test_samples = 10
                
                has_valid_high = test_result['n_high'] >= min_test_samples
                has_valid_low = test_result['n_low'] >= min_test_samples
                
                # Skip this step only if BOTH strategies have insufficient samples
                if not has_valid_high and not has_valid_low:
                    if verbose:
                        print(f"  Skipping step {step}: both strategies have insufficient samples (n_high={test_result['n_high']}, n_low={test_result['n_low']})")
                    continue
                
                # Store results (use 0.0 for strategies with insufficient samples)
                step_results['pf_high'].append(test_result['pf_high'] if has_valid_high else 0.0)
                step_results['pf_low'].append(test_result['pf_low'] if has_valid_low else 0.0)
                step_results['pf_best'].append(max(
                    test_result['pf_high'] if has_valid_high else 0.0,
                    test_result['pf_low'] if has_valid_low else 0.0
                ))
                step_results['n_high'].append(test_result['n_high'] if has_valid_high else 0)
                step_results['n_low'].append(test_result['n_low'] if has_valid_low else 0)
                
                # Collect returns for cumulative plotting - include ALL trades for complete history
                # (even if < 10 samples, though PF calculations still use the 10-sample minimum)
                if test_result['n_high'] > 0:
                    step_results['high_returns'].append(test_result['high_returns'])
                if test_result['n_low'] > 0:
                    step_results['low_returns'].append(test_result['low_returns'])
            
            # Only include features that have at least one valid step
            if len(step_results['pf_best']) == 0:
                if verbose:
                    print(f"  Skipping '{feature_col}': no valid steps")
                continue
            
            # Concatenate all returns for cumulative plotting - separate for long and short
            if step_results['high_returns']:
                high_returns_series = pd.concat(step_results['high_returns'], axis=0)
            else:
                high_returns_series = pd.Series([], dtype=float)
            
            if step_results['low_returns']:
                low_returns_series = pd.concat(step_results['low_returns'], axis=0)
            else:
                low_returns_series = pd.Series([], dtype=float)
            
            # Compute final OOS profit factor from pooled returns (not mean per fold)
            # This matches the C++ implementation: single PF from all accumulated wins/losses
            final_pf_high = compute_profit_factor(high_returns_series.values) if len(high_returns_series) > 0 else 0.0
            final_pf_low = compute_profit_factor(low_returns_series.values) if len(low_returns_series) > 0 else 0.0
            final_pf_best = max(final_pf_high, final_pf_low)
            
            # Compute Sortino ratios (annualized, assuming 252 trading days)
            # Sortino = (mean_return / downside_std) * sqrt(252)
            # Better than Sharpe because it only penalizes downside volatility
            sortino_high = compute_sortino_ratio(high_returns_series, annualize=True)
            sortino_low = compute_sortino_ratio(low_returns_series, annualize=True)
            sortino_best = sortino_high if final_pf_high >= final_pf_low else sortino_low
            
            # Aggregate results
            results.append({
                'feature': feature_col,
                'n_valid_steps': len(step_results['pf_best']),
                'oos_pf_high': final_pf_high,  # Final OOS PF from pooled returns
                'oos_pf_low': final_pf_low,    # Final OOS PF from pooled returns
                'oos_pf_best': final_pf_best,  # Best of the two strategies
                'oos_sortino_high': sortino_high,  # Annualized Sortino ratio for long
                'oos_sortino_low': sortino_low,    # Annualized Sortino ratio for short
                'oos_sortino_best': sortino_best,  # Sortino for best strategy
                'mean_pf_high_per_fold': np.mean(step_results['pf_high']),  # Keep for reference
                'mean_pf_low_per_fold': np.mean(step_results['pf_low']),    # Keep for reference
                'std_pf_high': np.std(step_results['pf_high']),
                'std_pf_low': np.std(step_results['pf_low']),
                'mean_n_high': np.mean(step_results['n_high']),
                'mean_n_low': np.mean(step_results['n_low']),
                'high_returns_series': high_returns_series,  # Store for plotting
                'low_returns_series': low_returns_series     # Store for plotting
            })
            
            if verbose:
                print(f"  Valid steps: {len(step_results['pf_best'])}/{num_steps}, Final OOS PF: {final_pf_best:.2f}")
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to validate feature '{feature_col}': {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Handle empty results
    if len(results_df) == 0:
        if verbose:
            print(f"\n{'='*80}")
            print("WALK-FORWARD VALIDATION COMPLETE")
            print(f"{'='*80}")
            print("\nNo features could be validated successfully.")
            print("Check that your datetime parameters match the DataFrame index timezone.")
        return results_df, {}
    
    # Sort by final out-of-sample best profit factor
    results_df = results_df.sort_values('oos_pf_best', ascending=False).reset_index(drop=True)
    
    # Compute aggregate performance with equal weighting
    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING AGGREGATE PERFORMANCE")
        print(f"{'='*80}")
    
    # Collect all returns from all features with equal weighting
    all_high_returns = []
    all_low_returns = []
    
    if verbose:
        print(f"\nCollecting returns from {len(results)} features...")
    
    for result in results:
        high_returns = result.get('high_returns_series', pd.Series([], dtype=float))
        low_returns = result.get('low_returns_series', pd.Series([], dtype=float))
        
        if verbose:
            print(f"  Feature '{result['feature']}': {len(high_returns)} high returns, {len(low_returns)} low returns")
        
        if len(high_returns) > 0:
            all_high_returns.append(high_returns)
        if len(low_returns) > 0:
            all_low_returns.append(low_returns)
    
    if verbose:
        print(f"\nTotal collected: {len(all_high_returns)} high return series, {len(all_low_returns)} low return series")
    
    # Compute buy/hold baseline (always predict 1)
    test_start_date = train_end
    test_end_date = train_end + timedelta(days=test_step * num_steps)
    buyhold_returns = df[target_col].loc[
        (df.index >= test_start_date) & (df.index < test_end_date)
    ]
    
    # Aggregate with constant position sizing:
    # Each strategy gets equal weight (1/N), where N = num_features + 1 (for buy/hold)
    # On days without a feature signal, that weight goes to buy/hold instead
    if verbose:
        print(f"\nAggregating long strategies:")
        print(f"  Features with high returns: {len(all_high_returns)}")
        print(f"  Buy/hold returns: {len(buyhold_returns)}")
    
    if all_high_returns and len(buyhold_returns) > 0:
        # Signal combination with proportional position sizing:
        # - Position size = (number of active signals) / (total possible signals)
        # - Max risk (100%) when all signals agree, reduced risk when fewer signals
        # - All strategies trade the same underlying asset (target returns)
        
        if verbose:
            print(f"  Using signal combination with proportional position sizing")
            print(f"  Total possible signals: {len(all_high_returns) + 1} (features + buy/hold)")
        
        # Count how many signals are active on each day
        # Buy/hold is always active (always says "long")
        n_active_signals = pd.Series(1.0, index=buyhold_returns.index)  # Start with 1 for buy/hold
        
        # Check which features have signals on each day
        for feature_returns in all_high_returns:
            # Align feature returns with buy/hold index (NaN where no signal)
            aligned_feature = feature_returns.reindex(buyhold_returns.index)
            
            # Increment signal count where feature has a signal
            mask = ~aligned_feature.isna()
            n_active_signals[mask] += 1.0
        
        # Position size = (active signals) / (total signals)
        # This gives us the fraction of max risk to use
        total_signals = len(all_high_returns) + 1  # features + buy/hold
        position_size = n_active_signals / total_signals
        
        # Apply position sizing to target returns
        # When all signals agree: position_size = 1.0 (100% position)
        # When only buy/hold: position_size = 1/N (reduced position)
        aggregate_high_returns = buyhold_returns * position_size
        
        if verbose:
            days_with_features = (n_active_signals > 1).sum()
            days_without_features = (n_active_signals == 1).sum()
            print(f"  Days with feature signals: {days_with_features}")
            print(f"  Days without feature signals: {days_without_features}")
            if days_with_features > 0:
                sample_day_with = (n_active_signals > 1).idxmax()
                print(f"  Sample day WITH signal: {n_active_signals[sample_day_with]:.0f} signals active, position size={position_size[sample_day_with]:.4f}")
            if days_without_features > 0:
                sample_day_without = (n_active_signals == 1).idxmax()
                print(f"  Sample day WITHOUT signal: {n_active_signals[sample_day_without]:.0f} signals active, position size={position_size[sample_day_without]:.4f}")
        
        aggregate_pf_high = compute_profit_factor(aggregate_high_returns.values)
        
        if verbose:
            print(f"  Aggregate PF: {aggregate_pf_high:.4f}")
            print(f"  Buy/hold PF: {compute_profit_factor(buyhold_returns.values):.4f}")
            
            # Direct comparison check
            diff = (aggregate_high_returns - buyhold_returns).abs()
            print(f"  Max difference between aggregate and buy/hold: {diff.max():.6f}")
            print(f"  Mean difference: {diff.mean():.6f}")
            print(f"  Are they identical? {np.allclose(aggregate_high_returns.values, buyhold_returns.values)}")
            
            # Check a specific day with RSI signal
            if days_with_features > 0:
                sample_day = (n_active_signals > 1).idxmax()
                print(f"  Sample day WITH signal ({sample_day}):")
                print(f"    Target return: {buyhold_returns.loc[sample_day]:.6f}")
                print(f"    Active signals: {n_active_signals.loc[sample_day]:.0f}/{total_signals}")
                print(f"    Position size: {position_size.loc[sample_day]:.4f}")
                print(f"    Aggregate return: {aggregate_high_returns.loc[sample_day]:.6f}")
                print(f"    Expected: {buyhold_returns.loc[sample_day]:.6f} × {position_size.loc[sample_day]:.4f} = {buyhold_returns.loc[sample_day] * position_size.loc[sample_day]:.6f}")
    elif all_high_returns:
        # Only features, no buy/hold
        combined_high = pd.concat(all_high_returns, axis=1)
        aggregate_high_returns = combined_high.mean(axis=1, skipna=True)
        aggregate_pf_high = compute_profit_factor(aggregate_high_returns.values)
    elif len(buyhold_returns) > 0:
        # Only buy/hold
        aggregate_high_returns = buyhold_returns
        aggregate_pf_high = compute_profit_factor(aggregate_high_returns.values)
    else:
        aggregate_high_returns = pd.Series([], dtype=float)
        aggregate_pf_high = 0.0
    
    if all_low_returns:
        combined_low = pd.concat(all_low_returns, axis=1)
        aggregate_low_returns = combined_low.mean(axis=1, skipna=True)
        aggregate_pf_low = compute_profit_factor(aggregate_low_returns.values)
    else:
        aggregate_low_returns = pd.Series([], dtype=float)
        aggregate_pf_low = 0.0
    
    # Add aggregate and buy/hold to results
    if len(aggregate_high_returns) > 0:
        # Compute Sortino for aggregate long
        agg_sortino_high = compute_sortino_ratio(aggregate_high_returns, annualize=True)
        
        results_df.loc[len(results_df)] = {
            'feature': 'AGGREGATE_LONG',
            'n_valid_steps': num_steps,
            'oos_pf_high': aggregate_pf_high,
            'oos_pf_low': 0.0,
            'oos_pf_best': aggregate_pf_high,
            'oos_sortino_high': agg_sortino_high,
            'oos_sortino_low': 0.0,
            'oos_sortino_best': agg_sortino_high,
            'mean_pf_high_per_fold': 0.0,
            'mean_pf_low_per_fold': 0.0,
            'std_pf_high': 0.0,
            'std_pf_low': 0.0,
            'mean_n_high': len(aggregate_high_returns),
            'mean_n_low': 0,
            'high_returns_series': aggregate_high_returns,
            'low_returns_series': pd.Series([], dtype=float)
        }
    
    if len(aggregate_low_returns) > 0:
        # Compute Sortino for aggregate short
        agg_sortino_low = compute_sortino_ratio(aggregate_low_returns, annualize=True)
        
        results_df.loc[len(results_df)] = {
            'feature': 'AGGREGATE_SHORT',
            'n_valid_steps': num_steps,
            'oos_pf_high': 0.0,
            'oos_pf_low': aggregate_pf_low,
            'oos_pf_best': aggregate_pf_low,
            'oos_sortino_high': 0.0,
            'oos_sortino_low': agg_sortino_low,
            'oos_sortino_best': agg_sortino_low,
            'mean_pf_high_per_fold': 0.0,
            'mean_pf_low_per_fold': 0.0,
            'std_pf_high': 0.0,
            'std_pf_low': 0.0,
            'mean_n_high': 0,
            'mean_n_low': len(aggregate_low_returns),
            'high_returns_series': pd.Series([], dtype=float),
            'low_returns_series': aggregate_low_returns
        }
    
    if len(buyhold_returns) > 0:
        buyhold_pf = compute_profit_factor(buyhold_returns.values)
        # Compute Sortino for buy/hold
        buyhold_sortino = compute_sortino_ratio(buyhold_returns, annualize=True)
        
        results_df.loc[len(results_df)] = {
            'feature': 'BUY_HOLD',
            'n_valid_steps': num_steps,
            'oos_pf_high': buyhold_pf,
            'oos_pf_low': 0.0,
            'oos_pf_best': buyhold_pf,
            'oos_sortino_high': buyhold_sortino,
            'oos_sortino_low': 0.0,
            'oos_sortino_best': buyhold_sortino,
            'mean_pf_high_per_fold': 0.0,
            'mean_pf_low_per_fold': 0.0,
            'std_pf_high': 0.0,
            'std_pf_low': 0.0,
            'mean_n_high': len(buyhold_returns),
            'mean_n_low': 0,
            'high_returns_series': buyhold_returns,
            'low_returns_series': pd.Series([], dtype=float)
        }
    
    if verbose:
        print(f"\n{'='*80}")
        print("WALK-FORWARD VALIDATION COMPLETE")
        print(f"{'='*80}")
        n_features = len([r for r in results if r['feature'] not in ['AGGREGATE_LONG', 'AGGREGATE_SHORT', 'BUY_HOLD']])
        print(f"\nSuccessfully validated {n_features} features")
        print(f"\nTop 5 features by final out-of-sample profit factor:")
        top5 = results_df[~results_df['feature'].isin(['AGGREGATE_LONG', 'AGGREGATE_SHORT', 'BUY_HOLD'])].head(5)[['feature', 'n_valid_steps', 'oos_pf_best', 'oos_sortino_best']]
        print(top5.to_string(index=False))
        print(f"\n{'='*80}")
        print("AGGREGATE PERFORMANCE (Equal-Weighted)")
        print(f"{'='*80}")
        if len(aggregate_high_returns) > 0:
            agg_sortino_high = compute_sortino_ratio(aggregate_high_returns, annualize=True)
            print(f"Long Strategy:  PF={aggregate_pf_high:.2f}, Sortino={agg_sortino_high:.2f}, Trades={len(aggregate_high_returns)}")
        if len(aggregate_low_returns) > 0:
            agg_sortino_low = compute_sortino_ratio(aggregate_low_returns, annualize=True)
            print(f"Short Strategy: PF={aggregate_pf_low:.2f}, Sortino={agg_sortino_low:.2f}, Trades={len(aggregate_low_returns)}")
        if len(buyhold_returns) > 0:
            buyhold_sortino = compute_sortino_ratio(buyhold_returns, annualize=True)
            print(f"Buy & Hold:     PF={buyhold_pf:.2f}, Sortino={buyhold_sortino:.2f}, Trades={len(buyhold_returns)}")
    
    # Generate cumulative return plots if requested
    figures = {}
    if plot_cumulative and len(results_df) > 0:
        if verbose:
            print(f"\nGenerating cumulative return plots for top {plot_top_n} features...")
        
        # Get top N features (including aggregate strategies)
        top_features = results_df.head(plot_top_n)['feature'].tolist()
        
        for feature_name in top_features:
            # Find this feature in results or results_df (for aggregate strategies)
            feature_data = [r for r in results if r['feature'] == feature_name]
            
            # If not found in results, check if it's an aggregate strategy in results_df
            if not feature_data:
                if feature_name in ['AGGREGATE_LONG', 'AGGREGATE_SHORT', 'BUY_HOLD']:
                    row = results_df[results_df['feature'] == feature_name]
                    if len(row) > 0:
                        feature_data = [{
                            'feature': feature_name,
                            'high_returns_series': row.iloc[0]['high_returns_series'],
                            'low_returns_series': row.iloc[0]['low_returns_series'],
                            'oos_pf_high': row.iloc[0]['oos_pf_high'],
                            'oos_pf_low': row.iloc[0]['oos_pf_low'],
                            'mean_n_high': row.iloc[0]['mean_n_high'],
                            'mean_n_low': row.iloc[0]['mean_n_low']
                        }]
                    else:
                        continue
                else:
                    continue
            
            # Get the stored returns for both strategies
            high_returns = feature_data[0].get('high_returns_series', pd.Series([], dtype=float))
            low_returns = feature_data[0].get('low_returns_series', pd.Series([], dtype=float))
            
            # Skip if no returns for either strategy
            if len(high_returns) == 0 and len(low_returns) == 0:
                continue
            
            # Create plot with two subplots (long and short)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot long strategy (high)
            if len(high_returns) > 0:
                # Long position: profit from positive returns (position = +1)
                # Convert log returns to raw percentage returns
                # cumsum gives cumulative log return, exp() converts to price ratio, -1 gives percentage
                cum_high_log = high_returns.cumsum()
                cum_high_pct = (np.exp(cum_high_log) - 1) * 100  # Convert to percentage
                final_high_pct = cum_high_pct.iloc[-1] if len(cum_high_pct) > 0 else 0
                color_high = '#2ecc71' if final_high_pct >= 0 else '#e74c3c'
                
                ax1.plot(cum_high_pct.index, cum_high_pct.values, color=color_high, linewidth=2, label='Long')
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax1.set_ylabel('Cumulative Return (%)')
                ax1.set_title(f"{feature_name} - Long Strategy\nFinal: {final_high_pct:.2f}%, OOS PF: {feature_data[0]['oos_pf_high']:.2f}, Trades: {len(high_returns)}")
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No valid long trades', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f"{feature_name} - Long Strategy (No Data)")
            
            # Plot short strategy (low)
            if len(low_returns) > 0:
                # Short position: profit from negative returns (position = -1)
                # Invert returns so negative market returns become positive strategy returns
                short_strategy_returns = -low_returns
                
                # Convert log returns to raw percentage returns
                # cumsum gives cumulative log return, exp() converts to price ratio, -1 gives percentage
                cum_low_log = short_strategy_returns.cumsum()
                cum_low_pct = (np.exp(cum_low_log) - 1) * 100  # Convert to percentage
                final_low_pct = cum_low_pct.iloc[-1] if len(cum_low_pct) > 0 else 0
                color_low = '#2ecc71' if final_low_pct >= 0 else '#e74c3c'
                
                ax2.plot(cum_low_pct.index, cum_low_pct.values, color=color_low, linewidth=2, label='Short')
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Cumulative Return (%)')
                ax2.set_title(f"{feature_name} - Short Strategy\nFinal: {final_low_pct:.2f}%, OOS PF: {feature_data[0]['oos_pf_low']:.2f}, Trades: {len(low_returns)}")
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No valid short trades', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f"{feature_name} - Short Strategy (No Data)")
            
            fig.tight_layout()
            figures[feature_name] = fig
        
        if verbose and len(figures) > 0:
            print(f"Generated {len(figures)} cumulative return plots")
    
    return results_df, figures 

