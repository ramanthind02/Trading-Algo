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


def apply_threshold_strategy(
    feature: pd.Series,
    target: pd.Series,
    high_thresh: float,
    low_thresh: float,
    sign_flipped: bool
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
        sign_flipped: Whether feature sign was flipped during training
        
    Returns:
        Dict with profit factors for long, short, and best strategies
    """
    # Apply sign flip if it was done during training
    if sign_flipped:
        # Masks are reversed when sign is flipped
        high_mask = feature <= high_thresh
        low_mask = feature > low_thresh
    else:
        high_mask = feature >= high_thresh
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
        
        # Store fold results
        fold_results = {
            'pf_high': [],
            'pf_low': [],
            'pf_best': [],
            'n_high': [],
            'n_low': []
        }
        
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
                    min_kept=min_kept
                )
                
                # Apply thresholds to test data
                test_result = apply_threshold_strategy(
                    test_feature,
                    test_target,
                    train_result['high_thresh'],
                    train_result['low_thresh'],
                    train_result['sign_flipped']
                )
                
                # Skip this fold if sample sizes are too small for reliable estimates
                # Require at least 10 samples for each strategy to avoid extreme profit factors
                min_test_samples = 10
                
                if test_result['n_high'] < min_test_samples or test_result['n_low'] < min_test_samples:
                    if verbose:
                        print(f"  Skipping fold {fold_idx} for '{feature_col}': insufficient samples (n_high={test_result['n_high']}, n_low={test_result['n_low']})")
                    continue
                
                # Store results
                fold_results['pf_high'].append(test_result['pf_high'])
                fold_results['pf_low'].append(test_result['pf_low'])
                fold_results['pf_best'].append(test_result['pf_best'])
                fold_results['n_high'].append(test_result['n_high'])
                fold_results['n_low'].append(test_result['n_low'])
            
            # Compute in-sample performance on full dataset
            min_kept_full = int(floor * len(df) + 0.5)
            full_result = optimize_threshold(
                df[feature_col],
                df[target_col],
                min_kept=min_kept_full
            )
            
            # Only include features that have at least one valid fold
            if len(fold_results['pf_best']) == 0:
                if verbose:
                    print(f"  Skipping '{feature_col}': no valid folds (all had insufficient samples)")
                continue
            
            # Aggregate results
            results.append({
                'feature': feature_col,
                'n_valid_folds': len(fold_results['pf_best']),
                'mean_pf_high': np.mean(fold_results['pf_high']),
                'std_pf_high': np.std(fold_results['pf_high']),
                'mean_pf_low': np.mean(fold_results['pf_low']),
                'std_pf_low': np.std(fold_results['pf_low']),
                'mean_pf_best': np.mean(fold_results['pf_best']),
                'std_pf_best': np.std(fold_results['pf_best']),
                'mean_n_high': np.mean(fold_results['n_high']),
                'mean_n_low': np.mean(fold_results['n_low']),
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
    
    # Sort by mean out-of-sample best profit factor
    results_df = results_df.sort_values('mean_pf_best', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nSuccessfully validated {len(results_df)} features")
        print(f"\nTop 5 features by mean out-of-sample profit factor:")
        top5 = results_df.head(5)[['feature', 'mean_pf_best', 'std_pf_best', 'in_sample_pf_best']]
        print(top5.to_string(index=False))
    
    return results_df


def walkforward_validate_features(
    df: pd.DataFrame,
    target_col: str,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    floor: float = 0.02,
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
    
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
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
                    min_kept=min_kept
                )
                
                # Apply thresholds to test data
                test_result = apply_threshold_strategy(
                    test_feature,
                    test_target,
                    train_result['high_thresh'],
                    train_result['low_thresh'],
                    train_result['sign_flipped']
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
                
                # Collect returns for cumulative plotting - track both strategies separately
                if has_valid_high:
                    step_results['high_returns'].append(test_result['high_returns'])
                if has_valid_low:
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
            
            # Aggregate results
            results.append({
                'feature': feature_col,
                'n_valid_steps': len(step_results['pf_best']),
                'mean_pf_high': np.mean(step_results['pf_high']),
                'std_pf_high': np.std(step_results['pf_high']),
                'mean_pf_low': np.mean(step_results['pf_low']),
                'std_pf_low': np.std(step_results['pf_low']),
                'mean_pf_best': np.mean(step_results['pf_best']),
                'std_pf_best': np.std(step_results['pf_best']),
                'mean_n_high': np.mean(step_results['n_high']),
                'mean_n_low': np.mean(step_results['n_low']),
                'high_returns_series': high_returns_series,  # Store for plotting
                'low_returns_series': low_returns_series     # Store for plotting
            })
            
            if verbose:
                print(f"  Valid steps: {len(step_results['pf_best'])}/{num_steps}, Mean PF: {np.mean(step_results['pf_best']):.2f}")
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to validate feature '{feature_col}': {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'='*80}")
        print("WALK-FORWARD VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nSuccessfully validated {len(results_df)} features")
    
    # Handle empty results
    if len(results_df) == 0:
        if verbose:
            print("\nNo features could be validated successfully.")
            print("Check that your datetime parameters match the DataFrame index timezone.")
        return results_df
    
    # Sort by mean out-of-sample best profit factor
    results_df = results_df.sort_values('mean_pf_best', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\nTop 5 features by mean out-of-sample profit factor:")
        top5 = results_df.head(5)[['feature', 'n_valid_steps', 'mean_pf_best', 'std_pf_best']]
        print(top5.to_string(index=False))
    
    # Generate cumulative return plots if requested
    figures = {}
    if plot_cumulative and len(results_df) > 0:
        if verbose:
            print(f"\nGenerating cumulative return plots for top {plot_top_n} features...")
        
        # Get top N features
        top_features = results_df.head(plot_top_n)['feature'].tolist()
        
        for feature_name in top_features:
            # Find this feature in results
            feature_data = [r for r in results if r['feature'] == feature_name]
            if not feature_data:
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
                cum_high = high_returns.cumsum()
                final_high = cum_high.iloc[-1] if len(cum_high) > 0 else 0
                color_high = '#2ecc71' if final_high >= 0 else '#e74c3c'
                
                ax1.plot(cum_high.index, cum_high.values, color=color_high, linewidth=2, label='Long')
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax1.set_ylabel('Cumulative Return')
                ax1.set_title(f"{feature_name} - Long Strategy\nFinal: {final_high:.4f}, Mean PF: {feature_data[0]['mean_pf_high']:.2f}, Trades: {int(feature_data[0]['mean_n_high'])}")
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No valid long trades', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f"{feature_name} - Long Strategy (No Data)")
            
            # Plot short strategy (low)
            if len(low_returns) > 0:
                cum_low = low_returns.cumsum()
                final_low = cum_low.iloc[-1] if len(cum_low) > 0 else 0
                color_low = '#2ecc71' if final_low >= 0 else '#e74c3c'
                
                ax2.plot(cum_low.index, cum_low.values, color=color_low, linewidth=2, label='Short')
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Cumulative Return')
                ax2.set_title(f"{feature_name} - Short Strategy\nFinal: {final_low:.4f}, Mean PF: {feature_data[0]['mean_pf_low']:.2f}, Trades: {int(feature_data[0]['mean_n_low'])}")
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

