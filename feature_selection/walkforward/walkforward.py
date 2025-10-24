"""
Walk-Forward Analysis with Flexible Model Interface

This module implements walk-forward analysis using a flexible BaseModel interface.
Any model following the sklearn-style fit/predict pattern can be used.

Author: Trading Research Team
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from datetime import datetime

from utils.walkforward import apply_function_to_walkforward
from feature_selection.base_model import BaseModel


def walkforward_analysis(
    feature_data: pd.Series,
    target_data: pd.Series,
    model: BaseModel,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    verbose: bool = True,
    raw_return: pd.Series = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Perform walk-forward analysis using any BaseModel.
    
    For each step:
    1. Train model on historical window
    2. Generate predictions for both long and short strategies
    3. Track performance and selected bins/rules
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values with DatetimeIndex
    target_data : pd.Series
        Target values with DatetimeIndex (can be ATR-normalized)
    model : BaseModel
        Model instance to use for training/prediction (e.g., QuantileBinningModel, DecisionTreeBinningModel)
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int, default=252
        Number of days for test period (~1 year)
    num_steps : int, default=10
        Number of walk-forward steps
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
        
    Examples
    --------
    >>> from feature_selection.base_model import QuantileBinningModel
    >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
    >>> results_df, step_info = walkforward_analysis(
    ...     feature_data=feature_series,
    ...     target_data=target_series,
    ...     model=model,
    ...     train_start=datetime(2000, 1, 1),
    ...     train_end=datetime(2010, 1, 1),
    ...     test_step=252,
    ...     num_steps=15
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
        
        try:
            # Fit model on training data
            model.fit(train_data['feature'], train_data['target'])
            
            # Get bin statistics
            bin_stats = model.get_bin_stats()
            
            if verbose:
                model_name = model.__class__.__name__
                print(f"  Model: {model_name}")
                print(f"  Selected LONG bin: {model.best_long_bin_} (mean: {bin_stats[model.best_long_bin_]['mean_return']:.6f})")
                print(f"  Selected SHORT bin: {model.best_short_bin_} (mean: {bin_stats[model.best_short_bin_]['mean_return']:.6f})")
                
                print(f"  Bin statistics:")
                for bin_idx, stats in bin_stats.items():
                    long_marker = " <-- LONG" if bin_idx == model.best_long_bin_ else ""
                    short_marker = " <-- SHORT" if bin_idx == model.best_short_bin_ else ""
                    marker = long_marker + short_marker
                    sortino_str = f"sortino={stats['sortino_metric']:.2f}, " if 'sortino_metric' in stats else ""
                    print(f"    Bin {bin_idx}: {sortino_str}mean={stats['mean_return']:.6f}, "
                          f"downside_std={stats['downside_std']:.6f}, count={stats['count']}, "
                          f"range=[{stats['feature_min']:.3f}, {stats['feature_max']:.3f}]{marker}")
            
            # Generate predictions for LONG strategy
            test_signal_long = model.predict(test_data['feature'], strategy='long')
            
            # Generate predictions for SHORT strategy
            test_signal_short = model.predict(test_data['feature'], strategy='short')
            
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
                'thresholds': model.thresholds_,
                'best_long_bin': model.best_long_bin_,
                'best_short_bin': model.best_short_bin_,
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
    Plot walk-forward results for both LONG and SHORT strategies.
    
    Creates two separate figures:
    Figure 1: Bin selection, mean return, Sortino ratio, and number of trades (Long vs Short)
    Figure 2: Cumulative equity curves (Long vs Short comparison)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from walkforward_analysis (must include both long and short columns)
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
    from feature_selection.base_model import QuantileBinningModel, DecisionTreeBinningModel
    
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
    
    # Example 1: Quantile binning
    print("="*70)
    print("EXAMPLE 1: Quantile Binning Model")
    print("="*70)
    quantile_model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
    results_df_q, step_info_q = walkforward_analysis(
        feature_data=feature_data,
        target_data=target_data,
        model=quantile_model,
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2010, 1, 1),
        test_step=252,
        num_steps=15,
        verbose=True
    )
    
    # Example 2: Decision tree binning
    print("\n" + "="*70)
    print("EXAMPLE 2: Decision Tree Binning Model")
    print("="*70)
    tree_model = DecisionTreeBinningModel(n_bins=3, selection_metric='sortino', min_samples_leaf_pct=0.05)
    results_df_t, step_info_t = walkforward_analysis(
        feature_data=feature_data,
        target_data=target_data,
        model=tree_model,
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2010, 1, 1),
        test_step=252,
        num_steps=15,
        verbose=True
    )
    
    # Plot results
    fig_metrics_q, fig_equity_q = plot_walkforward_results(results_df_q, step_info_q, f"{feature_name} (Quantile)")
    fig_metrics_t, fig_equity_t = plot_walkforward_results(results_df_t, step_info_t, f"{feature_name} (Tree)")
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print("\nQuantile Binning Model:")
    print(f"  LONG - Avg return: {results_df_q['test_mean_return_long'].mean():.6f}, Avg Sortino: {results_df_q['test_sortino_long'].mean():.2f}")
    print(f"  SHORT - Avg return: {results_df_q['test_mean_return_short'].mean():.6f}, Avg Sortino: {results_df_q['test_sortino_short'].mean():.2f}")
    
    print("\nDecision Tree Binning Model:")
    print(f"  LONG - Avg return: {results_df_t['test_mean_return_long'].mean():.6f}, Avg Sortino: {results_df_t['test_sortino_long'].mean():.2f}")
    print(f"  SHORT - Avg return: {results_df_t['test_mean_return_short'].mean():.6f}, Avg Sortino: {results_df_t['test_sortino_short'].mean():.2f}")
