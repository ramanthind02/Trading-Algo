"""
Rolling Decile Analysis with Whisker Plots

This module provides rolling window analysis of feature-target relationships,
showing how decile behavior changes over time using whisker plots.

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings

# Import the abstract walkforward utility
from utils.evaluation.walkforward import apply_function_to_rolling_windows


def compute_decile_stats(
    feature_data: pd.Series,
    target_data: pd.Series,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute decile statistics for a single window.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values
    target_data : pd.Series
        Target values
    n_bins : int, default=10
        Number of bins
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: decile, mean_target, std_target, count, feature_min, feature_max
    """
    # Create clean subset
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if len(df) < n_bins * 2:  # Need at least 2 samples per bin
        return pd.DataFrame()
    
    # Create bins
    try:
        df['bin'] = pd.qcut(df['feature'], n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Try with fewer bins
        for n in range(n_bins - 1, 1, -1):
            try:
                df['bin'] = pd.qcut(df['feature'], n, labels=False, duplicates='drop')
                break
            except ValueError:
                continue
        else:
            return pd.DataFrame()
    
    # Calculate statistics for each bin including feature bounds
    bin_stats = df.groupby('bin').agg({
        'target': ['mean', 'std', 'count'],
        'feature': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    bin_stats.columns = ['bin', 'mean_target', 'std_target', 'count', 'feature_min', 'feature_max']
    bin_stats['decile'] = bin_stats['bin'] + 1
    bin_stats = bin_stats[['decile', 'mean_target', 'std_target', 'count', 'feature_min', 'feature_max']]
    
    return bin_stats


def plot_rolling_decile_whiskers(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    window_size: int = 252,
    step_size: int = 63,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    remove_outliers: bool = True,
    outlier_threshold: float = 1.5
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a rolling decile whisker plot showing variance over time.
    
    This plot shows how the mean target return for each decile changes
    over rolling windows, with whiskers showing the standard deviation.
    This helps identify if the feature-target relationship is stable over time.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values (must have DatetimeIndex)
    target_data : pd.Series
        Target values (must have DatetimeIndex)
    feature_name : str
        Name of the feature for labeling
    window_size : int, default=252
        Number of days in each rolling window (252 = ~1 year)
    step_size : int, default=63
        Number of days to roll forward (~3 months for quarterly analysis)
    n_bins : int, default=10
        Number of decile bins
    figsize : Tuple[int, int], default=(14, 8)
        Figure size
    save_path : Optional[str], default=None
        If provided, save the figure to this path
        
    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        Figure object and DataFrame with all rolling window results
    """
    # Combine into DataFrame
    df = pd.DataFrame({'feature': feature_data, 'target': target_data})
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex for rolling analysis")
    
    # Define function to compute deciles for each window
    def compute_window_deciles(window_data, window_info):
        decile_stats = compute_decile_stats(
            window_data['feature'],
            window_data['target'],
            n_bins=n_bins
        )
        
        # Print bin bounds for this window
        if len(decile_stats) > 0:
            window_num = window_info.get('window_num', '?')
            start_date = window_info.get('start', 'N/A')
            end_date = window_info.get('end', 'N/A')
            print(f"\n  Window {window_num}: {start_date} to {end_date}")
            print(f"  {'Decile':<8} {'Feature Range':<30} {'Mean Target':<15} {'Count':<8}")
            print(f"  {'-'*8} {'-'*30} {'-'*15} {'-'*8}")
            for _, row in decile_stats.iterrows():
                decile = int(row['decile'])
                feat_range = f"[{row['feature_min']:.4f}, {row['feature_max']:.4f}]"
                mean_target = f"{row['mean_target']:.6f}"
                count = int(row['count'])
                print(f"  {decile:<8} {feat_range:<30} {mean_target:<15} {count:<8}")
        
        return {'decile_stats': decile_stats}
    
    # Apply to rolling windows
    print(f"Computing rolling deciles for {feature_name}...")
    print(f"Window size: {window_size} days, Step size: {step_size} days")
    results = apply_function_to_rolling_windows(
        df,
        func=compute_window_deciles,
        window_size=window_size,
        step_size=step_size,
        min_samples=n_bins * 10,  # Need enough samples for meaningful deciles
        verbose=False  # We handle our own verbose output above
    )
    
    if len(results) == 0:
        raise ValueError("No valid rolling windows generated")
    
    print(f"\n{'='*70}")
    print(f"Generated {len(results)} rolling windows total")
    print(f"{'='*70}")
    
    # Aggregate results across all windows for each decile
    decile_aggregates = []
    for decile_num in range(1, n_bins + 1):
        decile_means = []
        decile_stds = []
        
        for result in results:
            decile_stats = result['decile_stats']
            if len(decile_stats) > 0:
                decile_row = decile_stats[decile_stats['decile'] == decile_num]
                if len(decile_row) > 0:
                    decile_means.append(decile_row['mean_target'].values[0])
                    decile_stds.append(decile_row['std_target'].values[0])
        
        if len(decile_means) > 0:
            decile_aggregates.append({
                'decile': decile_num,
                'mean_of_means': np.mean(decile_means),
                'std_of_means': np.std(decile_means),
                'mean_of_stds': np.mean(decile_stds),
                'n_windows': len(decile_means),
                'min_mean': np.min(decile_means),
                'max_mean': np.max(decile_means),
                'q25_mean': np.percentile(decile_means, 25),
                'q75_mean': np.percentile(decile_means, 75)
            })
    
    # Prepare data for strip plot - collect all window results
    all_window_data = []
    for result in results:
        decile_stats = result['decile_stats']
        for _, row in decile_stats.iterrows():
            all_window_data.append({
                'decile': row['decile'],
                'mean_target': row['mean_target']
            })
    
    df_all = pd.DataFrame(all_window_data)
    
    # Remove outliers if requested (using IQR method)
    if remove_outliers:
        df_filtered = []
        n_outliers_removed = 0
        
        for decile in range(1, n_bins + 1):
            decile_subset = df_all[df_all['decile'] == decile].copy()
            
            if len(decile_subset) > 4:  # Need at least 5 points for IQR
                # Calculate IQR
                q1 = decile_subset['mean_target'].quantile(0.25)
                q3 = decile_subset['mean_target'].quantile(0.75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                
                # Filter outliers
                before_count = len(decile_subset)
                decile_subset = decile_subset[
                    (decile_subset['mean_target'] >= lower_bound) &
                    (decile_subset['mean_target'] <= upper_bound)
                ]
                n_outliers_removed += (before_count - len(decile_subset))
            
            df_filtered.append(decile_subset)
        
        df_all = pd.concat(df_filtered, ignore_index=True)
        
        if n_outliers_removed > 0:
            print(f"  Removed {n_outliers_removed} outlier windows (IQR method, threshold={outlier_threshold})")
    
    # Create strip plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each decile's distribution as dots (each dot = one rolling window)
    for decile in range(1, n_bins + 1):
        decile_data = df_all[df_all['decile'] == decile]['mean_target'].values
        
        if len(decile_data) == 0:
            continue
        
        # Add jitter to x-coordinates for visibility
        np.random.seed(42 + decile)  # Reproducible jitter
        x_jitter = np.random.normal(decile, 0.08, size=len(decile_data))
        
        # Color based on median value
        median_val = np.median(decile_data)
        color = '#2ecc71' if median_val >= 0 else '#e74c3c'
        
        # Plot dots (each dot = one rolling window)
        ax.scatter(x_jitter, decile_data, alpha=0.5, s=40, color=color,
                  edgecolors='black', linewidth=0.5, zorder=5)
        
        # Add median line (thick black horizontal line)
        ax.plot([decile - 0.35, decile + 0.35], [median_val, median_val],
                color='black', linewidth=3, zorder=10)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Target Return (per window)', fontsize=12, fontweight='bold')
    
    title = f'{feature_name} Rolling Decile Strip Plot\n'
    title += f'Window: {window_size} days, Step: {step_size} days, N_windows: {len(results)}\n'
    title += f'Each dot = one rolling window | Black line = median'
    if remove_outliers:
        title += f' | Outliers removed (IQR {outlier_threshold}x)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(range(1, n_bins + 1))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rolling decile plot saved to: {save_path}")
    
    # Compute summary statistics for return value
    summary_stats = df_all.groupby('decile')['mean_target'].agg([
        ('median', 'median'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('p25', lambda x: np.percentile(x, 25)),
        ('p75', lambda x: np.percentile(x, 75)),
        ('n_windows', 'count')
    ]).reset_index()
    
    return fig, summary_stats


def plot_rolling_decile_heatmap(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    window_size: int = 252,
    step_size: int = 63,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, List[pd.DataFrame]]:
    """
    Create a heatmap showing how decile returns evolve over rolling windows.
    
    This provides a time-series view of decile performance, showing if
    certain deciles become more/less predictive over time.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values (must have DatetimeIndex)
    target_data : pd.Series
        Target values (must have DatetimeIndex)
    feature_name : str
        Name of the feature for labeling
    window_size : int, default=252
        Number of days in each rolling window
    step_size : int, default=63
        Number of days to roll forward
    n_bins : int, default=10
        Number of decile bins
    figsize : Tuple[int, int], default=(14, 10)
        Figure size
    save_path : Optional[str], default=None
        If provided, save the figure to this path
        
    Returns
    -------
    Tuple[plt.Figure, List[pd.DataFrame]]
        Figure object and list of decile stats for each window
    """
    # Combine into DataFrame
    df = pd.DataFrame({'feature': feature_data, 'target': target_data})
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex for rolling analysis")
    
    # Define function to compute deciles for each window
    def compute_window_deciles(window_data, window_info):
        decile_stats = compute_decile_stats(
            window_data['feature'],
            window_data['target'],
            n_bins=n_bins
        )
        return {'decile_stats': decile_stats}
    
    # Apply to rolling windows
    print(f"Computing rolling deciles for heatmap...")
    results = apply_function_to_rolling_windows(
        df,
        func=compute_window_deciles,
        window_size=window_size,
        step_size=step_size,
        min_samples=n_bins * 10,
        verbose=True
    )
    
    if len(results) == 0:
        raise ValueError("No valid rolling windows generated")
    
    # Create matrix: rows = windows, columns = deciles
    n_windows = len(results)
    heatmap_data = np.full((n_windows, n_bins), np.nan)
    window_labels = []
    
    for i, result in enumerate(results):
        decile_stats = result['decile_stats']
        window_labels.append(result['start_date'].strftime('%Y-%m'))
        
        for _, row in decile_stats.iterrows():
            decile_idx = int(row['decile']) - 1
            heatmap_data[i, decile_idx] = row['mean_target']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(heatmap_data))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels([str(i+1) for i in range(n_bins)])
    ax.set_xlabel('Decile', fontsize=12, fontweight='bold')
    
    # Show every Nth window label to avoid crowding
    label_step = max(1, n_windows // 20)
    ax.set_yticks(np.arange(0, n_windows, label_step))
    ax.set_yticklabels([window_labels[i] for i in range(0, n_windows, label_step)])
    ax.set_ylabel('Rolling Window (Start Date)', fontsize=12, fontweight='bold')
    
    # Title
    ax.set_title(
        f'{feature_name} Rolling Decile Heatmap\n'
        f'Window: {window_size} days, Step: {step_size} days',
        fontsize=14, fontweight='bold'
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Target Return', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rolling decile heatmap saved to: {save_path}")
    
    # Return all decile stats for further analysis
    all_stats = [r['decile_stats'] for r in results]
    
    return fig, all_stats
