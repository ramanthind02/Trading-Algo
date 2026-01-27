"""
Decile and Bin Plotting Functions for Feature Analysis

This module contains plotting functions for analyzing feature-target relationships
through binned analysis (deciles, 2-bins, etc.).

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Tuple, Optional
import warnings


def _create_quantile_bins_robust(feature_data: pd.Series, n_bins: int) -> pd.Series:
    """
    Create quantile-based bins ensuring n_bins are created even with duplicate values.
    
    This is a robust version of pd.qcut that handles cases where many duplicate values
    cause bins to collapse. It uses a hybrid approach:
    1. Try qcut first
    2. If fewer bins created, use unique value boundaries to ensure n_bins
    3. If still not enough, use equal-width binning on the full range
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values to bin
    n_bins : int
        Number of bins to create
        
    Returns
    -------
    pd.Series
        Bin assignments for each sample
    """
    try:
        bins = pd.qcut(feature_data, n_bins, labels=False, duplicates='drop')
        n_created_bins = bins.nunique()
        
        # If we got fewer bins than requested, use a different approach
        if n_created_bins < n_bins:
            # Get unique values and create bins based on unique value boundaries
            unique_vals = feature_data.unique()
            n_unique = len(unique_vals)
            
            if n_unique <= n_bins:
                # Not enough unique values: assign each unique value to its own bin
                # Map unique values to bin indices
                sorted_unique = np.sort(unique_vals)
                bin_map = {val: idx for idx, val in enumerate(sorted_unique)}
                bins = feature_data.map(bin_map).astype(int)
            else:
                # Enough unique values: try to create bins using quantiles on unique values
                # This helps when many values are duplicates but we have enough unique values
                quantiles = np.linspace(0, 1, n_bins + 1)
                unique_quantiles = np.quantile(unique_vals, quantiles)
                # Remove duplicates from quantiles
                unique_quantiles = np.unique(unique_quantiles)
                
                if len(unique_quantiles) - 1 >= n_bins:
                    # Use quantile-based thresholds on unique values
                    bins = pd.cut(feature_data, bins=unique_quantiles, labels=False, include_lowest=True, duplicates='drop')
                    n_created_bins = bins.nunique()
                
                # If still not enough bins, use equal-width binning on the full range
                # This ensures we get the requested number of bins even if distribution is skewed
                if n_created_bins < n_bins:
                    bins = pd.cut(feature_data, bins=n_bins, labels=False, duplicates='drop', include_lowest=True)
                    n_created_bins = bins.nunique()
                    
                    # If equal-width still doesn't work (very few unique values), 
                    # split the data range into n_bins equal-width intervals
                    if n_created_bins < n_bins:
                        min_val = feature_data.min()
                        max_val = feature_data.max()
                        # Create n_bins equal-width intervals
                        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
                        # Ensure first and last edges include all values
                        bin_edges[0] = min_val - 1e-10
                        bin_edges[-1] = max_val + 1e-10
                        bins = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
    except (ValueError, TypeError):
        # Fallback: use equal-width bins on the full range
        min_val = feature_data.min()
        max_val = feature_data.max()
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_edges[0] = min_val - 1e-10
        bin_edges[-1] = max_val + 1e-10
        bins = pd.cut(feature_data, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
    
    return bins


def plot_decile_analysis(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    save_path: Optional[str] = None,
    selected_bin: Optional[int] = None,
    strategy: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a decile plot showing target behavior across feature value buckets.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values
    target_data : pd.Series
        Target values (must have same index as feature_data)
    feature_name : str
        Name of the feature for labeling
    n_bins : int, default=10
        Number of bins/buckets to create
    figsize : Tuple[int, int], default=(12, 8)
        Figure size for the plot
    plot_type : str, default="bar"
        Type of plot: "bar" or "line"
    save_path : Optional[str], default=None
        If provided, save the figure to this path
    selected_bin : Optional[int], default=None
        Bin index to highlight (0-indexed). If provided, this bin will be highlighted.
    strategy : Optional[str], default=None
        Strategy type ('long' or 'short') for title annotation if selected_bin is provided
        
    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        Figure object and DataFrame with decile bin data
    """
    # Create a clean subset with no NaNs
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if df.empty:
        raise ValueError("No valid data after dropping NaNs")
    
    # Create bins/buckets using robust quantile binning that handles duplicates
    df['bin'] = _create_quantile_bins_robust(df['feature'], n_bins)
    
    # Calculate statistics for each bin
    bin_stats = df.groupby('bin')['target'].agg(['mean', 'std', 'count'])
    bin_edges = df.groupby('bin')['feature'].agg(['min', 'max'])
    
    # Create decile data table
    decile_table = pd.DataFrame({
        'Decile': [f"{i+1}" for i in range(len(bin_stats))],
        'Feature_Min': bin_edges['min'].values,
        'Feature_Max': bin_edges['max'].values,
        'Mean_Target': bin_stats['mean'].values,
        'Std_Target': bin_stats['std'].values,
        'Count': bin_stats['count'].values.astype(int)
    })
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot the mean target value for each bin
    if plot_type == "bar":
        # Color based on value (green for positive, red for negative)
        bar_colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in bin_stats['mean']]
        
        # Highlight selected bin if provided
        edgecolors = []
        linewidths = []
        for bin_idx in bin_stats.index:
            if selected_bin is not None and bin_idx == selected_bin:
                edgecolors.append('#FFD700')  # Gold color for selected bin
                linewidths.append(3.0)  # Thicker edge
            else:
                edgecolors.append('black')
                linewidths.append(1.0)
        
        bars = ax.bar(bin_stats.index + 1, bin_stats['mean'], color=bar_colors, alpha=0.8, 
                     edgecolor=edgecolors, linewidth=linewidths)
        
        # Add annotation for selected bin
        if selected_bin is not None and selected_bin in bin_stats.index:
            strategy_str = f" [{strategy.upper()}]" if strategy else ""
            selected_mean = bin_stats['mean'].iloc[bin_stats.index.get_loc(selected_bin)]
            ax.text(selected_bin + 1, selected_mean,
                   f"SELECTED{strategy_str}", ha='center', va='bottom' if selected_mean >= 0 else 'top',
                   fontsize=9, fontweight='bold', color='#FFD700',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    else:  # line plot
        for i, (bin_idx, val) in enumerate(zip(bin_stats.index, bin_stats['mean'])):
            color = '#2ecc71' if val >= 0 else '#e74c3c'
            marker = 'D' if (selected_bin is not None and bin_idx == selected_bin) else 'o'
            markersize = 12 if (selected_bin is not None and bin_idx == selected_bin) else 10
            ax.plot(i+1, val, marker, color=color, markersize=markersize, 
                   markeredgecolor='#FFD700' if (selected_bin is not None and bin_idx == selected_bin) else 'black',
                   markeredgewidth=2 if (selected_bin is not None and bin_idx == selected_bin) else 1)
        ax.plot(bin_stats.index + 1, bin_stats['mean'], '-', color='#555555', alpha=0.5, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(f"{feature_name} Decile", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Target Return", fontsize=12, fontweight='bold')
    ax.set_title(f"{feature_name} Decile Plot", fontsize=14, fontweight='bold')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add second axis with feature value ranges
    ax2 = ax.twiny()
    tick_positions = ax.get_xticks()
    valid_positions = [pos for pos in tick_positions if 1 <= pos <= len(bin_stats)]
    ax2.set_xticks(valid_positions)
    
    # Create labels showing feature value ranges
    if len(bin_edges) > 0:
        tick_to_bin = {int(pos): int(pos-1) for pos in valid_positions if 1 <= pos <= len(bin_edges)}
        range_labels = []
        for pos in valid_positions:
            if int(pos) in tick_to_bin and tick_to_bin[int(pos)] < len(bin_edges):
                bin_idx = tick_to_bin[int(pos)]
                min_val = bin_edges['min'].iloc[bin_idx]
                max_val = bin_edges['max'].iloc[bin_idx]
                
                # Format values
                if abs(min_val) < 0.01 or abs(min_val) > 1000:
                    min_str = f"{min_val:.2e}"
                else:
                    min_str = f"{min_val:.3f}"
                
                if abs(max_val) < 0.01 or abs(max_val) > 1000:
                    max_str = f"{max_val:.2e}"
                else:
                    max_str = f"{max_val:.3f}"
                
                range_labels.append(f"[{min_str}, {max_str}]")
            else:
                range_labels.append("")
        
        ax2.set_xticklabels(range_labels, rotation=45, ha='left', fontsize=9)
        ax2.set_xlabel(f"{feature_name} Value Ranges", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decile plot saved to: {save_path}")
    
    return fig, decile_table


def plot_2bin_analysis(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a 2-bin plot: one bin for positive feature values, one for negative.
    
    This is useful for features that are expected to have symmetric behavior
    around zero, or to quickly assess if positive vs negative values have
    different target characteristics.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values
    target_data : pd.Series
        Target values (must have same index as feature_data)
    feature_name : str
        Name of the feature for labeling
    figsize : Tuple[int, int], default=(10, 6)
        Figure size for the plot
    save_path : Optional[str], default=None
        If provided, save the figure to this path
        
    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        Figure object and DataFrame with 2-bin data
    """
    # Create a clean subset with no NaNs
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if df.empty:
        raise ValueError("No valid data after dropping NaNs")
    
    # Create 2 bins: negative and positive
    df['bin'] = (df['feature'] >= 0).astype(int)  # 0 = negative, 1 = positive
    
    # Calculate statistics for each bin
    bin_stats = df.groupby('bin')['target'].agg(['mean', 'std', 'count'])
    bin_edges = df.groupby('bin')['feature'].agg(['min', 'max'])
    
    # Create bin data table
    bin_labels = ['Negative', 'Positive']
    bin_table = pd.DataFrame({
        'Bin': [bin_labels[i] for i in range(len(bin_stats))],
        'Feature_Min': bin_edges['min'].values,
        'Feature_Max': bin_edges['max'].values,
        'Mean_Target': bin_stats['mean'].values,
        'Std_Target': bin_stats['std'].values,
        'Count': bin_stats['count'].values.astype(int)
    })
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot bars
    x_positions = [0, 1]
    bar_colors = ['#e74c3c', '#2ecc71']  # Red for negative, green for positive
    bars = ax.bar(x_positions, bin_stats['mean'].values, color=bar_colors, alpha=0.8, edgecolor='black', width=0.6)
    
    # Set labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Target Return", fontsize=12, fontweight='bold')
    ax.set_title(f"{feature_name} 2-Bin Analysis (Negative vs Positive)", fontsize=14, fontweight='bold')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, bin_stats['mean'].values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, bin_stats['count'].values)):
        ax.text(bar.get_x() + bar.get_width()/2., 0,
                f'n={count}',
                ha='center', va='center',
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2-bin plot saved to: {save_path}")
    
    return fig, bin_table


def plot_uniform_binning(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a uniform binning plot showing target behavior across equal-width feature value buckets.
    
    Unlike decile plots which use equal-frequency bins (pd.qcut), uniform binning uses
    equal-width bins (pd.cut), where each bin spans the same range of feature values.
    This is useful for understanding behavior at specific feature value ranges.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature values
    target_data : pd.Series
        Target values (must have same index as feature_data)
    feature_name : str
        Name of the feature for labeling
    n_bins : int, default=10
        Number of bins/buckets to create
    figsize : Tuple[int, int], default=(12, 8)
        Figure size for the plot
    plot_type : str, default="bar"
        Type of plot: "bar" or "line"
    save_path : Optional[str], default=None
        If provided, save the figure to this path
        
    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        Figure object and DataFrame with uniform bin data
    """
    # Create a clean subset with no NaNs
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if df.empty:
        raise ValueError("No valid data after dropping NaNs")
    
    # Create bins/buckets based on equal-width intervals (uniform binning)
    try:
        df['bin'] = pd.cut(df['feature'], bins=n_bins, labels=False, duplicates='drop', include_lowest=True)
    except ValueError as e:
        warnings.warn(f"Could not create {n_bins} bins due to duplicate values. Using fewer bins.")
        # Try with fewer bins
        for n in range(n_bins - 1, 1, -1):
            try:
                df['bin'] = pd.cut(df['feature'], bins=n, labels=False, duplicates='drop', include_lowest=True)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Could not create bins even with reduced number of bins")
    
    # Remove NaN bins (can occur if all values fall outside the range)
    df = df.dropna(subset=['bin'])
    
    if df.empty or df['bin'].nunique() == 0:
        raise ValueError("No valid bins created after uniform binning")
    
    # Calculate statistics for each bin
    bin_stats = df.groupby('bin')['target'].agg(['mean', 'std', 'count'])
    bin_edges = df.groupby('bin')['feature'].agg(['min', 'max'])
    
    # Create uniform bin data table
    uniform_table = pd.DataFrame({
        'Bin': [f"{i+1}" for i in range(len(bin_stats))],
        'Feature_Min': bin_edges['min'].values,
        'Feature_Max': bin_edges['max'].values,
        'Mean_Target': bin_stats['mean'].values,
        'Std_Target': bin_stats['std'].values,
        'Count': bin_stats['count'].values.astype(int)
    })
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot the mean target value for each bin
    if plot_type == "bar":
        # Color based on value (green for positive, red for negative)
        bar_colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in bin_stats['mean']]
        ax.bar(bin_stats.index + 1, bin_stats['mean'], color=bar_colors, alpha=0.8, edgecolor='black')
    else:  # line plot
        for i, val in enumerate(bin_stats['mean']):
            color = '#2ecc71' if val >= 0 else '#e74c3c'
            ax.plot(i+1, val, 'o', color=color, markersize=10)
        ax.plot(bin_stats.index + 1, bin_stats['mean'], '-', color='#555555', alpha=0.5, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(f"{feature_name} Uniform Bin", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Target Return", fontsize=12, fontweight='bold')
    ax.set_title(f"{feature_name} Uniform Binning Plot (Equal-Width Bins)", fontsize=14, fontweight='bold')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add second axis with feature value ranges
    ax2 = ax.twiny()
    tick_positions = ax.get_xticks()
    valid_positions = [pos for pos in tick_positions if 1 <= pos <= len(bin_stats)]
    ax2.set_xticks(valid_positions)
    
    # Create labels showing feature value ranges
    if len(bin_edges) > 0:
        tick_to_bin = {int(pos): int(pos-1) for pos in valid_positions if 1 <= pos <= len(bin_edges)}
        range_labels = []
        for pos in valid_positions:
            if int(pos) in tick_to_bin and tick_to_bin[int(pos)] < len(bin_edges):
                bin_idx = tick_to_bin[int(pos)]
                min_val = bin_edges['min'].iloc[bin_idx]
                max_val = bin_edges['max'].iloc[bin_idx]
                
                # Format values
                if abs(min_val) < 0.01 or abs(min_val) > 1000:
                    min_str = f"{min_val:.2e}"
                else:
                    min_str = f"{min_val:.3f}"
                
                if abs(max_val) < 0.01 or abs(max_val) > 1000:
                    max_str = f"{max_val:.2e}"
                else:
                    max_str = f"{max_val:.3f}"
                
                range_labels.append(f"[{min_str}, {max_str}]")
            else:
                range_labels.append("")
        
        ax2.set_xticklabels(range_labels, rotation=45, ha='left', fontsize=9)
        ax2.set_xlabel(f"{feature_name} Value Ranges", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uniform binning plot saved to: {save_path}")
    
    return fig, uniform_table
