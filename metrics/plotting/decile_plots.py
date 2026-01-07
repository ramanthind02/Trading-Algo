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


def plot_decile_analysis(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    save_path: Optional[str] = None
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
        
    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        Figure object and DataFrame with decile bin data
    """
    # Create a clean subset with no NaNs
    df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
    
    if df.empty:
        raise ValueError("No valid data after dropping NaNs")
    
    # Create bins/buckets based on feature quantiles
    try:
        df['bin'] = pd.qcut(df['feature'], n_bins, labels=False, duplicates='drop')
    except ValueError as e:
        warnings.warn(f"Could not create {n_bins} bins due to duplicate values. Using fewer bins.")
        # Try with fewer bins
        for n in range(n_bins - 1, 1, -1):
            try:
                df['bin'] = pd.qcut(df['feature'], n, labels=False, duplicates='drop')
                break
            except ValueError:
                continue
        else:
            raise ValueError("Could not create bins even with reduced number of bins")
    
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
        ax.bar(bin_stats.index + 1, bin_stats['mean'], color=bar_colors, alpha=0.8, edgecolor='black')
    else:  # line plot
        for i, val in enumerate(bin_stats['mean']):
            color = '#2ecc71' if val >= 0 else '#e74c3c'
            ax.plot(i+1, val, 'o', color=color, markersize=10)
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
