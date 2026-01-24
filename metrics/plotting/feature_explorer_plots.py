"""
Pure plotting functions extracted from FeatureExplorer.

This module provides pure functions for visualizing feature analysis results.
All functions are stateless and follow functional programming principles.

Author: Trading Research Team
Date: 2025-01-XX
"""

from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrics.plotting.decile_plots import (
    plot_decile_analysis,
    plot_2bin_analysis,
    plot_uniform_binning,
)
from metrics.plotting.distribution import (
    plot_feature_distribution,
    plot_feature_timeseries,
)


def plot_all_feature_deciles(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = 'log_return',
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot decile analysis for all features.
    
    Pure function version of FeatureExplorer.plot_all_deciles().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    targets_df : pd.DataFrame
        DataFrame containing target columns
    feature_names : List[str]
        List of feature names to plot
    target_col : str, default='log_return'
        Target column to use
    n_bins : int, default=10
        Number of bins for decile analysis
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    plot_type : str, default="bar"
        Type of plot: "bar" or "line"
    save_dir : Optional[str], default=None
        Directory to save plots
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping feature names to figure objects
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Plotting decile analysis for {len(feature_names)} features")
        print(f"Target: {target_col}")
        print(f"{'='*70}")
    
    figures = {}
    for i, feature_name in enumerate(feature_names, 1):
        if verbose:
            print(f"\n[{i}/{len(feature_names)}] {feature_name}")
        
        try:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feature_name}_deciles.png")
            
            feature_data = features_df[feature_name]
            target_data = targets_df[target_col]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                if verbose:
                    print(f"  ✗ Skipped (non-numeric)")
                continue
            
            fig, _ = plot_decile_analysis(
                feature_data=feature_data,
                target_data=target_data,
                feature_name=feature_name,
                n_bins=n_bins,
                figsize=figsize,
                plot_type=plot_type,
                save_path=save_path
            )
            figures[feature_name] = fig
            if verbose:
                print(f"  ✓ Complete")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(figures)}/{len(feature_names)} features")
        print(f"{'='*70}")
    return figures


def plot_feature_2bin(
    feature_data: pd.Series,
    target_data: pd.Series,
    feature_name: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2-bin analysis (positive vs negative feature values).
    
    Pure function version of FeatureExplorer.plot_2bin().
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature data series
    target_data : pd.Series
        Target data series
    feature_name : str
        Name of the feature
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if not pd.api.types.is_numeric_dtype(feature_data):
        raise TypeError(f"Feature '{feature_name}' is non-numeric")
    
    fig, _ = plot_2bin_analysis(
        feature_data=feature_data,
        target_data=target_data,
        feature_name=feature_name,
        figsize=figsize,
        save_path=save_path
    )
    return fig


def plot_all_feature_uniform_bins(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = 'log_return',
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot uniform binning analysis for all features.
    
    Pure function version of FeatureExplorer.plot_all_uniform_bins().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    targets_df : pd.DataFrame
        DataFrame containing target columns
    feature_names : List[str]
        List of feature names to plot
    target_col : str, default='log_return'
        Target column to use
    n_bins : int, default=10
        Number of bins to create
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    plot_type : str, default="bar"
        Type of plot: "bar" or "line"
    save_dir : Optional[str], default=None
        Directory to save plots
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping feature names to figure objects
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Plotting uniform binning analysis for {len(feature_names)} features")
        print(f"Target: {target_col}")
        print(f"{'='*70}")
    
    figures = {}
    for i, feature_name in enumerate(feature_names, 1):
        if verbose:
            print(f"\n[{i}/{len(feature_names)}] {feature_name}")
        
        try:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feature_name}_uniform_bins.png")
            
            feature_data = features_df[feature_name]
            target_data = targets_df[target_col]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                if verbose:
                    print(f"  ✗ Skipped (non-numeric)")
                continue
            
            fig, _ = plot_uniform_binning(
                feature_data=feature_data,
                target_data=target_data,
                feature_name=feature_name,
                n_bins=n_bins,
                figsize=figsize,
                plot_type=plot_type,
                save_path=save_path
            )
            figures[feature_name] = fig
            if verbose:
                print(f"  ✓ Complete")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(figures)}/{len(feature_names)} features")
        print(f"{'='*70}")
    return figures


def plot_feature_target_correlations(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = 'log_return',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    annot: bool = True,
    fmt: str = '.2f',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of feature-target correlations using Spearman rank correlation.
    
    Pure function version of FeatureExplorer.plot_feature_target_correlations().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    targets_df : pd.DataFrame
        DataFrame containing target columns
    feature_names : List[str]
        List of feature names to include
    target_col : str, default='log_return'
        Target column to use
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    cmap : str, default='RdBu_r'
        Colormap for heatmap
    annot : bool, default=True
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format string for annotations
    save_path : Optional[str], default=None
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if target_col not in targets_df.columns:
        raise ValueError(f"Target '{target_col}' not found")
    
    target_data = targets_df[target_col]
    
    # Compute correlations using list comprehension
    correlations = {
        name: (
            features_df[name].corr(target_data, method='spearman')
            if pd.api.types.is_numeric_dtype(features_df[name])
            else np.nan
        )
        for name in feature_names
    }
    
    corr_series = pd.Series(correlations, name=f'spearman_correlation')
    corr_series = corr_series.dropna()
    
    if corr_series.empty:
        raise ValueError("No valid correlations computed (all features are non-numeric)")
    
    corr_df = pd.DataFrame({target_col: corr_series})
    corr_df = corr_df.reindex(corr_series.abs().sort_values(ascending=False).index)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Spearman Correlation'},
        ax=ax
    )
    ax.set_title(
        f'Feature-Target Correlations (Spearman)\n'
        f'Target: {target_col}\n'
        f'Captures monotonic relationships',
        fontsize=12,
        pad=20
    )
    ax.set_xlabel('')
    ax.set_ylabel('Features (sorted by |correlation|)')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_correlation_matrix(
    features_df: pd.DataFrame,
    feature_names: List[str],
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    annot: bool = False,
    fmt: str = '.2f',
    save_path: Optional[str] = None,
    mask_diagonal: bool = True
) -> plt.Figure:
    """
    Plot heatmap of intra-feature correlations using Pearson correlation.
    
    Pure function version of FeatureExplorer.plot_feature_correlations().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    feature_names : List[str]
        List of feature names to include
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    cmap : str, default='coolwarm'
        Colormap for heatmap
    annot : bool, default=False
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format string for annotations
    save_path : Optional[str], default=None
        Path to save the figure
    mask_diagonal : bool, default=True
        Whether to mask the diagonal (upper triangle)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Filter numeric features using list comprehension
    numeric_features = [
        name for name in feature_names
        if pd.api.types.is_numeric_dtype(features_df[name])
    ]
    
    if len(numeric_features) < 2:
        raise ValueError("Need at least 2 numeric features for correlation matrix")
    
    corr_matrix = features_df[numeric_features].corr(method='pearson')
    mask = None
    if mask_diagonal:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    ax.set_title(
        f'Intra-Feature Correlations (Pearson)\n'
        f'{len(numeric_features)} features\n'
        f'Measures linear relationships',
        fontsize=12,
        pad=20
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_feature_distributions(
    features_df: pd.DataFrame,
    feature_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    show_stats: bool = True,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot distribution analysis for all features.
    
    Pure function version of FeatureExplorer.plot_distributions().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    feature_names : List[str]
        List of feature names to plot
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    bins : int, default=50
        Number of bins for histogram
    show_stats : bool, default=True
        Whether to show statistics on plot
    save_dir : Optional[str], default=None
        Directory to save plots
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping feature names to figure objects
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Plotting distributions for {len(feature_names)} features")
        print(f"{'='*70}")
    
    figures = {}
    for i, feature_name in enumerate(feature_names, 1):
        if verbose:
            print(f"\n[{i}/{len(feature_names)}] {feature_name}")
        
        try:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feature_name}_distribution.png")
            
            feature_data = features_df[feature_name]
            fig = plot_feature_distribution(
                feature_data=feature_data,
                feature_name=feature_name,
                figsize=figsize,
                bins=bins,
                show_stats=show_stats,
                save_path=save_path
            )
            figures[feature_name] = fig
            if verbose:
                print(f"  ✓ Complete")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(figures)}/{len(feature_names)} features")
        print(f"{'='*70}")
    return figures


def plot_all_feature_timeseries(
    features_df: pd.DataFrame,
    feature_names: List[str],
    figsize: Tuple[int, int] = (14, 6),
    show_rolling_mean: bool = True,
    rolling_window: int = 20,
    show_rolling_std: bool = True,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot time series analysis for all features.
    
    Pure function version of FeatureExplorer.plot_timeseries().
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    feature_names : List[str]
        List of feature names to plot
    figsize : Tuple[int, int], default=(14, 6)
        Figure size
    show_rolling_mean : bool, default=True
        Whether to show rolling mean
    rolling_window : int, default=20
        Window size for rolling statistics
    show_rolling_std : bool, default=True
        Whether to show rolling standard deviation
    save_dir : Optional[str], default=None
        Directory to save plots
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping feature names to figure objects
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Plotting time series for {len(feature_names)} features")
        print(f"{'='*70}")
    
    figures = {}
    for i, feature_name in enumerate(feature_names, 1):
        if verbose:
            print(f"\n[{i}/{len(feature_names)}] {feature_name}")
        
        try:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feature_name}_timeseries.png")
            
            feature_data = features_df[feature_name]
            fig = plot_feature_timeseries(
                feature_data=feature_data,
                feature_name=feature_name,
                figsize=figsize,
                show_rolling_mean=show_rolling_mean,
                rolling_window=rolling_window,
                show_rolling_std=show_rolling_std,
                save_path=save_path
            )
            figures[feature_name] = fig
            if verbose:
                print(f"  ✓ Complete")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(figures)}/{len(feature_names)} features")
        print(f"{'='*70}")
    return figures


def plot_feature_signal_cumsum(
    gated_returns: pd.Series,
    feature_name: str,
    strategy: str = 'long',
    metric_value: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot cumulative sum of target returns gated by signals.
    
    Pure function for plotting signal-gated cumulative returns.
    Signal generation and gated returns computation should be handled separately.
    
    Parameters
    ----------
    gated_returns : pd.Series
        Gated returns series (target * signals, already computed)
    feature_name : str
        Name of the feature
    strategy : str, default='long'
        Strategy label for title
    metric_value : Optional[float], default=None
        Optional metric value to display in title
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # CRITICAL: Sort by datetime index before calculating cumulative sum
    # This ensures chronological order for proper cumulative calculation
    if isinstance(gated_returns.index, pd.DatetimeIndex):
        gated_returns = gated_returns.sort_index()
    elif hasattr(gated_returns.index, 'sort_values'):
        # If index has sort_values method, try to sort
        try:
            gated_returns = gated_returns.sort_index()
        except Exception:
            # If sorting fails, proceed with original order
            pass
    
    cum_returns = gated_returns.cumsum()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cum_returns.index, cum_returns.values, label='Cumulative Return')
    ax.axhline(0.0, color='black', linewidth=1, alpha=0.5)
    
    metric_str = f"{metric_value:.4f}" if metric_value is not None and not np.isnan(metric_value) else 'NA'
    strategy_label = strategy.replace('-', ' ').title().replace(' ', '-')
    ax.set_title(
        f"{feature_name} | CumSum(target * signal) [{strategy_label}]\n"
        f"Final: {cum_returns.iloc[-1]:.4f} | Metric: {metric_str}"
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Sum')
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
    
    return fig


def combine_decile_plots(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = 'log_return',
    n_bins: int = 10,
    n_cols: int = 4,
    figsize_per_plot: Tuple[int, int] = (6, 4),
    plot_type: str = "bar",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Combine multiple decile plots into a single figure with subplots.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    targets_df : pd.DataFrame
        DataFrame containing target columns
    feature_names : List[str]
        List of feature names to plot
    target_col : str, default='log_return'
        Target column to use
    n_bins : int, default=10
        Number of bins for decile analysis
    n_cols : int, default=4
        Number of columns in the grid
    figsize_per_plot : Tuple[int, int], default=(6, 4)
        Size of each subplot
    plot_type : str, default="bar"
        Type of plot: "bar" or "line"
    save_path : Optional[str], default=None
        Path to save the combined figure
        
    Returns
    -------
    plt.Figure
        Combined figure with all decile plots
    """
    if not feature_names:
        raise ValueError("No features to plot")
    
    n_plots = len(feature_names)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    # Ensure axes is always a flat array for indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        # When n_rows == 1, axes is a 1D array, ensure it's flat
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]
        
        try:
            feature_data = features_df[feature_name]
            target_data = targets_df[target_col]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                ax.text(0.5, 0.5, f'{feature_name}\n(non-numeric)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature_name, fontsize=9)
                continue
            
            # Create decile plot data
            df = pd.DataFrame({'feature': feature_data, 'target': target_data}).dropna()
            
            if df.empty:
                ax.text(0.5, 0.5, f'{feature_name}\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature_name, fontsize=9)
                continue
            
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
                    ax.text(0.5, 0.5, f'{feature_name}\n(cannot bin)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(feature_name, fontsize=9)
                    continue
            
            # Calculate statistics
            bin_stats = df.groupby('bin')['target'].agg(['mean', 'std', 'count'])
            
            # Plot
            if plot_type == "bar":
                bar_colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in bin_stats['mean']]
                ax.bar(bin_stats.index + 1, bin_stats['mean'], color=bar_colors, alpha=0.8, edgecolor='black')
            else:
                for i, val in enumerate(bin_stats['mean']):
                    color = '#2ecc71' if val >= 0 else '#e74c3c'
                    ax.plot(i+1, val, 'o', color=color, markersize=6)
                ax.plot(bin_stats.index + 1, bin_stats['mean'], '-', color='#555555', alpha=0.5, linewidth=1.5)
            
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_title(feature_name, fontsize=9)
            ax.set_xlabel('Decile', fontsize=7)
            ax.set_ylabel('Mean Target', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, len(bin_stats) + 1))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feature_name}\n(error)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_name, fontsize=9)
            continue
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def combine_signal_cumsum_plots(
    gated_returns_dict: Dict[str, pd.Series],
    n_cols: int = 4,
    figsize_per_plot: Tuple[int, int] = (6, 3),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Combine multiple signal cumsum plots into a single figure with subplots.
    
    Parameters
    ----------
    gated_returns_dict : Dict[str, pd.Series]
        Dictionary mapping feature names to gated returns series
    n_cols : int, default=4
        Number of columns in the grid
    figsize_per_plot : Tuple[int, int], default=(6, 3)
        Size of each subplot
    save_path : Optional[str], default=None
        Path to save the combined figure
        
    Returns
    -------
    plt.Figure
        Combined figure with all signal cumsum plots
    """
    if not gated_returns_dict:
        raise ValueError("No data to plot")
    
    n_plots = len(gated_returns_dict)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    # Ensure axes is always a flat list/array for indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        # When n_rows == 1, axes is a 1D array, convert to list for consistent indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, (feature_name, gated_returns) in enumerate(gated_returns_dict.items()):
        ax = axes[idx]
        
        try:
            # Sort by index for proper cumulative calculation
            if isinstance(gated_returns.index, pd.DatetimeIndex):
                gated_returns = gated_returns.sort_index()
            
            cum_returns = gated_returns.cumsum()
            
            ax.plot(cum_returns.index, cum_returns.values, linewidth=1.5)
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_title(feature_name, fontsize=9)
            ax.set_xlabel('Date', fontsize=7)
            ax.set_ylabel('CumSum', fontsize=7)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feature_name}\n(error)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_name, fontsize=9)
            continue
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def combine_distribution_plots(
    features_df: pd.DataFrame,
    feature_names: List[str],
    n_cols: int = 4,
    figsize_per_plot: Tuple[int, int] = (5, 3),
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Combine multiple distribution plots into a single figure with subplots.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns
    feature_names : List[str]
        List of feature names to plot
    n_cols : int, default=4
        Number of columns in the grid
    figsize_per_plot : Tuple[int, int], default=(5, 3)
        Size of each subplot
    bins : int, default=50
        Number of bins for histogram
    save_path : Optional[str], default=None
        Path to save the combined figure
        
    Returns
    -------
    plt.Figure
        Combined figure with all distribution plots
    """
    if not feature_names:
        raise ValueError("No features to plot")
    
    n_plots = len(feature_names)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    # Ensure axes is always a flat array for indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        # When n_rows == 1, axes is a 1D array, ensure it's flat
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]
        
        try:
            feature_data = features_df[feature_name].dropna()
            
            if len(feature_data) == 0:
                ax.text(0.5, 0.5, f'{feature_name}\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature_name, fontsize=9)
                continue
            
            ax.hist(feature_data, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(feature_name, fontsize=9)
            ax.set_xlabel('Value', fontsize=7)
            ax.set_ylabel('Frequency', fontsize=7)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feature_name}\n(error)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_name, fontsize=9)
            continue
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def combine_timeseries_plots(
    features_df: pd.DataFrame,
    feature_names: List[str],
    n_cols: int = 4,
    figsize_per_plot: Tuple[int, int] = (6, 3),
    rolling_window: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Combine multiple timeseries plots into a single figure with subplots.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns with datetime index
    feature_names : List[str]
        List of feature names to plot
    n_cols : int, default=4
        Number of columns in the grid
    figsize_per_plot : Tuple[int, int], default=(6, 3)
        Size of each subplot
    rolling_window : int, default=20
        Window size for rolling mean
    save_path : Optional[str], default=None
        Path to save the combined figure
        
    Returns
    -------
    plt.Figure
        Combined figure with all timeseries plots
    """
    if not feature_names:
        raise ValueError("No features to plot")
    
    n_plots = len(feature_names)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    # Ensure axes is always a flat array for indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        # When n_rows == 1, axes is a 1D array, ensure it's flat
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]
        
        try:
            feature_data = features_df[feature_name].dropna()
            
            if len(feature_data) == 0:
                ax.text(0.5, 0.5, f'{feature_name}\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature_name, fontsize=9)
                continue
            
            # Plot time series
            ax.plot(feature_data.index, feature_data.values, alpha=0.6, linewidth=0.8, label='Value')
            
            # Add rolling mean if enough data
            if len(feature_data) > rolling_window:
                rolling_mean = feature_data.rolling(window=rolling_window).mean()
                ax.plot(rolling_mean.index, rolling_mean.values, 
                       color='red', linewidth=1.5, alpha=0.8, label=f'MA({rolling_window})')
            
            ax.set_title(feature_name, fontsize=9)
            ax.set_xlabel('Date', fontsize=7)
            ax.set_ylabel('Value', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feature_name}\n(error)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_name, fontsize=9)
            continue
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

