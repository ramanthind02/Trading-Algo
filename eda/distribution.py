"""Distribution and Time Series Plotting Module

This module provides visualization functions for feature distributions and time series.

Author: Trading Research Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import scipy.stats as stats


def plot_feature_distribution(
    feature_data: pd.Series,
    feature_name: str,
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of a feature with histogram and KDE.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature data to plot
    feature_name : str
        Name of the feature
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    bins : int, default=50
        Number of histogram bins
    show_stats : bool, default=True
        Whether to show statistical information
    save_path : Optional[str], default=None
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
        
    Raises
    ------
    TypeError
        If feature_data is not numeric
    ValueError
        If feature_data is empty or all NaN
    """
    # Validate input
    if not isinstance(feature_data, pd.Series):
        raise TypeError("feature_data must be a pandas Series")
    
    if not pd.api.types.is_numeric_dtype(feature_data):
        raise TypeError(
            f"Feature '{feature_name}' is non-numeric (dtype: {feature_data.dtype}). "
            f"Distribution plots require numeric features."
        )
    
    # Remove NaN values
    clean_data = feature_data.dropna()
    
    if len(clean_data) == 0:
        raise ValueError(
            f"Feature '{feature_name}' has no valid (non-NaN) values. "
            f"Cannot plot distribution."
        )
    
    # Create single larger figure for distribution
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Histogram with KDE
    ax.hist(clean_data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add KDE if we have enough data points
    if len(clean_data) > 10:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(clean_data)
            x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend(loc='upper right')
        except Exception:
            # KDE can fail for certain distributions, just skip it
            pass
    
    # Zoom in to show main distribution (exclude extreme outliers)
    # Use percentiles to set reasonable x-axis limits
    p1, p99 = np.percentile(clean_data, [1, 99])
    data_range = p99 - p1
    margin = data_range * 0.1  # Add 10% margin
    ax.set_xlim(p1 - margin, p99 + margin)
    
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Distribution of {feature_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box if requested
    if show_stats:
        # Compute additional statistics
        from scipy.stats import entropy, jarque_bera
        
        # Compute entropy (using histogram bins)
        hist_counts, _ = np.histogram(clean_data, bins=bins)
        hist_probs = hist_counts / hist_counts.sum()
        hist_probs = hist_probs[hist_probs > 0]  # Remove zeros
        feature_entropy = entropy(hist_probs)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = jarque_bera(clean_data)
        
        # Coefficient of variation
        cv = (clean_data.std() / abs(clean_data.mean())) if clean_data.mean() != 0 else np.inf
        
        # Percentiles
        p5, p25, p75, p95 = np.percentile(clean_data, [5, 25, 75, 95])
        
        stats_text = (
            f"n = {len(clean_data):,}\n"
            f"Mean = {clean_data.mean():.4f}\n"
            f"Median = {clean_data.median():.4f}\n"
            f"Std = {clean_data.std():.4f}\n"
            f"CV = {cv:.2f}\n"
            f"\n"
            f"Skew = {clean_data.skew():.4f}\n"
            f"Kurt = {clean_data.kurtosis():.4f}\n"
            f"Entropy = {feature_entropy:.4f}\n"
            f"\n"
            f"Min = {clean_data.min():.4f}\n"
            f"5% = {p5:.4f}\n"
            f"25% = {p25:.4f}\n"
            f"75% = {p75:.4f}\n"
            f"95% = {p95:.4f}\n"
            f"Max = {clean_data.max():.4f}\n"
            f"\n"
            f"JB Test p = {jb_pvalue:.4f}\n"
            f"{'Normal' if jb_pvalue > 0.05 else 'Non-normal'}"
        )
        
        # Add text box to plot
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_timeseries(
    feature_data: pd.Series,
    feature_name: str,
    figsize: Tuple[int, int] = (14, 6),
    show_rolling_mean: bool = True,
    rolling_window: int = 20,
    show_rolling_std: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series of a feature with optional rolling statistics.
    
    Parameters
    ----------
    feature_data : pd.Series
        Feature data to plot (must have datetime index)
    feature_name : str
        Name of the feature
    figsize : Tuple[int, int], default=(14, 6)
        Figure size
    show_rolling_mean : bool, default=True
        Whether to show rolling mean
    rolling_window : int, default=20
        Window size for rolling statistics
    show_rolling_std : bool, default=True
        Whether to show rolling standard deviation bands
    save_path : Optional[str], default=None
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
        
    Raises
    ------
    TypeError
        If feature_data is not numeric or doesn't have datetime index
    ValueError
        If feature_data is empty or all NaN
    """
    # Validate input
    if not isinstance(feature_data, pd.Series):
        raise TypeError("feature_data must be a pandas Series")
    
    if not pd.api.types.is_numeric_dtype(feature_data):
        raise TypeError(
            f"Feature '{feature_name}' is non-numeric (dtype: {feature_data.dtype}). "
            f"Time series plots require numeric features."
        )
    
    if not isinstance(feature_data.index, pd.DatetimeIndex):
        raise TypeError(
            f"Feature data must have a DatetimeIndex for time series plotting. "
            f"Current index type: {type(feature_data.index)}"
        )
    
    # Remove NaN values but keep index
    clean_data = feature_data.dropna()
    
    if len(clean_data) == 0:
        raise ValueError(
            f"Feature '{feature_name}' has no valid (non-NaN) values. "
            f"Cannot plot time series."
        )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main time series
    ax.plot(clean_data.index, clean_data.values, linewidth=1, alpha=0.7, label=feature_name)
    
    # Add rolling mean if requested
    if show_rolling_mean and len(clean_data) > rolling_window:
        rolling_mean = clean_data.rolling(window=rolling_window, center=True).mean()
        ax.plot(
            rolling_mean.index, rolling_mean.values,
            linewidth=2, color='red', alpha=0.8,
            label=f'{rolling_window}-period MA'
        )
    
    # Add rolling std bands if requested
    if show_rolling_std and len(clean_data) > rolling_window:
        rolling_mean = clean_data.rolling(window=rolling_window, center=True).mean()
        rolling_std = clean_data.rolling(window=rolling_window, center=True).std()
        
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        
        ax.fill_between(
            rolling_mean.index,
            lower_band.values,
            upper_band.values,
            alpha=0.3, color='lightcoral',
            label=f'±2σ bands'
        )
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel(feature_name)
    ax.set_title(f'Time Series: {feature_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Compute additional time series statistics
    # Autocorrelation at lag 1
    autocorr_1 = clean_data.autocorr(lag=1) if len(clean_data) > 1 else np.nan
    
    # Stationarity indicator (simple heuristic: rolling mean variance)
    if len(clean_data) > 100:
        rolling_means = clean_data.rolling(window=50).mean()
        mean_stability = rolling_means.std() / clean_data.std() if clean_data.std() > 0 else np.nan
    else:
        mean_stability = np.nan
    
    # Add statistics text box
    stats_text = (
        f"Period: {clean_data.index.min().date()} to {clean_data.index.max().date()}\n"
        f"n = {len(clean_data):,}\n"
        f"\n"
        f"Mean = {clean_data.mean():.4f}\n"
        f"Std = {clean_data.std():.4f}\n"
        f"Min = {clean_data.min():.4f}\n"
        f"Max = {clean_data.max():.4f}\n"
        f"\n"
        f"Autocorr(1) = {autocorr_1:.4f}\n"
        f"Mean Stability = {mean_stability:.4f}" if not np.isnan(mean_stability) else ""
    )
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig