"""
Robustness Testing Visualization Module

This module provides plotting functions for robustness test results,
showing equity curve distributions and statistical annotations.

Author: Trading Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

from utils.enums import ResamplingMethod


def plot_robustness_curves(
    original_curve: pd.Series,
    resampled_curves: List[pd.Series],
    stats: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 8),
    show_original: bool = True,
    alpha: float = 0.05,
    method: ResamplingMethod = ResamplingMethod.MONTE_CARLO,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot robustness test results showing original vs resampled equity curves.

    Parameters
    ----------
    original_curve : pd.Series
        Original equity curve with datetime index
    resampled_curves : List[pd.Series]
        List of resampled equity curves
    stats : Dict[str, Any]
        Statistics dictionary from robustness_test
    figsize : Tuple[int, int], default=(14, 8)
        Figure size
    show_original : bool, default=True
        Whether to highlight original curve
    alpha : float, default=0.05
        Significance level for highlighting
    method : ResamplingMethod
        Resampling method used (for title)
    save_path : Optional[str], default=None
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot resampled curves (gray, semi-transparent)
    for curve in resampled_curves:
        ax.plot(
            curve.index,
            curve.values,
            color='gray',
            alpha=0.1,
            linewidth=0.5,
            label='_nolegend_'
        )

    # Create array of all curves for percentile calculation
    all_curves = np.array([curve.values for curve in resampled_curves])
    p5 = np.percentile(all_curves, 5, axis=0)
    p25 = np.percentile(all_curves, 25, axis=0)
    p75 = np.percentile(all_curves, 75, axis=0)
    p95 = np.percentile(all_curves, 95, axis=0)
    median = np.percentile(all_curves, 50, axis=0)

    dates = original_curve.index

    # Fill between 5th and 95th percentile
    ax.fill_between(
        dates, p5, p95,
        color='lightgray',
        alpha=0.5,
        label='5th-95th percentile'
    )

    # Fill between 25th and 75th percentile (darker)
    ax.fill_between(
        dates, p25, p75,
        color='darkgray',
        alpha=0.3,
        label='25th-75th percentile'
    )

    # Plot median resampled curve
    ax.plot(
        dates, median,
        color='#555555',
        linewidth=2,
        linestyle='--',
        label='Median resampled'
    )

    # Plot original curve (highlighted)
    if show_original:
        # Color based on significance
        if stats['p_value'] <= alpha:
            curve_color = '#2ecc71'  # Green for significant
        else:
            curve_color = '#e74c3c'  # Red for not significant

        ax.plot(
            original_curve.index,
            original_curve.values,
            color=curve_color,
            linewidth=2.5,
            label=f'Original (p={stats["p_value"]:.4f})'
        )

    # Add horizontal line at y=1 (initial value)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add statistics text box (following distribution.py pattern)
    stats_text = _format_stats_text(stats, method)
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
    )

    # Labels and title
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Equity (Starting = 1.0)', fontsize=11)

    method_name = {
        ResamplingMethod.MONTE_CARLO: 'Monte Carlo Permutation',
        ResamplingMethod.BOOTSTRAP: 'Bootstrap',
        ResamplingMethod.BLOCK_BOOTSTRAP: 'Block Bootstrap'
    }.get(method, method.value)

    ax.set_title(
        f'Robustness Test: {method_name}\n'
        f'n={stats["n_samples"]} resamples',
        fontsize=13,
        fontweight='bold'
    )

    # Legend
    ax.legend(loc='lower right', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness plot saved to: {save_path}")

    return fig


def _format_stats_text(stats: Dict[str, Any], method: ResamplingMethod) -> str:
    """Format statistics for display in text box."""
    text_lines = [
        f"Original Cum. Return: {stats['original_cumulative_return']:.2%}",
        f"",
        f"Resampled Distribution:",
        f"  Mean: {stats['mean_cumulative_return']:.2%}",
        f"  Median: {stats['median_cumulative_return']:.2%}",
        f"  Std: {stats['std_cumulative_return']:.2%}",
        f"",
        f"Percentiles:",
        f"  5th: {stats['p5_cumulative_return']:.2%}",
        f"  25th: {stats['p25_cumulative_return']:.2%}",
        f"  75th: {stats['p75_cumulative_return']:.2%}",
        f"  95th: {stats['p95_cumulative_return']:.2%}",
        f"",
        f"P-value: {stats['p_value']:.4f}",
        f"Significant (5%): {'Yes' if stats['p_value'] <= 0.05 else 'No'}",
    ]

    # Add autocorrelation info for block bootstrap
    if method == ResamplingMethod.BLOCK_BOOTSTRAP:
        text_lines.extend([
            f"",
            f"Autocorrelation:",
            f"  Original: {stats['original_autocorr_lag1']:.4f}",
            f"  Resampled Mean: {stats['mean_resampled_autocorr']:.4f}",
            f"  Preserved: {'Yes' if stats['preserved_autocorr'] else 'No'}",
        ])

    return '\n'.join(text_lines)
