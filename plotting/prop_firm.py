"""
Prop Firm Challenge Simulator Visualization Module

This module provides plotting functions for prop firm challenge simulation results,
showing equity curves, pass/fail distributions, and cost analysis.

Author: Trading Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

from utils.simulation.prop_firm_simulator.data_structures import (
    ChallengeResult,
    ChallengePass,
    ChallengeFail,
    SimulationStatistics,
    SimulationConfig
)


def plot_challenge_results(
    results: List[ChallengeResult],
    stats: SimulationStatistics,
    config: SimulationConfig,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a 2x2 dashboard showing challenge simulation results.

    Layout:
    - Top left: Pass/Fail pie chart
    - Top right: Days to pass distribution
    - Bottom left: Sample equity curves
    - Bottom right: Cost analysis

    Parameters
    ----------
    results : List[ChallengeResult]
        List of challenge results
    stats : SimulationStatistics
        Aggregate statistics
    config : SimulationConfig
        Simulation configuration used
    figsize : Tuple[int, int], default=(14, 10)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Pass/Fail pie chart
    _plot_pass_fail_pie(axes[0, 0], stats)

    # Top right: Days to pass distribution
    passes = [r for r in results if r.passed]
    _plot_days_distribution(axes[0, 1], passes, stats)

    # Bottom left: Sample equity curves
    _plot_sample_curves(axes[1, 0], results, config, n_samples=20)

    # Bottom right: Cost analysis
    _plot_cost_breakdown(axes[1, 1], results, stats)

    # Overall title
    fig.suptitle(
        f'Prop Firm Challenge Simulation Results\n'
        f'{config.method.value} - {stats.n_simulations} simulations',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Challenge results plot saved to: {save_path}")

    return fig


def _plot_pass_fail_pie(ax: plt.Axes, stats: SimulationStatistics) -> None:
    """Plot pass/fail pie chart."""
    if stats.n_simulations == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('Pass/Fail Distribution')
        return

    labels = ['Pass', 'Fail']
    sizes = [stats.n_passed, stats.n_failed]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%\n({int(p * stats.n_simulations / 100)})',
        startangle=90,
        textprops={'fontsize': 10}
    )
    ax.set_title('Pass/Fail Distribution', fontsize=12, fontweight='bold')


def _plot_days_distribution(
    ax: plt.Axes,
    passes: List[ChallengePass],
    stats: SimulationStatistics
) -> None:
    """Plot days to pass distribution."""
    if not passes:
        ax.text(0.5, 0.5, 'No passes', ha='center', va='center')
        ax.set_title('Days to Pass Distribution')
        return

    days = [p.days_to_pass for p in passes]

    ax.hist(days, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(
        stats.days_to_pass_distribution['mean'],
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Mean: {stats.days_to_pass_distribution['mean']:.1f}"
    )
    ax.axvline(
        stats.days_to_pass_distribution['median'],
        color='orange',
        linestyle=':',
        linewidth=2,
        label=f"Median: {stats.days_to_pass_distribution['median']:.1f}"
    )

    ax.set_xlabel('Days to Pass', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Days to Pass Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_sample_curves(
    ax: plt.Axes,
    results: List[ChallengeResult],
    config: SimulationConfig,
    n_samples: int = 20
) -> None:
    """Plot sample equity curves from results."""
    if not results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('Sample Equity Curves')
        return

    # Sample results
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(results),
        size=min(n_samples, len(results)),
        replace=False
    )
    samples = [results[i] for i in sample_indices]

    for result in samples:
        curve = result.equity_curve
        if len(curve) > 0:
            color = '#2ecc71' if result.passed else '#e74c3c'
            alpha = 0.7 if result.passed else 0.3
            ax.plot(
                range(len(curve)),
                curve.values,
                color=color,
                alpha=alpha,
                linewidth=1
            )

    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(
        y=config.rules.profit_target_pct,
        color='green',
        linestyle='--',
        linewidth=1.5,
        label=f'Target: {config.rules.profit_target_pct:.1%}'
    )
    ax.axhline(
        y=-config.rules.max_drawdown_pct,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Max DD: -{config.rules.max_drawdown_pct:.1%}'
    )

    ax.set_xlabel('Days', fontsize=10)
    ax.set_ylabel('Equity (%)', fontsize=10)
    ax.set_title('Sample Equity Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)


def _plot_cost_breakdown(
    ax: plt.Axes,
    results: List[ChallengeResult],
    stats: SimulationStatistics
) -> None:
    """Plot cost analysis."""
    if not results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('Cost Analysis')
        return

    costs = [r.total_cost for r in results]

    ax.hist(costs, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.axvline(
        stats.expected_total_cost,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Expected: ${stats.expected_total_cost:.2f}'
    )

    ax.set_xlabel('Total Cost ($)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Cost Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add stats text box
    stats_text = (
        f"Expected Cost: ${stats.expected_total_cost:.2f}\n"
        f"Expected Resets: {stats.expected_n_resets:.2f}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )


def plot_equity_curves_sample(
    results: List[ChallengeResult],
    config: SimulationConfig,
    n_samples: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sample equity curves with percentile bands.

    Parameters
    ----------
    results : List[ChallengeResult]
        List of challenge results
    config : SimulationConfig
        Simulation configuration
    n_samples : int, default=20
        Number of sample curves to plot
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Sample results for individual curves
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(results),
        size=min(n_samples, len(results)),
        replace=False
    )

    # Find max curve length
    max_len = max(len(r.equity_curve) for r in results)

    # Pad all curves to same length for percentile calculation
    all_curves = []
    for r in results:
        curve = r.equity_curve.values
        if len(curve) < max_len:
            # Pad with NaN
            padded = np.full(max_len, np.nan)
            padded[:len(curve)] = curve
            all_curves.append(padded)
        else:
            all_curves.append(curve)

    all_curves = np.array(all_curves)

    # Calculate percentiles
    with np.errstate(all='ignore'):
        p5 = np.nanpercentile(all_curves, 5, axis=0)
        p25 = np.nanpercentile(all_curves, 25, axis=0)
        p75 = np.nanpercentile(all_curves, 75, axis=0)
        p95 = np.nanpercentile(all_curves, 95, axis=0)
        median = np.nanpercentile(all_curves, 50, axis=0)

    days = np.arange(max_len)

    # Fill percentile bands
    ax.fill_between(
        days, p5, p95,
        color='lightgray', alpha=0.5,
        label='5th-95th percentile'
    )
    ax.fill_between(
        days, p25, p75,
        color='darkgray', alpha=0.3,
        label='25th-75th percentile'
    )

    # Plot median
    ax.plot(days, median, color='#555555', linewidth=2, linestyle='--', label='Median')

    # Plot sample curves
    for i in sample_indices:
        result = results[i]
        curve = result.equity_curve
        color = '#2ecc71' if result.passed else '#e74c3c'
        ax.plot(range(len(curve)), curve.values, color=color, alpha=0.5, linewidth=0.8)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(
        y=config.rules.profit_target_pct,
        color='green', linestyle='--', linewidth=1.5,
        label=f'Target: {config.rules.profit_target_pct:.1%}'
    )
    ax.axhline(
        y=-config.rules.max_drawdown_pct,
        color='red', linestyle='--', linewidth=1.5,
        label=f'Max DD: -{config.rules.max_drawdown_pct:.1%}'
    )

    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Equity (%)', fontsize=11)
    ax.set_title(
        f'Equity Curve Distribution\n{len(results)} simulations',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Equity curves plot saved to: {save_path}")

    return fig


def plot_days_to_pass_distribution(
    passes: List[ChallengePass],
    stats: SimulationStatistics,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram + KDE of days to pass.

    Parameters
    ----------
    passes : List[ChallengePass]
        List of passed challenge results
    stats : SimulationStatistics
        Aggregate statistics
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not passes:
        ax.text(0.5, 0.5, 'No passes to analyze', ha='center', va='center')
        return fig

    days = np.array([p.days_to_pass for p in passes])
    d = stats.days_to_pass_distribution

    # Histogram
    ax.hist(days, bins=30, density=True, color='#3498db', alpha=0.7, edgecolor='black')

    # Add vertical lines for statistics
    ax.axvline(d['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {d['mean']:.1f}")
    ax.axvline(d['median'], color='orange', linestyle=':', linewidth=2, label=f"Median: {d['median']:.1f}")
    ax.axvline(d['p5'], color='gray', linestyle='-.', linewidth=1, label=f"5th %ile: {d['p5']:.1f}")
    ax.axvline(d['p95'], color='gray', linestyle='-.', linewidth=1, label=f"95th %ile: {d['p95']:.1f}")

    ax.set_xlabel('Days to Pass', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(
        f'Days to Pass Distribution\n{len(passes)} passes out of {stats.n_simulations} simulations',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Days distribution plot saved to: {save_path}")

    return fig


def plot_cost_analysis(
    results: List[ChallengeResult],
    stats: SimulationStatistics,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cost distribution and breakdown.

    Parameters
    ----------
    results : List[ChallengeResult]
        List of challenge results
    stats : SimulationStatistics
        Aggregate statistics
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if not results:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Left: Cost histogram
    costs = np.array([r.total_cost for r in results])

    axes[0].hist(costs, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0].axvline(
        stats.expected_total_cost,
        color='red', linestyle='--', linewidth=2,
        label=f'Expected: ${stats.expected_total_cost:.2f}'
    )
    axes[0].axvline(
        np.median(costs),
        color='orange', linestyle=':', linewidth=2,
        label=f'Median: ${np.median(costs):.2f}'
    )

    axes[0].set_xlabel('Total Cost ($)', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title('Cost Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: Reset count histogram
    resets = np.array([r.n_resets for r in results])

    reset_counts = {}
    for n in range(int(resets.max()) + 1):
        reset_counts[n] = np.sum(resets == n)

    axes[1].bar(
        reset_counts.keys(),
        reset_counts.values(),
        color='#e67e22', alpha=0.7, edgecolor='black'
    )
    axes[1].axvline(
        stats.expected_n_resets,
        color='red', linestyle='--', linewidth=2,
        label=f'Expected: {stats.expected_n_resets:.2f}'
    )

    axes[1].set_xlabel('Number of Resets', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title('Reset Count Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Cost Analysis - {stats.n_simulations} simulations',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost analysis plot saved to: {save_path}")

    return fig
