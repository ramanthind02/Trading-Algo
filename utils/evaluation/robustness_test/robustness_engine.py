"""
Robustness Test Engine

This module provides the core robustness testing functionality:
1. Orchestrates resampling across multiple iterations
2. Computes equity curves and statistics
3. Returns comprehensive results for analysis

Architecture follows PermutationEngine pattern from permutation_test/permutation_engine.py

Author: Trading Research Team
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.core.enums import ResamplingMethod
from utils.evaluation.robustness_test.resampling_strategy import (
    ResamplingStrategy,
    MonteCarloStrategy,
    BootstrapStrategy,
    BlockBootstrapStrategy
)
from metrics.equity import cumulative_returns


def robustness_test(
    returns: pd.Series,
    n_samples: int = 1000,
    method: ResamplingMethod = ResamplingMethod.MONTE_CARLO,
    block_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[None, List[pd.Series], Dict[str, Any]]:
    """
    Perform robustness test on a return series using resampling methods.

    This function tests whether observed cumulative returns are statistically
    significant by comparing against a distribution of resampled return paths.

    Parameters
    ----------
    returns : pd.Series
        Return series with datetime index (decimal form, e.g., 0.01 for 1%)
    n_samples : int, default=1000
        Number of resampled paths to generate
    method : ResamplingMethod, default=MONTE_CARLO
        Resampling method to use:
        - MONTE_CARLO: Random permutation (destroys autocorrelation)
        - BOOTSTRAP: Sampling with replacement
        - BLOCK_BOOTSTRAP: Block-wise sampling (preserves autocorrelation)
    block_size : Optional[int], default=None
        Block size for block bootstrap. Required if method=BLOCK_BOOTSTRAP.
        Recommended: sqrt(n) or based on autocorrelation decay.
    random_seed : Optional[int], default=None
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress information

    Returns
    -------
    Tuple[None, List[pd.Series], Dict[str, Any]]
        - fig: Always ``None`` now that plotting has been removed
        - resampled_series: List of resampled return Series (not cumulative)
        - stats: Dictionary with comprehensive statistics

    Raises
    ------
    TypeError
        If returns is not pd.Series or doesn't have datetime index
    ValueError
        If returns is empty, contains NaN, or block_size invalid

    Examples
    --------
    >>> from utils.evaluation.robustness_test import robustness_test
    >>> from utils.core.enums import ResamplingMethod
    >>>
    >>> # Generate sample returns
    >>> returns = pd.Series(
    ...     np.random.normal(0.001, 0.02, 252),
    ...     index=pd.date_range('2024-01-01', periods=252)
    ... )
    >>>
    >>> # Run Monte Carlo robustness test
    >>> fig, series, stats = robustness_test(
    ...     returns,
    ...     n_samples=1000,
    ...     method=ResamplingMethod.MONTE_CARLO
    ... )
    >>> print(f"P-value: {stats['p_value']:.4f}")
    """
    # Handle NaN values
    if returns.isna().any():
        nan_count = returns.isna().sum()
        if verbose:
            print(f"Warning: Dropping {nan_count} NaN values from returns series")
        returns = returns.dropna()

    # Select strategy based on method
    strategy = _get_strategy(method)

    # Validate inputs
    strategy.validate_data(returns, block_size=block_size)

    if verbose:
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS TEST")
        print(f"{'='*70}")
        print(f"Method: {method.value}")
        print(f"Samples: {n_samples}")
        print(f"Series length: {len(returns)}")
        if method == ResamplingMethod.BLOCK_BOOTSTRAP:
            print(f"Block size: {block_size}")
        print(f"{'='*70}\n")

    # Set base random seed
    base_seed = random_seed if random_seed is not None else 42

    # Compute original equity curve statistics without materializing plots.
    original_cumulative = cumulative_returns(returns, initial_value=1.0).iloc[-1] - 1.0
    original_autocorr = returns.autocorr(lag=1)
    original_mean = returns.mean()
    original_volatility = returns.std()

    if verbose:
        print(f"Original cumulative return: {original_cumulative:.4%}")
        print(f"Original mean return: {original_mean:.6f}")
        print(f"Original volatility: {original_volatility:.6f}")
        print(f"Original autocorrelation (lag 1): {original_autocorr:.4f}")
        print(f"\nGenerating {n_samples} resampled paths...")

    # Generate resampled paths
    resampled_series: List[pd.Series] = []
    cumulative_returns_list: List[float] = []
    autocorr_list: List[float] = []

    for i in range(n_samples):
        # Generate seed for this iteration
        iter_seed = base_seed + i

        # Resample returns
        resampled_returns = strategy.resample(
            returns,
            random_seed=iter_seed,
            block_size=block_size
        )

        # Store resampled returns (not cumulative)
        resampled_series.append(resampled_returns)

        # Store statistics
        cumulative_returns_list.append(
            cumulative_returns(resampled_returns, initial_value=1.0).iloc[-1] - 1.0
        )
        autocorr_list.append(resampled_returns.autocorr(lag=1))

        # Progress update
        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")

    if verbose:
        print(f"Completed {n_samples} samples\n")

    # Compute statistics
    stats = _compute_statistics(
        original_cumulative=original_cumulative,
        original_autocorr=original_autocorr,
        original_mean=original_mean,
        original_volatility=original_volatility,
        cumulative_returns_list=cumulative_returns_list,
        autocorr_list=autocorr_list,
        method=method,
        block_size=block_size
    )

    if verbose:
        _print_statistics(stats)

    return None, resampled_series, stats


def _get_strategy(method: ResamplingMethod) -> ResamplingStrategy:
    """Get the appropriate resampling strategy for the method."""
    strategies = {
        ResamplingMethod.MONTE_CARLO: MonteCarloStrategy(),
        ResamplingMethod.BOOTSTRAP: BootstrapStrategy(),
        ResamplingMethod.BLOCK_BOOTSTRAP: BlockBootstrapStrategy(),
    }

    if method not in strategies:
        raise ValueError(f"Unknown resampling method: {method}")

    return strategies[method]


def _compute_statistics(
    original_cumulative: float,
    original_autocorr: float,
    original_mean: float,
    original_volatility: float,
    cumulative_returns_list: List[float],
    autocorr_list: List[float],
    method: ResamplingMethod,
    block_size: Optional[int] = None
) -> Dict[str, Any]:
    """Compute comprehensive statistics from resampling results."""
    cumulative_array = np.array(cumulative_returns_list)
    autocorr_array = np.array(autocorr_list)

    # P-value: fraction of resamples >= original
    # (one-sided test for positive returns being significant)
    if original_cumulative >= 0:
        p_value = np.mean(cumulative_array >= original_cumulative)
    else:
        # For negative returns, test if significantly negative
        p_value = np.mean(cumulative_array <= original_cumulative)

    stats = {
        # Method info
        'resampling_method': method.value,
        'block_size': block_size if method == ResamplingMethod.BLOCK_BOOTSTRAP else None,

        # Original statistics
        'original_cumulative_return': original_cumulative,
        'original_mean_return': original_mean,
        'original_volatility': original_volatility,
        'original_autocorr_lag1': original_autocorr if not np.isnan(original_autocorr) else 0.0,

        # Distribution statistics
        'mean_cumulative_return': np.mean(cumulative_array),
        'median_cumulative_return': np.median(cumulative_array),
        'std_cumulative_return': np.std(cumulative_array),

        # Percentiles
        'percentiles': {
            'p5': np.percentile(cumulative_array, 5),
            'p25': np.percentile(cumulative_array, 25),
            'p50': np.percentile(cumulative_array, 50),
            'p75': np.percentile(cumulative_array, 75),
            'p95': np.percentile(cumulative_array, 95),
        },
        'p5_cumulative_return': np.percentile(cumulative_array, 5),
        'p25_cumulative_return': np.percentile(cumulative_array, 25),
        'p50_cumulative_return': np.percentile(cumulative_array, 50),
        'p75_cumulative_return': np.percentile(cumulative_array, 75),
        'p95_cumulative_return': np.percentile(cumulative_array, 95),

        # Statistical significance
        'p_value': p_value,
        'n_samples': len(cumulative_returns_list),
        'n_valid_samples': len(cumulative_returns_list),

        # Autocorrelation preservation (for block bootstrap)
        'mean_resampled_autocorr': np.nanmean(autocorr_array),
        'preserved_autocorr': (
            abs(np.nanmean(autocorr_array) - original_autocorr) < 0.1
            if method == ResamplingMethod.BLOCK_BOOTSTRAP and not np.isnan(original_autocorr)
            else False
        ),
    }

    return stats


def _print_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted statistics to console."""
    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nOriginal Performance:")
    print(f"  Cumulative Return: {stats['original_cumulative_return']:.4%}")
    print(f"  Mean Return: {stats['original_mean_return']:.6f}")
    print(f"  Volatility: {stats['original_volatility']:.6f}")
    print(f"  Autocorrelation (lag 1): {stats['original_autocorr_lag1']:.4f}")

    print(f"\nResampled Distribution:")
    print(f"  Mean: {stats['mean_cumulative_return']:.4%}")
    print(f"  Median: {stats['median_cumulative_return']:.4%}")
    print(f"  Std Dev: {stats['std_cumulative_return']:.4%}")

    print(f"\nPercentiles:")
    print(f"  5th:  {stats['p5_cumulative_return']:.4%}")
    print(f"  25th: {stats['p25_cumulative_return']:.4%}")
    print(f"  50th: {stats['p50_cumulative_return']:.4%}")
    print(f"  75th: {stats['p75_cumulative_return']:.4%}")
    print(f"  95th: {stats['p95_cumulative_return']:.4%}")

    print(f"\nStatistical Significance:")
    print(f"  P-value: {stats['p_value']:.4f}")
    print(f"  Significant at 5%: {'Yes' if stats['p_value'] <= 0.05 else 'No'}")
    print(f"  Significant at 1%: {'Yes' if stats['p_value'] <= 0.01 else 'No'}")

    if stats['resampling_method'] == 'block_bootstrap':
        print(f"\nAutocorrelation Preservation:")
        print(f"  Mean Resampled Autocorr: {stats['mean_resampled_autocorr']:.4f}")
        print(f"  Autocorrelation Preserved: {'Yes' if stats['preserved_autocorr'] else 'No'}")

    print(f"{'='*70}\n")
