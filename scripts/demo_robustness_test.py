"""
Demo Script: Robustness Testing with Real Data

This script demonstrates the robustness testing functionality using
real ES (S&P 500 futures) daily data.

Usage:
    python scripts/demo_robustness_test.py

Output:
    - Prints comparison of all three resampling methods
    - Saves plots to outputs/robustness/ directory
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from utils.evaluation.robustness_test import robustness_test
from utils.core.enums import ResamplingMethod


def load_es_data(filepath: str) -> pd.DataFrame:
    """Load ES futures data from CSV file."""
    df = pd.read_csv(
        filepath,
        names=['date', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['date'],
        index_col='date'
    )
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()


def main():
    """Run robustness test demo with real ES data."""
    print("=" * 70)
    print("ROBUSTNESS TEST DEMO")
    print("Using ES (S&P 500 Futures) Daily Data")
    print("=" * 70)

    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'daily_data', 'ES.txt'
    )

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"\nLoading data from: {data_path}")
    df = load_es_data(data_path)
    print(f"Loaded {len(df)} days of data from {df.index.min()} to {df.index.max()}")

    # Use last 2 years of data
    two_years_ago = df.index.max() - pd.DateOffset(years=2)
    df = df.loc[df.index >= two_years_ago]
    print(f"Using last 2 years: {len(df)} observations")

    # Compute log returns
    returns = compute_log_returns(df['close'])
    returns.name = 'ES_log_returns'

    print(f"\nReturn Series Statistics:")
    print(f"  Mean daily return: {returns.mean():.6f} ({returns.mean() * 252:.2%} annualized)")
    print(f"  Volatility: {returns.std():.6f} ({returns.std() * np.sqrt(252):.2%} annualized)")
    print(f"  Sharpe Ratio: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
    print(f"  Autocorrelation (lag 1): {returns.autocorr(lag=1):.4f}")

    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'outputs', 'robustness'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Run all three methods
    n_samples = 500
    results = {}

    print("\n" + "=" * 70)
    print(f"Running {n_samples} resamples for each method...")
    print("=" * 70)

    # 1. Monte Carlo (Permutation)
    print("\n[1/3] Monte Carlo Permutation Test")
    fig_mc, series_mc, stats_mc = robustness_test(
        returns=returns,
        n_samples=n_samples,
        method=ResamplingMethod.MONTE_CARLO,
        random_seed=42,
        save_path=os.path.join(output_dir, 'monte_carlo.png'),
        verbose=True
    )
    results['Monte Carlo'] = stats_mc
    plt.close(fig_mc)

    # 2. Bootstrap
    print("\n[2/3] Bootstrap Test")
    fig_bs, series_bs, stats_bs = robustness_test(
        returns=returns,
        n_samples=n_samples,
        method=ResamplingMethod.BOOTSTRAP,
        random_seed=42,
        save_path=os.path.join(output_dir, 'bootstrap.png'),
        verbose=True
    )
    results['Bootstrap'] = stats_bs
    plt.close(fig_bs)

    # 3. Block Bootstrap
    block_size = int(np.sqrt(len(returns)))  # Rule of thumb: sqrt(n)
    print(f"\n[3/3] Block Bootstrap Test (block_size={block_size})")
    fig_bb, series_bb, stats_bb = robustness_test(
        returns=returns,
        n_samples=n_samples,
        method=ResamplingMethod.BLOCK_BOOTSTRAP,
        block_size=block_size,
        random_seed=42,
        save_path=os.path.join(output_dir, 'block_bootstrap.png'),
        verbose=True
    )
    results['Block Bootstrap'] = stats_bb
    plt.close(fig_bb)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON OF METHODS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'P-value':>10} {'Mean Cum. Ret':>15} {'Std':>12} {'Significant?':>12}")
    print("-" * 70)

    for method, stats in results.items():
        significant = "Yes" if stats['p_value'] <= 0.05 else "No"
        print(f"{method:<20} {stats['p_value']:>10.4f} {stats['mean_cumulative_return']:>15.2%} {stats['std_cumulative_return']:>12.2%} {significant:>12}")

    print(f"\nOriginal cumulative return: {stats_mc['original_cumulative_return']:.2%}")

    print(f"\n{'Method':<20} {'5th %ile':>12} {'Median':>12} {'95th %ile':>12}")
    print("-" * 70)

    for method, stats in results.items():
        print(f"{method:<20} {stats['p5_cumulative_return']:>12.2%} {stats['p50_cumulative_return']:>12.2%} {stats['p95_cumulative_return']:>12.2%}")

    # Autocorrelation comparison
    print(f"\n{'Method':<20} {'Orig. Autocorr':>15} {'Mean Resampled':>15} {'Preserved?':>12}")
    print("-" * 70)

    for method, stats in results.items():
        preserved = "Yes" if stats['preserved_autocorr'] else "No"
        print(f"{method:<20} {stats['original_autocorr_lag1']:>15.4f} {stats['mean_resampled_autocorr']:>15.4f} {preserved:>12}")

    print("\n" + "=" * 70)
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Monte Carlo (Permutation):
  - Tests if the ORDER of returns matters
  - All permutations have the same final cumulative return (multiplication is commutative)
  - Useful for path-dependent metrics like max drawdown, not final returns

Bootstrap:
  - Tests robustness by sampling WITH replacement
  - Creates variation in final cumulative returns
  - P-value indicates if original performance is significantly better than random

Block Bootstrap:
  - Preserves autocorrelation structure by sampling blocks
  - More realistic for time series with serial correlation
  - Better for strategies that exploit momentum or mean reversion
""")


if __name__ == '__main__':
    main()
