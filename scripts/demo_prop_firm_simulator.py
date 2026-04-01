"""
Demo Script: Prop Firm Challenge Simulator with Real Data

This script demonstrates the prop firm challenge simulator using
real ES (S&P 500 futures) daily data.

Usage:
    python scripts/demo_prop_firm_simulator.py

Output:
    - Prints simulation results and statistics
    - Saves plots to outputs/prop_firm/ directory
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

from utils.simulation.prop_firm_simulator import (
    PropFirmChallengeSimulator,
    SimulationConfig,
    SimulationMethod,
    ChallengeRules,
    ChallengeCosts,
    print_statistics
)
from plotting.prop_firm import (
    plot_challenge_results,
    plot_equity_curves_sample,
    plot_days_to_pass_distribution,
    plot_cost_analysis
)


def load_es_data(filepath: str) -> pd.DataFrame:
    """Load ES futures data from CSV file."""
    df = pd.read_csv(
        filepath,
        names=['date', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['date'],
        index_col='date'
    )
    return df


def compute_simple_returns(
    prices: pd.Series,
    cost_per_trade: float = 0.0
) -> pd.Series:
    """
    Compute simple percentage returns with optional trading costs.

    Parameters
    ----------
    prices : pd.Series
        Price series
    cost_per_trade : float, default=0.0
        Cost per trade as decimal (e.g., 0.0001 for 1 basis point).
        Applied as round-trip cost (entry + exit) per day.

    Returns
    -------
    pd.Series
        Simple returns adjusted for trading costs
    """
    # Simple percentage return: (P_t - P_{t-1}) / P_{t-1}
    returns = (prices - prices.shift(1)) / prices.shift(1)
    returns = returns.dropna()

    # Subtract round-trip trading cost (entry + exit)
    if cost_per_trade > 0:
        returns = returns - (2 * cost_per_trade)

    return returns


def main():
    """Run prop firm challenge simulator demo with real ES data."""
    # ==========================================================================
    # Configuration - Adjust these parameters
    # ==========================================================================
    INCLUDE_TRADING_COSTS = True          # Toggle trading costs on/off
    COST_PER_TRADE = 0.0002               # 2 basis points per trade (conservative)
    # For ES futures: ~0.5 tick spread + commission ≈ $15-20 round trip
    # On a $5000 ES position, that's roughly 0.3-0.4% or 3-4 bps
    # Using 2 bps is conservative for a well-executed strategy

    print("=" * 70)
    print("PROP FIRM CHALLENGE SIMULATOR DEMO")
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

    # Use last 3 years of data for more samples
    three_years_ago = df.index.max() - pd.DateOffset(years=3)
    df = df.loc[df.index >= three_years_ago]
    print(f"Using last 3 years: {len(df)} observations")

    # Compute simple returns (with optional trading costs)
    trading_cost = COST_PER_TRADE if INCLUDE_TRADING_COSTS else 0.0
    returns = compute_simple_returns(df['close'], cost_per_trade=trading_cost)
    returns.name = 'ES_simple_returns'

    print(f"\nTrading Costs:")
    if INCLUDE_TRADING_COSTS:
        print(f"  Cost per trade: {COST_PER_TRADE:.4%} ({COST_PER_TRADE * 10000:.1f} bps)")
        print(f"  Round-trip cost: {2 * COST_PER_TRADE:.4%} ({2 * COST_PER_TRADE * 10000:.1f} bps)")
    else:
        print(f"  Trading costs: DISABLED")

    print(f"\nReturn Series Statistics:")
    print(f"  Mean daily return: {returns.mean():.6f} ({returns.mean() * 252:.2%} annualized)")
    print(f"  Volatility: {returns.std():.6f} ({returns.std() * np.sqrt(252):.2%} annualized)")
    print(f"  Sharpe Ratio: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
    print(f"  Autocorrelation (lag 1): {returns.autocorr(lag=1):.4f}")

    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'outputs', 'prop_firm'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Define challenge rules (typical prop firm challenge)
    rules = ChallengeRules(
        max_drawdown_pct=0.10,          # 10% max drawdown
        profit_target_pct=0.08,          # 8% profit target
        min_trading_days=5,              # Minimum 5 trading days
        trailing_drawdown_pct=0.05,      # 5% trailing drawdown
        max_daily_drawdown_pct=0.04      # 4% max daily loss
    )

    # Define costs
    costs = ChallengeCosts(
        reset_fee=100.0,                 # $100 reset fee
        one_time_fee=500.0               # $500 challenge fee
    )

    print(f"\nChallenge Rules:")
    print(f"  Max Drawdown: {rules.max_drawdown_pct:.0%}")
    print(f"  Profit Target: {rules.profit_target_pct:.0%}")
    print(f"  Min Trading Days: {rules.min_trading_days}")
    print(f"  Trailing Drawdown: {rules.trailing_drawdown_pct:.0%}")
    print(f"  Max Daily Drawdown: {rules.max_daily_drawdown_pct:.0%}")

    print(f"\nChallenge Costs:")
    print(f"  Entry Fee: ${costs.one_time_fee:.0f}")
    print(f"  Reset Fee: ${costs.reset_fee:.0f}")

    # Run simulations with both methods
    n_simulations = 500
    results_all = {}

    # =========================================================================
    # 1. Historical Walk-Forward
    # =========================================================================
    print("\n" + "=" * 70)
    print("[1/2] HISTORICAL WALK-FORWARD SIMULATION")
    print("=" * 70)

    config_hist = SimulationConfig(
        method=SimulationMethod.HISTORICAL_WALKFORWARD,
        rules=rules,
        costs=costs,
        n_simulations=n_simulations,
        max_attempts_per_sim=5,
        random_seed=42
    )

    simulator_hist = PropFirmChallengeSimulator(returns, config_hist)
    results_hist, stats_hist = simulator_hist.run_simulation(verbose=True)
    results_all['Historical'] = (results_hist, stats_hist)

    print_statistics(stats_hist)

    # Save plots
    fig_hist = plot_challenge_results(
        results_hist, stats_hist, config_hist,
        save_path=os.path.join(output_dir, 'historical_dashboard.png')
    )
    plt.close(fig_hist)

    # =========================================================================
    # 2. Monte Carlo Block Bootstrap
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2/2] MONTE CARLO BLOCK BOOTSTRAP SIMULATION")
    print("=" * 70)

    block_size = int(np.sqrt(len(returns)))  # Rule of thumb
    print(f"Block size: {block_size}")

    config_mc = SimulationConfig(
        method=SimulationMethod.MONTE_CARLO_BLOCK,
        rules=rules,
        costs=costs,
        n_simulations=n_simulations,
        max_attempts_per_sim=5,
        block_length=block_size,
        path_length=252,  # 1 year of trading days
        random_seed=42
    )

    simulator_mc = PropFirmChallengeSimulator(returns, config_mc)
    results_mc, stats_mc = simulator_mc.run_simulation(verbose=True)
    results_all['Monte Carlo'] = (results_mc, stats_mc)

    print_statistics(stats_mc)

    # Save plots
    fig_mc = plot_challenge_results(
        results_mc, stats_mc, config_mc,
        save_path=os.path.join(output_dir, 'monte_carlo_dashboard.png')
    )
    plt.close(fig_mc)

    # Additional plots for Monte Carlo
    fig_curves = plot_equity_curves_sample(
        results_mc, config_mc, n_samples=30,
        save_path=os.path.join(output_dir, 'monte_carlo_curves.png')
    )
    plt.close(fig_curves)

    passes_mc = [r for r in results_mc if r.passed]
    if passes_mc:
        fig_days = plot_days_to_pass_distribution(
            passes_mc, stats_mc,
            save_path=os.path.join(output_dir, 'monte_carlo_days.png')
        )
        plt.close(fig_days)

    fig_cost = plot_cost_analysis(
        results_mc, stats_mc,
        save_path=os.path.join(output_dir, 'monte_carlo_cost.png')
    )
    plt.close(fig_cost)

    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON OF METHODS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Pass Rate':>12} {'Exp. Cost':>12} {'Exp. Resets':>12}")
    print("-" * 60)
    for method, (results, stats) in results_all.items():
        print(f"{method:<20} {stats.pass_probability:>12.1%} "
              f"${stats.expected_total_cost:>11.2f} "
              f"{stats.expected_n_resets:>12.2f}")

    # Failure analysis
    print(f"\n{'Method':<20} {'Max DD':>10} {'Trail DD':>10} {'Daily DD':>10} {'Path Exh.':>10}")
    print("-" * 60)
    for method, (results, stats) in results_all.items():
        fr = stats.failure_reasons
        total_fails = stats.n_failed if stats.n_failed > 0 else 1
        print(f"{method:<20} "
              f"{fr.get('max_drawdown', 0)/total_fails:>10.1%} "
              f"{fr.get('trailing_drawdown', 0)/total_fails:>10.1%} "
              f"{fr.get('daily_drawdown', 0)/total_fails:>10.1%} "
              f"{fr.get('path_exhausted', 0)/total_fails:>10.1%}")

    # Days to pass comparison
    print(f"\n{'Method':<20} {'Mean Days':>12} {'Median Days':>12} {'95th %ile':>12}")
    print("-" * 60)
    for method, (results, stats) in results_all.items():
        d = stats.days_to_pass_distribution
        if stats.n_passed > 0:
            print(f"{method:<20} {d['mean']:>12.1f} {d['median']:>12.1f} {d['p95']:>12.1f}")
        else:
            print(f"{method:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print("\n" + "=" * 70)
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Historical Walk-Forward:
  - Uses actual historical sequences from random start points
  - Reflects real market conditions and correlations
  - Results depend on specific historical period chosen

Monte Carlo Block Bootstrap:
  - Generates synthetic paths by resampling blocks
  - Preserves autocorrelation structure from original data
  - Provides broader range of possible market scenarios
  - Better for stress testing and worst-case analysis

Key Insights:
  - Compare pass rates across methods for robustness
  - If Historical >> Monte Carlo: strategy may be overfitting
  - If Monte Carlo >> Historical: strategy handles variety well
  - Check failure reasons to identify strategy weaknesses
  - Expected cost helps size risk allocation appropriately
""")

    # Risk allocation suggestion
    print("\n" + "=" * 70)
    print("RISK ALLOCATION SUGGESTION")
    print("=" * 70)

    # Use more conservative (lower) pass rate
    conservative_pass_rate = min(stats_hist.pass_probability, stats_mc.pass_probability)
    conservative_cost = max(stats_hist.expected_total_cost, stats_mc.expected_total_cost)

    print(f"\nConservative Pass Rate: {conservative_pass_rate:.1%}")
    print(f"Conservative Expected Cost: ${conservative_cost:.2f}")

    if conservative_pass_rate > 0:
        # Expected attempts to pass
        expected_attempts = 1 / conservative_pass_rate
        total_expected_cost = conservative_cost * expected_attempts
        print(f"\nExpected attempts to pass: {expected_attempts:.1f}")
        print(f"Total expected cost (including failures): ${total_expected_cost:.2f}")

        # Break-even analysis (assuming $10k funded account with 80% profit split)
        funded_account = 10000
        profit_split = 0.80
        payout = funded_account * rules.profit_target_pct * profit_split
        print(f"\nBreak-even Analysis (${funded_account:,} account, {profit_split:.0%} split):")
        print(f"  Payout per pass: ${payout:.2f}")
        print(f"  Total expected cost: ${total_expected_cost:.2f}")
        print(f"  Expected profit: ${payout - total_expected_cost:.2f}")
        print(f"  ROI: {(payout - total_expected_cost) / total_expected_cost:.1%}")


if __name__ == '__main__':
    main()
