"""
Statistics Computation for Prop Firm Simulator

This module provides functions for computing aggregate statistics
from simulation results.

Author: Trading Research Team
"""

import numpy as np
from typing import List, Dict

from utils.prop_firm_simulator.data_structures import (
    ChallengeResult,
    ChallengePass,
    ChallengeFail,
    SimulationConfig,
    SimulationStatistics
)


def compute_statistics(
    results: List[ChallengeResult],
    config: SimulationConfig
) -> SimulationStatistics:
    """
    Compute aggregate statistics from simulation results.

    Parameters
    ----------
    results : List[ChallengeResult]
        List of challenge results from simulation
    config : SimulationConfig
        Simulation configuration used

    Returns
    -------
    SimulationStatistics
        Aggregate statistics from all simulations
    """
    n_simulations = len(results)

    if n_simulations == 0:
        return _empty_statistics()

    # Separate passes and fails
    passes = [r for r in results if r.passed]
    fails = [r for r in results if not r.passed]

    n_passed = len(passes)
    n_failed = len(fails)

    # Compute probabilities
    pass_probability = n_passed / n_simulations
    fail_probability = n_failed / n_simulations

    # Days to pass distribution
    days_to_pass_distribution = _compute_days_distribution(passes)

    # Max equity peak distribution for failures
    max_equity_peak_failure_distribution = _compute_peak_distribution(fails)

    # Expected resets and cost
    all_resets = [r.n_resets for r in results]
    all_costs = [r.total_cost for r in results]
    expected_n_resets = np.mean(all_resets)
    expected_total_cost = np.mean(all_costs)

    # Failure reasons breakdown
    failure_reasons = _count_failure_reasons(fails)

    return SimulationStatistics(
        pass_probability=pass_probability,
        fail_probability=fail_probability,
        days_to_pass_distribution=days_to_pass_distribution,
        max_equity_peak_failure_distribution=max_equity_peak_failure_distribution,
        expected_n_resets=expected_n_resets,
        expected_total_cost=expected_total_cost,
        n_simulations=n_simulations,
        n_passed=n_passed,
        n_failed=n_failed,
        failure_reasons=failure_reasons
    )


def _empty_statistics() -> SimulationStatistics:
    """Return empty statistics for edge case of no results."""
    return SimulationStatistics(
        pass_probability=0.0,
        fail_probability=0.0,
        days_to_pass_distribution={
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0
        },
        max_equity_peak_failure_distribution={
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0
        },
        expected_n_resets=0.0,
        expected_total_cost=0.0,
        n_simulations=0,
        n_passed=0,
        n_failed=0,
        failure_reasons={}
    )


def _compute_days_distribution(passes: List[ChallengePass]) -> Dict[str, float]:
    """Compute distribution statistics for days to pass."""
    if not passes:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0
        }

    days = np.array([p.days_to_pass for p in passes])

    return {
        'mean': float(np.mean(days)),
        'median': float(np.median(days)),
        'std': float(np.std(days)),
        'p5': float(np.percentile(days, 5)),
        'p25': float(np.percentile(days, 25)),
        'p75': float(np.percentile(days, 75)),
        'p95': float(np.percentile(days, 95)),
        'min': float(np.min(days)),
        'max': float(np.max(days))
    }


def _compute_peak_distribution(fails: List[ChallengeFail]) -> Dict[str, float]:
    """Compute distribution statistics for max equity peak at failure."""
    if not fails:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'p5': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0
        }

    peaks = np.array([f.max_equity_peak for f in fails])

    return {
        'mean': float(np.mean(peaks)),
        'median': float(np.median(peaks)),
        'std': float(np.std(peaks)),
        'p5': float(np.percentile(peaks, 5)),
        'p25': float(np.percentile(peaks, 25)),
        'p75': float(np.percentile(peaks, 75)),
        'p95': float(np.percentile(peaks, 95)),
        'min': float(np.min(peaks)),
        'max': float(np.max(peaks))
    }


def _count_failure_reasons(fails: List[ChallengeFail]) -> Dict[str, int]:
    """Count occurrences of each failure reason."""
    reasons = {}
    for f in fails:
        reason = f.failure_reason
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def print_statistics(stats: SimulationStatistics, verbose: bool = True) -> None:
    """
    Print formatted statistics to console.

    Parameters
    ----------
    stats : SimulationStatistics
        Statistics to print
    verbose : bool, default=True
        Whether to print detailed statistics
    """
    print(f"\n{'='*70}")
    print("SIMULATION STATISTICS")
    print(f"{'='*70}")

    print(f"\nOverall Results:")
    print(f"  Simulations: {stats.n_simulations}")
    print(f"  Passed: {stats.n_passed} ({stats.pass_probability:.2%})")
    print(f"  Failed: {stats.n_failed} ({stats.fail_probability:.2%})")

    print(f"\nCost Analysis:")
    print(f"  Expected resets: {stats.expected_n_resets:.2f}")
    print(f"  Expected total cost: ${stats.expected_total_cost:.2f}")

    if stats.n_passed > 0 and verbose:
        print(f"\nDays to Pass Distribution:")
        d = stats.days_to_pass_distribution
        print(f"  Mean: {d['mean']:.1f} days")
        print(f"  Median: {d['median']:.1f} days")
        print(f"  Std Dev: {d['std']:.1f} days")
        print(f"  Range: {d['min']:.0f} - {d['max']:.0f} days")
        print(f"  Percentiles: 5th={d['p5']:.0f}, 25th={d['p25']:.0f}, "
              f"75th={d['p75']:.0f}, 95th={d['p95']:.0f}")

    if stats.n_failed > 0 and verbose:
        print(f"\nFailure Analysis:")
        for reason, count in sorted(stats.failure_reasons.items()):
            pct = count / stats.n_failed * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

        print(f"\nMax Equity Peak at Failure:")
        p = stats.max_equity_peak_failure_distribution
        print(f"  Mean: {p['mean']:.2%}")
        print(f"  Median: {p['median']:.2%}")

    print(f"{'='*70}\n")
