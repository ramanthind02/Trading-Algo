"""
Robustness Test Package

This package provides robustness testing functionality for trading strategies:
- resampling_strategy: Abstract base class and concrete resampling strategies
- robustness_engine: Main robustness_test() function

Resampling Methods:
- Monte Carlo: Random permutation (destroys autocorrelation)
- Bootstrap: Sampling with replacement
- Block Bootstrap: Block-wise sampling (preserves autocorrelation)

Example Usage:
    from utils.evaluation.robustness_test import robustness_test
    from utils.core.enums import ResamplingMethod

    _, resampled_series, stats = robustness_test(
        returns=my_returns,
        n_samples=1000,
        method=ResamplingMethod.MONTE_CARLO,
        random_seed=42
    )

    print(f"P-value: {stats['p_value']:.4f}")
"""

from utils.evaluation.robustness_test.robustness_engine import robustness_test
from utils.evaluation.robustness_test.resampling_strategy import (
    ResamplingStrategy,
    MonteCarloStrategy,
    BootstrapStrategy,
    BlockBootstrapStrategy
)

__all__ = [
    # Main function
    'robustness_test',

    # Strategy classes
    'ResamplingStrategy',
    'MonteCarloStrategy',
    'BootstrapStrategy',
    'BlockBootstrapStrategy',
]
