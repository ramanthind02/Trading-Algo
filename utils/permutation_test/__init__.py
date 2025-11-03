"""Permutation Test Package

This package contains all permutation testing functionality:
- permutation_engine: Core permutation test engine with strategy pattern
- permute_bars: Bar permutation and walk-forward validation
"""

# Import main classes and functions for easy access
from utils.permutation_test.permutation_engine import (
    PermutationEngine,
    PermutationStrategy,
    FeaturePermutationStrategy,
    BarPermutationStrategy,
    run_permutation_test
)

from utils.permutation_test.permute_bars import (
    BarPermute,
    BarPermuteWalkForward,
    WalkForwardValidator
)

__all__ = [
    # Core engine
    'PermutationEngine',
    'PermutationStrategy',
    'FeaturePermutationStrategy',
    'BarPermutationStrategy',
    'run_permutation_test',
    
    # Bar permutation utilities
    'BarPermute',
    'BarPermuteWalkForward',
    'WalkForwardValidator',
]