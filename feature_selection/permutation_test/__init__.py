"""Permutation Test Package

This package contains all permutation testing functionality:
- permutation_engine: Core permutation test engine with strategy pattern
- permute_bars: Bar permutation and walk-forward validation
- unified_permutation_test: Unified interface for all permutation test types
- perm_test: In-sample and cross-validated permutation tests
"""

# Import main classes and functions for easy access
from feature_selection.permutation_test.permutation_engine import (
    PermutationEngine,
    PermutationStrategy,
    FeaturePermutationStrategy,
    BarPermutationStrategy,
    run_permutation_test
)

from feature_selection.permutation_test.unified_permutation_test import (
    insample_permutation_test,
    walkforward_permutation_test
)

from feature_selection.permutation_test.perm_test import (
    permutation_test,
    cross_validate_permutation_test
)

from feature_selection.permutation_test.permute_bars import (
    BarPermute,
    BarPermuteWalkForward,
    WalkForwardValidator,
    bar_permutation_test
)

__all__ = [
    # Core engine
    'PermutationEngine',
    'PermutationStrategy',
    'FeaturePermutationStrategy',
    'BarPermutationStrategy',
    'run_permutation_test',
    
    # Unified interface
    'insample_permutation_test',
    'walkforward_permutation_test',
    
    # Legacy tests
    'permutation_test',
    'cross_validate_permutation_test',
    'bar_permutation_test',
    
    # Bar permutation utilities
    'BarPermute',
    'BarPermuteWalkForward',
    'WalkForwardValidator',
]