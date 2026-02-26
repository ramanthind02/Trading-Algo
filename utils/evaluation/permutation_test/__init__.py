"""Permutation Test Package

This package contains all permutation testing functionality:
- permutation_engine: Core permutation test engine with strategy pattern
- candle_shuffle: Canonical candle/bar permutation (source of truth)
"""

from utils.evaluation.permutation_test.permutation_engine import (
    PermutationEngine,
    PermutationStrategy,
    FeaturePermutationStrategy,
    BarPermutationStrategy,
    run_permutation_test,
)
from utils.evaluation.permutation_test.candle_shuffle import (
    CandleShuffler,
    CandleShuffleMode,
    GapType,
    IntradayGapConfig,
    classify_daily_gap,
    classify_intraday_gap,
)

__all__ = [
    'PermutationEngine',
    'PermutationStrategy',
    'FeaturePermutationStrategy',
    'BarPermutationStrategy',
    'run_permutation_test',
    'CandleShuffler',
    'CandleShuffleMode',
    'GapType',
    'IntradayGapConfig',
    'classify_daily_gap',
    'classify_intraday_gap',
]
