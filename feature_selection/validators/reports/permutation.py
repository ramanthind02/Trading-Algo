"""Permutation test report data structures."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PermutationReport:
    """Results from permutation testing (Stage 1 or Stage 2)."""

    stage: Literal['stage1_vector_shuffle', 'stage2_feature_shuffle', 'stage2_candle_shuffle']

    # Observed statistics (real data)
    observed_sharpe: float
    observed_t_stat: float
    observed_returns_mean: float

    # Permutation distribution
    permuted_sharpes: np.ndarray  # shape: (n_permutations,)
    permuted_t_stats: np.ndarray

    # Statistical test results
    p_value: float
    confidence_level: float  # e.g., 0.95
    critical_value: float  # threshold from permutation distribution

    # Verdict
    passed: bool
    margin: float  # observed - critical_value (safety margin)

    # Diagnostic plots
    permutation_histogram: Figure | Path | None
    qq_plot: Figure | Path | None

    # Metadata
    n_permutations: int
    random_seed: int | None
    execution_time: float  # seconds
