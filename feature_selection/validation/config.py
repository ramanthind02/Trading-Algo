"""Configuration dataclass for permutation test suite (T016)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PermutationTestConfig:
    """Configuration for permutation test suite (T013-T017)."""

    nreps: int = 1000
    alpha: float = 0.10
    metric_threshold: float = 0.0
    top_k: int = 3
    random_seed: Optional[int] = None
    permutation_mode_stage2: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle'
    min_folds_stable: int = 3
