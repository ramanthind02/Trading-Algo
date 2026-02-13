"""Validation configuration."""
from dataclasses import dataclass
from typing import Literal


FeatureType = Literal['continuous', 'rule_based']


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for validation pipeline."""

    feature_type: FeatureType
    n_permutations: int = 1000
    confidence_level: float = 0.95
    min_sharpe_threshold: float = 0.5
    min_t_stat_threshold: float = 2.0
    neighbor_steps: int = 1  # For grid smoothing
    top_k_per_fold: int = 5  # Walkforward stability
    random_seed: int | None = None
