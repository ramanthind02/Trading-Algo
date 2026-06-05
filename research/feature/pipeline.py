"""Compatibility façade for feature_research phase pipelines.

Public run functions keep stable import paths while canonical phase entrypoints
live in `feature_research.exploration`, `feature_research.validation`, and
`feature_research.portfolio_addition`.
"""

from research.feature.exploration import (
    run_exploration_permutation_pipeline as run_permutation_pipeline,
    run_exploration_pipeline as run_eda_pipeline,
    run_exploration_robustness_pipeline as run_robustness_pipeline,
    write_exploration_permutation_summary as write_permutation_summary,
    write_exploration_robustness_summary as write_robustness_summary,
)
from research.feature.portfolio_addition import (
    run_portfolio_addition_pipeline as run_oos_pipeline,
    run_portfolio_addition_pipeline_with_bundle as run_oos_pipeline_with_bundle,
)
from research.feature.validation import run_validation_pipeline

__all__ = [
    "run_eda_pipeline",
    "run_validation_pipeline",
    "run_oos_pipeline",
    "run_oos_pipeline_with_bundle",
    "run_robustness_pipeline",
    "write_robustness_summary",
    "run_permutation_pipeline",
    "write_permutation_summary",
]
