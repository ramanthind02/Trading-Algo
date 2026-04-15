"""Compatibility façade for feature_research phase pipelines.

Public run functions keep stable import paths while implementation lives in
`feature_research.pipelines.*`.
"""

from feature_research.pipelines.in_sample import run_eda_pipeline
from feature_research.pipelines.oos import run_oos_pipeline, run_oos_pipeline_with_bundle
from feature_research.pipelines.permutation import run_permutation_pipeline, write_permutation_summary
from feature_research.pipelines.validation import run_validation_pipeline

__all__ = [
    "run_eda_pipeline",
    "run_validation_pipeline",
    "run_oos_pipeline",
    "run_oos_pipeline_with_bundle",
    "run_permutation_pipeline",
    "write_permutation_summary",
]
