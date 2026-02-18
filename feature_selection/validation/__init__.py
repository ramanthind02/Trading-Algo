"""Feature validation package for permutation testing (T013-T017)."""

from feature_selection.validation.config import (
    InSamplePermutationConfig,
    OutOfSamplePermutationConfig,
    PermutationReportConfig,
    PermutationTestConfig,
    WalkforwardPermutationConfig,
)
from feature_selection.validation.objective_metrics import (
    ObjectiveMetricSpec,
    resolve_objective_metric,
)

__all__ = [
    'InSamplePermutationConfig',
    'OutOfSamplePermutationConfig',
    'PermutationReportConfig',
    'PermutationTestConfig',
    'WalkforwardPermutationConfig',
    'ObjectiveMetricSpec',
    'resolve_objective_metric',
]
