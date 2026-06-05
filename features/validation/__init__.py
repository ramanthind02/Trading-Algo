"""Feature validation package for permutation testing (T013-T017)."""

from features.validation.config import (
    InSamplePermutationConfig,
    OutOfSamplePermutationConfig,
    PermutationReportConfig,
    PermutationTestConfig,
)
from features.validation.objective_metrics import (
    ObjectiveMetricSpec,
    resolve_objective_metric,
)

__all__ = [
    'InSamplePermutationConfig',
    'OutOfSamplePermutationConfig',
    'PermutationReportConfig',
    'PermutationTestConfig',
    'ObjectiveMetricSpec',
    'resolve_objective_metric',
]
