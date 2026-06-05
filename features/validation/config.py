"""Configuration models for permutation test suite (T016)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from features.validation.objective_metrics import ObjectiveMetricSpec

OOSCandidateSource = Literal['stage2_passers', 'stable_intersection']
_DEFAULT_NREPS = 1000
_DEFAULT_ALPHA = 0.10
_DEFAULT_METRIC_THRESHOLD = 0.0


@dataclass(frozen=True)
class InSamplePermutationConfig:
    """In-sample vector-shuffle permutation (no pipeline second stage)."""

    nreps: int = _DEFAULT_NREPS
    alpha: float = _DEFAULT_ALPHA
    metric_threshold: float = _DEFAULT_METRIC_THRESHOLD
    run_stage1: bool = True
    # Reserved for callers that still thread pipeline-permutation reps; vector shuffle paths
    # ignore these and run sequential NumPy loops.
    n_jobs_combos: int = 8
    n_jobs_reps: int = 8


@dataclass(frozen=True)
class OutOfSamplePermutationConfig:
    """Out-of-sample permutation settings.

    When run_oos_permutation is False (e.g. in-sample phase), Phase 3 is skipped
    and ensemble candidates are vector-shuffle passers. ``candidate_source`` still
    uses the literal ``stage2_passers`` for backward compatibility.
    """

    objective_metric: ObjectiveMetricSpec = field(
        default_factory=lambda: ObjectiveMetricSpec(builtin='sharpe'),
    )
    candidate_source: OOSCandidateSource = 'stage2_passers'
    run_oos_permutation: bool = False


@dataclass(frozen=True)
class PermutationReportConfig:
    """Permutation report formatting settings."""

    decimal_places: int = 1


@dataclass(frozen=True, init=False)
class PermutationTestConfig:
    """Configuration for permutation test suite (vector shuffle only; no walkforward Stage 3)."""

    in_sample: InSamplePermutationConfig
    out_of_sample: OutOfSamplePermutationConfig
    report: PermutationReportConfig
    random_seed: Optional[int]

    def __init__(
        self,
        nreps: int = _DEFAULT_NREPS,
        alpha: float = _DEFAULT_ALPHA,
        metric_threshold: float = _DEFAULT_METRIC_THRESHOLD,
        random_seed: Optional[int] = None,
        run_stage1: bool = True,
        *,
        in_sample: InSamplePermutationConfig | None = None,
        out_of_sample: OutOfSamplePermutationConfig | None = None,
        report: PermutationReportConfig | None = None,
        objective_metric: ObjectiveMetricSpec | None = None,
        # Ignored (Stage 3 removed): top_k, min_folds_stable, run_stage3_walkforward, walkforward
        top_k: int = 3,
        min_folds_stable: int = 3,
        run_stage3_walkforward: bool = False,
        walkforward: object = None,
    ) -> None:
        if in_sample is not None and any(
            (
                nreps != _DEFAULT_NREPS,
                alpha != _DEFAULT_ALPHA,
                metric_threshold != _DEFAULT_METRIC_THRESHOLD,
                run_stage1 is not True,
            ),
        ):
            raise ValueError(
                'Provide in_sample or legacy in-sample scalar args, not both.',
            )

        if out_of_sample is not None and objective_metric is not None:
            raise ValueError('Provide objective_metric through out_of_sample or objective_metric, not both.')

        resolved_out_of_sample = out_of_sample or OutOfSamplePermutationConfig(
            objective_metric=objective_metric or ObjectiveMetricSpec(builtin='sharpe'),
        )
        resolved_in_sample = in_sample or InSamplePermutationConfig(
            nreps=nreps,
            alpha=alpha,
            metric_threshold=metric_threshold,
            run_stage1=run_stage1,
        )
        resolved_report = report or PermutationReportConfig()

        object.__setattr__(self, 'in_sample', resolved_in_sample)
        object.__setattr__(self, 'out_of_sample', resolved_out_of_sample)
        object.__setattr__(self, 'report', resolved_report)
        object.__setattr__(self, 'random_seed', random_seed)

    @property
    def nreps(self) -> int:
        return self.in_sample.nreps

    @property
    def alpha(self) -> float:
        return self.in_sample.alpha

    @property
    def metric_threshold(self) -> float:
        return self.in_sample.metric_threshold

    @property
    def run_stage1(self) -> bool:
        return self.in_sample.run_stage1

    @property
    def n_jobs_combos(self) -> int:
        return self.in_sample.n_jobs_combos

    @property
    def n_jobs_reps(self) -> int:
        return self.in_sample.n_jobs_reps

    @property
    def objective_metric(self) -> ObjectiveMetricSpec:
        return self.out_of_sample.objective_metric

    @property
    def run_oos_permutation(self) -> bool:
        return self.out_of_sample.run_oos_permutation


@dataclass(frozen=True)
class ValidationConfig:
    """Legacy validation config for tests; full pipeline uses PermutationTestConfig."""

    feature_type: Literal["signed_signal"]
    n_permutations: int = 1000
    confidence_level: float = 0.95
    min_sharpe_threshold: float = 0.5
    random_seed: Optional[int] = None
