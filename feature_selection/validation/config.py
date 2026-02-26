"""Configuration models for permutation test suite (T016)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from feature_selection.validation.objective_metrics import ObjectiveMetricSpec

PermutationModeStage2 = Literal['feature_shuffle', 'candle_shuffle']
OOSCandidateSource = Literal['stage2_passers', 'stable_intersection']
_DEFAULT_NREPS = 1000
_DEFAULT_ALPHA = 0.10
_DEFAULT_METRIC_THRESHOLD = 0.0
_DEFAULT_TOP_K = 3
_DEFAULT_PERMUTATION_MODE_STAGE2: PermutationModeStage2 = 'candle_shuffle'
_DEFAULT_MIN_FOLDS_STABLE = 3


@dataclass(frozen=True)
class InSamplePermutationConfig:
    """Stage 1/2 in-sample permutation settings."""

    nreps: int = _DEFAULT_NREPS
    alpha: float = _DEFAULT_ALPHA
    metric_threshold: float = _DEFAULT_METRIC_THRESHOLD
    permutation_mode_stage2: PermutationModeStage2 = _DEFAULT_PERMUTATION_MODE_STAGE2
    n_jobs_stage2_reps: int = 1
    run_stage1: bool = True
    run_stage2: bool = True
    run_stage3_walkforward: bool = True  # False in in-sample phase; True in walkforward phase


@dataclass(frozen=True)
class WalkforwardPermutationConfig:
    """Stage 3 walkforward stability settings."""

    top_k: int = _DEFAULT_TOP_K
    min_folds_stable: int = _DEFAULT_MIN_FOLDS_STABLE


@dataclass(frozen=True)
class OutOfSamplePermutationConfig:
    """Out-of-sample permutation settings.

    When run_oos_permutation is False (e.g. in-sample phase), Phase 3 is skipped
    and ensemble candidates are Stage 2 passers (or Stage 2 ∩ stable if Stage 3 ran).
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
    """Configuration for permutation test suite (T013-T017)."""

    in_sample: InSamplePermutationConfig
    walkforward: WalkforwardPermutationConfig
    out_of_sample: OutOfSamplePermutationConfig
    report: PermutationReportConfig
    random_seed: Optional[int]

    def __init__(
        self,
        nreps: int = _DEFAULT_NREPS,
        alpha: float = _DEFAULT_ALPHA,
        metric_threshold: float = _DEFAULT_METRIC_THRESHOLD,
        top_k: int = _DEFAULT_TOP_K,
        random_seed: Optional[int] = None,
        permutation_mode_stage2: PermutationModeStage2 = _DEFAULT_PERMUTATION_MODE_STAGE2,
        min_folds_stable: int = _DEFAULT_MIN_FOLDS_STABLE,
        n_jobs_stage2_reps: int = 1,
        run_stage1: bool = True,
        run_stage2: bool = True,
        run_stage3_walkforward: bool = True,
        *,
        in_sample: InSamplePermutationConfig | None = None,
        walkforward: WalkforwardPermutationConfig | None = None,
        out_of_sample: OutOfSamplePermutationConfig | None = None,
        report: PermutationReportConfig | None = None,
        objective_metric: ObjectiveMetricSpec | None = None,
    ) -> None:
        if in_sample is not None and any(
            (
                nreps != _DEFAULT_NREPS,
                alpha != _DEFAULT_ALPHA,
                metric_threshold != _DEFAULT_METRIC_THRESHOLD,
                permutation_mode_stage2 != _DEFAULT_PERMUTATION_MODE_STAGE2,
                run_stage1 is not True,
                run_stage2 is not True,
            ),
        ):
            raise ValueError(
                'Provide in_sample or legacy in-sample scalar args, not both.',
            )

        if walkforward is not None and any(
            (
                top_k != _DEFAULT_TOP_K,
                min_folds_stable != _DEFAULT_MIN_FOLDS_STABLE,
            ),
        ):
            raise ValueError(
                'Provide walkforward or legacy walkforward scalar args, not both.',
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
            permutation_mode_stage2=permutation_mode_stage2,
            n_jobs_stage2_reps=n_jobs_stage2_reps,
            run_stage1=run_stage1,
            run_stage2=run_stage2,
            run_stage3_walkforward=run_stage3_walkforward,
        )
        resolved_walkforward = walkforward or WalkforwardPermutationConfig(
            top_k=top_k,
            min_folds_stable=min_folds_stable,
        )
        resolved_report = report or PermutationReportConfig()

        object.__setattr__(self, 'in_sample', resolved_in_sample)
        object.__setattr__(self, 'walkforward', resolved_walkforward)
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
    def top_k(self) -> int:
        return self.walkforward.top_k

    @property
    def permutation_mode_stage2(self) -> PermutationModeStage2:
        return self.in_sample.permutation_mode_stage2

    @property
    def n_jobs_stage2_reps(self) -> int:
        return self.in_sample.n_jobs_stage2_reps

    @property
    def run_stage3_walkforward(self) -> bool:
        return self.in_sample.run_stage3_walkforward

    @property
    def run_stage1(self) -> bool:
        return self.in_sample.run_stage1

    @property
    def run_stage2(self) -> bool:
        return self.in_sample.run_stage2

    @property
    def min_folds_stable(self) -> int:
        return self.walkforward.min_folds_stable

    @property
    def objective_metric(self) -> ObjectiveMetricSpec:
        return self.out_of_sample.objective_metric

    @property
    def run_oos_permutation(self) -> bool:
        return self.out_of_sample.run_oos_permutation


@dataclass(frozen=True)
class ValidationConfig:
    """Legacy validation config for tests; full pipeline uses PermutationTestConfig."""

    feature_type: Literal["continuous", "rule_based"]
    n_permutations: int = 1000
    confidence_level: float = 0.95
    min_sharpe_threshold: float = 0.5
    random_seed: Optional[int] = None
