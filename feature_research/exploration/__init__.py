"""Exploration-phase public API."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from feature_research.shared import FeatureResearchPhase, ParameterGrid, PhaseArtifactMap
from feature_research.shared.phase_api import (
    normalize_parameter_grid,
    normalize_phase_artifact_map,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
    from feature_selection.validation.reports import PermutationTestSuite
    from feature_research.exploration.orchestrate import ExplorationPhaseResult
    from feature_research.pipelines.robustness import InSampleRobustnessReport

PHASE: Final[FeatureResearchPhase] = FeatureResearchPhase.EXPLORATION

_LAZY_ORCHESTRATE_EXPORTS: Final[frozenset[str]] = frozenset(
    {
        "ExplorationPhaseResult",
        "execute_exploration_phase",
        "exploration_permutation_enabled",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ORCHESTRATE_EXPORTS:
        from feature_research.exploration import orchestrate as _orchestrate

        return getattr(_orchestrate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_exploration_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> PhaseArtifactMap:
    """Run the current in-sample exploration pipeline via the new phase package."""
    from feature_research.pipelines.in_sample import run_eda_pipeline

    return run_eda_pipeline(config, output_dir)


def run_exploration_robustness_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> tuple["InSampleRobustnessReport", ParameterGrid]:
    """Run in-sample robustness from the exploration phase package."""
    from feature_research.pipelines.robustness import run_robustness_pipeline

    report, parameter_grid = run_robustness_pipeline(config, output_dir)
    return report, normalize_parameter_grid(parameter_grid)


def write_exploration_robustness_summary(
    report: "InSampleRobustnessReport",
    output_dir: Path,
) -> PhaseArtifactMap:
    """Write exploration robustness artifacts with the shared path mapping type."""
    from feature_research.pipelines.robustness import write_robustness_summary

    return normalize_phase_artifact_map(write_robustness_summary(report, output_dir))


def run_exploration_permutation_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
    *,
    selected_combo_name: str | None = None,
) -> tuple["PermutationTestSuite", ParameterGrid]:
    """Run in-sample permutation testing from the exploration phase package."""
    from feature_research.pipelines.permutation import run_permutation_pipeline

    suite, parameter_grid = run_permutation_pipeline(
        config,
        output_dir,
        selected_combo_name=selected_combo_name,
    )
    return suite, normalize_parameter_grid(parameter_grid)


def write_exploration_permutation_summary(
    suite: "PermutationTestSuite",
    output_dir: Path,
    *,
    objective_metric: "ObjectiveMetricSpec | None" = None,
    param_grid: ParameterGrid | None = None,
    full_grid_permutation: object | None = None,
    selection_metric_label: str | None = None,
) -> tuple[Path, Path]:
    """Write exploration permutation artifacts with the legacy summary signature."""
    from feature_research.pipelines.permutation import write_permutation_summary

    return write_permutation_summary(
        suite,
        output_dir,
        objective_metric=objective_metric,
        param_grid=param_grid,
        full_grid_permutation=full_grid_permutation,
        selection_metric_label=selection_metric_label,
    )


__all__ = [
    "PHASE",
    "ExplorationPhaseResult",
    "execute_exploration_phase",
    "exploration_permutation_enabled",
    "run_exploration_permutation_pipeline",
    "run_exploration_pipeline",
    "write_exploration_permutation_summary",
    "run_exploration_robustness_pipeline",
    "write_exploration_robustness_summary",
]
