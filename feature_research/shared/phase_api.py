"""Pure helpers for phase-level public APIs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from feature_research.shared.contracts import ParameterGrid, PhaseArtifactMap


def normalize_parameter_grid(
    parameter_grid: Sequence[Mapping[str, object]],
) -> ParameterGrid:
    """Return a stable JSON-friendly copy of a parameter grid."""

    return [{str(key): value for key, value in params.items()} for params in parameter_grid]


def normalize_phase_artifact_map(
    artifact_paths: Mapping[str, Path],
) -> PhaseArtifactMap:
    """Return a copied artifact-path mapping with normalized string keys."""

    return {str(key): Path(value) for key, value in artifact_paths.items()}


__all__ = ["normalize_parameter_grid", "normalize_phase_artifact_map"]
