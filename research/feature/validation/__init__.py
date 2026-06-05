"""Validation-phase public API."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from research.feature.pipelines.validation import (
    run_validation_pipeline as _run_validation_pipeline,
)
from research.feature.shared import FeatureResearchPhase
from research.feature.validation.robustness_runner import (
    run_and_write_validation_robustness,
    run_validation_robustness_pipeline,
)

if TYPE_CHECKING:
    from research.feature.config import ResearchConfig
    from research.evaluation.walkforward.runner import WalkforwardRunReport


PHASE: Final[FeatureResearchPhase] = FeatureResearchPhase.VALIDATION


def run_validation_pipeline(
    config: "ResearchConfig",
    output_dir: Path | None = None,
) -> "WalkforwardRunReport":
    """Run the current validation pipeline via the phase package."""

    return _run_validation_pipeline(config, output_dir)


__all__ = [
    "PHASE",
    "run_and_write_validation_robustness",
    "run_validation_pipeline",
    "run_validation_robustness_pipeline",
]
