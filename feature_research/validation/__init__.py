"""Validation-phase public API."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from feature_research.pipelines.validation import (
    run_validation_pipeline as _run_validation_pipeline,
)
from feature_research.shared import FeatureResearchPhase
from feature_research.validation.robustness_runner import (
    run_and_write_validation_robustness,
    run_validation_robustness_pipeline,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


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
