from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from feature_research.pipelines._shared import _run_evaluation_pipeline

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


def run_validation_pipeline(
    config: "ResearchConfig",
    output_dir: Path | None = None,
) -> "WalkforwardRunReport":
    """Run validation evaluation pipeline.

    This is a thin wrapper around the shared evaluation pipeline implementation,
    configured for the validation phase.

    Parameters
    ----------
    config : ResearchConfig
        Research configuration with validation_window.
    output_dir : Path, optional
        Override output directory. If None, computed from config.output_root.

    Returns
    -------
    WalkforwardRunReport
        Walkforward research report with folds, selection summary, and stability plots.
    """
    report, _bundle = _run_evaluation_pipeline(
        phase="validation",
        config=config,
        output_dir=str(output_dir) if output_dir is not None else None,
    )
    return report
