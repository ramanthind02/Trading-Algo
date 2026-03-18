from __future__ import annotations

from typing import TYPE_CHECKING

from feature_research.pipelines._shared import _run_evaluation_pipeline

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


def run_oos_pipeline(config: "ResearchConfig") -> "WalkforwardRunReport":
    """Run OOS evaluation pipeline.

    This is a thin wrapper around the shared evaluation pipeline implementation,
    configured for the OOS phase. It uses _effective_oos_window which combines
    validation and OOS windows if both are present.

    Parameters
    ----------
    config : ResearchConfig
        Research configuration with oos_window and optional validation_window.

    Returns
    -------
    WalkforwardRunReport
        Walkforward research report with folds, selection summary, and stability plots.
    """
    return _run_evaluation_pipeline(phase="oos", config=config)
