from __future__ import annotations

from typing import TYPE_CHECKING

from feature_research.pipelines._shared import _run_evaluation_pipeline

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from feature_research.shared.contracts import OosCorrelationBundle
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


def run_oos_pipeline(config: "ResearchConfig") -> "WalkforwardRunReport":
    """Run the portfolio-addition evaluation pipeline on train + validation data."""

    report, _bundle = _run_evaluation_pipeline(phase="oos", config=config)
    return report


def run_oos_pipeline_with_bundle(
    config: "ResearchConfig",
) -> "tuple[WalkforwardRunReport, OosCorrelationBundle | None]":
    """Run portfolio-addition evaluation and return the correlation sidecar bundle."""

    report, bundle = _run_evaluation_pipeline(phase="oos", config=config)
    return report, bundle
