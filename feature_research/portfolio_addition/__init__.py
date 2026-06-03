"""Portfolio-addition phase public API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from feature_research.pipelines.oos import (
    run_oos_pipeline as _run_oos_pipeline,
    run_oos_pipeline_with_bundle as _run_oos_pipeline_with_bundle,
)
from feature_research.portfolio_addition.gate_runner import (
    run_and_write_portfolio_addition_gate,
    run_portfolio_addition_gate_pipeline,
)
from feature_research.shared import (
    FeatureResearchPhase,
    OosCorrelationBundle,
    PortfolioAdditionBundle,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


PHASE: Final[FeatureResearchPhase] = FeatureResearchPhase.PORTFOLIO_ADDITION


def run_portfolio_addition_pipeline(
    config: "ResearchConfig",
) -> "WalkforwardRunReport":
    """Run the current OOS phase through the new portfolio-addition package."""

    return _run_oos_pipeline(config)


def run_portfolio_addition_pipeline_with_bundle(
    config: "ResearchConfig",
) -> tuple["WalkforwardRunReport", PortfolioAdditionBundle | None]:
    """Run portfolio-addition evaluation and return the correlation sidecar bundle."""

    return _run_oos_pipeline_with_bundle(config)


__all__ = [
    "OosCorrelationBundle",
    "PHASE",
    "PortfolioAdditionBundle",
    "run_and_write_portfolio_addition_gate",
    "run_portfolio_addition_gate_pipeline",
    "run_portfolio_addition_pipeline",
    "run_portfolio_addition_pipeline_with_bundle",
]
