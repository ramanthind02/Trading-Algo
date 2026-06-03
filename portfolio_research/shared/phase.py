"""Canonical portfolio research workflow phases."""
from __future__ import annotations

from enum import Enum


class PortfolioResearchPhase(str, Enum):
    """Docs-aligned phases for portfolio research workspace."""

    PORTFOLIO_TEST = "portfolio_test"
    STRATEGY_HOLDOUT = "strategy_holdout"
    PORTFOLIO_HOLDOUT = "portfolio_holdout"
    FULL_PIPELINE = "full_pipeline"
    PROP_FIRM_REPORTS = "prop_firm_reports"


def artifact_view_phases() -> tuple[PortfolioResearchPhase, ...]:
    """Phases that have a results-dashboard tab (excludes orchestration-only runs)."""

    return (
        PortfolioResearchPhase.PORTFOLIO_TEST,
        PortfolioResearchPhase.STRATEGY_HOLDOUT,
        PortfolioResearchPhase.PORTFOLIO_HOLDOUT,
        PortfolioResearchPhase.PROP_FIRM_REPORTS,
    )
