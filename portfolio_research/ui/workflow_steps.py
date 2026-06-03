"""Docs-driven workflow metadata for portfolio research UI."""
from __future__ import annotations

from portfolio_research.shared.phase import PortfolioResearchPhase
from portfolio_research.ui.contracts import UiWorkflowStep


def workflow_steps() -> tuple[UiWorkflowStep, ...]:
    return (
        UiWorkflowStep(
            phase=PortfolioResearchPhase.FULL_PIPELINE,
            title="Run all stages",
            summary=(
                "One command: portfolio test tearsheets, per-strategy holdout monitoring, "
                "and portfolio holdout analytics."
            ),
            gate="Recommended default. Runs stages 1–3 sequentially.",
            docs_path="docs/SaaS/robustness_tests/portfolio_holdout.md",
        ),
        UiWorkflowStep(
            phase=PortfolioResearchPhase.PORTFOLIO_TEST,
            title="1. Portfolio Test",
            summary=(
                "Portfolio QuantStats tearsheets on train, validation, and test windows, "
                "FundedNext prop-firm account simulation reports, and holdout return matrices."
            ),
            gate=(
                "Artifacts: train/validation/test tearsheets, "
                "{phase}/prop_firm/fundednext/ HTML reports, and holdout/returns CSVs."
            ),
            docs_path="docs/SaaS/robustness_tests/portfolio_holdout.md",
        ),
        UiWorkflowStep(
            phase=PortfolioResearchPhase.STRATEGY_HOLDOUT,
            title="2. Strategy Holdout Monitoring",
            summary=(
                "Validation-parity monitoring charts (CUSUM, rolling Sharpe, equity bands) "
                "for each vault strategy on the holdout window."
            ),
            gate="Requires portfolio test return CSVs.",
            docs_path="docs/SaaS/robustness_tests/monitoring.md",
        ),
        UiWorkflowStep(
            phase=PortfolioResearchPhase.PORTFOLIO_HOLDOUT,
            title="3. Portfolio Holdout",
            summary=(
                "Portfolio-level holdout analytics: correlation realisation, drawdown "
                "correlation, IDM calibration, and contribution concentration."
            ),
            gate="Advisory diagnostics only; culling follows monitoring doctrine.",
            docs_path="docs/SaaS/robustness_tests/portfolio_holdout.md",
        ),
        UiWorkflowStep(
            phase=PortfolioResearchPhase.PROP_FIRM_REPORTS,
            title="4. Prop Firm Reports",
            summary=(
                "FundedNext CFD multi-account portfolio simulation for train, validation, "
                "and test windows. Produces HTML/MD reports and CSV exports."
            ),
            gate=(
                "Requires Portfolio Test to have run. Reports under "
                "{phase}/prop_firm/fundednext/."
            ),
            docs_path="docs/library/Testing/prop_firms.md",
        ),
    )
