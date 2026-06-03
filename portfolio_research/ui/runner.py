"""Orchestration for portfolio research UI phase runs."""
from __future__ import annotations

from portfolio_research.config import PortfolioResearchConfig
from portfolio_research.holdout.pipeline import run_portfolio_holdout_pipeline
from portfolio_research.holdout.returns_export import load_return_matrices
from portfolio_research.holdout.rolling_eval import run_holdout_evaluation
from portfolio_research.pipelines.portfolio_test import run_portfolio_test_pipeline
from portfolio_research.holdout.strategy_monitoring import run_strategy_holdout_monitoring
from portfolio_research.shared.phase import PortfolioResearchPhase
from portfolio_research.ui.contracts import PortfolioResearchUiRequest, PhaseRunPlan
from portfolio_research.ui.pipeline import run_full_portfolio_research_pipeline
from portfolio_research.ui.planner import apply_ui_request


def render_phase_plan(plan: PhaseRunPlan) -> str:
    lines = [
        plan.title,
        "",
        plan.description,
        "",
        *(f"- {field.label}: {field.value}" for field in plan.fields),
        "",
        "What will run:",
        *(f"- {step}" for step in plan.will_run),
        "",
        f"Command: {plan.command}",
    ]
    return "\n".join(lines)


def execute_phase_request(
    config: PortfolioResearchConfig,
    request: PortfolioResearchUiRequest,
) -> int:
    configured = apply_ui_request(config, request)
    match request.phase:
        case PortfolioResearchPhase.PORTFOLIO_TEST:
            run_portfolio_test_pipeline(configured)
            run_holdout_evaluation(configured, emit_tearsheets=False)
            print(
                "Portfolio test complete. Tearsheets: "
                f"{configured.output_root}/{{train,validation,test}}/portfolio; "
                f"holdout returns: {configured.output_root / 'holdout' / 'returns'}"
            )
        case PortfolioResearchPhase.STRATEGY_HOLDOUT:
            matrices = load_return_matrices(configured.output_root)
            run_strategy_holdout_monitoring(
                configured,
                research_strategy_returns=matrices["strategy_research"],  # type: ignore[arg-type]
                holdout_strategy_returns=matrices["strategy_holdout"],  # type: ignore[arg-type]
            )
            print("Strategy holdout monitoring complete.")
        case PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
            run_portfolio_holdout_pipeline(configured)
            print("Portfolio holdout pipeline complete.")
        case PortfolioResearchPhase.FULL_PIPELINE:
            run_full_portfolio_research_pipeline(configured)
        case PortfolioResearchPhase.PROP_FIRM_REPORTS:
            run_portfolio_test_pipeline(configured)
            print(
                "Prop firm reports complete. "
                f"Artifacts: {configured.output_root}/{{train,validation,test}}/prop_firm/"
            )
    return 0
