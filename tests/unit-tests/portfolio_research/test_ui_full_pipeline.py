from __future__ import annotations

from portfolio_research.config import load_config
from portfolio_research.shared.phase import PortfolioResearchPhase, artifact_view_phases
from portfolio_research.ui.planner import build_phase_plan, build_ui_defaults, build_ui_request


def test_full_pipeline_plan_lists_all_stage_steps() -> None:
    config = load_config()
    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=PortfolioResearchPhase.FULL_PIPELINE.value,
        tickers_text="ES,NQ",
        fallback_tickers=defaults.default_tickers,
    )
    plan = build_phase_plan(config, request)
    assert plan.phase is PortfolioResearchPhase.FULL_PIPELINE
    assert plan.title == "Full portfolio research pipeline"
    assert len(plan.will_run) >= 9
    assert "full_pipeline" in plan.command
    assert str(config.output_root) in plan.output_path.as_posix()


def test_artifact_view_phases_excludes_full_pipeline() -> None:
    phases = artifact_view_phases()
    assert PortfolioResearchPhase.FULL_PIPELINE not in phases
    assert len(phases) == 3
