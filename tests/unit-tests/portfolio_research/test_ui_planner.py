from __future__ import annotations

from portfolio_research.config import load_config
from portfolio_research.shared.phase import PortfolioResearchPhase
from portfolio_research.ui.planner import build_phase_plan, build_ui_defaults, build_ui_request


def test_build_portfolio_phase_plan() -> None:
    config = load_config()
    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=PortfolioResearchPhase.PORTFOLIO_HOLDOUT.value,
        tickers_text="ES,NQ",
        fallback_tickers=defaults.default_tickers,
    )
    plan = build_phase_plan(config, request)
    assert plan.phase == PortfolioResearchPhase.PORTFOLIO_HOLDOUT
    assert "holdout" in plan.output_path.as_posix()
