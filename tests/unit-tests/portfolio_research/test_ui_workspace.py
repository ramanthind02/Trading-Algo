from __future__ import annotations

from datetime import datetime
from pathlib import Path

from research.portfolio.config import PortfolioFitMode, PortfolioResearchConfig, ResearchWindow
from research.portfolio.shared.phase import PortfolioResearchPhase
from research.portfolio.ui.contracts import PortfolioResearchUiRequest
from research.portfolio.ui.workspace import build_workspace_view
from lib.core.enums import Ticker, TimeFrame


def test_build_workspace_view_payload_shape(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout"
    (holdout / "visualization" / "portfolio").mkdir(parents=True)
    (holdout / "visualization" / "portfolio" / "portfolio_holdout_report.json").write_text(
        '{"passed": true, "correlation_realisation": {"mean_pairwise_corr": 0.2}}',
        encoding="utf-8",
    )
    config = PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2026, 1, 1),
        use_cache=False,
        train_window=ResearchWindow(datetime(2000, 1, 1), datetime(2017, 12, 31)),
        validation_window=ResearchWindow(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2025, 12, 31)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.ROLLING_HOLDOUT,
        output_root=tmp_path,
    )
    request = PortfolioResearchUiRequest(
        phase=PortfolioResearchPhase.PORTFOLIO_HOLDOUT,
        tickers=(Ticker.ES,),
        portfolio_fit_mode=PortfolioFitMode.ROLLING_HOLDOUT,
    )
    view = build_workspace_view(config, request)
    phase = next(entry for entry in view["phases"] if entry["phase"] == "portfolio_holdout")
    assert phase["default_section_id"] is not None
    assert phase["summary_cards"]
    assert phase["report_sections"]
    assert phase["has_results"] is True
