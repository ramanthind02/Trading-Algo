from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_research.config import (
    PortfolioFitMode,
    PortfolioResearchConfig,
    ResearchWindow,
)
from portfolio_research.shared.phase import PortfolioResearchPhase
from portfolio_research.ui.monitoring_dashboard import (
    build_monitoring_dashboard,
    inject_monitoring_section,
)
from utils.core.enums import Ticker, TimeFrame


def _config(tmp_path: Path) -> PortfolioResearchConfig:
    return PortfolioResearchConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2026, 5, 13),
        use_cache=False,
        train_window=ResearchWindow(datetime(2000, 1, 1), datetime(2017, 12, 31)),
        validation_window=ResearchWindow(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        test_window=ResearchWindow(datetime(2023, 1, 1), datetime(2026, 5, 13)),
        ensemble_dirs={"demo": "vault/D/demo"},
        portfolio_fit_mode=PortfolioFitMode.SINGLE_FIT,
        output_root=tmp_path,
    )


def test_build_monitoring_dashboard_from_rollup(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout"
    strategies = holdout / "strategies" / "demo"
    strategies.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy": "demo",
                "traffic_light": "YELLOW",
                "tests_failed": 2,
                "advisory_weight_fraction": 0.5,
                "effective_weight_fraction": 0.5,
                "eval_start": "2025-05-13",
                "eval_end": "2026-05-13",
                "traffic_light_prev_month": "GREEN",
                "traffic_light_2mo_ago": "GREEN",
                "all_passed": False,
                "reference_sigma_method": "pooled_train_validation",
                "sigma_relative_shift": 0.1,
            }
        ]
    ).to_csv(holdout / "monitoring_rollup.csv", index=False)
    pd.DataFrame(
        [
            {
                "as_of_date": "2025-12-31",
                "traffic_light": "GREEN",
                "tests_failed": 1,
            }
        ]
    ).to_csv(strategies / "monitoring_history.csv", index=False)

    config = _config(tmp_path)
    dashboard = build_monitoring_dashboard(
        config,
        PortfolioResearchPhase.STRATEGY_HOLDOUT,
        repo_root=tmp_path,
    )
    assert dashboard is not None
    assert dashboard["strategy_count"] == 1
    assert dashboard["traffic_light_counts"]["YELLOW"] == 1
    assert len(dashboard["histories"]) == 1

    sections = [{"id": "robustness", "label": "Robustness"}]
    enriched, default_id = inject_monitoring_section(
        sections,
        dashboard=dashboard,
        default_section_id="robustness",
    )
    assert default_id == "monitoring_status"
    assert enriched[0]["id"] == "monitoring_status"
    assert enriched[0]["panels"][0]["view_type"] == "monitoring_rollup"


def test_build_monitoring_dashboard_skips_empty_history_csv(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout"
    strategies = holdout / "strategies" / "demo"
    strategies.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy": "demo",
                "traffic_light": "GREEN",
                "tests_failed": 0,
                "advisory_weight_fraction": 1.0,
                "effective_weight_fraction": 1.0,
                "eval_start": "2025-05-13",
                "eval_end": "2026-05-13",
                "traffic_light_prev_month": "GREEN",
                "traffic_light_2mo_ago": "GREEN",
                "all_passed": True,
                "reference_sigma_method": "pooled_train_validation",
                "sigma_relative_shift": 0.0,
            }
        ]
    ).to_csv(holdout / "monitoring_rollup.csv", index=False)
    (strategies / "monitoring_history.csv").write_text("\n", encoding="utf-8")

    dashboard = build_monitoring_dashboard(
        _config(tmp_path),
        PortfolioResearchPhase.STRATEGY_HOLDOUT,
        repo_root=tmp_path,
    )
    assert dashboard is not None
    assert dashboard["histories"] == []
