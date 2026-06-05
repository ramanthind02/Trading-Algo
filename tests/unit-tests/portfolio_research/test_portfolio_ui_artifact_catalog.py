from __future__ import annotations

from datetime import datetime
from pathlib import Path

from research.portfolio.config import (
    PortfolioFitMode,
    PortfolioResearchConfig,
    ResearchWindow,
)
from research.portfolio.shared.phase import PortfolioResearchPhase
from research.portfolio.ui.artifact_catalog import (
    WorkspaceArtifactCategory,
    _portfolio_friendly_panel_title,
    build_phase_report_sections,
    categorize_artifact_path,
    discover_artifact_records,
    group_artifact_records,
    phase_discovery_roots,
)
from lib.core.enums import Ticker, TimeFrame


def _config(tmp_path: Path) -> PortfolioResearchConfig:
    holdout = tmp_path / "holdout"
    (holdout / "test" / "portfolio").mkdir(parents=True)
    (holdout / "returns").mkdir(parents=True)
    (holdout / "strategies" / "demo" / "matplotlib").mkdir(parents=True)
    (holdout / "strategies" / "other" / "matplotlib").mkdir(parents=True)
    (holdout / "visualization" / "portfolio" / "matplotlib").mkdir(parents=True)
    (holdout / "test" / "portfolio" / "Portfolio_Holdout_test_tearsheet.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    (holdout / "strategies" / "demo" / "holdout_robustness_report.json").write_text("{}", encoding="utf-8")
    (holdout / "strategies" / "demo" / "matplotlib" / "holdout_equity_bands.png").write_bytes(b"png")
    (holdout / "strategies" / "other" / "matplotlib" / "holdout_equity_bands.png").write_bytes(b"png")
    (holdout / "visualization" / "portfolio" / "portfolio_holdout_report.json").write_text(
        '{"passed": true}', encoding="utf-8"
    )
    (holdout / "fold_manifest.json").write_text('{"folds": []}', encoding="utf-8")
    return PortfolioResearchConfig(
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


def test_weight_layer_report_friendly_title() -> None:
    title = _portfolio_friendly_panel_title(
        Path("portfolio_research/results/validation/portfolio/weight_layer_report.json")
    )
    assert title == "Weight layer — validation"


def test_build_phase_report_sections_with_weight_layer_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report_path = tmp_path / "validation" / "portfolio"
    report_path.mkdir(parents=True)
    (report_path / "weight_layer_report.json").write_text("{}", encoding="utf-8")
    records = discover_artifact_records(config, PortfolioResearchPhase.PORTFOLIO_TEST)
    grouped = group_artifact_records(records, PortfolioResearchPhase.PORTFOLIO_TEST)
    sections, _ = build_phase_report_sections(grouped, PortfolioResearchPhase.PORTFOLIO_TEST)
    weight_section = next(
        section
        for section in sections
        if section["category"] == WorkspaceArtifactCategory.WEIGHT_LAYER.value
    )
    assert any(
        panel["title"] == "Weight layer — validation"
        for panel in weight_section["panels"]
    )


def test_categorize_holdout_paths() -> None:
    assert (
        categorize_artifact_path(Path("holdout/test/portfolio/foo_tearsheet.html"))
        is WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET
    )
    assert (
        categorize_artifact_path(Path("holdout/strategies/x/holdout_robustness_report.json"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("holdout/visualization/portfolio/portfolio_holdout_report.json"))
        is WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS
    )


def test_strategy_matplotlib_panels_include_strategy_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    records = discover_artifact_records(config, PortfolioResearchPhase.STRATEGY_HOLDOUT)
    grouped = group_artifact_records(records, PortfolioResearchPhase.STRATEGY_HOLDOUT)
    sections, _ = build_phase_report_sections(grouped, PortfolioResearchPhase.STRATEGY_HOLDOUT)
    matplotlib = next(
        section
        for section in sections
        if section["category"] == WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value
    )
    titles = {panel["title"] for panel in matplotlib["panels"]}
    panel_ids = {panel["panel_id"] for panel in matplotlib["panels"]}
    assert "demo — Holdout Equity Bands" in titles
    assert "other — Holdout Equity Bands" in titles
    assert len(panel_ids) == len(matplotlib["panels"])


def test_portfolio_test_discovery_roots_include_research_windows(tmp_path: Path) -> None:
    for window in ("train", "validation", "test"):
        (tmp_path / window / "portfolio").mkdir(parents=True)
        (tmp_path / window / "portfolio" / f"Portfolio_{window.title()}_window_tearsheet.html").write_text(
            "<html></html>",
            encoding="utf-8",
        )
    config = _config(tmp_path)
    roots = phase_discovery_roots(config, PortfolioResearchPhase.PORTFOLIO_TEST)
    assert tmp_path / "train" in roots
    assert tmp_path / "validation" in roots
    assert tmp_path / "test" in roots


def test_build_phase_report_sections_has_panels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    records = discover_artifact_records(config, PortfolioResearchPhase.PORTFOLIO_HOLDOUT)
    grouped = group_artifact_records(records, PortfolioResearchPhase.PORTFOLIO_HOLDOUT)
    sections, default_id = build_phase_report_sections(grouped, PortfolioResearchPhase.PORTFOLIO_HOLDOUT)
    assert default_id == WorkspaceArtifactCategory.PORTFOLIO_ANALYTICS.value
    assert sections
    assert all("panels" in section for section in sections)
    assert phase_discovery_roots(config, PortfolioResearchPhase.STRATEGY_HOLDOUT) == (
        tmp_path / "holdout" / "strategies",
    )
    portfolio_test_roots = phase_discovery_roots(config, PortfolioResearchPhase.PORTFOLIO_TEST)
    assert tmp_path / "holdout" / "strategies" not in portfolio_test_roots
