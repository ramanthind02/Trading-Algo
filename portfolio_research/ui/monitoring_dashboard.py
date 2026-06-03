"""Build monitoring dashboard payloads for the portfolio research workspace UI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio_research.config import PortfolioResearchConfig
from portfolio_research.shared.phase import PortfolioResearchPhase
from portfolio_research.shared.visualization_paths import holdout_root, holdout_strategy_dir


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_monitoring_csv(path: Path) -> pd.DataFrame | None:
    """Load a monitoring CSV, returning ``None`` when the file is missing or empty."""
    if not path.is_file():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty:
        return None
    return frame


def _load_rollup(repo_root: Path, output_root: Path) -> tuple[pd.DataFrame, str] | None:
    rollup_path = holdout_root(output_root) / "monitoring_rollup.csv"
    frame = _read_monitoring_csv(rollup_path)
    if frame is None:
        return None
    return frame, _relative(repo_root, rollup_path)


def _load_histories(
    repo_root: Path,
    output_root: Path,
    strategies: list[str],
) -> list[dict[str, object]]:
    histories: list[dict[str, object]] = []
    for strategy in strategies:
        history_path = holdout_strategy_dir(output_root, strategy) / "monitoring_history.csv"
        frame = _read_monitoring_csv(history_path)
        if frame is None:
            continue
        histories.append(
            {
                "strategy": strategy,
                "relative_path": _relative(repo_root, history_path),
                "rows": frame.fillna("").astype(str).to_dict(orient="records"),
                "row_count": int(frame.shape[0]),
            }
        )
    return histories


def build_monitoring_dashboard(
    config: PortfolioResearchConfig,
    phase: PortfolioResearchPhase,
    *,
    repo_root: Path,
) -> dict[str, object] | None:
    """Structured monitoring status for strategy/portfolio holdout phases."""

    if phase not in {
        PortfolioResearchPhase.STRATEGY_HOLDOUT,
        PortfolioResearchPhase.PORTFOLIO_HOLDOUT,
        PortfolioResearchPhase.FULL_PIPELINE,
    }:
        return None

    loaded = _load_rollup(repo_root, config.output_root)
    if loaded is None:
        return None
    rollup, rollup_path = loaded
    if rollup.empty:
        return None

    strategies = [str(value) for value in rollup["strategy"].tolist()]
    counts = {
        light: int((rollup["traffic_light"] == light).sum())
        for light in ("GREEN", "YELLOW", "RED")
    }
    payload: dict[str, object] = {
        "rollup_path": rollup_path,
        "rollup_rows": rollup.fillna("").astype(str).to_dict(orient="records"),
        "strategy_count": len(rollup),
        "traffic_light_counts": counts,
        "histories": _load_histories(repo_root, config.output_root, strategies),
    }

    portfolio_status_path = (
        holdout_root(config.output_root)
        / "visualization"
        / "portfolio"
        / "monitoring_status.json"
    )
    if portfolio_status_path.is_file():
        status = json.loads(portfolio_status_path.read_text(encoding="utf-8"))
        payload["portfolio_monitoring"] = {
            "relative_path": _relative(repo_root, portfolio_status_path),
            "status": status,
        }
    return payload


def inject_monitoring_section(
    sections: list[dict[str, object]],
    *,
    dashboard: dict[str, object] | None,
    default_section_id: str | None,
) -> tuple[list[dict[str, object]], str | None]:
    """Prepend a dedicated monitoring section when rollup data exists."""

    if dashboard is None:
        return sections, default_section_id

    rollup_path = str(dashboard["rollup_path"])
    monitoring_section = {
        "id": "monitoring_status",
        "label": "Monitoring status",
        "description": (
            "Traffic-light rollup from trailing 12-month evaluation vs validation μ "
            "and train+validation σ. Advisory weights are display-only."
        ),
        "category": "monitoring_status",
        "priority": -1,
        "panels": [
            {
                "panel_id": "monitoring_rollup_dashboard",
                "title": "Strategy traffic lights",
                "view_type": "monitoring_rollup",
                "panel_kind": "primary",
                "kind": "table",
                "relative_path": rollup_path,
                "category": "monitoring_status",
                "is_primary": True,
            }
        ],
        "raw_panel_count": 0,
        "primary_panel_id": "monitoring_rollup_dashboard",
        "is_raw_data": False,
        "monitoring_dashboard": dashboard,
    }
    return [monitoring_section, *sections], "monitoring_status"
