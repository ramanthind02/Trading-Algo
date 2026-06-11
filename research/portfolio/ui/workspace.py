"""Workspace view builder for portfolio research UI."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.shared.phase import PortfolioResearchPhase, artifact_view_phases
from research.portfolio.shared.visualization_paths import holdout_root
from research.portfolio.ui.artifact_catalog import (
    build_phase_report_sections,
    discover_artifact_records,
    group_artifact_records,
)
from research.portfolio.ui.monitoring_dashboard import (
    build_monitoring_dashboard,
    inject_monitoring_section,
)
from research.portfolio.ui.contracts import PortfolioResearchUiRequest
from research.portfolio.ui.planner import apply_ui_request
from research.workspace.preview import load_artifact_preview as _load_preview
from research.workspace.preview import resolve_artifact_path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_workspace_view(
    config: PortfolioResearchConfig,
    request: PortfolioResearchUiRequest,
    *,
    current_job: dict[str, object] | None = None,
) -> dict[str, object]:
    configured = apply_ui_request(config, request)
    view_phases = artifact_view_phases()
    selected = (
        request.phase
        if request.phase in view_phases
        else PortfolioResearchPhase.PORTFOLIO_HOLDOUT
    )
    phases = [_phase_view(configured, phase, phase == selected) for phase in view_phases]
    return {
        "selected_phase": selected.value,
        "job": current_job,
        "phases": phases,
    }


def _phase_view(
    config: PortfolioResearchConfig,
    phase: PortfolioResearchPhase,
    selected: bool,
) -> dict[str, object]:
    artifacts = discover_artifact_records(config, phase)
    grouped = group_artifact_records(artifacts, phase)
    report_sections, default_section_id = build_phase_report_sections(grouped, phase)
    monitoring_dashboard = build_monitoring_dashboard(config, phase, repo_root=_REPO_ROOT)
    report_sections, default_section_id = inject_monitoring_section(
        report_sections,
        dashboard=monitoring_dashboard,
        default_section_id=default_section_id,
    )
    return {
        "phase": phase.value,
        "label": phase.value.replace("_", " ").title(),
        "selected": selected,
        "summary_cards": _summary_cards(config, phase, artifacts),
        "artifact_counts": dict(Counter(str(artifact["kind"]) for artifact in artifacts)),
        "category_counts": dict(Counter(str(artifact["category"]) for artifact in artifacts)),
        "report_sections": report_sections,
        "default_section_id": default_section_id,
        "has_results": bool(report_sections),
        "monitoring_dashboard": monitoring_dashboard,
    }


def load_artifact_preview(relative_path: str) -> dict[str, object]:
    return _load_preview(_REPO_ROOT, relative_path)


def resolve_workspace_artifact_path(relative_path: str) -> Path:
    return resolve_artifact_path(_REPO_ROOT, relative_path)


def _summary_cards(
    config: PortfolioResearchConfig,
    phase: PortfolioResearchPhase,
    artifacts: list[dict[str, object]],
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = [{"label": "Artifacts", "value": str(len(artifacts))}]
    path_lookup = {str(artifact["relative_path"]): artifact for artifact in artifacts}
    plot_count = sum(1 for artifact in artifacts if artifact["kind"] in {"html", "image"})
    if plot_count:
        cards.append({"label": "Plots & reports", "value": str(plot_count)})

    manifest_path = _first_path_ending(path_lookup, "fold_manifest.json")
    if manifest_path is not None:
        payload = json.loads(resolve_workspace_artifact_path(manifest_path).read_text(encoding="utf-8"))
        folds = payload.get("folds", [])
        cards.append({"label": "Fit mode", "value": str(payload.get("fit_mode", "unknown"))})
        cards.append({"label": "Holdout folds", "value": str(len(folds))})

    if phase is PortfolioResearchPhase.STRATEGY_HOLDOUT:
        rollup_path = _first_path_ending(path_lookup, "monitoring_rollup.csv")
        if rollup_path is not None:
            rollup = pd.read_csv(resolve_workspace_artifact_path(rollup_path))
            cards.append({"label": "Strategies monitored", "value": str(len(rollup))})
            green = int((rollup["traffic_light"] == "GREEN").sum())
            yellow = int((rollup["traffic_light"] == "YELLOW").sum())
            red = int((rollup["traffic_light"] == "RED").sum())
            cards.append({"label": "Green", "value": str(green)})
            cards.append({"label": "Yellow", "value": str(yellow)})
            cards.append({"label": "Red", "value": str(red)})
        else:
            reports = [
                path
                for path in path_lookup
                if path.endswith("holdout_robustness_report.json")
            ]
            if reports:
                cards.append({"label": "Strategies monitored", "value": str(len(reports))})

    if phase is PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
        report_path = _first_path_ending(path_lookup, "portfolio_holdout_report.json")
        if report_path is not None:
            payload = json.loads(
                resolve_workspace_artifact_path(report_path).read_text(encoding="utf-8")
            )
            corr = payload.get("correlation_realisation", {})
            if isinstance(corr, dict):
                cards.append(
                    {
                        "label": "Mean pairwise corr",
                        "value": f"{float(corr.get('mean_pairwise_corr', 0.0)):.2f}",
                    }
                )
            cards.append(
                {
                    "label": "Portfolio holdout",
                    "value": "PASS" if payload.get("passed") else "FAIL",
                }
            )

    returns_holdout = _first_path_ending(path_lookup, "portfolio_returns_holdout.csv")
    if returns_holdout is not None and phase is PortfolioResearchPhase.PORTFOLIO_TEST:
        frame = pd.read_csv(resolve_workspace_artifact_path(returns_holdout))
        if not frame.empty:
            col = next((c for c in frame.columns if c != "datetime"), frame.columns[0])
            cards.append({"label": "Holdout bars", "value": str(len(frame))})

    if phase is PortfolioResearchPhase.PROP_FIRM_REPORTS:
        html_paths = [p for p in path_lookup if p.endswith(".html")]
        if html_paths:
            cards.append({"label": "Phase reports", "value": str(len(html_paths))})
        rolling_json_paths = [
            path
            for path in path_lookup
            if path.endswith("_rolling_batch_statistics.json")
        ]
        if rolling_json_paths:
            expected_ncf_values = [
                float(payload["expected_net_cashflow"])
                for path in rolling_json_paths
                for payload in [
                    json.loads(
                        resolve_workspace_artifact_path(path).read_text(encoding="utf-8")
                    )
                ]
                if payload.get("expected_net_cashflow") is not None
            ]
            cards.append(
                {"label": "Rolling reports", "value": str(len(rolling_json_paths))}
            )
            if expected_ncf_values:
                mean_ncf = sum(expected_ncf_values) / len(expected_ncf_values)
                cards.append(
                    {
                        "label": "Mean rolling E[net cashflow]",
                        "value": f"${mean_ncf:,.0f}",
                    }
                )
        firms = {
            Path(p).parent.name
            for p in path_lookup
            if "/prop_firm/" in p.replace("\\", "/")
        } - {"prop_firm"}
        if firms:
            cards.append({"label": "Provider", "value": ", ".join(sorted(firms))})
        phases_found = sorted(
            {
                part
                for p in path_lookup
                for part in Path(p).parts
                if part in {"train", "validation", "test"}
            }
        )
        if phases_found:
            cards.append({"label": "Windows", "value": ", ".join(phases_found)})

    return cards


def _first_path_ending(path_lookup: dict[str, object], suffix: str) -> str | None:
    return next((path for path in path_lookup if path.endswith(suffix)), None)
