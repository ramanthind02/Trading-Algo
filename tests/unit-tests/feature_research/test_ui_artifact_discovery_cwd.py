from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_research.config import load_config
from feature_research.shared import FeatureResearchPhase
from feature_research.ui.artifact_catalog import phase_discovery_roots
from feature_research.ui.planner import build_ui_defaults, build_ui_request
from feature_research.ui.workspace import build_workspace_view
from feature_research.ui.workspace_manifest import validation_artifacts_current


def test_validation_discovery_works_when_cwd_is_not_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    frontend_cwd = repo_root / "frontend"
    monkeypatch.chdir(frontend_cwd)

    config = load_config()
    assert validation_artifacts_current(config)

    roots = phase_discovery_roots(config, FeatureResearchPhase.VALIDATION)
    assert roots, "validation discovery roots should resolve from repo root, not cwd"

    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=FeatureResearchPhase.VALIDATION.value,
        tickers_text=",".join(ticker.name for ticker in defaults.default_tickers),
        fallback_tickers=defaults.default_tickers,
    )
    workspace = build_workspace_view(config, request, current_job=None)
    validation = next(phase for phase in workspace["phases"] if phase["phase"] == "validation")

    assert int(validation["summary_cards"][0]["value"]) > 0
    assert validation["has_results"] is True


def test_validation_discovery_finds_repo_relative_walkforward_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    frontend_cwd = repo_root / "frontend"
    monkeypatch.chdir(frontend_cwd)

    config = load_config()
    module_name = str(config.eval_bias_spec["module_name"])
    walkforward_dir = (
        repo_root
        / "feature_research"
        / "shared_results"
        / "signed_signal"
        / module_name
        / "validation"
    )
    walkforward_dir.mkdir(parents=True, exist_ok=True)
    report_path = walkforward_dir / "report.json"
    if not report_path.is_file():
        report_path.write_text(
            json.dumps({"module_name": module_name, "objective_metric_name": "t_stat"}),
            encoding="utf-8",
        )

    roots = phase_discovery_roots(config, FeatureResearchPhase.VALIDATION)
    assert any(str(root).endswith(f"signed_signal/{module_name}/validation") for root in roots)
