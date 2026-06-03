from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from feature_research.config import VaultSaveConfig, load_config
from feature_research.shared import FeatureResearchPhase
from feature_research.shared.visualization_paths import walkforward_visualization_csv_dir
from feature_research.ui.contracts import FeatureResearchUiRequest
from feature_research.ui.vault_save import (
    VaultGateStatus,
    assess_vault_save_eligibility,
    build_vault_commit_view,
    execute_vault_save,
    load_portfolio_gate_status,
)


def _request() -> FeatureResearchUiRequest:
    config = load_config()
    return FeatureResearchUiRequest(
        phase=FeatureResearchPhase.VALIDATION,
        tickers=tuple(config.tickers),
    )


def _write_gate_report(
    config,
    tmp_path: Path,
    *,
    passed: bool | None = None,
    skipped: bool = False,
) -> None:
    report_dir = walkforward_visualization_csv_dir(tmp_path, "validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": passed,
        "skipped": skipped,
        "interpretation": "test gate",
    }
    (report_dir / "portfolio_addition_report.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_assess_vault_save_eligibility_requires_gate_report(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    eligibility = assess_vault_save_eligibility(config)
    assert eligibility["configured"] is True
    assert eligibility["ready"] is False
    assert load_portfolio_gate_status(config) is VaultGateStatus.MISSING
    assert "Run Validation" in eligibility["blockers"][0]


def test_assess_vault_save_eligibility_unlocks_after_gate_pass(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    _write_gate_report(config, tmp_path, passed=True)
    eligibility = assess_vault_save_eligibility(config)
    assert eligibility["ready"] is True
    assert eligibility["gate_status"] == VaultGateStatus.PASSED.value


def test_assess_vault_save_eligibility_blocks_after_gate_fail(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    _write_gate_report(config, tmp_path, passed=False)
    eligibility = assess_vault_save_eligibility(config)
    assert eligibility["ready"] is False
    assert "failed" in eligibility["blockers"][0].lower()


def test_build_vault_commit_view_includes_preview(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    _write_gate_report(config, tmp_path, passed=True)
    view = build_vault_commit_view(config, _request())
    assert view["preview"] is not None
    assert view["preview"]["feature_column"]
    assert view["preview"]["ensemble_dir_repo_relative"]


def test_execute_vault_save_dry_run_does_not_write(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    _write_gate_report(config, tmp_path, passed=True)
    result = execute_vault_save(config, dry_run=True)
    assert result.dry_run is True
    assert result.model_id is not None
    assert not Path(result.ensemble_dir).joinpath("features").exists()


def test_execute_vault_save_write_rejects_failed_gate(tmp_path: Path) -> None:
    config = replace(load_config(), output_root=tmp_path)
    _write_gate_report(config, tmp_path, passed=False)
    with pytest.raises(ValueError, match="failed"):
        execute_vault_save(config, dry_run=False)


def test_execute_vault_save_write_creates_feature_file(tmp_path: Path) -> None:
    base = load_config()
    vault_root = tmp_path / "vault_root"
    vault_save = VaultSaveConfig(
        direction=base.vault_save.direction if base.vault_save else "long",
        ensemble_name="ui_commit_test",
        weight_hierarchy_group="momentum",
        vault_root=str(vault_root),
        dry_run=False,
    )
    config = replace(
        base,
        output_root=tmp_path / "results",
        vault_save=vault_save,
    )
    _write_gate_report(config, config.output_root, passed=True)
    result = execute_vault_save(config, dry_run=False)
    feature_dir = Path(result.ensemble_dir) / "features"
    feature_files = list(feature_dir.glob("*.json"))
    assert len(feature_files) == 1
    payload = json.loads(feature_files[0].read_text(encoding="utf-8"))
    assert payload["weight_hierarchy_group"] == "momentum"
