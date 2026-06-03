"""Vault commit helpers for the feature_research workspace UI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ensemble.vault.manager import generate_model_id
from feature_research.config import ResearchConfig, VaultSaveConfig, vault_save_effective_vault_root
from feature_research.inclusion_gates import _inject_weight_hierarchy_group_into_features
from feature_research.save_feature_to_vault import (
    _normalize_bias_spec_for_model,
    _resolve_ensemble_dir,
)
from feature_research.shared.visualization_paths import (
    canonical_in_sample_visualization_dir,
    walkforward_visualization_csv_dir,
)
from feature_research.ui.workspace_manifest import validation_artifacts_current
from feature_research.ui.planner import resolve_ui_config
from feature_research.ui.contracts import FeatureResearchUiRequest
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.helpers import build_feature_column_name
from utils.core.enums import Ticker, TimeFrame, coerce_direction

_REPO_ROOT = Path(__file__).resolve().parents[2]


class VaultGateStatus(str, Enum):
    """Portfolio addition gate state relevant to vault commit."""

    MISSING = "missing"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class VaultSaveExecution:
    """Result of a vault save preview or write."""

    dry_run: bool
    vault_root: str
    ensemble_dir: str
    ensemble_dir_repo_relative: str
    tickers: tuple[str, ...]
    direction: str
    feature_column: str
    model_id: str | None
    bias_spec: dict[str, Any]
    vault_profile: str | None
    weight_hierarchy_group: str | None


def portfolio_addition_report_path(config: ResearchConfig) -> Path:
    """Canonical portfolio addition gate JSON for the current research output root."""

    return (
        walkforward_visualization_csv_dir(config.output_root, "validation")
        / "portfolio_addition_report.json"
    )


def load_portfolio_gate_status(config: ResearchConfig) -> VaultGateStatus:
    """Return portfolio gate status from persisted validation artifacts."""

    if not validation_artifacts_current(config):
        return VaultGateStatus.MISSING
    report_path = portfolio_addition_report_path(config)
    if not report_path.is_file():
        return VaultGateStatus.MISSING
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("skipped"):
        return VaultGateStatus.SKIPPED
    if payload.get("passed"):
        return VaultGateStatus.PASSED
    return VaultGateStatus.FAILED


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix().replace("\\", "/")


def _timeframe_from_bias_spec(bias_spec: dict[str, Any]) -> TimeFrame:
    first_tf_name = bias_spec["timeframes"][0]
    return TimeFrame[first_tf_name] if isinstance(first_tf_name, str) else first_tf_name


def _feature_column_from_bias_spec(bias_spec: dict[str, Any]) -> str:
    first_timeframe = bias_spec["timeframes"][0]
    timeframe = (
        TimeFrame[first_timeframe]
        if isinstance(first_timeframe, str)
        else first_timeframe
    )
    return build_feature_column_name(
        module=str(bias_spec["module_name"]),
        feature="signal",
        tf=timeframe,
        params=dict(bias_spec.get("params", {})),
    )


def _vault_save_target_dict(vault_save: VaultSaveConfig) -> dict[str, object]:
    return {
        "direction": coerce_direction(vault_save.direction, field_name="direction").value,
        "ensemble_name": vault_save.ensemble_name,
        "existing_ensemble_dir": (
            None
            if vault_save.existing_ensemble_dir is None
            else str(vault_save.existing_ensemble_dir)
        ),
        "weight_hierarchy_group": vault_save.weight_hierarchy_group,
        "vault_profile": vault_save.vault_profile or "prop",
        "vault_root": (
            None
            if vault_save.vault_root is None
            else str(vault_save.vault_root)
        ),
    }


def assess_vault_save_eligibility(config: ResearchConfig) -> dict[str, object]:
    """Return whether the workspace can commit the frozen eval combo to the vault."""

    blockers: list[str] = []
    vault_save = config.vault_save
    if vault_save is None:
        blockers.append("Configure vault_save on ResearchConfig in feature_research/config.py.")
        return {
            "configured": False,
            "ready": False,
            "gate_status": load_portfolio_gate_status(config).value,
            "blockers": blockers,
            "target": None,
        }

    gate_status = load_portfolio_gate_status(config)
    if gate_status is VaultGateStatus.MISSING:
        blockers.append("Run Validation to produce the portfolio addition gate report.")
    elif gate_status is VaultGateStatus.FAILED:
        blockers.append(
            "Portfolio addition gate failed. Do not save this strategy to the vault."
        )

    module_name: str | None = None
    try:
        bias_spec = _normalize_bias_spec_for_model(dict(config.eval_bias_spec))
        module_name = str(bias_spec["module_name"])
    except ValueError as exc:
        blockers.append(str(exc))

    return {
        "configured": True,
        "ready": not blockers,
        "gate_status": gate_status.value,
        "blockers": blockers,
        "target": _vault_save_target_dict(vault_save),
        "module_name": module_name,
        "cli_command": "python -m feature_research.save_feature_to_vault --write",
    }


def build_vault_commit_view(
    config: ResearchConfig,
    request: FeatureResearchUiRequest,
) -> dict[str, object]:
    """Build vault commit status for the current UI selection."""

    configured, _selection = resolve_ui_config(config, request)
    eligibility = assess_vault_save_eligibility(configured)
    preview = None
    preview_error: str | None = None
    if eligibility["configured"]:
        try:
            preview = vault_save_execution_to_dict(
                execute_vault_save(configured, dry_run=True),
            )
        except (ValueError, FileNotFoundError) as exc:
            preview_error = str(exc)

    return {
        **eligibility,
        "preview": preview,
        "preview_error": preview_error,
    }


def execute_vault_save(
    config: ResearchConfig,
    *,
    dry_run: bool,
) -> VaultSaveExecution:
    """Preview or write the frozen eval combo to the configured vault target."""

    vault_save = config.vault_save
    if vault_save is None:
        raise ValueError(
            "Set vault_save=VaultSaveConfig(...) on ResearchConfig in feature_research/config.py."
        )

    eligibility = assess_vault_save_eligibility(config)
    if not eligibility["ready"] and not dry_run:
        blockers = eligibility.get("blockers", [])
        message = blockers[0] if blockers else "Vault save is not ready."
        raise ValueError(message)

    bias_spec = _normalize_bias_spec_for_model(dict(config.eval_bias_spec))
    tickers = (
        list(vault_save.tickers)
        if vault_save.tickers is not None
        else list(config.tickers)
    )
    direction = coerce_direction(vault_save.direction, field_name="direction")
    timeframe = _timeframe_from_bias_spec(bias_spec)
    feature_column = _feature_column_from_bias_spec(bias_spec)
    model_id = generate_model_id("signed_signal", {}, bias_node_params=bias_spec)

    ensemble_dir = _resolve_ensemble_dir(
        vault_save=vault_save,
        timeframe=timeframe,
        direction=direction,
        tickers=tickers,
        dry_run=dry_run,
    )
    vault_root = vault_save_effective_vault_root(vault_save)

    saved_model_id: str | None = None
    if not dry_run:
        if vault_save.init_vault:
            from ensemble.vault_manager import initialize_vault

            initialize_vault(str(vault_root))
        feature_config: dict[str, Any] = {
            "bias_node_spec": bias_spec,
            "strategy": direction,
        }
        model = BaseModel(feature_config, tickers=tickers)
        saved_model_id = model.save_to_vault(ensemble_dir, tickers=tickers)
        if vault_save.weight_hierarchy_group:
            _inject_weight_hierarchy_group_into_features(
                ensemble_dir,
                str(vault_save.weight_hierarchy_group),
            )

    return VaultSaveExecution(
        dry_run=dry_run,
        vault_root=str(vault_root),
        ensemble_dir=ensemble_dir,
        ensemble_dir_repo_relative=_repo_relative_path(Path(ensemble_dir)),
        tickers=tuple(ticker.name for ticker in tickers),
        direction=coerce_direction(direction, field_name="direction").value,
        feature_column=feature_column,
        model_id=saved_model_id if not dry_run else model_id,
        bias_spec=bias_spec,
        vault_profile=vault_save.vault_profile,
        weight_hierarchy_group=vault_save.weight_hierarchy_group,
    )


def vault_save_execution_to_dict(result: VaultSaveExecution) -> dict[str, object]:
    """Serialize a vault save result for Flask endpoints."""

    return {
        "dry_run": result.dry_run,
        "vault_root": result.vault_root,
        "ensemble_dir": result.ensemble_dir,
        "ensemble_dir_repo_relative": result.ensemble_dir_repo_relative,
        "tickers": list(result.tickers),
        "direction": result.direction,
        "feature_column": result.feature_column,
        "model_id": result.model_id,
        "bias_spec": result.bias_spec,
        "vault_profile": result.vault_profile or "prop",
        "weight_hierarchy_group": result.weight_hierarchy_group,
    }
