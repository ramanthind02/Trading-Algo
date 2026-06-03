"""Run manifest helpers so the workspace UI ignores stale phase artifacts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from feature_research._internal.bias_spec_catalog import first_bias_spec
from feature_research.config import ResearchConfig
from feature_research.shared import FeatureResearchPhase
from feature_research.shared.visualization_paths import walkforward_visualization_csv_dir

MANIFEST_FILENAME = "workspace_manifest.json"
_SCHEMA_VERSION = "workspace-manifest-v1"


def build_workspace_manifest(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> dict[str, Any]:
    """Serialize the config slice that identifies one workspace run."""

    eval_spec = config.eval_bias_spec
    return {
        "schema_version": _SCHEMA_VERSION,
        "phase": phase.value,
        "written_at": datetime.now(tz=UTC).isoformat(),
        "module_name": str(eval_spec.get("module_name", "")).strip(),
        "timeframe": config.timeframe.name,
        "tickers": sorted(ticker.name for ticker in config.tickers),
        "eval_params": _normalize_params(dict(eval_spec.get("params", {}))),
        "reports_dir": config.reports_dir.as_posix(),
    }


def write_workspace_manifest(
    path: Path,
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> Path:
    """Write a manifest next to phase artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_workspace_manifest(config, phase)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_workspace_manifest(path: Path) -> dict[str, Any] | None:
    """Load a manifest when present and readable."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def manifest_matches_config(
    manifest: dict[str, Any] | None,
    config: ResearchConfig,
) -> bool:
    """True when ``manifest`` describes the current configured research target."""

    if manifest is None:
        return False
    expected = build_workspace_manifest(config, FeatureResearchPhase.VALIDATION)
    keys = ("module_name", "timeframe", "tickers", "eval_params")
    return all(manifest.get(key) == expected.get(key) for key in keys)


def _viz_matches_config(
    manifest: dict[str, Any] | None,
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> bool:
    """Match module/timeframe/eval_params with optional ticker subset overlap."""

    if manifest is None:
        return False
    expected = build_workspace_manifest(config, phase)
    keys = ("module_name", "timeframe", "eval_params")
    if phase is FeatureResearchPhase.EXPLORATION:
        keys = (*keys, "reports_dir")
    if not all(manifest.get(key) == expected.get(key) for key in keys):
        return False
    manifest_tickers = {
        str(ticker).strip()
        for ticker in manifest.get("tickers", [])
        if str(ticker).strip()
    }
    config_tickers = {ticker.name for ticker in config.tickers}
    if not manifest_tickers or not config_tickers:
        return True
    return bool(manifest_tickers & config_tickers)


def exploration_viz_matches_config(
    manifest: dict[str, Any] | None,
    config: ResearchConfig,
) -> bool:
    """True when canonical exploration viz artifacts match the active research target.

    Ticker lists may differ when a run used a subset (for example ``--tickers NQ``)
    while the UI still exposes the full default universe.
    """

    return _viz_matches_config(manifest, config, FeatureResearchPhase.EXPLORATION)


def validation_viz_matches_config(
    manifest: dict[str, Any] | None,
    config: ResearchConfig,
) -> bool:
    """True when validation viz artifacts match the active research target.

    Same relaxed ticker rules as :func:`exploration_viz_matches_config`.
    """

    return _viz_matches_config(manifest, config, FeatureResearchPhase.VALIDATION)


def validation_manifest_path(config: ResearchConfig) -> Path:
    """Manifest co-located with validation / portfolio gate CSV outputs."""

    from utils.repo_bootstrap import require_repo_root, resolve_repo_path

    repo_root = require_repo_root(Path(__file__))
    return resolve_repo_path(
        walkforward_visualization_csv_dir(config.output_root, "validation") / MANIFEST_FILENAME,
        repo_root=repo_root,
    )


def validation_artifacts_current(config: ResearchConfig) -> bool:
    """True when persisted validation artifacts belong to the active config."""

    return validation_viz_matches_config(
        read_workspace_manifest(validation_manifest_path(config)),
        config,
    )


def exploration_manifest_paths(config: ResearchConfig) -> tuple[Path, ...]:
    """Manifest locations written after exploration."""

    from feature_research.shared.visualization_paths import canonical_in_sample_visualization_dir

    return (
        config.reports_dir / MANIFEST_FILENAME,
        canonical_in_sample_visualization_dir() / MANIFEST_FILENAME,
    )


def write_exploration_manifests(
    config: ResearchConfig,
) -> tuple[Path, ...]:
    """Stamp exploration outputs so the UI can ignore stale shared viz from other modules."""

    return tuple(
        write_workspace_manifest(path, config, FeatureResearchPhase.EXPLORATION)
        for path in exploration_manifest_paths(config)
    )


def write_validation_manifest(config: ResearchConfig) -> Path:
    """Stamp validation / portfolio gate outputs for the active config."""

    return write_workspace_manifest(
        validation_manifest_path(config),
        config,
        FeatureResearchPhase.VALIDATION,
    )


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Stable JSON-friendly param dict for manifest equality checks."""

    normalized: dict[str, Any] = {}
    for key in sorted(params):
        if str(key).startswith("_"):
            continue
        value = params[key]
        if isinstance(value, list):
            normalized[key] = [_normalize_scalar(item) for item in value]
        else:
            normalized[key] = _normalize_scalar(value)
    return normalized


def _normalize_scalar(value: object) -> object:
    if hasattr(value, "name"):
        return str(getattr(value, "name"))
    if isinstance(value, float):
        return round(value, 10)
    return value
