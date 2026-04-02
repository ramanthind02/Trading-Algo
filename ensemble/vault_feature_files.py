from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import utils.core.helpers as helpers
from feature_selection.domain_discrete import (
    load_domain_discrete_spec,
    raise_legacy_feature_artifact,
)
from utils.core.enums import TimeFrame


def extract_feature_name(feature_config: dict[str, Any], fallback_stem: str) -> str:
    """Resolve feature name from canonical or legacy payload keys."""
    return (
        feature_config.get("feature_name")
        or feature_config.get("feature_column")
        or fallback_stem
    )


def validate_domain_discrete_bias_node_spec(
    bias_node_spec: dict[str, Any],
    *,
    prefix: str,
) -> None:
    if not isinstance(bias_node_spec, dict):
        raise ValueError(f"{prefix}bias_node_spec must be a dictionary")
    if bias_node_spec.get("module_name") != "domain_discrete":
        raise_legacy_feature_artifact(
            f"{prefix}bias_node_spec.module_name must be 'domain_discrete'."
        )
    params = bias_node_spec.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{prefix}bias_node_spec.params must be a dictionary")
    domain_spec = load_domain_discrete_spec(params)
    raw_timeframes = bias_node_spec.get("timeframes", [])
    if raw_timeframes:
        stored_timeframes = [
            tf if isinstance(tf, TimeFrame) else TimeFrame[str(tf)]
            for tf in raw_timeframes
        ]
        if stored_timeframes != list(domain_spec.source_bias_node_spec.timeframes):
            raise ValueError(
                f"{prefix}bias_node_spec.timeframes must match source_bias_node_spec.timeframes"
            )


def validate_domain_discrete_feature_config(
    feature_config: dict[str, Any],
    *,
    feature_file: Path,
) -> None:
    required_keys = ["feature_name", "bias_node_spec", "base_models"]
    missing_keys = [key for key in required_keys if key not in feature_config]
    if missing_keys:
        raise ValueError(f"Feature file {feature_file} missing required keys: {missing_keys}")
    if not isinstance(feature_config["base_models"], list):
        raise ValueError(f"Feature file {feature_file} base_models must be a list")
    if len(feature_config["base_models"]) != 1:
        raise ValueError(
            f"Feature file {feature_file} must contain exactly one base model; "
            f"found {len(feature_config['base_models'])}"
        )
    if "feature_column" in feature_config:
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} still stores legacy top-level feature_column."
        )

    validate_domain_discrete_bias_node_spec(
        feature_config["bias_node_spec"],
        prefix=f"{feature_file}: ",
    )

    model_entry = feature_config["base_models"][0]
    legacy_keys = {
        key
        for key in (
            "binning_model_type",
            "binning_model_params",
            "requires_fit",
            "is_fitted",
            "fitted_params",
        )
        if key in model_entry
    }
    if legacy_keys:
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} contains legacy model keys: {sorted(legacy_keys)}"
        )

    expected_feature_name = extract_feature_name(feature_config, feature_file.stem)
    if model_entry.get("feature_column") not in {expected_feature_name, feature_file.stem}:
        raise ValueError(
            f"Feature file {feature_file} base model feature_column must match feature name"
        )
    if model_entry.get("model_type") != "domain_discrete":
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} base model model_type must be 'domain_discrete'."
        )
    if model_entry.get("bias_node_spec") != feature_config["bias_node_spec"]:
        raise ValueError(
            f"Feature file {feature_file} base model bias_node_spec must match top-level bias_node_spec"
        )


def _candidate_feature_names(raw_name: str) -> list[str]:
    """Generate likely feature-file stems for canonical and legacy names."""
    candidates = [raw_name]
    parsed = helpers.parse_feature_column_name(raw_name)
    tf = parsed.get("tf")
    tf_name = tf.name if hasattr(tf, "name") else str(tf) if tf else None
    if parsed.get("module") and parsed.get("feature") and tf_name:
        canonical = f"{parsed['module']}_{parsed['feature']}_{tf_name}"
        if canonical not in candidates:
            candidates.append(canonical)
    return candidates


def resolve_feature_file_path(features_dir: Path, raw_name: str) -> Path | None:
    """Find feature JSON path by filename or payload-level feature keys."""
    for candidate in _candidate_feature_names(raw_name):
        path = features_dir / f"{candidate}.json"
        if path.exists():
            return path
    for path in sorted(features_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if payload.get("feature_name") == raw_name or payload.get("feature_column") == raw_name:
            return path
    return None


def load_validated_feature_config(feature_file: Path) -> dict[str, Any]:
    with open(feature_file, "r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
    return feature_config


def iter_validated_feature_configs(
    features_dir: Path,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    for feature_file in sorted(features_dir.glob("*.json")):
        yield feature_file, load_validated_feature_config(feature_file)


def build_feature_listing(features_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": resolved_feature_name,
            "feature_column": resolved_feature_name,
            "n_base_models": 1,
            "n_fitted": 0,
            "created_at": feature_config.get("created_at", ""),
            "updated_at": feature_config.get("updated_at", ""),
        }
        for feature_file, feature_config in iter_validated_feature_configs(features_dir)
        for resolved_feature_name in [extract_feature_name(feature_config, feature_file.stem)]
    ]


def collect_bias_node_specs(features_dir: Path) -> list[dict[str, Any]]:
    return [
        dict(feature_config["bias_node_spec"])
        for _, feature_config in iter_validated_feature_configs(features_dir)
    ]


def collect_base_model_names(features_dir: Path) -> list[str]:
    return [
        str(feature_config["base_models"][0]["model_name"])
        for _, feature_config in iter_validated_feature_configs(features_dir)
    ]


def validate_feature_configs_for_ensemble(
    features_dir: Path,
    *,
    expected_ticker_names: list[str],
) -> None:
    for feature_file in sorted(features_dir.glob("*.json")):
        try:
            feature_config = load_validated_feature_config(feature_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in feature file {feature_file}: {exc}") from exc

        feature_ticker_names = sorted(feature_config.get("tickers", []))
        if feature_ticker_names != expected_ticker_names:
            raise ValueError(
                f"Feature '{feature_file.stem}' has tickers {feature_ticker_names} "
                f"but ensemble expects {expected_ticker_names}."
            )
        if (
            sorted(feature_config["base_models"][0]["bias_node_spec"]["params"]["ticker_scope"]["tickers"])
            != feature_ticker_names
        ):
            raise ValueError(
                f"Feature '{feature_file.stem}' ticker_scope must match stored tickers"
            )


def consolidate_feature_files_in_dir(features_dir: Path) -> list[str]:
    if not features_dir.exists():
        return []
    return [str(feature_file) for feature_file, _ in iter_validated_feature_configs(features_dir)]


def migrate_legacy_feature_members_schema_in_dir(features_dir: Path) -> list[str]:
    if not features_dir.exists():
        return []

    updated_paths: list[str] = []
    for feature_file in sorted(features_dir.glob("*.json")):
        with open(feature_file, "r", encoding="utf-8") as handle:
            feature_config = json.load(handle)

        base_models = feature_config.get("base_models", [])
        if not isinstance(base_models, list) or not base_models:
            continue
        model_entry = base_models[0]
        if "members" not in model_entry:
            continue
        if model_entry["members"]:
            raise_legacy_feature_artifact(
                f"Feature file {feature_file} still contains non-empty legacy members schema payloads."
            )

        model_entry.pop("members", None)
        feature_config["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(feature_file, "w", encoding="utf-8") as handle:
            json.dump(feature_config, handle, indent=2)
            handle.write("\n")
        updated_paths.append(str(feature_file))

    return updated_paths
