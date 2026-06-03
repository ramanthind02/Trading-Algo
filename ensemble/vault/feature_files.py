from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Linux filename limit is 255; inclusion temp paths add ~100+ chars before ``features/``.
MAX_FEATURE_FILE_STEM_CHARS = 120

import utils.core.helpers as helpers
from utils.cache.runtime.cache_paths import win32_extended_path


LEGACY_MODEL_KEYS = frozenset(
    {"binning_model_type", "binning_model_params", "requires_fit", "is_fitted", "fitted_params"}
)


def extract_feature_name(feature_config: dict[str, Any], fallback_stem: str) -> str:
    return feature_config.get("feature_name") or feature_config.get("feature_column") or fallback_stem


def feature_json_stem(
    feature_name: str,
    *,
    max_stem_chars: int = MAX_FEATURE_FILE_STEM_CHARS,
) -> str:
    """On-disk ``features/*.json`` stem; hashes when the canonical column name is too long."""
    name = str(feature_name).strip()
    if len(name) <= max_stem_chars:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:16]
    return f"feat_{digest}"


def validate_signed_signal_bias_node_spec(
    bias_node_spec: dict[str, Any],
    *,
    prefix: str,
) -> None:
    if not isinstance(bias_node_spec, dict):
        raise ValueError(f"{prefix}bias_node_spec must be a dictionary")
    module_name = str(bias_node_spec.get("module_name") or "").strip()
    if not module_name:
        raise ValueError(f"{prefix}bias_node_spec.module_name must be non-empty")
    if module_name == "domain_discrete":
        raise ValueError(f"{prefix}domain_discrete bias nodes are no longer supported")
    timeframes = bias_node_spec.get("timeframes")
    if not isinstance(timeframes, list) or not timeframes:
        raise ValueError(f"{prefix}bias_node_spec.timeframes must be a non-empty list")
    params = bias_node_spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{prefix}bias_node_spec.params must be a dictionary")


def validate_signed_signal_feature_config(
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

    validate_signed_signal_bias_node_spec(
        feature_config["bias_node_spec"],
        prefix=f"{feature_file}: ",
    )

    model_entry = feature_config["base_models"][0]
    legacy_keys = sorted(LEGACY_MODEL_KEYS.intersection(model_entry))
    if legacy_keys:
        raise ValueError(
            f"Feature file {feature_file} contains legacy model keys: {legacy_keys}"
        )

    expected_feature_name = extract_feature_name(feature_config, feature_file.stem)
    if model_entry.get("feature_column") not in {expected_feature_name, feature_file.stem}:
        raise ValueError(
            f"Feature file {feature_file} base model feature_column must match feature name"
        )
    if model_entry.get("model_type") != "signed_signal":
        raise ValueError(
            f"Feature file {feature_file} base model model_type must be 'signed_signal'."
        )
    if model_entry.get("bias_node_spec") != feature_config["bias_node_spec"]:
        raise ValueError(
            f"Feature file {feature_file} base model bias_node_spec must match top-level bias_node_spec"
        )


def _candidate_feature_names(raw_name: str) -> list[str]:
    candidates = [raw_name]
    parsed = helpers.parse_feature_column_name(raw_name)
    tf = parsed.get("tf")
    tf_name = tf.name if hasattr(tf, "name") else str(tf) if tf else None
    if parsed.get("module") and parsed.get("feature") and tf_name:
        canonical = f"{parsed['module']}_{parsed['feature']}_{tf_name}"
        if canonical not in candidates:
            candidates.append(canonical)
    return candidates


def feature_file_exists(path: Path) -> bool:
    """``Path.exists()`` that works for long Windows paths (``MAX_PATH`` / ``\\\\?\\``)."""
    return os.path.exists(win32_extended_path(path))


def resolve_feature_file_path(features_dir: Path, raw_name: str) -> Path | None:
    for candidate in _candidate_feature_names(raw_name):
        path = features_dir / f"{candidate}.json"
        if feature_file_exists(path):
            return path
    for path in sorted(features_dir.glob("*.json")):
        try:
            with open(win32_extended_path(path), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if payload.get("feature_name") == raw_name or payload.get("feature_column") == raw_name:
            return path
    return None


def load_validated_feature_config(feature_file: Path) -> dict[str, Any]:
    with open(win32_extended_path(feature_file), "r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    validate_signed_signal_feature_config(feature_config, feature_file=feature_file)
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
    return [dict(feature_config["bias_node_spec"]) for _, feature_config in iter_validated_feature_configs(features_dir)]


def collect_base_model_names(features_dir: Path) -> list[str]:
    return [str(feature_config["base_models"][0]["model_name"]) for _, feature_config in iter_validated_feature_configs(features_dir)]


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


def consolidate_feature_files_in_dir(features_dir: Path) -> list[str]:
    if not features_dir.exists():
        return []
    return [str(feature_file) for feature_file, _ in iter_validated_feature_configs(features_dir)]


def _merge_model_bias_node_params_into_top_spec(
    feature_config: dict[str, Any],
    model_entry: dict[str, Any],
) -> bool:
    """Fold legacy ``base_models[0].bias_node_params`` into top-level ``bias_node_spec.params``."""
    raw = model_entry.get("bias_node_params")
    if not isinstance(raw, dict) or not raw:
        return False
    top = feature_config.get("bias_node_spec")
    if not isinstance(top, dict):
        raise ValueError("bias_node_spec must be a dictionary")
    existing = top.get("params")
    if not isinstance(existing, dict):
        existing = {}
    top["params"] = {**existing, **raw}
    return True


def migrate_legacy_signed_signal_feature_config(
    feature_config: dict[str, Any],
    *,
    feature_file: Path,
) -> bool:
    """
    Strip binning-era keys from ``base_models[0]`` and align with signed-signal vault schema.

    Merges ``bias_node_params`` into the top-level ``bias_node_spec`` (same as current writers).
    Returns True if the document was modified.
    """
    base_models = feature_config.get("base_models")
    if not isinstance(base_models, list) or len(base_models) != 1:
        return False
    model_entry = base_models[0]
    if not isinstance(model_entry, dict):
        return False

    expected_name = extract_feature_name(feature_config, feature_file.stem)
    merged_params = _merge_model_bias_node_params_into_top_spec(feature_config, model_entry)
    legacy_present = bool(LEGACY_MODEL_KEYS.intersection(model_entry))
    had_bias_node_params_key = "bias_node_params" in model_entry

    for key in LEGACY_MODEL_KEYS:
        model_entry.pop(key, None)
    model_entry.pop("bias_node_params", None)

    top_spec = feature_config["bias_node_spec"]
    desired_model_spec = copy.deepcopy(top_spec)

    needs_fixup = (
        merged_params
        or legacy_present
        or had_bias_node_params_key
        or model_entry.get("model_type") != "signed_signal"
        or model_entry.get("feature_column") != expected_name
        or model_entry.get("bias_node_spec") != desired_model_spec
    )
    if not needs_fixup:
        return False

    model_entry["model_type"] = "signed_signal"
    model_entry["feature_column"] = expected_name
    model_entry["bias_node_spec"] = desired_model_spec

    validate_signed_signal_feature_config(feature_config, feature_file=feature_file)
    return True


def migrate_legacy_signed_signal_feature_file(
    feature_file: Path,
    *,
    dry_run: bool = False,
) -> bool:
    """Load, migrate when needed, validate, and write. Returns True if the file was updated."""
    with open(win32_extended_path(feature_file), "r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    if not migrate_legacy_signed_signal_feature_config(
        feature_config,
        feature_file=feature_file,
    ):
        return False
    if dry_run:
        return True
    feature_config["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(win32_extended_path(feature_file), "w", encoding="utf-8") as handle:
        json.dump(feature_config, handle, indent=2)
        handle.write("\n")
    return True


def migrate_legacy_signed_signal_feature_files_under_vault(
    vault_root: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Migrate every ``vault/**/features/*.json`` that still uses legacy base-model keys."""
    updated: list[str] = []
    for feature_file in sorted(vault_root.rglob("*.json")):
        if feature_file.parent.name != "features":
            continue
        try:
            changed = migrate_legacy_signed_signal_feature_file(feature_file, dry_run=dry_run)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            raise RuntimeError(f"Failed migrating {feature_file}: {exc}") from exc
        if changed:
            updated.append(str(feature_file))
    return updated


def migrate_legacy_feature_members_schema_in_dir(features_dir: Path) -> list[str]:
    if not features_dir.exists():
        return []

    updated_paths: list[str] = []
    for feature_file in sorted(features_dir.glob("*.json")):
        with open(win32_extended_path(feature_file), "r", encoding="utf-8") as handle:
            feature_config = json.load(handle)

        base_models = feature_config.get("base_models", [])
        if not isinstance(base_models, list) or not base_models:
            continue
        model_entry = base_models[0]
        if "members" not in model_entry:
            continue
        if model_entry["members"]:
            raise ValueError(
                f"Feature file {feature_file} still contains non-empty legacy members schema payloads."
            )

        model_entry.pop("members", None)
        feature_config["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(win32_extended_path(feature_file), "w", encoding="utf-8") as handle:
            json.dump(feature_config, handle, indent=2)
            handle.write("\n")
        updated_paths.append(str(feature_file))

    return updated_paths
