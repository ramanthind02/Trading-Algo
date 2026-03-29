"""Save a frozen domain-discrete research handoff to the vault.

Run after freezing the discrete contract in research. Configure ``vault_save``
in ``feature_research.config.load_config()`` then run:

    python -m feature_research.save_to_vault
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_repo_hint = Path(__file__).resolve().parents[1]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import load_config
from feature_research.in_sample.config import load_config as load_in_sample_config
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    populate_cache_if_needed,
)
from feature_selection.domain_discrete import (
    DomainDiscreteSpec,
    build_domain_discrete_bias_node_spec,
    load_domain_discrete_spec,
)
from utils.core.enums import coerce_direction


def _resolve_param_combos(
    bias_spec: dict[str, Any],
    params_to_save: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve one or more frozen domain-discrete specs for vault save."""
    expanded = expand_bias_specs(bias_spec)
    if not expanded:
        raise ValueError(
            "No param combo to save: bias_spec has no params or expand_bias_specs returned empty. "
            "Set vault_save.params_to_save in feature_research.config.load_config()."
        )

    if params_to_save is None or params_to_save == {}:
        raise ValueError(
            "vault_save.params_to_save must now provide frozen domain-discrete specs. "
            "Pass one dict or a list of dicts with source_bias_node_spec, ticker_scope, "
            "edges, n_bins, long_bins, short_bins, direction, and spec_version."
        )

    if isinstance(params_to_save, dict):
        return [params_to_save]

    if not isinstance(params_to_save, list):
        raise ValueError(
            "vault_save.params_to_save must be dict, list[dict], or None"
        )
    if not params_to_save:
        raise ValueError("vault_save.params_to_save list cannot be empty")
    if not all(isinstance(combo, dict) for combo in params_to_save):
        raise ValueError("vault_save.params_to_save list entries must be dicts")

    return list(params_to_save)


def _validate_frozen_spec_payload(payload: dict[str, Any]) -> DomainDiscreteSpec:
    missing = [
        key
        for key in (
            "source_bias_node_spec",
            "ticker_scope",
            "edges",
            "n_bins",
            "long_bins",
            "short_bins",
            "direction",
            "spec_version",
        )
        if key not in payload
    ]
    if missing:
        raise ValueError(
            "Frozen domain-discrete specs require keys: "
            f"{missing}. "
            "Legacy raw param combos are no longer accepted by save_to_vault."
        )
    return load_domain_discrete_spec(payload)


def _build_feature_name(spec: DomainDiscreteSpec) -> str:
    source = spec.source_bias_node_spec
    tf = source.timeframes[0]
    tf_name = tf.name if hasattr(tf, "name") else str(tf)
    return f"{source.module_name}_domain_discrete_{tf_name}_{spec.spec_version}"


def _write_feature_manifest(
    feature_file: Path,
    spec: DomainDiscreteSpec,
    feature_name: str,
    source_feature_column: str,
    tickers: list[str],
) -> None:
    payload = {
        "feature_name": feature_name,
        "source_feature_column": source_feature_column,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "model_type": "domain_discrete",
            "is_fit": False,
            "spec_version": spec.spec_version,
        },
        "tickers": tickers,
        "bias_node_spec": build_domain_discrete_bias_node_spec(spec),
        "base_models": [
            {
                "model_id": feature_name,
                "model_name": feature_name,
                "model_type": "domain_discrete",
                "strategy": spec.direction.value,
                "bias_node_spec": build_domain_discrete_bias_node_spec(spec),
                "bias_node_params": spec.to_mapping(),
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    }
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _run() -> None:
    base = load_config()
    if base.vault_save is None:
        print(
            "Vault save not configured; set vault_save in feature_research.config.load_config().",
            file=sys.stderr,
        )
        sys.exit(1)

    from ensemble.vault_manager import create_ensemble_directory

    research_config = load_in_sample_config()
    phase_defaults = base.in_sample_defaults.for_feature_type(base.feature_type)
    bias_spec = phase_defaults.bias_spec

    frozen_specs = _resolve_param_combos(bias_spec, base.vault_save.params_to_save)
    populate_cache_if_needed(research_config)

    vault_direction = coerce_direction(base.vault_save.direction, field_name="vault_save.direction")
    ensemble_dir = create_ensemble_directory(
        base.timeframe,
        base.vault_save.ensemble_name,
        vault_direction,
        research_config.tickers,
    )

    success_count = 0
    failed: list[str] = []
    for frozen_payload in frozen_specs:
        try:
            frozen_spec = _validate_frozen_spec_payload(frozen_payload)
            result = load_features_for_combo(frozen_spec.source_bias_node_spec.to_mapping(), research_config)
            if result is None:
                failed.append(str(frozen_spec.spec_version))
                continue

            _feature_series, _target_series, feature_col = result
            feature_name = _build_feature_name(frozen_spec)
            feature_file = Path(ensemble_dir) / "features" / f"{feature_name}.json"

            _write_feature_manifest(
                feature_file=feature_file,
                spec=frozen_spec,
                feature_name=feature_name,
                source_feature_column=feature_col,
                tickers=[ticker.name for ticker in research_config.tickers],
            )
            success_count += 1
            print(f"Saved frozen domain-discrete spec '{feature_name}' to vault: {ensemble_dir}")
        except Exception as exc:
            failed.append(f"{frozen_payload}: {exc}")

    if failed:
        print(
            f"Failed to save {len(failed)} spec(s): {failed}",
            file=sys.stderr,
        )
    if success_count == 0 or failed:
        sys.exit(1)


if __name__ == "__main__":
    _run()
