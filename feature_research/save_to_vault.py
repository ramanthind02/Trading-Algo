"""Save the research model from feature_research config to the vault.

Run after satisfying research results. Configure vault_save in
feature_research.config.load_config() then run:

    python -m feature_research.save_to_vault
"""
from __future__ import annotations

import sys
from typing import Any
from pathlib import Path

_repo_hint = Path(__file__).resolve().parents[1]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import BinningAnalysisConfig, FeatureType, load_config
from feature_research.in_sample.config import load_config as load_in_sample_config
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    populate_cache_if_needed,
)
from utils.core.enums import coerce_direction


def _resolve_param_combos(
    bias_spec: dict[str, Any],
    params_to_save: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve one or more param combos for vault save."""
    expanded = expand_bias_specs(bias_spec)
    if not expanded:
        raise ValueError(
            "No param combo to save: bias_spec has no params or expand_bias_specs returned empty. "
            "Set vault_save.params_to_save in feature_research.config.load_config()."
        )

    def _normalize_combo(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "module_name": bias_spec["module_name"],
            "timeframes": bias_spec.get("timeframes", []),
            "params": params,
        }

    if params_to_save is None or params_to_save == {}:
        return [
            {
                "module_name": combo["module_name"],
                "timeframes": combo.get("timeframes", []),
                "params": combo.get("params", {}),
            }
            for combo in expanded
        ]

    if isinstance(params_to_save, dict):
        return [_normalize_combo(params_to_save)]

    if not isinstance(params_to_save, list):
        raise ValueError(
            "vault_save.params_to_save must be dict, list[dict], or None"
        )
    if not params_to_save:
        raise ValueError("vault_save.params_to_save list cannot be empty")
    if not all(isinstance(combo, dict) for combo in params_to_save):
        raise ValueError("vault_save.params_to_save list entries must be dicts")

    return [_normalize_combo(combo) for combo in params_to_save]


def _binning_params_to_constructor_params(
    binning_params: BinningAnalysisConfig,
) -> dict[str, Any]:
    """Map BinningAnalysisConfig to ContinuousBinningModel constructor params."""
    return {
        "n_bins": binning_params.bin_counts[0] if binning_params.bin_counts else 10,
        "bin_counts": list(binning_params.bin_counts),
        "strategy": binning_params.strategy.value,
        "bin_index_min": binning_params.bin_index_min,
        "bin_index_max": binning_params.bin_index_max,
    }


def _run() -> None:
    base = load_config()
    if base.vault_save is None:
        print(
            "Vault save not configured; set vault_save in feature_research.config.load_config().",
            file=sys.stderr,
        )
        sys.exit(1)

    from feature_selection.base_models import BaseModel, ContinuousBinningModel, RuleBasedModel
    from ensemble.vault_manager import add_feature_to_ensemble, create_ensemble_directory

    research_config = load_in_sample_config()
    phase_defaults = base.in_sample_defaults.for_feature_type(base.feature_type)
    bias_spec = phase_defaults.bias_spec

    bias_specs_for_save = _resolve_param_combos(bias_spec, base.vault_save.params_to_save)
    populate_cache_if_needed(research_config)

    strategy = coerce_direction(research_config.strategy, field_name="research_config.strategy")
    vault_direction = coerce_direction(base.vault_save.direction, field_name="vault_save.direction")

    ensemble_dir = create_ensemble_directory(
        base.timeframe,
        base.vault_save.ensemble_name,
        vault_direction,
        research_config.tickers,
    )
    success_count = 0
    failed: list[str] = []
    for bias_spec_for_combo in bias_specs_for_save:
        combo_label = str(bias_spec_for_combo.get("params", {}))
        try:
            result = load_features_for_combo(bias_spec_for_combo, research_config)
            if result is None:
                failed.append(combo_label)
                continue
            feature_series, target_series, feature_col = result

            timeframes_raw = bias_spec_for_combo.get("timeframes", [base.timeframe])
            tf_list = timeframes_raw[0:1] if isinstance(timeframes_raw, list) else [timeframes_raw]
            tf = tf_list[0] if tf_list else base.timeframe
            tf_name = tf.name if hasattr(tf, "name") else str(tf)
            bias_node_spec = {
                "module_name": bias_spec_for_combo["module_name"],
                "timeframes": [tf_name],
                "params": bias_spec_for_combo["params"],
            }

            if base.feature_type == FeatureType.CONTINUOUS:
                constructor_params = _binning_params_to_constructor_params(research_config.binning_params)
                binning_model = ContinuousBinningModel(**constructor_params)
                binning_model.fit(feature_series, target_series)
            else:
                rule_params = {
                    "strategy": strategy,
                    "selection_metric": "t_stat",
                }
                binning_model = RuleBasedModel(**rule_params)

            feature_config = {
                "bias_node_spec": bias_node_spec,
                "model_type": "continuous_binning" if base.feature_type == FeatureType.CONTINUOUS else "rule_based",
                "constructor_params": binning_model.get_params(),
                "strategy": strategy.value,
            }
            base_model = BaseModel(
                feature_config=feature_config,
                tickers=research_config.tickers,
                binning_model=binning_model,
                use_cache=research_config.use_cache,
            )
            base_model.feature_column = feature_col

            add_feature_to_ensemble(
                feature_name=feature_col,
                bias_node_spec=bias_node_spec,
                base_model=base_model,
                ensemble_dir=ensemble_dir,
                tickers=research_config.tickers,
            )
            success_count += 1
            print(f"Saved feature '{feature_col}' to vault: {ensemble_dir}")
        except Exception as exc:
            failed.append(f"{combo_label}: {exc}")

    if failed:
        print(
            f"Failed to save {len(failed)} combo(s): {failed}",
            file=sys.stderr,
        )
    if success_count == 0 or failed:
        sys.exit(1)


if __name__ == "__main__":
    _run()
