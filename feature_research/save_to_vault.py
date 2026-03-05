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


def _resolve_single_combo(
    bias_spec: dict[str, Any],
    params_to_save: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the single param combo for vault save."""
    if params_to_save is not None:
        return {
            "module_name": bias_spec["module_name"],
            "timeframes": bias_spec.get("timeframes", []),
            "params": params_to_save,
        }
    expanded = expand_bias_specs(bias_spec)
    if not expanded:
        raise ValueError(
            "No param combo to save: bias_spec has no params or expand_bias_specs returned empty. "
            "Set vault_save.params_to_save in feature_research.config.load_config()."
        )
    first = expanded[0]
    return {
        "module_name": first["module_name"],
        "timeframes": first["timeframes"],
        "params": first.get("params", {}),
    }


def _binning_params_to_constructor_params(
    binning_params: BinningAnalysisConfig,
) -> dict[str, Any]:
    """Map BinningAnalysisConfig to ContinuousBinningModel constructor params."""
    return {
        "n_bins": binning_params.bin_counts[0] if binning_params.bin_counts else 10,
        "bin_counts": list(binning_params.bin_counts),
        "selection_metric": binning_params.selection_metric,
        "strategy": binning_params.strategy,
        "metric_threshold": binning_params.metric_threshold,
        "t_threshold": binning_params.t_threshold,
        "min_region_width": binning_params.min_region_width,
        "shrinkage_k": binning_params.shrinkage_k,
        "long_clip_min": binning_params.long_clip_min,
        "long_clip_max": binning_params.long_clip_max,
        "short_clip_min": binning_params.short_clip_min,
        "short_clip_max": binning_params.short_clip_max,
        "use_coverage_bonus": binning_params.use_coverage_bonus,
        "coverage_bonus_per_10pct": binning_params.coverage_bonus_per_10pct,
        "max_coverage_bonus": binning_params.max_coverage_bonus,
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

    bias_spec_for_combo = _resolve_single_combo(bias_spec, base.vault_save.params_to_save)
    populate_cache_if_needed(research_config)

    result = load_features_for_combo(bias_spec_for_combo, research_config)
    if result is None:
        print(
            "Failed to load feature/target for the chosen param combo; check cache and bias_spec.",
            file=sys.stderr,
        )
        sys.exit(1)
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
            "strategy": research_config.strategy,
            "selection_metric": research_config.binning_params.selection_metric,
        }
        binning_model = RuleBasedModel(**rule_params)

    feature_config = {
        "bias_node_spec": bias_node_spec,
        "model_type": "continuous_binning" if base.feature_type == FeatureType.CONTINUOUS else "rule_based",
        "constructor_params": binning_model.get_params(),
        "strategy": research_config.strategy,
    }
    base_model = BaseModel(
        feature_config=feature_config,
        tickers=research_config.tickers,
        binning_model=binning_model,
        use_cache=research_config.use_cache,
    )
    base_model.feature_column = feature_col

    ensemble_dir = create_ensemble_directory(
        base.timeframe,
        base.vault_save.ensemble_name,
        base.vault_save.direction,
        research_config.tickers,
    )
    add_feature_to_ensemble(
        feature_name=feature_col,
        bias_node_spec=bias_node_spec,
        base_model=base_model,
        ensemble_dir=ensemble_dir,
        tickers=research_config.tickers,
    )
    print(f"Saved feature '{feature_col}' to vault: {ensemble_dir}")


if __name__ == "__main__":
    _run()
