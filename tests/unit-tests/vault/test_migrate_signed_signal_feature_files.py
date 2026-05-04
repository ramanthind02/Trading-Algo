from __future__ import annotations

import json
from pathlib import Path

from ensemble.vault.feature_files import (
    migrate_legacy_signed_signal_feature_config,
    migrate_legacy_signed_signal_feature_file,
    validate_signed_signal_feature_config,
)


def test_migrate_legacy_signed_signal_strips_binning_keys() -> None:
    legacy = {
        "feature_name": "turnaroundtuesday_signal_D_mode_tue_wed",
        "created_at": "2026-03-07T05:54:47.000088+00:00",
        "updated_at": "2026-03-07T05:54:47.000099+00:00",
        "bias_node_spec": {
            "module_name": "turnaround_tuesday",
            "timeframes": ["D"],
        },
        "tickers": ["ES", "NQ"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "turnaroundtuesday_signal_D_mode_tue_wed::rule_based_3",
                "bias_node_params": {},
                "binning_model_type": "rule_based",
                "strategy": "long",
                "binning_model_params": {"n_bins": 3},
                "requires_fit": False,
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    }
    path = Path(
        "vault/D/mean_reversion_indices/mr_indices_long/features/"
        "turnaroundtuesday_signal_D_mode_tue_wed.json"
    )
    assert migrate_legacy_signed_signal_feature_config(legacy, feature_file=path) is True
    validate_signed_signal_feature_config(legacy, feature_file=path)
    m0 = legacy["base_models"][0]
    assert m0["model_type"] == "signed_signal"
    assert m0["feature_column"] == "turnaroundtuesday_signal_D_mode_tue_wed"
    assert m0["bias_node_spec"] == legacy["bias_node_spec"]
    assert "binning_model_type" not in m0


def test_migrate_merges_bias_node_params_into_top_spec() -> None:
    legacy = {
        "feature_name": "rebalancing_signal_D_crossTickers_TLT",
        "bias_node_spec": {
            "module_name": "rebalancing",
            "timeframes": ["D"],
        },
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3_cross_tickers_tlt",
                "model_name": "rebalancing_signal_D_crossTickers_TLT::rule_based_3_cross_tickers_tlt",
                "bias_node_params": {"cross_tickers": ["TLT"]},
                "binning_model_type": "rule_based",
                "strategy": "long",
                "binning_model_params": {"n_bins": 3},
                "requires_fit": False,
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    }
    path = Path("rebalancing.json")
    assert migrate_legacy_signed_signal_feature_config(legacy, feature_file=path) is True
    assert legacy["bias_node_spec"]["params"] == {"cross_tickers": ["TLT"]}
    assert legacy["base_models"][0]["bias_node_spec"]["params"] == {"cross_tickers": ["TLT"]}


def test_migrate_idempotent_on_new_schema(tmp_path: Path) -> None:
    path = tmp_path / "feat.json"
    modern = {
        "feature_name": "ibs_lower_band_signal_D_x",
        "created_at": "2026-04-10T10:48:45.307851+00:00",
        "updated_at": "2026-04-10T10:48:45.307851+00:00",
        "bias_node_spec": {
            "module_name": "ibs_lower_band",
            "timeframes": ["D"],
            "params": {"hl_mean_lookback": 25},
        },
        "tickers": ["NQ"],
        "base_models": [
            {
                "model_id": "signed_signal_x",
                "model_name": "ibs_lower_band_signal_D_x::signed_signal_x",
                "model_type": "signed_signal",
                "feature_column": "ibs_lower_band_signal_D_x",
                "strategy": "long",
                "bias_node_spec": {
                    "module_name": "ibs_lower_band",
                    "timeframes": ["D"],
                    "params": {"hl_mean_lookback": 25},
                },
            }
        ],
    }
    path.write_text(json.dumps(modern), encoding="utf-8")
    assert migrate_legacy_signed_signal_feature_file(path) is False


def test_migrate_legacy_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    legacy = {
        "feature_name": "f",
        "bias_node_spec": {"module_name": "m", "timeframes": ["D"]},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "rule_based_3",
                "model_name": "f::rule_based_3",
                "bias_node_params": {},
                "binning_model_type": "rule_based",
                "strategy": "long",
                "binning_model_params": {},
                "requires_fit": False,
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert migrate_legacy_signed_signal_feature_file(path) is True
    assert migrate_legacy_signed_signal_feature_file(path) is False
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_signed_signal_feature_config(loaded, feature_file=path)


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    raw = '{"feature_name": "f", "bias_node_spec": {"module_name": "m", "timeframes": ["D"]}, "tickers": ["ES"], "base_models": [{"model_id": "rule_based_3", "model_name": "f::rule_based_3", "bias_node_params": {}, "binning_model_type": "rule_based", "strategy": "long", "binning_model_params": {}, "requires_fit": false, "is_fitted": false, "fitted_params": null}]}'
    path.write_text(raw, encoding="utf-8")
    assert migrate_legacy_signed_signal_feature_file(path, dry_run=True) is True
    assert path.read_text(encoding="utf-8") == raw
