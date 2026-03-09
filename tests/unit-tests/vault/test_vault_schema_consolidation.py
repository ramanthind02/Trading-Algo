"""Tests for vault feature-file consolidation into canonical schema."""

import json
from pathlib import Path

from ensemble.vault_manager import consolidate_feature_files


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def test_consolidate_feature_files_merges_split_variants(tmp_path) -> None:
    ensemble_dir = tmp_path / "vault" / "D" / "mean-reversion_indices_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    file_a = features_dir / "rsi_signal_D_lookback_2.json"
    file_b = features_dir / "rsi_signal_D_lookback_5.json"

    _write_json(
        file_a,
        {
            "feature_column": "rsi_signal_D_lookback_2",
            "bias_node_spec": {
                "module_name": "rsi",
                "timeframes": ["D"],
                "params": {"lookback": 2},
            },
            "tickers": ["ES", "NQ"],
            "base_models": [
                {
                    "model_id": "legacy_q_10",
                    "model_name": "rsi_signal_D_lookback_2::legacy_q_10",
                    "model_type": "QuantileBinningModel",
                    "strategy": "long",
                    "constructor_params": {"n_bins": 10},
                    "is_fitted": True,
                    "fitted_params": {"thresholds": [0.1, 0.2]},
                }
            ],
        },
    )
    _write_json(
        file_b,
        {
            "feature_name": "rsi_signal_D_lookback_5",
            "bias_node_spec": {
                "module_name": "rsi",
                "timeframes": ["D"],
                "params": {"lookback": [5]},
            },
            "tickers": ["RTY"],
            "base_models": [
                {
                    "model_id": "rb",
                    "model_name": "rsi_signal_D_lookback_5::rb",
                    "binning_model_type": "rule_based",
                    "strategy": "long",
                    "binning_model_params": {"selection_metric": "t_stat"},
                    "is_fitted": False,
                    "fitted_params": None,
                }
            ],
        },
    )

    out = consolidate_feature_files(str(ensemble_dir))

    canonical_file = features_dir / "rsi_signal_D.json"
    assert str(canonical_file) in out
    assert canonical_file.exists()
    assert not file_a.exists()
    assert not file_b.exists()

    with open(canonical_file, "r") as handle:
        payload = json.load(handle)

    assert payload["feature_name"] == "rsi_signal_D"
    assert payload["bias_node_spec"]["module_name"] == "rsi"
    assert len(payload["base_models"]) == 2

    by_type = {m["binning_model_type"]: m for m in payload["base_models"]}
    assert "continuous_binning" in by_type
    assert "rule_based" in by_type

    cont = by_type["continuous_binning"]
    assert cont["requires_fit"] is True
    assert cont["is_fitted"] is False
    assert cont["fitted_params"] is None
    assert cont["bias_node_params"]["lookback"] == 2

    rb = by_type["rule_based"]
    assert rb["requires_fit"] is False
    assert rb["bias_node_params"]["lookback"] == 5
