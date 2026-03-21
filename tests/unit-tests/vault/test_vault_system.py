"""Unit tests for vault-manager schema-v2 behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ensemble.vault_manager import (
    add_feature_to_ensemble,
    create_ensemble_directory,
    generate_model_id,
    get_all_base_model_names,
    get_bias_node_specs,
    list_features,
    load_feature_base_models,
    update_base_model_fitted_params,
    validate_ensemble_directory,
)
from feature_selection.base_models import ContinuousBinningModel
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Direction, TimeFrame, Ticker


def _make_base_model(lookback: int = 2) -> BaseModel:
    bias_node_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": lookback},
    }
    model = BaseModel(
        feature_config={"bias_node_spec": bias_node_spec},
        tickers=[Ticker.ES],
        binning_model=ContinuousBinningModel(n_bins=3, strategy="long"),
        use_cache=False,
    )
    idx = pd.date_range("2020-01-01", periods=150, freq="D")
    candles = pd.DataFrame(
        {
            "datetime": idx,
            "open": 100.0 + np.arange(len(idx)) * 0.1,
            "high": 101.0 + np.arange(len(idx)) * 0.1,
            "low": 99.0 + np.arange(len(idx)) * 0.1,
            "close": 100.2 + np.arange(len(idx)) * 0.1,
            "volume": 1_000_000.0,
            "ticker": Ticker.ES,
            "timeframe": TimeFrame.D,
        }
    )
    rng = np.random.default_rng(42)
    target = pd.Series(0.01 + rng.normal(0, 0.002, len(idx)), index=idx)
    model.fit(candles, target)
    return model


def test_generate_model_id_appends_bias_node_params() -> None:
    model_id = generate_model_id(
        "rule_based",
        {"selection_metric": "t_stat"},
        bias_node_params={"lookback": 2},
    )
    assert model_id == "rule_based_lookback_2"


def test_add_feature_writes_feature_name_only(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="schema_v2",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=2)
    bias_node_spec = model.bias_node_spec

    model_id = add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={
            "module_name": bias_node_spec["module_name"],
            "timeframes": bias_node_spec["timeframes"],
        },
        bias_node_params={"lookback": 2},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )

    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    assert feature_file.exists()
    with open(feature_file, "r") as handle:
        payload = json.load(handle)
    assert payload["feature_name"] == "rsi_signal_D"
    assert "feature_column" not in payload
    assert payload["base_models"][0]["model_id"] == model_id
    assert payload["base_models"][0]["bias_node_params"] == {"lookback": 2}
    assert payload["base_models"][0]["requires_fit"] is True


def test_load_models_merges_bias_node_params_single_model(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="load_members",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=5)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 5},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )

    loaded = load_feature_base_models(
        feature_name="rsi_signal_D",
        ensemble_dir=ensemble_dir,
    )
    assert len(loaded) == 1
    loaded_model = next(iter(loaded.values()))
    assert loaded_model.bias_node_spec["params"]["lookback"] == 5


def test_update_fitted_params_path(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="member_update",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=2)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 2},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )
    model_id = Path(ensemble_dir).joinpath("features", "rsi_signal_D.json")
    with open(model_id, "r") as handle:
        payload = json.load(handle)
    base_model_id = payload["base_models"][0]["model_id"]

    fitted_payload = model.binning_model.get_fitted_params()
    update_base_model_fitted_params(
        ensemble_dir=ensemble_dir,
        feature_name="rsi_signal_D",
        model_id=base_model_id,
        fitted_params=fitted_payload,
        train_start="2020-01-01",
        train_end="2020-12-31",
    )
    with open(model_id, "r") as handle:
        updated = json.load(handle)
    entry = updated["base_models"][0]
    assert entry["is_fitted"] is True
    assert entry["fitted_params"]["model_version"] == "binning_v2"


def test_list_specs_names_and_validate(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="specs_names",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=2)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 2},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )

    features_df = list_features(ensemble_dir)
    assert "feature_name" in features_df.columns
    assert features_df.iloc[0]["feature_name"] == "rsi_signal_D"

    specs = get_bias_node_specs(ensemble_dir)
    assert len(specs) == 1
    assert specs[0]["params"]["lookback"] == 2

    names = get_all_base_model_names(ensemble_dir)
    assert len(names) == 1
    assert names[0].startswith("rsi_signal_D::")

    validate_ensemble_directory(ensemble_dir)


def test_validate_rejects_legacy_members_schema(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="reject_members",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=2)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 2},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r") as handle:
        payload = json.load(handle)
    payload["base_models"][0]["members"] = []
    with open(feature_file, "w") as handle:
        json.dump(payload, handle, indent=2)

    with pytest.raises(ValueError, match="legacy members schema"):
        validate_ensemble_directory(ensemble_dir)


def test_validate_rejects_multiple_base_models_in_feature_file(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="reject_multi_models",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_base_model(lookback=2)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 2},
        base_model=model,
        ensemble_dir=ensemble_dir,
        tickers=[Ticker.ES],
    )
    feature_file = Path(ensemble_dir) / "features" / "rsi_signal_D.json"
    with open(feature_file, "r") as handle:
        payload = json.load(handle)
    payload["base_models"].append(dict(payload["base_models"][0], model_id="dup", model_name="rsi_signal_D::dup"))
    with open(feature_file, "w") as handle:
        json.dump(payload, handle, indent=2)

    with pytest.raises(ValueError, match="exactly one base model"):
        validate_ensemble_directory(ensemble_dir)
