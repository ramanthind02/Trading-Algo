"""Tests for control-file member schema validation."""

import json
import tempfile
from pathlib import Path

import pytest

from ensemble.ensemble_utils import (
    parse_control_file,
    validate_base_model_config,
    validate_control_file,
)


def _base_model_config() -> dict:
    return {
        "name": "test_model",
        "model_type": "continuous_binning",
        "feature_column": "test_feature",
        "strategy": "long",
        "constructor_params": {"n_bins": 3},
    }


def test_base_model_config_allows_missing_members() -> None:
    config = _base_model_config()
    validate_base_model_config(config, index=0)


def test_base_model_config_allows_empty_members() -> None:
    config = _base_model_config()
    config["members"] = []
    validate_base_model_config(config, index=0)


def test_base_model_members_must_be_list() -> None:
    config = _base_model_config()
    config["members"] = "not_a_list"
    with pytest.raises(ValueError, match="members"):
        validate_base_model_config(config, index=0)


def test_base_model_members_accept_new_and_legacy_shapes() -> None:
    config = _base_model_config()
    config["members"] = [
        {
            "member_name": "cb10",
            "binning_model_type": "continuous_binning",
            "binning_model_params": {"n_bins": 10},
        },
        {
            "member_id": "legacy_1",
            "params": {"n_bins": 5},
        },
    ]
    validate_base_model_config(config, index=0)


def test_control_file_accepts_legacy_schema_without_members() -> None:
    control_file = {
        "metadata": {"is_fit": False, "version": "2.0.0"},
        "base_models": [_base_model_config()],
    }
    validate_control_file(control_file)


def test_parse_control_file_with_members() -> None:
    control_file = {
        "metadata": {"is_fit": False, "version": "2.0.0"},
        "base_models": [
            {
                **_base_model_config(),
                "members": [
                    {
                        "member_name": "cb10",
                        "binning_model_type": "continuous_binning",
                        "binning_model_params": {"n_bins": 10},
                    }
                ],
            }
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(control_file, f)
        filepath = f.name
    try:
        result = parse_control_file(filepath)
        assert "base_models" in result
        assert len(result["base_models"]) == 1
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_control_file_is_fit_allows_missing_selection_method() -> None:
    control_file = {
        "metadata": {"is_fit": True, "version": "2.0.0"},
        "base_models": [_base_model_config()],
        "fitted_base_models": {},
        "fitted_ensemble": {
            "weights": {"test_model": 1.0},
            "exposure_fractions": {"test_model": 0.5},
            "feature_names": ["test_model"],
            "target_volatility": 0.15,
            "unique_tickers": ["ES"],
            "instrument_weights": {"ES": 1.0},
            "n_tickers": 1,
        },
    }
    validate_control_file(control_file)
