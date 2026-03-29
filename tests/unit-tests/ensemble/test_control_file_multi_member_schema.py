"""Tests for single-feature control-file schema validation."""

import json
import tempfile
from pathlib import Path

import pytest

from feature_selection.domain_discrete import DomainDiscreteMigrationError
from ensemble.ensemble_utils import (
    parse_control_file,
    validate_base_model_config,
    validate_control_file,
)


def _base_model_config() -> dict:
    return {
        "name": "test_model",
        "model_type": "domain_discrete",
        "feature_column": "test_feature",
        "strategy": "long",
        "bias_node_spec": {
            "module_name": "domain_discrete",
            "timeframes": ["D"],
            "params": {
                "source_bias_node_spec": {
                    "module_name": "rsi",
                    "timeframes": ["D"],
                    "params": {"lookback": 14},
                },
                "ticker_scope": {"tickers": ["ES"], "scope_name": "ES"},
                "edges": [-0.5, 0.5],
                "n_bins": 3,
                "long_bins": [2],
                "short_bins": [],
                "direction": "long",
                "spec_version": "v1",
            },
        },
    }


def test_base_model_config_allows_domain_discrete_schema() -> None:
    config = _base_model_config()
    validate_base_model_config(config, index=0)


def test_base_model_config_rejects_members_key_even_if_empty() -> None:
    config = _base_model_config()
    config["members"] = []
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_base_model_config(config, index=0)


def test_base_model_config_rejects_members_key_when_non_list() -> None:
    config = _base_model_config()
    config["members"] = "not_a_list"
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_base_model_config(config, index=0)


def test_base_model_config_rejects_member_payload_shape() -> None:
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
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_base_model_config(config, index=0)


def test_control_file_accepts_domain_discrete_schema() -> None:
    control_file = {
        "metadata": {"is_fit": False, "version": "2.0.0"},
        "base_models": [_base_model_config()],
    }
    validate_control_file(control_file)


def test_parse_control_file_with_members_fails() -> None:
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
        with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
            parse_control_file(filepath)
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_control_file_is_not_fit_allows_missing_selection_method() -> None:
    control_file = {
        "metadata": {"is_fit": False, "version": "2.0.0"},
        "base_models": [_base_model_config()],
    }
    validate_control_file(control_file)


def test_control_file_rejects_top_level_fitted_base_models() -> None:
    control_file = {
        "metadata": {"is_fit": True, "version": "2.0.0"},
        "base_models": [_base_model_config()],
        "fitted_base_models": {},
    }
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_control_file(control_file)


def test_base_model_config_rejects_legacy_continuous_schema() -> None:
    config = _base_model_config()
    config["model_type"] = "continuous_binning"
    config["constructor_params"] = {"n_bins": 3}
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_base_model_config(config, index=0)


def test_base_model_config_rejects_legacy_rule_based_schema() -> None:
    config = _base_model_config()
    config["model_type"] = "rule_based"
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        validate_base_model_config(config, index=0)
