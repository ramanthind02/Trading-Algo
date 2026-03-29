"""Unit tests for save_to_vault script and helpers."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from feature_research.config import (
    load_config as load_base_config,
)
from feature_research.save_to_vault import (
    _resolve_param_combos,
    _validate_frozen_spec_payload,
    _run,
)


def _frozen_payload(spec_version: str = "v1") -> dict[str, object]:
    return {
        "source_bias_node_spec": {
            "module_name": "cyclical_rsi",
            "timeframes": ["D"],
            "params": {"short_period": 4, "long_period": 120, "rsi_period": 2},
        },
        "ticker_scope": {"kind": "specific", "tickers": ["ES"]},
        "edges": [-1.0, 0.0, 1.0],
        "n_bins": 2,
        "long_bins": [1],
        "short_bins": [0],
        "direction": "long_short",
        "spec_version": spec_version,
    }


def test_resolve_param_combos_uses_params_to_save_when_set() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"short_period": [4, 8], "long_period": [120], "rsi_period": [2]},
    }
    params_to_save = _frozen_payload()
    combos = _resolve_param_combos(bias_spec, params_to_save)
    assert len(combos) == 1
    assert combos[0] == params_to_save


def test_resolve_param_combos_raises_when_params_to_save_none() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"short_period": [4], "long_period": [120], "rsi_period": [2]},
    }
    with pytest.raises(ValueError, match="must now provide frozen domain-discrete specs"):
        _resolve_param_combos(bias_spec, None)


def test_resolve_param_combos_raises_when_no_combos() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {},
    }
    with pytest.raises(ValueError, match="must now provide frozen domain-discrete specs"):
        _resolve_param_combos(bias_spec, None)
    bias_spec_empty_lists = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"x": []},
    }
    with pytest.raises(ValueError, match="No param combo to save"):
        _resolve_param_combos(bias_spec_empty_lists, None)


def test_resolve_param_combos_returns_multiple_when_list_of_frozen_specs() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"short_period": [4, 8], "long_period": [120], "rsi_period": [2]},
    }
    combos = _resolve_param_combos(bias_spec, [_frozen_payload("v1"), _frozen_payload("v2")])
    assert len(combos) == 2
    assert [combo["spec_version"] for combo in combos] == ["v1", "v2"]


def test_validate_frozen_spec_payload_rejects_legacy_raw_params() -> None:
    with pytest.raises(ValueError, match="Frozen domain-discrete specs require keys"):
        _validate_frozen_spec_payload({"lookback": 5})


def test_run_exits_when_vault_save_none() -> None:
    base = load_base_config()
    config_without_vault = replace(base, vault_save=None)
    with patch("feature_research.save_to_vault.load_config", return_value=config_without_vault):
        with pytest.raises(SystemExit) as exc_info:
            _run()
    assert exc_info.value.code == 1
