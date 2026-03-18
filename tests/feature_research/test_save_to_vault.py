"""Unit tests for save_to_vault script and helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from feature_research.config import (
    BaseResearchConfig,
    BinningAnalysisConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    PermutationResearchConfig,
    VaultSaveConfig,
    load_config as load_base_config,
)
from feature_research.save_to_vault import (
    _binning_params_to_constructor_params,
    _resolve_single_combo,
    _run,
)


def test_resolve_single_combo_uses_params_to_save_when_set() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"short_period": [4, 8], "long_period": [120], "rsi_period": [2]},
    }
    params_to_save = {"short_period": 4, "long_period": 120, "rsi_period": 2}
    combo = _resolve_single_combo(bias_spec, params_to_save)
    assert combo["module_name"] == "cyclical_rsi"
    assert combo["params"] == params_to_save
    assert combo["timeframes"] == ["D"]


def test_resolve_single_combo_uses_first_expanded_when_params_to_save_none() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"short_period": [4], "long_period": [120], "rsi_period": [2]},
    }
    combo = _resolve_single_combo(bias_spec, None)
    assert combo["module_name"] == "cyclical_rsi"
    assert combo["params"] == {"short_period": 4, "long_period": 120, "rsi_period": 2}


def test_resolve_single_combo_raises_when_no_combos() -> None:
    bias_spec = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {},
    }
    expanded = _resolve_single_combo(bias_spec, None)
    assert expanded["params"] == {}
    bias_spec_empty_lists = {
        "module_name": "cyclical_rsi",
        "timeframes": ["D"],
        "params": {"x": []},
    }
    with pytest.raises(ValueError, match="No param combo to save"):
        _resolve_single_combo(bias_spec_empty_lists, None)


def test_binning_params_to_constructor_params() -> None:
    binning_params = BinningAnalysisConfig(
        bin_counts=[10, 8],
        strategy="long",
        bin_index_min=0,
        bin_index_max=2,
    )
    out = _binning_params_to_constructor_params(binning_params)
    assert out["n_bins"] == 10
    assert out["bin_counts"] == [10, 8]
    assert out["strategy"] == "long"
    assert out["bin_index_min"] == 0
    assert out["bin_index_max"] == 2


def test_run_exits_when_vault_save_none() -> None:
    base = load_base_config()
    config_without_vault = replace(base, vault_save=None)
    with patch("feature_research.save_to_vault.load_config", return_value=config_without_vault):
        with pytest.raises(SystemExit) as exc_info:
            _run()
    assert exc_info.value.code == 1
