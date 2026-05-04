from __future__ import annotations

from utils.core.enums import TimeFrame
from feature_research.bias_spec_catalog import first_bias_spec
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    expanded_combo_param_value,
    expanded_spec_combo_label,
    param_combo_label,
)


def test_expand_bias_specs_single_param():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": [2, 3]},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 2
    assert result[0]["params"] == {"rsi_period": 2}
    assert result[1]["params"] == {"rsi_period": 3}
    assert all(r["module_name"] == "rsi_signal" for r in result)


def test_expand_bias_specs_grid():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": [2, 3], "oversold": [20.0, 25.0]},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 4  # 2 × 2


def test_expand_bias_specs_scalar_params():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": 2, "oversold": 25.0},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 1
    assert result[0]["params"] == {"rsi_period": 2, "oversold": 25.0}


def test_expand_bias_specs_list_catalog_concatenates() -> None:
    a = {
        "module_name": "donchian_long_only",
        "timeframes": [TimeFrame.D],
        "params": {"channel_lookback": [10, 20]},
    }
    b = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": [2, 3]},
    }
    out = expand_bias_specs([a, b])
    assert len(out) == 4
    assert {r["module_name"] for r in out} == {"donchian_long_only", "rsi_signal"}


def test_expand_bias_specs_multi_module_name_cartesian() -> None:
    spec = {
        "module_name": ["filter_gate", "filter_gate_entry_only"],
        "timeframes": [TimeFrame.D],
        "params": {
            "filter_module": "atr_percentile_filter",
            "filter_params": {"atr_period": [10, 14]},
            "signal_module": "donchian_long_only",
            "signal_params": {"channel_lookback": 20, "sma_period": 350},
        },
    }
    out = expand_bias_specs(spec)
    assert len(out) == 4
    assert sum(1 for r in out if r["module_name"] == "filter_gate") == 2


def test_first_bias_spec() -> None:
    assert first_bias_spec({"module_name": "x", "params": {}})["module_name"] == "x"
    assert first_bias_spec([{"module_name": "a"}, {"module_name": "b"}])["module_name"] == "a"


def test_expanded_combo_param_value_top_level_and_nested() -> None:
    assert expanded_combo_param_value({"channel_lookback": 20}, "channel_lookback") == 20
    nested = {
        "signal_module": "donchian_long_only",
        "signal_params": {"channel_lookback": 30, "sma_period": 350},
    }
    assert expanded_combo_param_value(nested, "channel_lookback") == 30
    assert expanded_combo_param_value(nested, "missing") is None


def test_expanded_spec_combo_label_distinguishes_gate_variants() -> None:
    params = {"channel_lookback": 20, "sma_period": 350}
    a = expanded_spec_combo_label(
        {"module_name": "filter_gate", "params": params},
    )
    b = expanded_spec_combo_label(
        {"module_name": "filter_gate_entry_only", "params": params},
    )
    assert a != b
    assert a.startswith("filter_gate__")
    assert b.startswith("filter_gate_entry_only__")


def test_param_combo_label_single():
    assert param_combo_label({"rsi_period": 2}) == "rsi_period_2"


def test_param_combo_label_multi():
    label = param_combo_label({"rsi_period": 2, "oversold": 25.0})
    assert label == "oversold_25.0__rsi_period_2"  # sorted alphabetically
