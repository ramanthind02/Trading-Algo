from __future__ import annotations

from utils.enums import TimeFrame
from feature_research.in_sample.rule_based.data_loader import (
    expand_bias_specs,
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


def test_param_combo_label_single():
    assert param_combo_label({"rsi_period": 2}) == "rsi_period_2"


def test_param_combo_label_multi():
    label = param_combo_label({"rsi_period": 2, "oversold": 25.0})
    assert label == "oversold_25.0__rsi_period_2"  # sorted alphabetically
