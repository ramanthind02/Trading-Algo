from itertools import product

import pytest

from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    param_combo_label,
)
from utils.enums import TimeFrame


def test_expand_bias_specs_single_param():
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 2
    assert result[0]["params"] == {"lookback": 2}
    assert result[1]["params"] == {"lookback": 3}
    assert all(r["module_name"] == "rsi" for r in result)


def test_expand_bias_specs_grid():
    spec = {
        "module_name": "cmma",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [20, 50], "atr_length": [14, 21]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 4  # 2x2 grid


def test_expand_bias_specs_scalar_params():
    """Scalar param values (not lists) should be treated as single-element lists."""
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": 5},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 1
    assert result[0]["params"] == {"lookback": 5}


def test_param_combo_label_single():
    assert param_combo_label({"lookback": 5}) == "lookback_5"


def test_param_combo_label_multi():
    label = param_combo_label({"lookback": 20, "atr_length": 14})
    # deterministic: sorted keys
    assert label == "atr_length_14__lookback_20"
