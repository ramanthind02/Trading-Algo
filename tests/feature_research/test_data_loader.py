from itertools import product
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from feature_research.in_sample.config import ResearchConfig
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
)
from utils.core.enums import TimeFrame


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


def test_load_features_for_combo_rejects_raw_return_with_multi_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_df = pd.DataFrame(
        {
            "rsi_signal_D_lookback_5": [45.0, 55.0],
            "ticker": ["ES", "NQ"],
        }
    )
    targets_df = pd.DataFrame(
        {
            "log_return": [0.001, 0.002],
            "ticker": ["ES", "NQ"],
        }
    )

    def _fake_extract_features_for_bias_node(**_: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        return features_df, targets_df

    monkeypatch.setattr(
        "feature_research.in_sample.data_loader.extract_features_for_bias_node",
        _fake_extract_features_for_bias_node,
    )

    from feature_research.config import FeatureType
    config = cast(
        ResearchConfig,
        SimpleNamespace(
            tickers=["ES", "NQ"],
            start=None,
            end=None,
            target_col="log_return",
            use_cache=False,
            feature_type=FeatureType.CONTINUOUS,
        ),
    )

    with pytest.raises(ValueError, match="Raw return targets must not be mixed"):
        load_features_for_combo(
            single_combo_spec={
                "module_name": "rsi",
                "params": {"lookback": 5},
                "timeframes": [TimeFrame.D],
            },
            config=config,
        )
