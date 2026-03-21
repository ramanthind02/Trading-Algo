"""Tests for BaseModel single-combo contract."""

from __future__ import annotations

import pytest

from feature_selection.base_models import RuleBasedModel
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Ticker, TimeFrame


def _make_feature_config(params: dict) -> dict:
    return {
        "bias_node_spec": {
            "module_name": "buy_hold",
            "timeframes": [TimeFrame.D],
            "params": params,
        }
    }


def test_base_model_rejects_multi_value_param_lists() -> None:
    with pytest.raises(ValueError, match="requires a single param combo"):
        BaseModel(
            feature_config=_make_feature_config({"lookback": [2, 4]}),
            tickers=[Ticker.ES],
            binning_model=RuleBasedModel(strategy="long"),
            use_cache=False,
        )


def test_base_model_has_no_member_api() -> None:
    model = BaseModel(
        feature_config=_make_feature_config({}),
        tickers=[Ticker.ES],
        binning_model=RuleBasedModel(strategy="long"),
        use_cache=False,
    )
    assert not hasattr(model, "members")
    assert not hasattr(model, "add_member")
    assert not hasattr(model, "predict_members_from_candles")
