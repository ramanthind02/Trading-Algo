"""Tests for BaseModel member fit/predict behavior."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd

from feature_selection.base_models import RuleBasedModel
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Ticker, TimeFrame


def _make_candles(periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": 100.0 + np.arange(periods) * 0.1,
            "high": 101.0 + np.arange(periods) * 0.1,
            "low": 99.0 + np.arange(periods) * 0.1,
            "close": 100.5 + np.arange(periods) * 0.1,
            "volume": 1_000_000.0,
            "ticker": Ticker.ES,
            "timeframe": TimeFrame.D,
        }
    )


def _make_target(candles_df: pd.DataFrame) -> pd.Series:
    idx = pd.to_datetime(candles_df["datetime"])
    values = np.random.default_rng(7).normal(0.0, 0.01, len(idx))
    return pd.Series(values, index=idx)


def _make_base_model() -> BaseModel:
    bias_node_spec = {
        "module_name": "buy_hold",
        "timeframes": [TimeFrame.D],
        "params": {},
    }
    return BaseModel(
        feature_config={"bias_node_spec": bias_node_spec},
        tickers=[Ticker.ES],
        binning_model=RuleBasedModel(strategy="long"),
        use_cache=False,
    )


def test_member_fit_skip_semantics_requires_fit_vs_prefit() -> None:
    base_model = _make_base_model()
    candles_df = _make_candles()
    target_data = _make_target(candles_df)

    static_member = Mock()
    static_member.is_fitted_ = True
    static_member.model_type = "rule_based"
    static_member.requires_fit = False

    refit_member = Mock()
    refit_member.is_fitted_ = False
    refit_member.model_type = "continuous_binning"
    refit_member.requires_fit = True

    base_model.add_member("static_rule", static_member)
    base_model.add_member("dynamic_cont", refit_member)

    base_model.fit(candles_df, target_data)

    static_member.fit.assert_not_called()
    refit_member.fit.assert_called_once()


def test_predict_members_from_candles_returns_member_columns() -> None:
    base_model = _make_base_model()
    candles_df = _make_candles()
    target_data = _make_target(candles_df)
    base_model.fit(candles_df, target_data)

    member = Mock()
    member.is_fitted_ = True
    member.model_type = "rule_based"
    member.requires_fit = False

    def _predict(feature_data: pd.Series, strategy: str = "long") -> pd.Series:
        return pd.Series(np.ones(len(feature_data)), index=feature_data.index)

    member.predict.side_effect = _predict
    base_model.add_member("rb_member", member)

    out = base_model.predict_members_from_candles(candles_df, strategy="long")

    assert isinstance(out, pd.DataFrame)
    assert "rb_member" in out.columns
    assert len(out) > 0
