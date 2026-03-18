from __future__ import annotations

import json
from datetime import datetime
from math import log

import numpy as np
import pandas as pd
from feature_research.config import FeatureType
from feature_research.in_sample.config import BinningAnalysisConfig
from feature_selection.base_models.rule_based import RuleBasedModel
from utils.evaluation.walkforward.portfolio_evaluator import (
    _enforce_rule_based_signed_multipliers,
    _calculate_oos_returns_from_positions,
    build_research_portfolio,
    ensure_portfolio_candle_columns,
    RULE_BASED_BIN_COUNT,
)
from utils.core.enums import TimeFrame, Ticker


def _install_fake_ensemble(monkeypatch) -> None:
    class _FakeEnsemble:
        def __init__(self, *args, **kwargs) -> None:
            control_file_path = kwargs["control_file_path"]
            with open(control_file_path, encoding="utf-8") as handle:
                self.control_file_data = json.load(handle)
            self.base_models = {}
            for model in self.control_file_data.get("base_models", []):
                self.base_models[model["name"]] = object()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.DiversifiedEnsemble",
        _FakeEnsemble,
    )


def test_build_research_portfolio_creates_expected_models(monkeypatch) -> None:
    _install_fake_ensemble(monkeypatch)
    portfolio = build_research_portfolio(
        selected_params=[{"lookback": 5, "bin_count": 4}, {"lookback": 7, "bin_count": 6}],
        binning_config=BinningAnalysisConfig(strategy="long"),
        tickers=[Ticker.ES],
        trading_timeframe=TimeFrame.D,
        module_name="rsi",
        feature_type=FeatureType.CONTINUOUS,
    )

    assert len(portfolio.ensembles) == 1
    ensemble = portfolio.ensembles[0]
    assert "rsi_signal_D_lookback_5_long" in ensemble.base_models
    assert "rsi_signal_D_lookback_7_long" in ensemble.base_models
    control_file = ensemble.control_file_data or {}
    base_models = control_file.get("base_models", [])
    assert base_models
    members = base_models[0]["members"]
    assert all("member_name" in member for member in members)
    assert all("params" in member for member in members)
    assert not portfolio.is_fitted_


def test_build_research_portfolio_rule_based_has_three_members_and_bias_node_spec(monkeypatch) -> None:
    """Rule-based control payload uses RULE_BASED_BIN_COUNT members and includes bias_node_spec."""
    _install_fake_ensemble(monkeypatch)
    portfolio = build_research_portfolio(
        selected_params=[{"oversold": 25, "overbought": 75, "rsi_period": 2}],
        binning_config=BinningAnalysisConfig(strategy="long"),
        tickers=[Ticker.ES],
        trading_timeframe=TimeFrame.D,
        module_name="rsi_signal",
        feature_type=FeatureType.RULE_BASED,
    )
    assert len(portfolio.ensembles) == 1
    ensemble = portfolio.ensembles[0]
    control_file = ensemble.control_file_data or {}
    base_models = control_file.get("base_models", [])
    assert len(base_models) == 1
    cfg = base_models[0]
    assert cfg.get("model_type") == "rule_based"
    members = cfg["members"]
    assert len(members) == RULE_BASED_BIN_COUNT
    assert [m["bin_index"] for m in members] == list(range(RULE_BASED_BIN_COUNT))
    bias_spec = cfg.get("bias_node_spec")
    assert bias_spec is not None
    assert bias_spec.get("module_name") == "rsi_signal"
    assert "timeframes" in bias_spec
    assert bias_spec.get("params") == {"oversold": 25, "overbought": 75, "rsi_period": 2}


def test_ensure_portfolio_candle_columns_adds_volume_and_timeframe() -> None:
    candles = pd.DataFrame(
        {
            "datetime": pd.date_range(datetime(2020, 1, 1), periods=3, freq="D"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "ticker": ["ES", "ES", "ES"],
        }
    )

    normalized = ensure_portfolio_candle_columns(candles, trading_timeframe=TimeFrame.D)

    assert "volume" in normalized.columns
    assert "timeframe" in normalized.columns
    assert (normalized["volume"] == 0.0).all()
    assert (normalized["timeframe"] == TimeFrame.D).all()


def test_calculate_oos_returns_from_positions_preserves_multi_ticker_aggregation() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    candles = pd.DataFrame(
        {
            "datetime": [dates[0], dates[1], dates[2], dates[0], dates[1], dates[2]],
            "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"],
            "open": [0.0] * 6,
            "high": [0.0] * 6,
            "low": [0.0] * 6,
            "close": [100.0, 110.0, 99.0, 200.0, 190.0, 209.0],
            "volume": [0.0] * 6,
            "timeframe": [TimeFrame.D] * 6,
        }
    )
    positions = pd.DataFrame(
        {
            "datetime": [dates[0], dates[1], dates[0], dates[1]],
            "ticker": ["ES", "ES", "NQ", "NQ"],
            "position_fraction": [1.0, -0.25, 0.5, 1.0],
            "forecast_score": [10.0, -2.5, 5.0, 10.0],
        }
    )

    returns = _calculate_oos_returns_from_positions(
        positions_df=positions,
        candles_df=candles,
        series_name="portfolio_returns",
    )

    # After reindex to full candle calendar, dates[0] is filled with 0.0
    expected = pd.Series(
        [
            0.0,
            1.0 * log(110.0 / 100.0) + 0.5 * log(190.0 / 200.0),
            -0.25 * log(99.0 / 110.0) + 1.0 * log(209.0 / 190.0),
        ],
        index=pd.DatetimeIndex([dates[0], dates[1], dates[2]]),
        name="portfolio_returns",
    )

    pd.testing.assert_series_equal(returns, expected)


def test_enforce_rule_based_signed_multipliers_preserves_cached_signal_direction() -> None:
    index = pd.date_range("2020-01-01", periods=240, freq="D")
    feature = pd.Series(np.tile([-1, 1], 120), index=index, name="seasonal_signal_D")
    target = pd.Series(-0.01, index=index)

    model = RuleBasedModel(strategy="long_short")
    model.fit(feature, target)
    _enforce_rule_based_signed_multipliers(model, "long_short")
    pred = model.predict(feature, strategy="long_short")

    assert (pred[feature == 1] > 0).all()
    assert (pred[feature == -1] < 0).all()
