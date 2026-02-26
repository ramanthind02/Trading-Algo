from __future__ import annotations

from datetime import datetime
from math import log

import pandas as pd
from feature_research.config import FeatureType
from feature_research.in_sample.config import BinningAnalysisConfig
from feature_research.walkforward.portfolio_evaluator import (
    _calculate_oos_returns_from_positions,
    build_research_portfolio,
    ensure_portfolio_candle_columns,
    RULE_BASED_BIN_COUNT,
)
from utils.core.enums import TimeFrame, Ticker


def test_build_research_portfolio_creates_expected_models() -> None:
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


def test_build_research_portfolio_rule_based_has_three_members_and_bias_node_spec() -> None:
    """Rule-based control payload uses RULE_BASED_BIN_COUNT members and includes bias_node_spec."""
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

    expected = pd.Series(
        [
            1.0 * log(110.0 / 100.0) + 0.5 * log(190.0 / 200.0),
            -0.25 * log(99.0 / 110.0) + 1.0 * log(209.0 / 190.0),
        ],
        index=pd.DatetimeIndex([dates[1], dates[2]]),
        name="portfolio_returns",
    )
    expected.index.name = "ret_datetime"

    pd.testing.assert_series_equal(returns, expected)
