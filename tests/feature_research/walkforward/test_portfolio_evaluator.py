from __future__ import annotations

from datetime import datetime

import pandas as pd
from feature_research.in_sample.continuous_binning.config import BinningAnalysisConfig
from feature_research.walkforward.portfolio_evaluator import (
    build_research_portfolio,
    ensure_portfolio_candle_columns,
)
from utils.enums import TimeFrame, Ticker

def test_build_research_portfolio_creates_expected_models() -> None:
    portfolio = build_research_portfolio(
        selected_params=[{"lookback": 5, "bin_count": 4}, {"lookback": 7, "bin_count": 6}],
        binning_config=BinningAnalysisConfig(strategy="long"),
        tickers=[Ticker.ES],
        trading_timeframe=TimeFrame.D,
        module_name="rsi",
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
