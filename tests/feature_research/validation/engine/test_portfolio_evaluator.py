from __future__ import annotations

from datetime import datetime
from math import log

import pandas as pd
import pytest

from utils.core.enums import TimeFrame, Ticker
from utils.evaluation.walkforward.portfolio_evaluator import (
    _calculate_oos_returns_from_positions,
    build_research_portfolio,
    ensure_portfolio_candle_columns,
    evaluate_fold_portfolio,
)


def test_build_research_portfolio_raises_for_legacy_fitting() -> None:
    with pytest.raises(RuntimeError, match="frozen-signal"):
        build_research_portfolio()


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
            0.0,
            1.0 * log(110.0 / 100.0) + 0.5 * log(190.0 / 200.0),
            -0.25 * log(99.0 / 110.0) + 1.0 * log(209.0 / 190.0),
        ],
        index=pd.DatetimeIndex([dates[0], dates[1], dates[2]]),
        name="portfolio_returns",
    )

    pd.testing.assert_series_equal(returns, expected)


def test_evaluate_fold_portfolio_aggregates_frozen_signals() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    candles = pd.DataFrame(
        {
            "datetime": index,
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "ticker": ["ES", "NQ", "ES", "NQ"],
        }
    )
    target = pd.Series([0.01, -0.02, 0.03, 0.04], index=index)
    feature_data_by_combo = {
        tuple(sorted({"lookback": 5}.items())): pd.DataFrame(
            {"signal": pd.Series([1.0, 1.0, 1.0, 1.0], index=index), "target": target}
        ),
    }

    result = evaluate_fold_portfolio(
        train_candles=candles.iloc[:2],
        test_candles=candles.iloc[2:],
        selected_params=[{"lookback": 5}],
        target_series=target,
        binning_config=object(),
        tickers=[Ticker.ES, Ticker.NQ],
        feature_data_by_combo=feature_data_by_combo,
    )

    assert result.n_params_selected == 1
    assert result.per_signal_oos_returns is not None
    assert result.oos_portfolio_returns.tolist() == [0.03, 0.04]
