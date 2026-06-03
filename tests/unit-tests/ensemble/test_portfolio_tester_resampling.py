"""Unit tests for portfolio_tester resampling and intraday aggregation helpers."""

import math

import pandas as pd

from ensemble.portfolio_impl.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
    calculate_strategy_returns_from_positions,
    filter_candles_to_position_tickers,
    resample_positions_to_daily,
)


def test_resample_positions_to_daily_forward_fills_and_prepends_zeros() -> None:
    positions_df = pd.DataFrame(
        {
            "ticker": ["ES", "ES", "NQ"],
            "datetime": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-03"),
            ],
            "position_fraction": [1.0, -0.5, 0.25],
        }
    )
    daily_dates = {
        "ES": pd.date_range("2024-01-01", "2024-01-06", freq="D"),
        "NQ": pd.date_range("2024-01-01", "2024-01-06", freq="D"),
    }

    result = resample_positions_to_daily(positions_df, daily_dates)

    es_values = result[result["ticker"] == "ES"]["position_fraction"].tolist()
    nq_values = result[result["ticker"] == "NQ"]["position_fraction"].tolist()

    assert es_values == [0.0, 1.0, 1.0, 1.0, -0.5, -0.5]
    assert nq_values == [0.0, 0.0, 0.25, 0.25, 0.25, 0.25]


def test_aggregate_intraday_returns_to_daily_sums_intraday_bars() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 16:00:00",
                "2024-01-02 09:30:00",
            ]
        ),
        name="strategy_return",
    )

    aggregated = aggregate_intraday_returns_to_daily(returns)

    expected_index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    expected = pd.Series([-0.01, 0.03], index=expected_index, name="strategy_return")
    pd.testing.assert_series_equal(aggregated, expected)


def test_aggregate_intraday_returns_to_daily_noop_for_unique_day_frequencies() -> None:
    for freq in ("D", "W", "ME"):
        idx = pd.date_range("2024-01-01", periods=4, freq=freq)
        returns = pd.Series([0.1, -0.2, 0.05, 0.03], index=idx, name="baseline_return")
        result = aggregate_intraday_returns_to_daily(returns)
        pd.testing.assert_series_equal(result, returns)


def test_calculate_strategy_returns_log_vs_simple_differs_on_synthetic_candles() -> None:
    """Simple returns match pct-change; log uses diff(log(close)); both use same lookahead merge."""
    candles = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["ES", "ES"],
            "close": [100.0, 110.0],
        }
    )
    positions = pd.DataFrame(
        {
            "ticker": ["ES"],
            "datetime": pd.to_datetime(["2024-01-01"]),
            "position_fraction": [1.0],
        }
    )
    log_ret = calculate_strategy_returns_from_positions(
        positions, candles, instrument_return_kind="log"
    )
    simple_ret = calculate_strategy_returns_from_positions(
        positions, candles, instrument_return_kind="simple"
    )

    assert len(log_ret) == 1
    assert len(simple_ret) == 1
    expected_simple = 0.1
    expected_log = math.log(110.0 / 100.0)
    assert abs(float(simple_ret.iloc[0]) - expected_simple) < 1e-12
    assert abs(float(log_ret.iloc[0]) - expected_log) < 1e-12
    assert abs(float(simple_ret.iloc[0]) - float(log_ret.iloc[0])) > 1e-6


def test_filter_candles_to_position_tickers_limits_return_instrument() -> None:
    """Strategy returns must use the position ticker's instrument move, not a peer."""
    import math

    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    positions = pd.DataFrame(
        {
            "ticker": ["GC", "GC"],
            "datetime": dates[:2],
            "position_fraction": [1.0, 1.0],
        }
    )
    candles = pd.DataFrame(
        {
            "ticker": ["GC", "GC", "ES", "ES", "ES"],
            "datetime": [
                dates[0],
                dates[1],
                dates[0],
                dates[1],
                dates[2],
            ],
            "close": [100.0, 110.0, 200.0, 240.0, 240.0],
        }
    )
    scoped = filter_candles_to_position_tickers(positions, candles)
    returns = calculate_strategy_returns_from_positions(
        positions,
        scoped,
        instrument_return_kind="simple",
    )
    expected = 0.1
    assert len(returns) == 1
    assert abs(float(returns.iloc[0]) - expected) < 1e-12
