"""Unit tests for portfolio_tester resampling and intraday aggregation helpers."""

import pandas as pd

from ensemble.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
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
