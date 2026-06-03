from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from utils.compute.daily_ewsd_volatility import DailyEWSDVolatilityService


def _daily_candles(
    ticker: str,
    start: str,
    closes: list[float],
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "ticker": [ticker] * len(closes),
            "close": closes,
        }
    )


def test_compute_daily_series_outputs_required_schema_and_positive_vol() -> None:
    candles = pd.concat(
        [
            _daily_candles("ES", "2024-01-01", [100.0, 101.0, 102.0, 101.5]),
            _daily_candles("NQ", "2024-01-01", [200.0, 198.0, 201.0, 202.0]),
        ],
        ignore_index=True,
    )
    service = DailyEWSDVolatilityService()
    out = service.compute_daily_series(candles)

    assert list(out.columns) == ["datetime", "ticker", "ewsd_annual_vol"]
    assert len(out) == len(candles)
    assert (out["ewsd_annual_vol"] > 0).all()
    assert set(out["ticker"].unique()) == {"ES", "NQ"}


def test_align_daily_volatility_to_intraday_candles_ffills_per_ticker() -> None:
    service = DailyEWSDVolatilityService()
    daily_volatility = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "ticker": ["ES", "ES"],
            "ewsd_annual_vol": [0.20, 0.30],
        }
    )
    intraday = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-02 10:00:00",
                    "2024-01-03 10:00:00",
                ]
            ),
            "ticker": ["ES", "ES", "ES"],
        }
    )

    aligned = service.align_daily_volatility_to_candles(daily_volatility, intraday)
    assert aligned["ewsd_annual_vol"].tolist() == [0.20, 0.20, 0.30]


def test_align_daily_volatility_ffills_month_end_calendar_dates() -> None:
    """Month-end candle dates not in the daily vol index still receive last trading-day vol."""
    service = DailyEWSDVolatilityService()
    daily_volatility = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-03-28", "2024-04-01"]),
            "ticker": ["ES", "ES"],
            "ewsd_annual_vol": [0.18, 0.19],
        }
    )
    monthly_candle = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-03-31"]),
            "ticker": ["ES"],
        }
    )
    aligned = service.align_daily_volatility_to_candles(daily_volatility, monthly_candle)
    assert aligned["ewsd_annual_vol"].tolist() == [pytest.approx(0.18)]


def test_align_daily_volatility_backfills_before_first_vol_observation() -> None:
    """Dates before the first EWSD row use the earliest available vol (rolled-fit windows)."""
    service = DailyEWSDVolatilityService()
    daily_volatility = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2005-01-03", "2005-01-04"]),
            "ticker": ["GC", "GC"],
            "ewsd_annual_vol": [0.22, 0.23],
        }
    )
    candles = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2004-12-31", "2005-01-03"]),
            "ticker": ["GC", "GC"],
        }
    )
    aligned = service.align_daily_volatility_to_candles(daily_volatility, candles)
    assert aligned["ewsd_annual_vol"].tolist() == [pytest.approx(0.22), pytest.approx(0.22)]


def test_align_daily_volatility_raises_for_unresolved_gaps() -> None:
    service = DailyEWSDVolatilityService()
    daily_volatility = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "ticker": ["ES"],
            "ewsd_annual_vol": [0.20],
        }
    )
    candles = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "ticker": ["ES", "NQ"],
        }
    )

    with pytest.raises(ValueError, match="Missing daily EWSD volatility for ticker 'NQ'"):
        service.align_daily_volatility_to_candles(daily_volatility, candles)


def test_update_incremental_appends_only_new_dates(tmp_path) -> None:
    service = DailyEWSDVolatilityService(store_dir=str(tmp_path))
    first_batch = _daily_candles("ES", "2024-01-01", [100.0, 101.0, 102.0])
    second_batch = _daily_candles("ES", "2024-01-01", [100.0, 101.0, 102.0, 103.0])

    out_1 = service.update_incremental(first_batch)
    out_2 = service.update_incremental(second_batch)

    assert len(out_1) == 3
    assert len(out_2) == 4
    assert out_2["datetime"].max() == pd.Timestamp("2024-01-04")


def test_long_run_recompute_is_monthly(tmp_path) -> None:
    service = DailyEWSDVolatilityService(store_dir=str(tmp_path))
    jan = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["ES", "ES"],
            "close": [100.0, 103.0],
        }
    )
    service.update_incremental(jan)
    state_after_jan2 = service._load_state()["ES"]
    sigma_jan = state_after_jan2.sigma_long
    assert state_after_jan2.last_long_run_month == "2024-01"

    # Same month update should not recompute long-run sigma.
    jan_same_month = pd.DataFrame(
        {
            "datetime": [datetime(2024, 1, 3)],
            "ticker": ["ES"],
            "close": [95.0],
        }
    )
    service.update_incremental(jan_same_month)
    state_after_jan3 = service._load_state()["ES"]
    assert state_after_jan3.sigma_long == pytest.approx(sigma_jan)
    assert state_after_jan3.last_long_run_month == "2024-01"

    # First observed day in next month should trigger long-run recompute.
    feb = pd.DataFrame(
        {
            "datetime": [datetime(2024, 2, 1)],
            "ticker": ["ES"],
            "close": [97.0],
        }
    )
    service.update_incremental(feb)
    state_after_feb = service._load_state()["ES"]
    assert state_after_feb.last_long_run_month == "2024-02"
    assert state_after_feb.sigma_long != pytest.approx(sigma_jan)


def test_state_round_trip_continues_after_restart(tmp_path) -> None:
    first = DailyEWSDVolatilityService(store_dir=str(tmp_path))
    first.update_incremental(_daily_candles("ES", "2024-01-01", [100.0, 101.0, 99.0]))

    second = DailyEWSDVolatilityService(store_dir=str(tmp_path))
    out = second.update_incremental(
        pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2024-01-04")],
                "ticker": ["ES"],
                "close": [100.5],
            }
        )
    )

    assert len(out) == 4
    assert out["datetime"].max() == pd.Timestamp("2024-01-04")
