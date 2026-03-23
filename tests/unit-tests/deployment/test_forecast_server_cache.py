from __future__ import annotations

import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import pandas as pd

from utils.cache.central_cache import CentralCacheStore
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _install_forecast_server_stubs() -> None:
    if "schedule" not in sys.modules:
        schedule_module = ModuleType("schedule")

        class _Job:
            day = property(lambda self: self)
            sunday = property(lambda self: self)

            def at(self, *_args, **_kwargs) -> "_Job":
                return self

            def do(self, *_args, **_kwargs) -> None:
                return None

        schedule_module.every = lambda *_args, **_kwargs: _Job()
        sys.modules["schedule"] = schedule_module

    if "MetaTrader5" not in sys.modules:
        sys.modules["MetaTrader5"] = SimpleNamespace(
            TIMEFRAME_D1=1,
            TIMEFRAME_W1=2,
            TIMEFRAME_MN1=3,
            initialize=lambda *_args, **_kwargs: True,
            login=lambda *_args, **_kwargs: True,
            last_error=lambda: "",
        )


_install_forecast_server_stubs()

from deployment.forecast_server import ForecastServer


def _frame(start: datetime, periods: int) -> pd.DataFrame:
    dates = [start + timedelta(days=offset) for offset in range(periods)]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + offset for offset in range(periods)],
            "high": [101.0 + offset for offset in range(periods)],
            "low": [99.0 + offset for offset in range(periods)],
            "close": [100.5 + offset for offset in range(periods)],
            "volume": [1_000.0 + offset for offset in range(periods)],
            "ticker": [Ticker.NQ.name] * periods,
            "timeframe": [TimeFrame.D] * periods,
        }
    )


def _candle(at: datetime, close: float) -> Candle:
    return Candle(
        datetime=at,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1_500.0,
        ticker=Ticker.NQ,
        tf=TimeFrame.D,
    )


def test_upsert_cross_ticker_candle_preserves_cached_history(tmp_path) -> None:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=str(tmp_path))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    server = ForecastServer.__new__(ForecastServer)
    server.candle_buffers = {}

    store.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 1, 1), 3))
    server._upsert_cross_ticker_candle(Ticker.NQ, TimeFrame.D, _candle(datetime(2024, 1, 4), 104.5))

    updated = store.query_candles(Ticker.NQ, TimeFrame.D)

    assert len(updated) == 4
    assert list(updated.index) == list(pd.date_range("2024-01-01", periods=4, freq="D"))


def test_upsert_cross_ticker_candle_replaces_matching_timestamp(tmp_path) -> None:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=str(tmp_path))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    server = ForecastServer.__new__(ForecastServer)
    server.candle_buffers = {}

    store.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 1, 1), 3))
    server._upsert_cross_ticker_candle(Ticker.NQ, TimeFrame.D, _candle(datetime(2024, 1, 2), 222.5))

    updated = store.query_candles(Ticker.NQ, TimeFrame.D)

    assert len(updated) == 3
    assert updated.loc[pd.Timestamp("2024-01-02"), "close"] == 222.5
