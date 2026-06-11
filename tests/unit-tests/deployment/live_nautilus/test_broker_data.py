"""Unit tests for deployment.live.broker_data (isolated tmp store, fake loader)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deployment.live import broker_data
from cache.runtime.central_cache import CentralCacheStore
from lib.core.enums import Ticker, TimeFrame

_TRUE = pd.Series(
    1000.0 * (1.0 + 0.0005 * np.arange(1600)),
    index=pd.bdate_range("2016-01-01", periods=1600),
)


def _raw(series: pd.Series) -> pd.DataFrame:
    """A Darwinex-loader-shaped frame: [datetime, OHLCV], NO ticker/timeframe col.

    Mirrors ``cfd_candles.load_cfd_candles_raw`` output (the real ``load_fn``).
    """
    close = series.to_numpy()
    return pd.DataFrame(
        {
            "datetime": series.index,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000,
        }
    )


def _fake_load(daily_series: pd.Series):
    """Build a ``load_fn(ticker, tf)`` stub returning D and M-resampled frames."""

    def load(_ticker, tf: TimeFrame) -> pd.DataFrame:
        raw = _raw(daily_series)
        if tf is TimeFrame.D:
            return raw
        if tf is TimeFrame.M:
            return (
                raw.set_index("datetime")
                .resample("ME")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna(subset=["close"])
                .reset_index()
            )
        raise ValueError(f"unexpected timeframe {tf!r}")

    return load


@pytest.fixture
def tmp_store(tmp_path):
    prev = CentralCacheStore._instance  # type: ignore[attr-defined]
    store = CentralCacheStore(cache_dir=str(tmp_path / "darwinex"))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]
    yield store
    CentralCacheStore._instance = prev  # type: ignore[attr-defined]


def test_signal_cache_root_is_darwinex() -> None:
    root = broker_data.signal_cache_root()
    assert "darwinex" in root.parts
    assert "broker_cache" in root.parts
    assert root == broker_data.broker_cache_root(broker_data.SIGNAL_BROKER)


def test_refresh_signal_daily_populates_store(tmp_store) -> None:
    report = broker_data.refresh_signal_daily(
        ("ES",), store=tmp_store, populate_bias=False, load_fn=_fake_load(_TRUE)
    )

    es = next(t for t in report.per_ticker if t.ticker == "ES")
    assert es.source == "darwinex"
    assert es.total_bars == len(_TRUE)

    # The shared store now holds the Darwinex daily + derived monthly.
    daily = tmp_store.query_candles(Ticker.ES, TimeFrame.D)
    assert len(daily) == len(_TRUE)
    monthly = tmp_store.query_candles(Ticker.ES, TimeFrame.M)
    assert not monthly.empty


def test_missing_darwinex_data_is_skipped_not_fatal(tmp_store) -> None:
    def missing_load(_ticker, _tf):
        raise FileNotFoundError("no bars_D1/bars_M1 for this symbol")

    report = broker_data.refresh_signal_daily(
        ("GC",), store=tmp_store, populate_bias=False, load_fn=missing_load
    )
    gc = next(t for t in report.per_ticker if t.ticker == "GC")
    assert gc.source == "missing"
    assert gc.total_bars == 0


def test_warm_cold_classification(tmp_store) -> None:
    report = broker_data.refresh_signal_daily(
        ("ES",), store=tmp_store, populate_bias=False, load_fn=_fake_load(_TRUE)
    )
    assert report.warm(min_bars=500) == ("ES",)
    assert report.cold(min_bars=500) == ()
    assert report.warm(min_bars=99999) == ()
    # A short series classifies as cold, not warm.
    short = broker_data.refresh_signal_daily(
        ("NQ",), store=tmp_store, populate_bias=False, load_fn=_fake_load(_TRUE.iloc[:120])
    )
    assert short.cold(min_bars=500) == ("NQ",)
    assert short.warm(min_bars=500) == ()
