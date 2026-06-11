from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cache.runtime.central_cache as central_cache_module
from cache.runtime.central_cache import CentralCacheStore, _candle_frame_semantically_equal
from cache.runtime.central_cache_errors import (
    ArtifactMissingError,
    CacheCoverageError,
)
from cache.runtime.central_cache_models import (
    ArtifactDescriptor,
    ArtifactRecord,
    ArtifactScope,
    CacheRequest,
    LookupMode,
)
from lib.core.enums import Ticker, TimeFrame


@pytest.fixture
def central_cache(tmp_path: pytest.TempPathFactory) -> CentralCacheStore:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=tmp_path)
    CentralCacheStore._instance = store  # type: ignore[attr-defined]
    yield store
    CentralCacheStore.reset()


def _frame(start: datetime, periods: int, freq: timedelta, base: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex([start + i * freq for i in range(periods)])
    return pd.DataFrame(
        {
            "open": [base + i for i in range(periods)],
            "high": [base + i + 1 for i in range(periods)],
            "low": [base + i - 1 for i in range(periods)],
            "close": [base + i + 0.5 for i in range(periods)],
            "volume": [1_000 + i for i in range(periods)],
        },
        index=index,
    )


def test_query_candle_exact_as_of_and_range(central_cache: CentralCacheStore) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 4, timedelta(days=1)))

    exact = central_cache.query_candle(Ticker.ES, TimeFrame.D, datetime(2024, 1, 2), LookupMode.EXACT)
    assert exact.close == pytest.approx(101.5)

    as_of = central_cache.query_candle(Ticker.ES, TimeFrame.D, datetime(2024, 1, 2, 12), LookupMode.AS_OF)
    assert as_of.close == pytest.approx(101.5)

    window = central_cache.query_candles(Ticker.ES, TimeFrame.D, datetime(2024, 1, 2), datetime(2024, 1, 3))
    assert list(window.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_default_layout_reserves_separate_runtime_cache_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    CentralCacheStore.reset()
    central_root = tmp_path / ".cache" / "trading_algo" / "central_cache"
    monkeypatch.setattr(central_cache_module, "default_central_cache_dir", lambda: central_root)
    monkeypatch.setattr(central_cache_module, "default_candle_cache_dir", lambda: central_root / "candles")
    monkeypatch.setattr(
        central_cache_module,
        "default_live_artifact_cache_dir",
        lambda: central_root / "artifacts" / "live",
    )
    monkeypatch.setattr(
        central_cache_module,
        "default_research_artifact_cache_dir",
        lambda: central_root / "artifacts" / "research",
    )

    store = CentralCacheStore()

    assert store.cache_dir == central_root
    assert store.candle_cache_dir == central_root / "candles"
    assert store.live_artifact_cache_dir == central_root / "artifacts" / "live"
    assert store.research_artifact_cache_dir == central_root / "artifacts" / "research"
    CentralCacheStore.reset()


def test_candle_missing_and_partial_coverage_errors(central_cache: CentralCacheStore) -> None:
    with pytest.raises(ArtifactMissingError):
        central_cache.query_candle(Ticker.NQ, TimeFrame.D, datetime(2024, 1, 1), LookupMode.EXACT)

    central_cache.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 2, 1), 3, timedelta(days=1)))

    with pytest.raises(CacheCoverageError):
        central_cache.query_candles(Ticker.NQ, TimeFrame.D, datetime(2024, 1, 31), datetime(2024, 2, 2))


def test_candles_reload_from_disk_after_memory_clear(central_cache: CentralCacheStore) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 4, timedelta(days=1)))
    central_cache.clear_candles()

    reloaded = central_cache.query_candles(Ticker.ES, TimeFrame.D, datetime(2024, 1, 2), datetime(2024, 1, 3))

    assert list(reloaded.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert set(reloaded["ticker"].unique()) == {"ES"}
    assert set(reloaded["timeframe"].unique()) == {TimeFrame.D}


def test_upsert_candles_appends_new_rows_without_losing_existing_data(
    central_cache: CentralCacheStore,
) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 3, timedelta(days=1)))

    central_cache.upsert_candles(
        Ticker.ES,
        TimeFrame.D,
        _frame(datetime(2024, 1, 4), 2, timedelta(days=1), base=200.0),
    )

    reloaded = central_cache.query_candles(Ticker.ES, TimeFrame.D, datetime(2024, 1, 1), datetime(2024, 1, 5))
    assert list(reloaded.index) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    record = central_cache.describe_candle(Ticker.ES, TimeFrame.D)
    assert record is not None
    assert record.coverage.end == pd.Timestamp("2024-01-05").to_pydatetime()


def test_upsert_candles_overwrites_existing_datetime_rows(
    central_cache: CentralCacheStore,
) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 3, timedelta(days=1)))

    central_cache.upsert_candles(
        Ticker.ES,
        TimeFrame.D,
        _frame(datetime(2024, 1, 2), 1, timedelta(days=1), base=500.0),
    )

    updated = central_cache.query_candle(Ticker.ES, TimeFrame.D, datetime(2024, 1, 2))
    assert updated.close == pytest.approx(500.5)


def test_artifact_read_exact_and_as_of(central_cache: CentralCacheStore) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 3, 1), 4, timedelta(days=1), base=200.0))

    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.LIVE,
        artifact_name="signal",
    )
    central_cache.write_artifact(descriptor, _frame(datetime(2024, 3, 1), 4, timedelta(days=1)))

    exact = central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 3, 2)))
    assert exact.iloc[0]["open"] == pytest.approx(101.0)

    as_of = central_cache.read_artifact(descriptor, CacheRequest(as_of_dt=datetime(2024, 3, 2, 12)))
    assert as_of.iloc[0]["open"] == pytest.approx(101.0)

    record = central_cache.describe_artifact(descriptor)
    assert isinstance(record, ArtifactRecord)
    assert record.coverage.start is not None
    assert record.coverage.end is not None


def test_artifact_reload_from_disk_after_memory_clear(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.LIVE,
        artifact_name="rsi",
    )
    payload = _frame(datetime(2024, 6, 1), 3, timedelta(days=1))
    central_cache.write_artifact(descriptor, payload)
    central_cache.clear()

    record = central_cache.describe_artifact(descriptor)
    assert record is not None
    assert record.coverage.start == pd.Timestamp("2024-06-01").to_pydatetime()


def test_write_artifact_overwrites_existing(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.LIVE,
        artifact_name="rsi",
    )
    central_cache.write_artifact(descriptor, _frame(datetime(2024, 5, 1), 3, timedelta(days=1)))
    central_cache.write_artifact(
        descriptor,
        _frame(datetime(2024, 5, 1), 3, timedelta(days=1), base=500.0),
    )

    refreshed = central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 5, 2)))
    assert refreshed.iloc[0]["open"] == pytest.approx(501.0)


def test_exact_lookup_miss_on_artifact(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.NQ,
        timeframe=TimeFrame.D,
        module_name="atr",
        params={"lookback": 20},
        scope=ArtifactScope.LIVE,
        artifact_name="signal",
    )
    central_cache.write_artifact(descriptor, _frame(datetime(2024, 7, 1), 2, timedelta(days=1)))

    with pytest.raises(ArtifactMissingError):
        central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 7, 5)))


def test_candle_frame_semantically_equal_accepts_datetime64_unit_mismatch() -> None:
    idx_ns = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    idx_us = idx_ns.astype("datetime64[us]")
    left = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.5, 102.5],
            "volume": [1000, 1100],
        },
        index=idx_ns,
    )
    right = pd.DataFrame(left, copy=True)
    right.index = idx_us
    assert not left.index.equals(right.index)
    assert _candle_frame_semantically_equal(left, right)


def test_candle_frame_semantically_equal_accepts_float_dtype_mismatch() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    left = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.5, 102.5],
            "volume": [1000, 1100],
        },
        index=idx,
    )
    right = left.astype(
        {"open": np.float32, "high": np.float32, "low": np.float32, "close": np.float32, "volume": np.float32}
    )
    assert _candle_frame_semantically_equal(left, right)
    assert not left.equals(right)


def test_set_candles_skips_rewrite_when_semantically_equal_to_disk(
    central_cache: CentralCacheStore,
) -> None:
    """Parquet round-trip must not force a candle re-persist."""
    df = _frame(datetime(2024, 1, 1), 3, timedelta(days=1))
    central_cache.set_candles(Ticker.ES, TimeFrame.D, df)

    central_cache._candle_frames.clear()
    central_cache._candle_records.clear()
    loaded = central_cache._read_candles_from_disk(Ticker.ES, TimeFrame.D)
    assert loaded is not None

    parquet_path = central_cache._candle_path(Ticker.ES, TimeFrame.D)
    mtime_before = parquet_path.stat().st_mtime

    central_cache.set_candles(Ticker.ES, TimeFrame.D, df)

    assert parquet_path.stat().st_mtime == mtime_before
