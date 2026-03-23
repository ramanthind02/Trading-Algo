from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from utils.cache.bootstrap_source_candles import bootstrap_source_candles
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_errors import ArtifactMissingError
from utils.cache.ingest_source_candles import ingest_source_candles
from utils.core.enums import Ticker, TimeFrame


def _source_frame(
    ticker: Ticker,
    timeframe: TimeFrame,
    start: str = "2024-01-01",
    periods: int = 5,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + idx for idx in range(periods)],
            "high": [101.0 + idx for idx in range(periods)],
            "low": [99.0 + idx for idx in range(periods)],
            "close": [100.5 + idx for idx in range(periods)],
            "volume": [1000.0 + idx for idx in range(periods)],
            "ticker": [ticker.name] * periods,
            "timeframe": [timeframe.name] * periods,
        }
    )


@pytest.fixture
def isolated_central_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


@pytest.fixture
def temp_source_dir(tmp_path: Path) -> Path:
    source_dir = tmp_path / "ohlc"
    for ticker, timeframe in [(Ticker.ES, TimeFrame.D), (Ticker.NQ, TimeFrame.W)]:
        ticker_dir = source_dir / ticker.name
        ticker_dir.mkdir(parents=True, exist_ok=True)
        _source_frame(ticker, timeframe).to_parquet(ticker_dir / f"{timeframe.name}_{ticker.name}.parquet")
    return source_dir


def test_bootstrap_source_candles_writes_to_central_cache(
    monkeypatch: pytest.MonkeyPatch,
    isolated_central_cache: None,
    temp_source_dir: Path,
) -> None:
    bootstrap_module = importlib.import_module("utils.cache.runtime.bootstrap_source_candles")
    from utils.cache.cache_manager import CacheManager

    monkeypatch.setattr(
        bootstrap_module,
        "CacheManager",
        lambda: CacheManager(
            cache_dir=str(CentralCacheStore.get_instance().live_artifact_cache_dir),
            candle_dir=str(temp_source_dir),
        ),
    )

    summary = bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 4),
    )

    assert summary["success"] == 1
    store = CentralCacheStore.get_instance()
    result = store.query_candles(
        Ticker.ES,
        TimeFrame.D,
        start=datetime(2024, 1, 2),
        end=datetime(2024, 1, 4),
    )
    assert list(result.index) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]


def test_bootstrap_source_candles_reset_existing_clears_prior_candle_cache(
    monkeypatch: pytest.MonkeyPatch,
    isolated_central_cache: None,
    temp_source_dir: Path,
) -> None:
    bootstrap_module = importlib.import_module("utils.cache.runtime.bootstrap_source_candles")
    from utils.cache.cache_manager import CacheManager

    monkeypatch.setattr(
        bootstrap_module,
        "CacheManager",
        lambda: CacheManager(
            cache_dir=str(CentralCacheStore.get_instance().live_artifact_cache_dir),
            candle_dir=str(temp_source_dir),
        ),
    )

    store = CentralCacheStore.get_instance()
    store.set_candles(
        Ticker.GC,
        TimeFrame.D,
        _source_frame(Ticker.GC, TimeFrame.D),
    )

    summary = bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        reset_existing=True,
    )

    assert summary["success"] == 1
    with pytest.raises(ArtifactMissingError):
        store.query_candles(
            Ticker.GC,
            TimeFrame.D,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )


def test_ingest_source_candles_alias_delegates_to_bootstrap_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"total": 1, "success": 1, "failed": 0, "details": []}
    captured: dict[str, object] = {}

    def _bootstrap(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return expected

    ingest_module = importlib.import_module("utils.cache.runtime.ingest_source_candles")
    monkeypatch.setattr(ingest_module, "bootstrap_source_candles", _bootstrap)

    with pytest.deprecated_call(match="deprecated"):
        summary = ingest_source_candles(
            tickers=[Ticker.ES],
            timeframes=[TimeFrame.D],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            reset_existing=True,
        )

    assert summary == expected
    assert captured == {
        "tickers": [Ticker.ES],
        "timeframes": [TimeFrame.D],
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 1, 2),
        "reset_existing": True,
    }
