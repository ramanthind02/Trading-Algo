from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import utils.cache.runtime.central_cache as central_cache_module
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_errors import (
    ArtifactLifecycleError,
    ArtifactMissingError,
    CacheCoverageError,
    SourceRevisionConflictError,
)
from utils.cache.central_cache_models import (
    ArtifactDescriptor,
    ArtifactLifecycleState,
    ArtifactRecord,
    ArtifactScope,
    CacheRequest,
    LookupMode,
)
from utils.core.enums import Ticker, TimeFrame


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

    initial_record = central_cache.describe_candle(Ticker.ES, TimeFrame.D)
    assert initial_record is not None

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
    updated_record = central_cache.describe_candle(Ticker.ES, TimeFrame.D)
    assert updated_record is not None
    assert updated_record.revision == initial_record.revision + 1


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


def test_upsert_candles_noop_does_not_increment_revision(
    central_cache: CentralCacheStore,
) -> None:
    payload = _frame(datetime(2024, 1, 1), 3, timedelta(days=1))
    central_cache.set_candles(Ticker.ES, TimeFrame.D, payload)

    initial_record = central_cache.describe_candle(Ticker.ES, TimeFrame.D)
    assert initial_record is not None

    central_cache.upsert_candles(Ticker.ES, TimeFrame.D, payload)

    updated_record = central_cache.describe_candle(Ticker.ES, TimeFrame.D)
    assert updated_record is not None
    assert updated_record.revision == initial_record.revision


def test_upsert_candles_marks_dependent_artifacts_stale_on_real_change(
    central_cache: CentralCacheStore,
) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 3, timedelta(days=1)))
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )
    central_cache.write_artifact(
        descriptor,
        _frame(datetime(2024, 1, 1), 3, timedelta(days=1)),
        depends_on=[(Ticker.ES, TimeFrame.D)],
    )

    central_cache.upsert_candles(
        Ticker.ES,
        TimeFrame.D,
        _frame(datetime(2024, 1, 4), 1, timedelta(days=1), base=250.0),
    )

    record = central_cache.describe_artifact(descriptor)
    assert record is not None
    assert record.lifecycle_state is ArtifactLifecycleState.STALE


def test_artifact_read_as_of_and_lifecycle_state(central_cache: CentralCacheStore) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 3, 1), 4, timedelta(days=1), base=200.0))

    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )
    central_cache.write_artifact(descriptor, _frame(datetime(2024, 3, 1), 4, timedelta(days=1)), depends_on=[(Ticker.ES, TimeFrame.D)])

    exact = central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 3, 2)))
    assert exact.iloc[0]["open"] == pytest.approx(101.0)

    as_of = central_cache.read_artifact(descriptor, CacheRequest(as_of_dt=datetime(2024, 3, 2, 12)))
    assert as_of.iloc[0]["open"] == pytest.approx(101.0)

    record = central_cache.describe_artifact(descriptor)
    assert isinstance(record, ArtifactRecord)
    assert record.lifecycle_state is ArtifactLifecycleState.FRESH

    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 3, 1), 4, timedelta(days=1), base=250.0))

    stale_record = central_cache.describe_artifact(descriptor)
    assert isinstance(stale_record, ArtifactRecord)
    assert stale_record.lifecycle_state is ArtifactLifecycleState.STALE

    with pytest.raises(ArtifactLifecycleError):
        central_cache.read_artifact(descriptor)


def test_artifact_dependency_metadata_survives_memory_clear(central_cache: CentralCacheStore) -> None:
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 4, 1), 3, timedelta(days=1)))

    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="atr",
        params={"lookback": 20},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )
    central_cache.write_artifact(
        descriptor,
        _frame(datetime(2024, 4, 1), 3, timedelta(days=1)),
        depends_on=[(Ticker.ES, TimeFrame.D)],
    )

    central_cache.clear()
    central_cache.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 4, 1), 3, timedelta(days=1), base=300.0))

    record = central_cache.describe_artifact(descriptor)
    assert isinstance(record, ArtifactRecord)
    assert record.lifecycle_state is ArtifactLifecycleState.STALE

    with pytest.raises(ArtifactLifecycleError):
        central_cache.read_artifact(descriptor)


def test_source_revision_conflict_rejects_older_artifact_write(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )
    payload = _frame(datetime(2024, 5, 1), 3, timedelta(days=1))
    central_cache.write_artifact(descriptor, payload, source_revision=8)

    with pytest.raises(SourceRevisionConflictError):
        central_cache.write_artifact(
            descriptor,
            _frame(datetime(2024, 5, 1), 3, timedelta(days=1), base=500.0),
            source_revision=7,
        )


def test_newer_source_revision_refresh_is_allowed(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 14},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )

    central_cache.write_artifact(descriptor, _frame(datetime(2024, 5, 1), 3, timedelta(days=1)), source_revision=7)
    central_cache.write_artifact(
        descriptor,
        _frame(datetime(2024, 5, 1), 3, timedelta(days=1), base=500.0),
        source_revision=8,
    )

    refreshed = central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 5, 2)))
    assert refreshed.iloc[0]["open"] == pytest.approx(501.0)
    assert central_cache.get_artifact_record(descriptor).source_revision == 8


def test_non_node_artifact_support_promotion_and_prune(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="portfolio",
        artifact_name="forecast_vectors",
        scope=ArtifactScope.RESEARCH,
    )
    payload = _frame(datetime(2024, 6, 1), 3, timedelta(days=1))
    central_cache.write_artifact(descriptor, payload)

    promoted = central_cache.promote_artifact(descriptor, target_scope=ArtifactScope.LIVE)
    assert promoted.scope is ArtifactScope.LIVE

    live_descriptor = descriptor.with_scope(ArtifactScope.LIVE)
    live_result = central_cache.read_artifact(live_descriptor, CacheRequest(exact_dt=datetime(2024, 6, 2)))
    assert live_result.iloc[0]["close"] == pytest.approx(101.5)

    records_before_prune = central_cache.list_artifacts()
    assert any(record.descriptor.scope is ArtifactScope.RESEARCH for record in records_before_prune)
    assert any(record.descriptor.scope is ArtifactScope.LIVE for record in records_before_prune)

    removed = central_cache.prune_scope(ArtifactScope.RESEARCH)
    assert descriptor in removed
    assert central_cache.describe_artifact(descriptor) is None
    assert central_cache.describe_artifact(live_descriptor) is not None


def test_exact_lookup_miss_on_artifact(central_cache: CentralCacheStore) -> None:
    descriptor = ArtifactDescriptor(
        family="signals",
        ticker=Ticker.NQ,
        timeframe=TimeFrame.D,
        module_name="atr",
        params={"lookback": 20},
        scope=ArtifactScope.RESEARCH,
        artifact_name="signal",
    )
    central_cache.write_artifact(descriptor, _frame(datetime(2024, 7, 1), 2, timedelta(days=1)))

    with pytest.raises(ArtifactMissingError):
        central_cache.read_artifact(descriptor, CacheRequest(exact_dt=datetime(2024, 7, 5)))
