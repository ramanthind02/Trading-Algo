# Central Cache API

## Module

- [`utils/cache/__init__.py`](../../../utils/cache/__init__.py)
- [`utils/cache/central_cache.py`](../../../utils/cache/central_cache.py)
- [`utils/cache/central_cache_models.py`](../../../utils/cache/central_cache_models.py)
- [`utils/cache/central_cache_errors.py`](../../../utils/cache/central_cache_errors.py)
- [`utils/cache/cross_ticker_store.py`](../../../utils/cache/cross_ticker_store.py)
- [`utils/cache/cache_manager.py`](../../../utils/cache/cache_manager.py)
- [`utils/cache/ingest_source_candles.py`](../../../utils/cache/ingest_source_candles.py)

## Public Surface

- `CentralCacheStore`
- `CentralCache` alias
- `ArtifactDescriptor`
- `ArtifactRecord`
- `CoverageWindow`
- `CacheRequest`
- `ArtifactScope`
- `ArtifactLifecycleState`
- `LookupMode`
- `CrossTickerDataStore`
- `CacheManager`
- `ingest_source_candles`
- `ArtifactMissingError`
- `CacheCoverageError`
- `ArtifactLifecycleError`
- `SourceRevisionConflictError`

## Core Service

### `CentralCacheStore`

Lifecycle and singleton helpers:

- `CentralCacheStore.get_instance()`
- `CentralCacheStore.reset()`
- `clear()`
- `clear_candles(purge_persisted: bool = False)`

Candle methods:

- `set_candles(ticker, timeframe, candles, scope=ArtifactScope.LIVE) -> None`
- `query_candle(ticker, timeframe, dt, lookup_mode=LookupMode.EXACT) -> Candle`
- `query_candles(ticker, timeframe, start=None, end=None) -> pd.DataFrame`
- `loaded_tickers() -> list[Ticker]`
- `is_candle_loaded(ticker, timeframe) -> bool`
- `describe_candle(ticker, timeframe) -> ArtifactRecord | None`

Artifact methods:

- `write_artifact(descriptor, data, depends_on=(), source_revision=0) -> None`
- `read_artifact(descriptor, request=None, lookup_mode=LookupMode.EXACT) -> pd.DataFrame`
- `describe_artifact(descriptor) -> ArtifactRecord | None`
- `get_artifact_record(descriptor) -> ArtifactRecord`
- `list_artifacts(scope=None) -> list[ArtifactRecord]`

Artifact lifecycle helpers:

- `mark_dependents_stale(ticker, timeframe) -> None`
- `promote_artifact(descriptor, target_scope=ArtifactScope.LIVE) -> ArtifactDescriptor`
- `prune_research_artifacts() -> list[ArtifactDescriptor]`
- `prune_scope(scope) -> list[ArtifactDescriptor]`

Behavior notes:

- `clear()` removes in-memory state only
- `clear_candles(purge_persisted=True)` also deletes persisted runtime candle cache files
- `read_artifact(...)` raises `ArtifactLifecycleError` if the descriptor exists but is not `FRESH`
- `query_candles(...)` and range artifact reads raise `CacheCoverageError` when requested coverage exceeds stored coverage

## Request And Metadata Types

### `ArtifactDescriptor`

Stable identity for a derived artifact.

Fields:

- `family: str`
- `ticker: Ticker | None`
- `timeframe: TimeFrame | None`
- `module_name: str | None`
- `params: Mapping[str, Any]`
- `scope: ArtifactScope`
- `artifact_name: str | None`

Helpers:

- `cache_key() -> tuple[str, str, str, str, str]`
- `with_scope(scope) -> ArtifactDescriptor`

### `CacheRequest`

Read selector for artifact access.

Fields:

- `start: datetime | None`
- `end: datetime | None`
- `exact_dt: datetime | None`
- `as_of_dt: datetime | None`

Constraint:

- `exact_dt` and `as_of_dt` are mutually exclusive

### `ArtifactRecord`

Metadata returned by `describe_*` and `list_artifacts(...)`.

Fields:

- `descriptor: ArtifactDescriptor`
- `coverage: CoverageWindow`
- `lifecycle_state: ArtifactLifecycleState`
- `revision: int`
- `source_revision: int`
- `depends_on: tuple[tuple[str, str], ...]`

### Enums

- `ArtifactScope`
  - `LIVE`
  - `RESEARCH`
- `ArtifactLifecycleState`
  - `FRESH`
  - `STALE`
  - `REBUILDING`
  - `FAILED`
- `LookupMode`
  - `EXACT`
  - `AS_OF`

## Errors

- `ArtifactMissingError`
  - Raised when a candle/artifact key or requested timestamp is missing
- `CacheCoverageError`
  - Raised when stored coverage does not span the requested range
- `ArtifactLifecycleError`
  - Raised when an artifact exists but is stale, rebuilding, or failed
- `SourceRevisionConflictError`
  - Raised when an incoming write is based on an older source revision than the current dependency lineage

## Cache-Owned Adapter

### `CrossTickerDataStore`

Primary methods:

- `get_instance()`
- `reset()`
- `load(ticker, tf, start=None, end=None)`
- `set_data(ticker, tf, df)`
- `query_candle(ticker, tf, dt, lookup_mode=LookupMode.EXACT) -> Candle`
- `get_candle(ticker, tf, dt) -> Candle | None`
- `get_candle_as_of(ticker, tf, dt) -> Candle | None`
- `is_loaded(ticker, tf) -> bool`
- `loaded_tickers() -> list[Ticker]`
- `clear()`

Notes:

- `query_candle(...)` is the canonical typed-error interface
- `get_candle(...)` and `get_candle_as_of(...)` exist for compatibility with older `None` fallback semantics
- implementation ownership lives in `utils/cache/`, even if older import paths still re-export it

## Operational Helpers

### `CacheManager`

Primary cache-orchestration helpers:

- `find_source_candle_path(ticker, tf) -> Path | None`
- `load_source_candles(ticker, tf, start_date=None, end_date=None) -> pd.DataFrame`
- `populate_cache(...) -> dict[str, Any]`
- `populate_cache_for_vault(...) -> dict[str, Any]`
- `ensure_vault_cache_coverage(vault_ensemble_dirs, start_date, end_date, refresh_mode="missing_stale_only") -> dict[str, Any]`

Notes:

- `ensure_vault_cache_coverage(...)` is the cache-first preflight path used by portfolio research
- it reads vault feature files, ingests required candles into the central cache, ensures daily EWSD coverage, and rebuilds only missing/stale/out-of-range live `family="bias"` artifacts
- bias-artifact rebuilds are stateless cold rebuilds: `CacheManager` instantiates a fresh node, prepends the node's machine-readable warmup window, and trims the saved artifact back to the requested coverage range

### `ingest_source_candles`

Bulk helper for one-time source-candle ingestion from `data/ohlc_data` into `.cache/trading_algo/central_cache/candles`.

Signature:

- `ingest_source_candles(tickers=None, timeframes=None, start_date=None, end_date=None, reset_existing=False) -> dict[str, Any]`

CLI:

- `python -m utils.cache.ingest_source_candles`

## Related

- [Portfolio Cache Query API](../ensemble/portfolio_cache_query.md) — cache-native portfolio query API
- [Central Cache Architecture](../../library/Cache/architecture.md) — design and ownership
- [Central Cache User Guide](../../library/Cache/user_guide.md) — task-oriented examples
