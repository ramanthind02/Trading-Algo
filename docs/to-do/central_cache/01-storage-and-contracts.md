# Storage And Contract Plan

## Objective

Define the central cache as the system of record for candles and cache-backed derived artifacts, with explicit coverage, invalidation semantics, and an extensible artifact model.

The design goal is a single simple public cache facade, even if the implementation is decomposed across multiple internal modules.

The implementation goal is to consolidate cache ownership into a small, easy-to-find part of the codebase rather than leaving core behavior scattered across unrelated modules.

## Primary Files Impacted

- `utils/cache/bias_node_cache.py`
- `utils/cache/cache_manager.py`
- `utils/data/cross_ticker_store.py`
- `nodes/__init__.py`
- `feature_extraction/feature_extractor.py`
- likely new modules under `utils/cache/` or `utils/data/`

## Required Design Decisions

### 1. Central store interface

Create one explicit interface for:

- writing candle bars
- querying candle coverage
- reading candles by exact datetime or range
- reading derived artifacts by exact datetime or range
- registering or addressing multiple artifact families through one stable contract
- surfacing typed coverage and cache miss errors
- invalidating or versioning derived artifacts when source candles change
- tracking artifact lifecycle state after source updates

Likely deliverables:

- `CentralCacheStore` protocol or concrete service
- typed error models such as `CacheCoverageError`, `ArtifactMissingError`, `SourceRevisionConflictError`
- immutable request models for keys and ranges
- artifact descriptors that are not hard-coded only to candles and bias nodes
- artifact lifecycle states such as `fresh`, `stale`, `rebuilding`, and `failed`

Public API principle:

- callers should primarily interact with one cache service/facade
- specialized internal modules may exist behind that facade
- legacy direct use of `BiasNodeCache`, `CacheManager`, and `CrossTickerDataStore` should be treated as migration debt, not the target interface

Implementation ownership principle:

- central cache behavior should be implemented in a small number of closely related modules
- legacy cache behavior left in distant modules should be reduced to adapters or deleted
- a maintainer should be able to find read/write/invalidation/refresh behavior without searching the whole repo

### 2. Key model and namespaces

Standardize keys for at least:

- candle series: `(ticker, timeframe)`
- bias artifacts: `(module_name, params, ticker, timeframe)`
- EWSD volatility artifacts: centrally keyed cache objects, not separately passed DataFrames
- optional ensemble/base-model artifacts if materialized later
- future portfolio-level artifacts such as forecast vectors, positions, and portfolio predictions
- artifact scope or retention class, for example `research` vs `vault/live`

Document whether params remain filename-derived, hash-derived, or split into manifest metadata plus artifact path.

The key model should be extensible enough to support artifact families with different identities, for example:

- candle data keyed primarily by `(ticker, timeframe)`
- node artifacts keyed by `(module_name, params, ticker, timeframe)`
- portfolio prediction artifacts keyed by `(portfolio_id or strategy_id, datetime grid/range, version)`

Retention and source-of-truth policy for this refactor:

- vault-selected bias nodes are the source of truth for artifacts required by live trading and deployment
- research-phase bias-node artifacts live in a separate disposable scope or namespace
- research artifacts may be pruned without affecting vault-backed live artifacts
- research caches should be local and gitignored; they are not deployment artifacts

Volatility policy for this refactor:

- remove the public `daily_volatility_df` invariant from downstream APIs
- read any required volatility through EWSD cache lineage
- keep EWSD keying and coverage rules inside the same central cache contract as other artifacts

### 3. Coverage semantics

The target design needs stronger guarantees than the current `BiasNodeCache` behavior.

Define:

- full-coverage vs partial-coverage reads
- whether partial coverage is ever allowed
- exact-datetime vs as-of lookup rules
- how coverage metadata is stored and queried

Current problem:

- `BiasNodeCache` only warns on partial interval coverage
- `CrossTickerDataStore.get_candle()` conflates "not loaded", "load failed", and "true gap"

### 4. Invalidation and versioning

Current writes are plain parquet writes with no lineage or revision tracking.

Needed:

- source-candle revision policy
- invalidation when candles are corrected/restated
- automatic stale-marking of dependent artifacts when OHLCV changes
- dependency tracking from source candle series to derived artifacts
- managed refresh policy for append-only updates vs corrected historical updates
- atomic write strategy
- metadata for artifact provenance: source window, source revision, creation timestamp, schema version
- artifact-type metadata so future readers can interpret stored portfolio predictions or other derived outputs safely

Recommended default behavior:

- candle updates automatically invalidate or mark dependent artifacts stale
- refresh is managed by an orchestrator or background job layer, not by recomputing every dependent artifact inline on every write
- append-only updates should refresh incrementally where possible
- historical corrections may trigger targeted rebuilds or full invalidation depending on dependency depth

Do not make synchronous "recompute everything immediately" the default write-path behavior.

### 5. Cross-ticker and cross-timeframe ownership

The current `CrossTickerDataStore` should either:

- be absorbed into the central cache, or
- become a thin in-memory implementation detail behind it

Do not keep lazy auto-load and `None` returns as the public contract.

### 6. Research vs vault artifact retention

The cache needs explicit retention classes so EDA does not bloat the same storage used for live trading.

Recommended model:

- `vault` or `live` scope:
  - artifacts referenced by selected bias nodes in the vault
  - retained and treated as the production source of truth
- `research` scope:
  - artifacts generated during exploration of many parameter combinations
  - local, gitignored, disposable, and safe to prune

Required capabilities:

- mark or derive whether an artifact belongs to `research` or `vault/live`
- promote an artifact or spec from research scope into vault/live scope when selected
- prune research artifacts by scope, age, size, or explicit researcher command
- prevent research cleanup from deleting vault-backed artifacts

## Implementation Tasks

### Phase A: contract layer

- add central cache request/result models
- add typed miss and coverage exceptions
- define exact and as-of lookup APIs
- define artifact metadata schema
- define EWSD cache query contract used by downstream consumers
- define an extensible artifact descriptor or namespace model for future artifact families
- define artifact lifecycle states and dependency metadata
- define artifact scope/retention metadata for `research` vs `vault/live`

### Phase B: storage layer

- implement candle storage reads/writes
- implement artifact reads/writes
- implement EWSD artifact reads/writes behind the same contract
- persist coverage metadata
- add atomic write or temp-file replace behavior
- prove the contract can store at least one non-node derived artifact shape without redesign
- implement dependency-aware stale marking when source OHLCV changes
- implement managed refresh hooks or job handoff for dependent artifacts
- implement separate storage or namespace handling for research vs vault/live artifacts

### Phase C: compatibility layer

- wrap `BiasNodeCache` reads behind the new artifact interface
- wrap or replace `CrossTickerDataStore`
- evolve `CacheManager` from batch-population helper into cache population orchestrator
- reuse existing cache-population behavior from `utils/cache/cache_manager.py` rather than introducing a second orchestration path with overlapping responsibilities
- add researcher-facing cleanup/prune operations for disposable research artifacts
- collapse scattered cache logic into the central cache area once adapters are in place

## DRY Reuse Guidance

Do not create a brand new cache population stack if the existing one can be split into reusable pieces.

Preferred reuse targets in this layer:

- cache population and auxiliary-spec handling from `utils/cache/cache_manager.py`
- artifact read/write behavior from `utils/cache/bias_node_cache.py`
- existing cache-manager tests in `tests/unit-tests/utils/test_cache_manager.py` as the baseline behavior contract

## Test Requirements

Replace or expand:

- `tests/unit-tests/utils/test_bias_node_cache.py`
- `tests/unit-tests/utils/test_cache_manager.py`
- `tests/nodes/test_cross_ticker.py`
- `tests/integration/test_cache_integration.py`

Add new tests for:

- exact missing-key errors
- partial coverage failures
- source revision invalidation
- exact vs as-of semantics
- live/set-data writes using the same central interface as backtest/parquet loads
- storage and retrieval of a future-facing artifact type such as portfolio predictions or forecast vectors
- candle updates automatically marking dependent artifacts stale
- managed refresh behavior for append-only updates and corrected historical bars
- pruning research artifacts without affecting vault/live artifacts
- promotion or selection flow from research scope into vault-backed live scope

## Exit Criteria

- one public cache contract exists for candles and derived artifacts
- callers no longer need to select among multiple low-level cache/storage classes directly
- cache implementation ownership is consolidated enough that maintainers do not need to hunt across unrelated modules to fix cache behavior
- the contract is not specialized to only candle and bias-node artifacts
- partial coverage cannot slip through silently
- cross-ticker reads no longer return raw `None` for operational misses
- EWSD volatility is queryable from the central cache without passing `daily_volatility_df`
- dependent artifacts are automatically invalidated or marked stale when source OHLCV changes
- vault-selected bias-node artifacts are clearly separated from disposable research artifacts
- artifact metadata can explain how a cached object was produced
