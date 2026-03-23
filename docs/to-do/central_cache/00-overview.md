# Central Cache Refactor Overview

## Scope

This plan implements the target architecture described in `docs/library/Cache/architecture.md`:

- one canonical store for OHLCV and bias-derived artifacts
- datetime- or range-driven reads instead of `candles_per_tf` call chains
- explicit coverage semantics
- typed, recoverable cache misses instead of silent fallback
- shared ingestion/query behavior between backtest and live
- one simple public cache/storage API even if the internal implementation remains modular
- cache-related implementation should be consolidated enough that engineers can find and fix behavior without chasing logic across the codebase

## Current State

The current system is split across multiple cache and state mechanisms:

| Area                   | Current mechanism                            | Current behavior                                                           | Mismatch vs target                                          |
| ------------------------| ----------------------------------------------| ----------------------------------------------------------------------------| -------------------------------------------------------------|
| Bias outputs           | `utils/cache/bias_node_cache.py`             | Per-node parquet files keyed by `(module, ticker, tf, params)`             | Not a candle SSOT, weak coverage enforcement                |
| Cache population       | `utils/cache/cache_manager.py`               | Offline batch population by streaming candles through nodes                | Not a runtime orchestrator, no unified lineage              |
| Cross-ticker candles   | `utils/cache/cross_ticker_store.py`          | Centralized cache-owned singleton; legacy `utils/data/` path is now a shim | Compatibility `None` fallback still exists on `get_candle` |
| Node runtime           | `nodes/__init__.py`                          | Streamed candle execution plus optional per-node persistent cache          | Cache bootstrap is optional and per-node                    |
| Feature extraction     | `feature_extraction/feature_extractor.py` + `utils/cache/feature_pipeline_support.py` | Entry points remain in feature extraction, but cache/cross-ticker helpers are centralized under `utils/cache` | Candles remain the primary driver even when reading cache   |
| Portfolio APIs         | `ensemble/portfolio.py`                      | `GlobalPortfolio.fit/predict(candles_per_tf, daily_volatility_df, ...)`    | Not datetime/range driven                                   |
| Live orchestration     | `deployment/forecast_server.py`              | Candle buffers + `MLManager` + `CrossTickerDataStore` + separate vol state | Multiple parallel stores, not one SSOT                      |
| Training orchestration | `deployment/production_training_pipeline.py` | Reads parquet directly and computes features locally                       | Does not treat the cache as the primary training contract   |

Implementation note:

- cache-specific support code should live under `utils/cache`
- legacy locations such as `utils/data/cross_ticker_store.py` and `feature_extraction/backtest.py` should be compatibility facades only

## Goals For The Refactor

1. Introduce a central cache abstraction that owns candles, artifact coverage metadata, and cache miss semantics.
2. Move cross-ticker and future cross-timeframe lookups behind that abstraction.
3. Make feature extraction read from the central store instead of switching between two execution models.
4. Make `DiversifiedEnsemble`, `TFPortfolio`, and `GlobalPortfolio` query by datetime/range/grid.
5. Unify volatility lineage with the rest of the pipeline.
6. Replace permissive fallbacks with explicit, typed failures.
7. Automatically invalidate and manage refresh of dependent artifacts when source OHLCV changes.
8. Separate production/live artifacts from disposable research artifacts, with vault-selected bias nodes as the live source of truth.
9. Replace today’s scattered cache entrypoints with one simple public cache API.
10. Consolidate cache-related ownership so behavior is easy to locate, reason about, and change.
11. Migrate research code and every portfolio consumer to the new cache/portfolio API.
12. Preserve deterministic replay and backtest/live parity.

## Non-Goals

- Rewriting model math, weight-layer formulas, or IDM/FDM logic
- Changing vault schema semantics unless required by the new query contracts
- Collapsing all caches into one file format immediately; an adapter layer is acceptable during migration

## Success Criteria

The refactor is complete when all of the following are true:

- candles and bias artifacts are addressed through one central cache contract
- coverage is explicit for each `(ticker, timeframe, artifact, date range)`
- missing required data raises a typed error with the missing key/range
- `GlobalPortfolio.fit` and `predict` no longer require `candles_per_tf`
- volatility reads come from the same lineage or a single documented table
- updates to OHLCV automatically mark dependent bias and derived artifacts stale
- users do not need manual cache-maintenance steps after OHLCV updates
- vault-selected bias nodes are the retained source of truth for live trading artifacts
- research-only bias-node artifacts can be isolated and cleaned up without affecting live or deployment state
- callers interact with one easy-to-use cache facade rather than choosing among multiple storage classes
- cache logic is no longer scattered across unrelated modules without a clear ownership boundary
- research workflows and portfolio consumers no longer rely on legacy candle-frame portfolio APIs
- live and backtest both follow ingest first, then query
- existing candle-frame entrypoints are either deleted or clearly deprecated and isolated

## Highest-Risk Areas

- implicit fallback behavior in `CrossTickerDataStore` and `DiversifiedEnsemble`
- partial coverage currently treated as acceptable in `BiasNodeCache`
- automatic rebuild fan-out could become expensive or opaque if refresh policy is too eager
- internal migration may keep too many legacy entrypoints alive and fail to simplify the public API
- implementation may remain too scattered internally and stay hard to debug even if the facade improves
- live runtime currently owns its own state model instead of using a central store
- test suite pins the old API contracts in many places

## API Principle

The refactor may keep multiple internal modules, files, or helper classes, but it should expose one public cache/storage API to callers.

In practical terms, a caller should not need to choose between:

- `BiasNodeCache`
- `CacheManager`
- `CrossTickerDataStore`
- separate volatility readers
- separate research vs live storage interfaces

If a caller still has to know which of those components to use directly, the unification is incomplete.

## Ownership Principle

The refactor should improve implementation discoverability, not just caller ergonomics.

That means:

- cache-related logic should live in a small, obvious part of the codebase
- ownership boundaries should be clear enough that a maintainer knows where to go to change cache behavior
- legacy scattered implementations should be removed or reduced to thin adapters during migration
