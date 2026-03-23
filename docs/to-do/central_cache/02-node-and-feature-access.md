# Node And Feature Access Plan

## Objective

Move node lookup and feature extraction onto the central cache contract so candles are no longer passed around as the universal orchestration primitive.

## Primary Files Impacted

- `nodes/__init__.py`
- `feature_extraction/feature_extractor.py`
- `nodes/pairs/spread.py`
- `nodes/pairs/rebalancing.py`
- `nodes/pairs/rebalancing_cross.py`
- `utils.core.helpers` helpers that create nodes or preload cross-ticker dependencies
- `utils/evaluation/walkforward/portfolio_evaluator.py`

## Current Gaps

### `BiasNode`

`BiasNode` currently has:

- in-process memoization for the last candle
- optional persistent cache bootstrap via `_init_cache_after_params()`
- vectorized cache access only if subclasses correctly initialize their own cache

This is brittle. Cache availability should not depend on every node subclass remembering to call a bootstrapping method.

### Feature extraction

`feature_extraction/feature_extractor.py` currently has two models:

- cached reads from `BiasNodeCache`
- streaming candles through live node instances

Even the cached path still aligns outputs back onto candle-derived indexes and preloads cross-ticker data separately via `CrossTickerDataStore`.

### Cross-ticker access

Cross-ticker nodes call `CrossTickerDataStore` directly and accept permissive miss behavior. That conflicts with the central-cache design.

## Required Changes

### 1. Replace per-node cache bootstrap

Refactor `BiasNode` so node classes do not own persistent cache initialization.

Options:

- nodes remain pure streaming calculators and the orchestration layer handles all cache reads/writes
- nodes receive a query dependency/provider instead of constructing cache objects themselves

Recommended direction:

- keep nodes focused on pure or near-pure candle transformation
- move persistent storage concerns out of node constructors

### 2. Rework vectorized feature extraction

Replace `use_cache=True/False` as a top-level branching model with:

- a central-store-backed feature read path for the normal case
- an explicit artifact-population path when coverage is missing

Do not silently fall back from cache reads to streamed recomputation inside the same call.

This should be a refactor of `feature_extraction/feature_extractor.py`, not a new parallel extraction stack. Reuse existing pieces where possible:

- parameter-grid expansion
- cross-ticker preload helpers
- cached alignment logic
- EWSD-aware forward-return extraction flow

### 3. Unify dependency lookup semantics

Cross-ticker and future cross-timeframe lookups should use the same provider contract as primary candle reads:

- exact match for same-timeframe cross-ticker lookups
- as-of lookup for future cross-timeframe dependencies
- typed miss errors when required coverage is absent

### 4. Preserve research workflows

Permutation, override-candle, and walkforward workflows still need local overrides. The new store must support:

- in-memory or temporary dataset overlays
- test fixtures writing directly into the same query contract
- deterministic replay without direct filesystem assumptions

## Implementation Tasks

### Task group A: node contract cleanup

- audit nodes that call `_init_cache_after_params()`
- decide whether to remove `BiasNode.get_cached_values()` / `get_cached_dataframe()` or rebase them on the new store
- remove direct storage concerns from node subclasses over time

### Task group B: feature extraction migration

- refactor `feature_extractor` to query central artifacts by `(module, params, ticker, timeframe, range)`
- route cross-ticker dependency resolution through the central store
- make cache misses explicit and actionable
- add one sanctioned artifact-population path
- extract shared helpers from `feature_extraction/feature_extractor.py` instead of duplicating them in new cache readers

### Task group C: research and evaluation adapters

- update walkforward and portfolio evaluation utilities
- update permutation workflows that currently patch `CrossTickerDataStore`
- preserve testability for synthetic fixture-based unit tests

## Test Requirements

Revisit:

- `tests/nodes/test_rebalancing.py`
- `tests/nodes/test_cross_ticker.py`
- `tests/unit-tests/validators/permutation/test_data_loader_candle_override_unit.py`
- `tests/integration/test_cache_integration.py`

Add tests for:

- central-store-driven cross-ticker reads
- exact and as-of lookup enforcement
- feature extraction failure on incomplete coverage
- temporary override datasets that do not leak into persisted state

## Exit Criteria

- node subclasses no longer need to bootstrap persistent caches manually
- feature extraction has one primary cache-backed read contract
- cross-ticker and cross-timeframe dependency access use the same semantics as the rest of the store
- permutation and research workflows still work through explicit adapters
