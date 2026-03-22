# Portfolio API Cutover Plan

## Objective

Replace candle-frame-driven portfolio and ensemble entrypoints with cache-native datetime/range queries, without duplicating the old candle-routing logic behind a new API.

## Primary Files Impacted

- `ensemble/diversified_ensemble.py`
- `ensemble/portfolio.py`
- `ensemble/portfolio_manager.py`
- `ensemble/portfolio_tester.py`
- direct callers in tests and deployment

## Current Public Contracts

### `GlobalPortfolio`

Current signatures:

- `fit(candles_per_tf, instrument_returns, daily_volatility_df)`
- `predict(candles_per_tf, daily_volatility_df)`

Current behavior:

- loops over `TFPortfolio` instances
- calls `fit_from_candles()` / `predict_base_model_vectors_from_candles()`
- aligns forecast streams to a daily grid
- combines them with the global `WeightLayer`

### `TFPortfolio`

Current behavior is still candle-driven via:

- `fit_from_candles()`
- `predict_from_candles()`
- `predict_base_model_vectors_from_candles()`

### `DiversifiedEnsemble`

Current behavior:

- computes returns from candle DataFrames
- aligns volatility from an externally passed `daily_volatility_df`
- asks base models to predict from per-ticker candle slices
- may retry with `use_cache=False` after cache miss

That fallback is incompatible with the target fail-fast semantics.

## Target Contract Shape

The exact API names can change, but the contract should move toward:

- `fit(query_spec, instrument_returns or returns_query, fit_range)`
- `predict(query_spec, prediction_grid)`

Where `query_spec` identifies:

- instruments
- timeframes
- required artifacts
- exact vs as-of semantics
- optional environment or cache namespace

The important part is that portfolio layers query the central store for already-defined lineages instead of reconstructing features from candle frames at call time.

## Required Changes

### 1. `DiversifiedEnsemble`

Add cache-native methods that:

- read feature artifacts and EWSD volatility lineage from the central store
- raise typed coverage errors on missing requirements
- optionally materialize ensemble/base-model vectors if that is chosen as a storage policy
- remain compatible with future persistence of portfolio-level artifacts such as prediction grids

Remove behavior where a cache miss silently triggers uncached recomputation inside prediction.

### 2. `TFPortfolio`

Split responsibilities:

- query or receive per-timeframe forecast vectors
- combine vectors and compute per-timeframe IDM/instrument weighting

Do not keep `TFPortfolio` responsible for both candle ingestion and portfolio combination once the new store exists.

### 3. `GlobalPortfolio`

Keep the daily-grid alignment and global combine logic if it still fits, but change inputs to:

- range-driven historical streams for `fit`
- datetime/grid-driven streams for `predict`

The new `GlobalPortfolio` should not know how to rebuild candles or fetch volatility tables manually.
The new portfolio stack should resolve volatility through EWSD cache reads only, not through an externally supplied `daily_volatility_df`.

### 4. Wrappers and tooling

`PortfolioManager` and `PortfolioTester` currently preserve the old candle-routing contract. Decide whether they:

- become adapters around the new query API, or
- are deprecated and deleted after caller migration

## Migration Order

1. Introduce central-store-backed ensemble query methods.
2. Refactor `TFPortfolio` to consume cache-native vectors.
3. Refactor `GlobalPortfolio` to consume query specs and grids.
4. Add temporary adapter shims for callers that still pass candles.
5. Remove candle-frame APIs once tests and deployment move over.

## Primary Tests To Update

- `tests/unit-tests/ensemble/test_global_portfolio.py`
- `tests/unit-tests/ensemble/test_volatility_input_contract.py`
- `tests/integration/test_portfolio_integration.py`
- `tests/portfolio_research/test_run_portfolio_test_multitimeframe.py`

Add tests for:

- datetime-grid prediction with complete coverage
- typed failure on missing feature or EWSD coverage
- fit/predict parity across the same cached lineage
- forward-fill and higher-timeframe carry logic still matching current methodology

## Critical Risks

- changing `GlobalPortfolio` first will only hide the old candle coupling
- fit and predict may diverge if returns, vol, and rebalance grids do not come from one documented lineage
- wrappers may reintroduce candles as an unofficial backdoor if left vague

## Exit Criteria

- `GlobalPortfolio` no longer requires `candles_per_tf`
- `daily_volatility_df` is no longer a required external portfolio input
- EWSD cache is the only sanctioned volatility source for portfolio and ensemble prediction
- ensemble and portfolio layers fail explicitly on missing required cache coverage
- candle-routing APIs are either deleted or clearly marked as temporary adapters
