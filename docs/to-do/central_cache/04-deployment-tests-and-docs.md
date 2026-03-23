# Deployment, Tests, And Docs Plan

## Objective

Move live and training orchestration to the central cache model, then cut verification and documentation over to the new contracts.

## Primary Files Impacted

### Deployment and training

- `deployment/forecast_server.py`
- `deployment/production_training_pipeline.py`
- `deployment/mt5_data_connector.py`

### Verification

- `tests/integration/test_cache_integration.py`
- `tests/integration/test_portfolio_integration.py`
- `tests/portfolio_research/test_run_portfolio_test_multitimeframe.py`
- `tests/feature_research/validation/engine/test_portfolio_evaluator.py`
- `tests/feature_research/validation/engine/test_runner.py`
- `tests/unit-tests/ensemble/test_global_portfolio.py`
- `tests/unit-tests/ensemble/test_volatility_input_contract.py`
- `tests/nodes/test_cross_ticker.py`
- `tests/nodes/test_rebalancing.py`

### Research and evaluation callers

- `utils/evaluation/walkforward/portfolio_evaluator.py`
- research flows under `tests/portfolio_research/`
- feature-research portfolio evaluation flows under `tests/feature_research/`

### Docs

- `docs/library/Cache/architecture.md`
- `docs/library/Cache/user_guide.md`
- `docs/library/Ensemble/portfolio.md`
- `docs/library/Ensemble/multi_timeframe.md`
- `docs/library/Deployment/live_multi_timeframe.md`
- `docs/library/Feature_selection/pipeline.md`
- relevant API docs under `docs/api/`

## Deployment Refactor Work

### 1. Live runtime

`deployment/forecast_server.py` currently maintains several parallel state stores:

- candle buffers
- `MLManager` feature state
- `CrossTickerDataStore`
- separate EWSD volatility persistence

Refactor goal:

- live ingest writes new bars into the central cache
- forecast reads query the central cache by datetime/range
- cross-ticker and EWSD volatility reads go through the same contract

Required decisions:

- whether live still keeps in-memory warmed state for latency
- whether that state is only an optimization behind the central cache contract

### 2. Training runtime

`deployment/production_training_pipeline.py` currently reads parquet directly and computes features locally. Refactor it so training becomes:

1. declare coverage and required artifacts
2. populate central cache if needed
3. train from cache-backed feature/target reads

Prefer reusing `feature_extraction/feature_extractor.py` and related helpers rather than introducing a second feature-generation path for training.
Training and live should read from vault-backed selected artifacts, not from disposable research cache scope.

### 3. Failure policy

The architecture doc and current live doc diverge today:

- target doc says typed fail-fast cache misses
- live doc currently allows stale forecast carry-forward on fetch failure

Decide and document:

- what happens on ingest failure
- what happens on query-time artifact miss
- whether stale forecasts are allowed, and at which layer
- whether queries block on rebuild, fail with a typed rebuild-pending error, or allow a documented stale-read mode

## Verification Plan

### Tests to preserve conceptually

- streamed-vs-cached equivalence
- no future data leakage
- daily-grid alignment and higher-TF carry behavior
- cross-ticker dependency correctness
- cache population behavior already covered in `tests/unit-tests/utils/test_cache_manager.py`
- portfolio behavior in research and walkforward evaluation paths

### Tests to rewrite

Rewrite tests that currently pin:

- `candles_per_tf` as the portfolio contract
- `daily_volatility_df` as a required external input
- `CrossTickerDataStore` lazy auto-load and `None` fallback behavior
- research and evaluation helpers that still build candle-frame portfolio inputs

When rewriting, preserve useful existing assertions from `tests/unit-tests/utils/test_cache_manager.py` instead of replacing them with a completely separate test model for the new infra.

### New integration tests

Add integration coverage for:

- ingest-to-query round trip on the same cache
- cache miss error payloads
- partial coverage rejection
- corrected-source invalidation or version bump
- backtest/live parity against the same stored artifacts
- future artifact-family support, for example persisted portfolio prediction outputs
- automatic stale-marking and managed refresh after OHLCV updates
- research-scope cleanup that does not affect vault-backed live artifacts
- portfolio research and walkforward evaluation using the new cache/portfolio API

## Documentation Cutover

Documentation is a required deliverable for this refactor, not a cleanup nice-to-have. The final state should leave future contributors with stable reference docs for design, usage, and public APIs.

### Update examples

Rewrite examples that still show:

- passing `candles_per_tf` into `GlobalPortfolio`
- loading/fetching candles directly in top-level portfolio calls
- treating volatility as a separate required table instead of an EWSD cache read
- describing lookback warm-up candles as the top-level live contract

### Update architecture notes

After implementation details settle, update:

- central cache architecture doc to reflect actual contract names
- live multi-timeframe doc to describe ingest/query rather than fetch/warm/combine
- feature-selection pipeline doc to show cache-backed training flow

### Add permanent reference docs

Create or update durable documentation in `docs/library` and `docs/api` for:

- design: central cache architecture, artifact model, coverage semantics, invalidation/versioning, and EWSD volatility lineage
- usage: how to ingest candles, populate artifacts, query candles/artifacts, handle typed cache misses, understand stale/rebuilding artifact states, separate research artifacts from vault/live artifacts, and use the cache from backtest, training, and live flows
- API: public interfaces, request models, error types, and migration notes for replaced candle-frame APIs

Recommended doc outputs after the refactor:

- one central cache design doc under `docs/library/`
- one usage/playbook doc under `docs/library/`
- updated downstream docs in `docs/library/Ensemble/`, `docs/library/Deployment/`, and `docs/library/Feature_selection/`
- relevant module/API pages under `docs/api/` for the new cache services and query contracts
- documented researcher cleanup workflow and gitignored research-cache policy

### Documentation acceptance criteria

The refactor is not done until:

- a future developer can understand the cache design without reading implementation code first
- a future developer can use the cache from research, training, and live flows by following docs in `docs/library`
- public cache-facing APIs and error contracts are documented in `docs/api/`
- old examples that teach `candles_per_tf` and external volatility-table usage are removed or clearly marked historical

## Exit Criteria

- deployment writes and reads through the central cache contract
- training uses cache-backed artifacts as the primary path
- verification covers miss semantics, coverage, and parity
- permanent design, usage, and API docs exist under `docs/library` and `docs/api/`
- research folders and portfolio evaluation paths use the new API
- docs no longer present the old candle-frame APIs as the primary architecture
