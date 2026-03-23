# Central Cache User Guide

> [!tip] Use this page when you just want to read or write cache data without learning the whole implementation first.

## Mental Model

- `data/ohlc_data` is a bootstrap and integration-test dataset
- `.cache/trading_algo/central_cache` is the writable runtime cache and query source
- `CentralCacheStore` is the main facade
- `ArtifactDescriptor` names derived artifacts such as EWSD or node outputs
- `CacheRequest` describes exact, as-of, or range reads

## Imports

```python
from datetime import datetime

import pandas as pd

from utils.cache import (
    ArtifactDescriptor,
    ArtifactScope,
    CacheRequest,
    CentralCacheStore,
    CrossTickerDataStore,
    LookupMode,
    bootstrap_source_candles,
)
from utils.core.enums import Ticker, TimeFrame
```

## 1. Write candles into the cache

Use `upsert_candles(...)` for normal runtime updates from live feeds or incremental corrections.

```python
cache = CentralCacheStore.get_instance()

cache.upsert_candles(
    Ticker.ES,
    TimeFrame.D,
    candles_df,
)
```

Notes:

- `candles_df` must be indexed by datetime
- candle writes go to `.cache/trading_algo/central_cache/candles`
- `upsert_candles(...)` merges by datetime and replaces any overlapping bar
- use `set_candles(...)` only when you want a full snapshot replacement
- do not write runtime candles back into `data/ohlc_data`

### Explicit bootstrap from the repository source dataset

Use the bootstrap helper when you want a one-time write from `data/ohlc_data` into the runtime cache.

```python
summary = bootstrap_source_candles(
    tickers=[Ticker.ES, Ticker.NQ],
    timeframes=[TimeFrame.D, TimeFrame.W],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    reset_existing=False,
)
```

CLI equivalent:

```bash
python -m utils.cache.bootstrap_source_candles --tickers ES NQ --timeframes D W --start 2020-01-01 --end 2024-12-31
```

`ingest_source_candles(...)` still exists as a deprecated compatibility alias, but new code should call `bootstrap_source_candles(...)`.

## 2. Read candles

### Exact bar

```python
bar = cache.query_candle(
    Ticker.ES,
    TimeFrame.D,
    datetime(2024, 1, 5),
)
```

### As-of bar

Use `LookupMode.AS_OF` when you need the last completed bar at or before a timestamp.

```python
bar = cache.query_candle(
    Ticker.ES,
    TimeFrame.D,
    datetime(2024, 1, 5, 12, 0),
    lookup_mode=LookupMode.AS_OF,
)
```

### Range

```python
window = cache.query_candles(
    Ticker.ES,
    TimeFrame.D,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 31),
)
```

## 3. Write a derived artifact

Use `ArtifactDescriptor` for anything derived from candles: node output, EWSD, forecast vectors, or future portfolio-level artifacts.

```python
descriptor = ArtifactDescriptor(
    family="bias",
    ticker=Ticker.ES,
    timeframe=TimeFrame.D,
    module_name="ewsd",
    params={"long_run_window": 2520},
    artifact_name="ewsd",
    scope=ArtifactScope.LIVE,
)

cache.write_artifact(
    descriptor,
    ewsd_df,
    depends_on=((Ticker.ES, TimeFrame.D),),
)
```

Why `depends_on` matters:

- candle updates can mark the artifact stale
- stale artifacts cannot be read as if they were still current
- source revisions stay attached to the artifact metadata

## 4. Read an artifact

### Full artifact

```python
feature_df = cache.read_artifact(descriptor)
```

### Range

```python
feature_df = cache.read_artifact(
    descriptor,
    request=CacheRequest(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 3, 31),
    ),
)
```

### Exact or as-of row

```python
exact_row = cache.read_artifact(
    descriptor,
    request=CacheRequest(exact_dt=datetime(2024, 3, 1)),
)

as_of_row = cache.read_artifact(
    descriptor,
    request=CacheRequest(as_of_dt=datetime(2024, 3, 1, 12, 0)),
)
```

## 5. Handle errors explicitly

The cache contract is fail-fast. Expect typed exceptions.

```python
from utils.cache import ArtifactMissingError, CacheCoverageError

try:
    feature_df = cache.read_artifact(descriptor, request=CacheRequest(start=start, end=end))
except ArtifactMissingError:
    # Materialize or load the missing artifact first.
    ...
except CacheCoverageError:
    # Extend the available coverage before retrying.
    ...
```

Use this rule of thumb:

- `ArtifactMissingError`: nothing usable exists for the requested key or timestamp
- `CacheCoverageError`: the artifact exists, but not over the full requested range

## 6. Use research vs live scopes correctly

Use `ArtifactScope.RESEARCH` for local experiments and `ArtifactScope.LIVE` for artifacts that support live or deployment paths.

```python
research_descriptor = descriptor.with_scope(ArtifactScope.RESEARCH)
cache.write_artifact(research_descriptor, research_df, depends_on=((Ticker.ES, TimeFrame.D),))

live_descriptor = cache.promote_artifact(research_descriptor, target_scope=ArtifactScope.LIVE)

cache.prune_scope(ArtifactScope.RESEARCH)
```

## 7. Cross-ticker lookups

Use the cache-owned adapter when node code needs another instrument's candles.

```python
store = CrossTickerDataStore.get_instance()

nq_bar = store.query_candle(
    Ticker.NQ,
    TimeFrame.D,
    datetime(2024, 3, 1),
)
```

Prefer:

- `query_candle(...)` for typed failure behavior
- `get_candle(...)` only when you are intentionally preserving older `None`-on-miss compatibility

## 8. Portfolio queries

Cache-native portfolio APIs consume `PortfolioCacheQuery` instead of large `candles_per_tf` payloads.

```python
from ensemble.portfolio import PortfolioCacheQuery

query = PortfolioCacheQuery(
    tickers=("ES", "NQ"),
    start=datetime(2023, 1, 1),
    end=datetime(2024, 1, 1),
    timeframes=(TimeFrame.D, TimeFrame.W),
)

global_portfolio.fit_from_cache(query, instrument_returns=returns_df)
positions = global_portfolio.predict_from_cache(query)
```

The cache-native portfolio path resolves volatility from cached EWSD artifacts. Callers should not supply `daily_volatility_df`.

## 9. Preflight vault cache coverage

Before running portfolio backtests, refresh the cache for the exact ensemble set and date window you need.

```python
from ensemble.vault_manager import ensure_vault_cache_coverage

summary = ensure_vault_cache_coverage(
    vault_ensemble_dirs=("vault/D/rebalancing_es_tlt_long", "vault/M/buy_hold_long"),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

What this does:

- migrates empty legacy `members` keys out of vault feature files
- reads bias-node specs from `vault/*/*/features/*.json`
- rebuilds only missing, stale, or out-of-range `family="bias"` artifacts
- always ensures daily EWSD coverage for the requested tickers

`portfolio_research/run_portfolio_test.py` now does two explicit steps for the full train-to-test window:

- bootstraps the exact required candle set from `data/ohlc_data` into `.cache/trading_algo/central_cache/candles`
- runs `ensure_vault_cache_coverage(...)` to rebuild only missing or stale live artifacts

`PortfolioResearchConfig.populate_cache` remains a deprecated no-op.

The refresh step assumes candle coverage already exists in the central cache. It does not auto-ingest from `data/ohlc_data`; if candle coverage is missing, bootstrap it first.

## 10. Portfolio Workflow

### Simplest portfolio workflow

If you are running the standard portfolio research entrypoint, use this mental model:

1. Run `python portfolio_research/run_portfolio_test.py`
2. Let the runner bootstrap the exact required candles from `data/ohlc_data`
3. Let the runner validate exact cache coverage and rebuild stale artifacts only

What the runner now does before fitting the portfolio:

- migrates vault feature files if they still contain empty legacy `members` keys
- bootstraps the portfolio tickers, ensemble tickers, cross-ticker dependencies, and required timeframes into `.cache/trading_algo/central_cache/candles`
- rebuilds only the live bias artifacts that are missing or stale
- ensures daily EWSD exists for the tickers used by the portfolio
- runs fit and predict from cache-native portfolio queries

This means you do not need a separate manual bootstrap step before a normal portfolio test run.

### Manual portfolio preflight

Use the manual preflight path when:

- you want to validate cache coverage before a long run
- you are debugging a specific ensemble set
- you want to inspect the refresh summary before running the portfolio

```python
from datetime import datetime

from ensemble.vault_manager import ensure_vault_cache_coverage

summary = ensure_vault_cache_coverage(
    vault_ensemble_dirs=(
        "vault/D/rebalancing_es_tlt_long",
        "vault/M/buy_hold_long",
    ),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
)

print(summary["rebuilt"], summary["validated"], summary["failed"])
```

Recommended sequence for manual runs:

1. Bootstrap the candle cache for the exact tickers and timeframes you need
2. Choose the exact ensemble directories and the exact window you plan to test
3. Run `ensure_vault_cache_coverage(...)`
4. Check that `failed == 0`
5. Run the portfolio test or your own `PortfolioCacheQuery` flow

### Cache-native portfolio usage

If you are calling the portfolio layer directly, preflight first, then fit and predict from cache.

```python
from datetime import datetime

from ensemble.portfolio import PortfolioCacheQuery
from ensemble.vault_manager import ensure_vault_cache_coverage
from utils.core.enums import TimeFrame

ensure_vault_cache_coverage(
    vault_ensemble_dirs=("vault/D/rebalancing_es_tlt_long",),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
)

query = PortfolioCacheQuery(
    tickers=("ES",),
    start=datetime(2020, 1, 1),
    end=datetime(2024, 12, 31),
    timeframes=(TimeFrame.D,),
)

portfolio.fit_from_cache(query, instrument_returns=instrument_returns_df)
positions = portfolio.predict_from_cache(query)
```

Rules of thumb:

- preflight over the full train-start to test-end window, not just the prediction slice
- treat cache misses as a setup problem, not as something to bypass with uncached fallback
- do not manually inject `daily_volatility_df` into the cache-native path; EWSD is resolved from cache

## 11. What To Do When New Data Arrives

### Cache-first updates

Recommended pattern:

1. Upsert the new or corrected bars into `.cache/trading_algo/central_cache/candles`
2. Rebuild stale artifacts for the affected ensembles or feature specs
3. Run portfolio research, feature research, or live prediction

Why this order matters:

- `CentralCacheStore.upsert_candles(...)` marks dependent artifacts stale
- stale artifacts cannot be read as if they were current
- the correct fix is to refresh them, not to ignore the lifecycle error

### Runtime update workflow

Use this when live candles simply gained new rows or a bar was corrected.

```bash
python -m utils.cache.bootstrap_source_candles --tickers ES TLT --timeframes D M
python portfolio_research/run_portfolio_test.py
```

Notes:

- the bootstrap step is one-time or occasional, not part of the normal runtime loop
- subsequent live updates should use `CentralCacheStore.upsert_candles(...)`
- stale bias artifacts are rebuilt statelessly: `CacheManager` instantiates a fresh node, prepends the node's cold-rebuild warmup window, and trims the saved artifact back to the requested range
- if you run `run_portfolio_test.py`, the portfolio preflight will validate exact coverage and rebuild stale artifacts for the requested window

### Full source bootstrap refresh

Use this when you replaced or corrected the source OHLC files and want to refresh the runtime candle cache from the repository dataset.

```bash
python -m utils.cache.bootstrap_source_candles --tickers ES TLT --timeframes D M --reset-existing
python portfolio_research/run_portfolio_test.py
```

Use `--reset-existing` when:

- you rewrote historical source files
- you corrected bad bars
- you changed the available date range materially

Do not use `--reset-existing` just because new rows were appended. For appended bars, upsert directly into the runtime cache.

### Ensemble-specific refresh after new data

If only part of the portfolio universe changed, refresh only the affected ensemble set.

```python
from datetime import datetime

from ensemble.vault_manager import ensure_vault_cache_coverage

summary = ensure_vault_cache_coverage(
    vault_ensemble_dirs=("vault/D/rebalancing_es_tlt_long",),
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 12, 31),
)
```

This is the right tool when:

- a daily ensemble got new bars but monthly buy-and-hold data did not change
- you want to warm only one vault before a targeted run
- you are troubleshooting one stale artifact family

## 12. Recommended Operational Workflows

### First-time bootstrap

Use this when the runtime cache is empty.

```bash
python -m utils.cache.bootstrap_source_candles --tickers ES TLT --timeframes D W M
python portfolio_research/run_portfolio_test.py
```

Expected result:

- candle cache is populated under `.cache/trading_algo/central_cache/candles`
- portfolio preflight validates exact coverage and builds the required live bias artifacts and EWSD
- portfolio test runs entirely from cache after preflight

### Normal portfolio research run

Use this day to day once the runtime candle cache is already bootstrapped and kept current via `upsert_candles(...)`.

```bash
python portfolio_research/run_portfolio_test.py
```

This is enough for most cases because the runner now performs the cache preflight automatically.

### Feature research after new data

`feature_research` uses the same central-cache lifecycle now. Its cache helper ensures the runtime candles are bootstrapped first, then refreshes missing or stale artifacts before loading features.

Typical sequence:

```bash
python feature_research/in_sample/run_is.py
python feature_research/validation/run_validation.py
python feature_research/oos/run_oos.py
```

You usually do not need a separate manual cache step here either, provided the central cache is already bootstrapped and updated.

## 13. Reset and cleanup

```python
cache.clear()  # in-memory state only
cache.clear_candles(purge_persisted=True)  # also deletes persisted candle cache files
CentralCacheStore.reset()  # resets the singleton
```

Use persisted cleanup carefully. It removes writable runtime cache files under `.cache/trading_algo/central_cache`, not the bootstrap dataset under `data/ohlc_data`.

## Common Patterns

### Backtest

1. Bootstrap or upsert the required candles into the runtime cache
2. Run `python portfolio_research/run_portfolio_test.py`, or preflight manually if you want the summary first
3. Let the cache rebuild only missing or stale artifacts
4. Run portfolio queries with `PortfolioCacheQuery`

### Training

1. Ensure candle coverage exists in the cache
2. Read cached features or populate them on miss
3. Train from cache-backed feature and volatility lineage

### Live

1. Write new bars into the runtime cache with `upsert_candles(...)`
2. Refresh stale artifacts for the exact live ensemble set using stateless cold rebuilds sized from each bias node's warmup metadata
3. Read the latest rows with exact or as-of requests
4. Run `predict_from_cache(...)`

## Related

- [[Cache/architecture]] — system design and ownership rules
- [[portfolio]] — portfolio layer behavior
- [[pipeline]] — feature extraction workflow
- [[live_multi_timeframe]] — live orchestration flow
