# Central Cache User Guide

> [!tip] Use this page when you just want to read or write cache data without learning the whole implementation first.

## Mental Model

- `data/ohlc_data` is a bootstrap and integration-test dataset
- `.cache/trading_algo/central_cache` is the writable runtime cache and query source
- `CentralCacheStore` is the main facade
- `ArtifactDescriptor` names derived artifacts such as EWSD or node outputs
- `CacheRequest` describes exact, as-of, or range reads
- **Inference vs fitting:** routine candle updates call for **inference** only—refresh stale **bias** outputs and run **predict** on base models and portfolio using **already fitted** state. **Fitting** (`fit_from_cache` for a research window, base model `fit`, vault saves) is a **research / refit** concern, not something that runs on every live bar. See the *Inference vs fitting (live vs research)* subsection in [[Cache/architecture]].

## Imports

```python
from datetime import datetime

import pandas as pd

from lib.cache import (
    ArtifactDescriptor,
    ArtifactScope,
    CacheRequest,
    CentralCacheStore,
    CrossTickerDataStore,
    LookupMode,
    bootstrap_source_candles,
)
from lib.core.enums import Ticker, TimeFrame
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
- if `deployment/config/live_cache_refresh.json` is enabled, LIVE candle writes also schedule the async live refresh flow for tracked `(ticker, timeframe)` keys

### IBKR `CONTFUT` vs Norgate `&*_CCB` (live append)

Repository dailies in `data/ohlc_data/` are built from **Norgate continuous back-adjusted** futures symbols (e.g. `&ES_CCB`); see [[Data/norgate]]. The TWS live path fetches **Interactive Brokers continuous futures** (`secType=CONTFUT` in `scripts/enigma_live_forecast.py`), which use **IB’s own roll and adjustment rules** — they will not match Norgate levels bar-for-bar on the same calendar date.

When `upsert_tws_candles` writes IB dailies into `CentralCacheStore`:

1. **Append-only:** only sessions **strictly after** the current cache’s last daily timestamp are kept, so a long IB lookback does not bulk-overwrite Norgate-backed overlap (see `prepare_ib_rows_for_central_cache_append` in `lib/cache/runtime/ib_candle_ratio_align.py`).
2. **Junction ratio:** the appended block's `open/high/low/close` are multiplied by a single factor `last_close_cache / first_new_ib_close` so the first new close lines up with the last pre-existing close; relative moves within the block are unchanged. Ratio alignment is applied for every IB append batch.

`upsert_candles` itself remains a generic merge-by-timestamp; the IB-specific policy lives in the TWS upsert helper above.

### Explicit bootstrap from the repository source dataset

Use the bootstrap helper when you want a one-time write from `data/ohlc_data` into the runtime cache.

`CacheManager.bootstrap_source_candles` (and the module wrapper) show a **tqdm** bar (`Bootstrap OHLC → cache`) over each `(ticker, timeframe)` series. `ensure_bias_cache_coverage` shows a `Bias / EWSD cache (check)` bar over the coverage scan and a `Bias / EWSD artifacts (rebuild)` bar over the artifacts it actually rebuilds.

```python
summary = bootstrap_source_candles(
    tickers=[Ticker.ES, Ticker.NQ],
    timeframes=[TimeFrame.D, TimeFrame.W],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    reset_existing=False,
)
```

CLI equivalent (the runnable module is under `runtime/`; there is no top-level `lib/cache/bootstrap_source_candles.py`):

```bash
python -m lib.cache.runtime.bootstrap_source_candles --tickers ES NQ --timeframes D W --start 2020-01-01 --end 2024-12-31
```

`bootstrap_source_candles(...)` is the only bootstrap entrypoint; there is no
`ingest_source_candles(...)` symbol in the current code.

### Automatic live refresh from candle writes

If the live refresh manifest is configured, LIVE candle writes become the single operator-maintained input:

- candle writes mark dependent artifacts stale
- the runtime coalesces dirty keys asynchronously
- stale live bias artifacts are rebuilt
- live base-model prediction parquet files are rematerialized
- live portfolio prediction parquet files are rematerialized

This path is inference only. It does not refit models and it does not create new portfolio snapshots.

Manual recovery:

```python
from lib.cache import run_live_cache_refresh_now

summary = run_live_cache_refresh_now(
    manifest_path="deployment/config/live_cache_refresh.json",
)
```

Status file:

```text
.cache/trading_algo/central_cache/live_refresh/last_run.json
```

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
from lib.cache import ArtifactMissingError, CacheCoverageError

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

For **live** ticks, use **`predict_from_cache`** after bias artifacts are current. **`fit_from_cache`** is for **research** calibration on a historical window (or explicit refit), not for each new bar—see [[Cache/architecture]] (*Inference vs fitting*).

If `deployment/config/live_cache_refresh.json` is enabled, LIVE candle writes keep the deployed `portfolio_id` snapshots current automatically by rematerializing the affected live outputs.

## 9. Preflight vault cache coverage

Before running portfolio backtests, refresh the cache for the exact ensemble set and date window you need.

```python
from ensemble.vault_manager import ensure_vault_cache_coverage

summary = ensure_vault_cache_coverage(
    vault_ensemble_dirs=(
        "vault/D/es_tlt/rebalancing_es_tlt_long",
        "vault/M/buy_hold/buy_hold_long",
    ),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

What this does:

- migrates empty legacy `members` keys out of vault feature files
- reads bias-node specs from nested `<vault_root>/<TF>/<group>/<ensemble>/features/*.json` (and legacy flat `<vault_root>/<TF>/<ensemble>/features/*.json`; prop tree is `vault/`, personal is `vault_personal/` — [[Vault/vault]])
- rebuilds only missing, stale, or out-of-range `family="bias"` artifacts
- always ensures daily EWSD coverage for the requested tickers

`research/portfolio/run_portfolio_test.py` now does two explicit steps for the full train-to-test window:

- bootstraps the exact required candle set from `data/ohlc_data` into `.cache/trading_algo/central_cache/candles`
- runs `ensure_vault_cache_coverage(...)` to rebuild only missing or stale live artifacts

`PortfolioResearchConfig.populate_cache` remains a deprecated no-op.

The refresh step assumes candle coverage already exists in the central cache. It does not auto-ingest from `data/ohlc_data`; if candle coverage is missing, bootstrap it first.

## 10. Portfolio Workflow

### Simplest portfolio workflow

If you are running the standard portfolio research entrypoint, use this mental model:

1. Run `python research/portfolio/run_portfolio_test.py`
2. Let the runner bootstrap the exact required candles from `data/ohlc_data`
3. Let the runner validate exact cache coverage and rebuild stale artifacts only

What the runner now does before fitting the portfolio:

- migrates vault feature files if they still contain empty legacy `members` keys
- bootstraps the portfolio tickers, ensemble tickers, cross-ticker dependencies, and required timeframes into `.cache/trading_algo/central_cache/candles`
- rebuilds only the live bias artifacts that are missing or stale
- ensures daily EWSD exists for the tickers used by the portfolio
- runs fit and predict from cache-native portfolio queries

This means you do not need a separate manual bootstrap step before a normal portfolio test run.

### Portfolio addition gate (candidate strategy)

After a strategy clears exploration and validation, use the **portfolio addition** phase to decide whether it should enter the portfolio at all. In current local code, some configs and commands still use the older term `inclusion`, but the target workflow is `exploration -> validation -> portfolio_addition`.

Local compatibility tooling still computes the familiar checks: per-peer validation forecast correlation, standalone metrics (Sharpe/Sortino/Calmar) for the candidate and each baseline ensemble, portfolio uplift on train / validation / train+validation, and an optional test-window check. Thresholds and CSV output currently live on `ResearchConfig.portfolio_inclusion`; baseline portfolio comes from `research.portfolio.config.load_config()`. Compatibility CLI: `python -m research.feature.run_inclusion_gates` (default candidate is `eval_bias_spec` from research config; use ``--candidate-mode vault_path`` and a path for an on-disk ensemble).

Canonical workflow reference: `docs/SaaS/robustness_tests/portfolio_addition.md`.

Implementation detail reference: [[Ensemble/portfolio]] (section **Portfolio research — portfolio addition gate**).

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
        "vault/D/es_tlt/rebalancing_es_tlt_long",
        "vault/M/buy_hold/buy_hold_long",
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
from lib.core.enums import TimeFrame

ensure_vault_cache_coverage(
    vault_ensemble_dirs=("vault/D/es_tlt/rebalancing_es_tlt_long",),
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
2. **Refresh** stale derived artifacts for the affected ensembles (bias outputs, then inference to forecasts)—this is **inference**, not refitting models or portfolio weights
3. Run portfolio research, feature research, or live **prediction**

**Live trading:** a new bar implies you need a new **forecast**, so you will run the inference chain (bias → base-model predict → `predict_from_cache`) for the deployed ensemble anyway. Orchestrate that after candle ingest; do **not** confuse it with **`fit_from_cache`**, which is for research windows and refits.

Why this order matters:

- `CentralCacheStore.upsert_candles(...)` marks dependent artifacts stale
- stale artifacts cannot be read as if they were current
- the correct fix is to refresh them (recompute derived columns from fixed specs and fitted state), not to ignore the lifecycle error

### Runtime update workflow

Use this when live candles simply gained new rows or a bar was corrected.

```bash
python -m lib.cache.runtime.bootstrap_source_candles --tickers ES TLT --timeframes D M
python research/portfolio/run_portfolio_test.py
```

Notes:

- the bootstrap step is one-time or occasional, not part of the normal runtime loop
- subsequent live updates should use `CentralCacheStore.upsert_candles(...)`
- stale bias artifacts are rebuilt statelessly: `CacheManager` instantiates a fresh node, prepends the node's cold-rebuild warmup window, and trims the saved artifact back to the requested range
- if you run `run_portfolio_test.py`, the portfolio preflight will validate exact coverage and rebuild stale artifacts for the requested window

### Full source bootstrap refresh

Use this when you replaced or corrected the source OHLC files and want to refresh the runtime candle cache from the repository dataset.

```bash
python -m lib.cache.runtime.bootstrap_source_candles --tickers ES TLT --timeframes D M --reset-existing
python research/portfolio/run_portfolio_test.py
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
    vault_ensemble_dirs=("vault/D/es_tlt/rebalancing_es_tlt_long",),
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
python -m lib.cache.runtime.bootstrap_source_candles --tickers ES TLT --timeframes D W M
python research/portfolio/run_portfolio_test.py
```

Expected result:

- candle cache is populated under `.cache/trading_algo/central_cache/candles`
- portfolio preflight validates exact coverage and builds the required live bias artifacts and EWSD
- portfolio test runs entirely from cache after preflight

### Normal portfolio research run

Use this day to day once the runtime candle cache is already bootstrapped and kept current via `upsert_candles(...)`.

```bash
python research/portfolio/run_portfolio_test.py
```

This is enough for most cases because the runner now performs the cache preflight automatically.

### Feature research after new data

`research.feature` uses the same central-cache lifecycle. Each pipeline calls `populate_cache_if_needed` up front (bootstrap candles, refresh missing/stale bias artifacts and EWSD), then loads features with cache-backed reads. Cache population uses the **full common OHLC span** available for required tickers—not `ResearchConfig.start` / `end`—so indicators warm up once at data inception; analysis phases slice to train/validation windows only when loading features. `ResearchConfig` does not expose `use_cache` / `populate_cache` toggles—that behavior is fixed.

The documentation target is now a three-phase package structure:

1. `exploration`
2. `validation`
3. `portfolio_addition`

The codebase is still in a compatibility-preserving migration, so the practical entrypoints you see today may still use older names.

Typical sequence during migration:

```bash
python research/feature/in_sample/run_is.py
python research/feature/validation/run_validation.py
```

Interpret these as:

- `in_sample/run_is.py` -> current exploration-era entrypoint
- `validation/run_validation.py` -> validation phase entrypoint
- portfolio-addition scaffolding is migrating separately; use the portfolio-addition docs above for the conceptual next step

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
2. Run `python research/portfolio/run_portfolio_test.py`, or preflight manually if you want the summary first
3. Let the cache rebuild only missing or stale artifacts
4. Run portfolio queries with `PortfolioCacheQuery`

### Training

1. Ensure candle coverage exists in the cache
2. Read cached features or populate them on miss
3. Train from cache-backed feature and volatility lineage
4. **Fit** portfolios and base models in research as needed (`fit_from_cache`, vault saves)—separate from the live inference loop below

### Live

1. Write new bars into the runtime cache with `upsert_candles(...)`
2. **Inference only:** refresh stale **bias** artifacts for the exact live ensemble set (stateless cold rebuilds sized from each bias node's warmup metadata), run base-model **predict** with fitted vault models, then `predict_from_cache(...)`—all using **existing** fitted weights and specs
3. If the live refresh manifest is enabled, this fanout is triggered automatically from the LIVE candle write boundary and the latest status lands in `.cache/trading_algo/central_cache/live_refresh/last_run.json`
4. Read the latest rows with exact or as-of requests as needed
5. Do **not** run portfolio or base-model **fit** on each tick; refits are explicit research or scheduled events

The live loop is “new OHLC → consistent derived features → next forecast,” not “refit the book on every bar.”

## Related

- [[Cache/architecture]] — system design and ownership rules
- [[Vault/architecture]] — vault-side persistence and snapshot architecture
- [[Vault/user_guide]] — vault-side practical usage guide
- [[Deployment/live_cache_refresh]] — manifest contract and automatic live refresh semantics
- [[portfolio]] — portfolio layer behavior
- [[pipeline]] — feature extraction workflow
- [[live_multi_timeframe]] — live orchestration flow

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
