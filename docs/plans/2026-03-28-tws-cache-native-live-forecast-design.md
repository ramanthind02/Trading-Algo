# Design: Cache-Native TWS Live Forecast Pipeline

**Date:** 2026-03-28
**Status:** Approved
**Branch:** abhi/ikbr-script

## Problem

The TWS live forecast script fetches data from IB, generates predictions, and sends Telegram notifications. Currently it passes DataFrames through the pipeline directly -- nothing is persisted to the central cache. This means:

- Bias node computations are thrown away after each run (no vectorized backtest reuse)
- No incremental data accumulation -- each run fetches a full year of history
- When the Norgate subscription expires, there's no mechanism to extend OHLC data from TWS

## Solution

Rewrite the script to use the central cache as the single source of truth:

1. **Bootstrap** from `data/ohlc_data/` on first run (auto-detected)
2. **Upsert** fresh TWS bars into the cache incrementally
3. **Refresh** stale bias node artifacts via `ensure_bias_cache_coverage()`
4. **Fit + predict** via `GlobalPortfolio.fit_from_cache()` / `predict_from_cache()`

## Data Flow

```
First run:
  data/ohlc_data/ ──bootstrap──> .cache/central_cache/candles/
                                         ↓
Every run:                        TWS daily bars
                                         ↓ upsert_candles()
                                  .cache/central_cache/candles/
                                         ↓ (marks dependents stale)
                                  ensure_bias_cache_coverage()
                                         ↓ (rebuilds stale artifacts)
                                  .cache/central_cache/artifacts/
                                         ↓
                                  GlobalPortfolio.fit_from_cache(query)
                                  GlobalPortfolio.predict_from_cache(query)
                                         ↓
                                  Telegram + Console output
```

**Key invariant:** `data/ohlc_data/` is never modified by the live script. It remains an immutable Norgate snapshot. The cache accumulates TWS bars incrementally on top of the bootstrapped baseline.

## Script Flow (step by step)

### Step 1: Build portfolio from vault (unchanged)
```python
portfolio = build_portfolio(config)
required_tickers = discover_required_tickers(portfolio)
```

### Step 2: Auto-bootstrap check
Check if the central cache has candles for the required tickers. If not, bootstrap from `data/ohlc_data/`.

```python
store = CentralCacheStore.get_instance()
manager = CacheManager()

needs_bootstrap = False
for ticker_str in required_tickers:
    ticker = Ticker[ticker_str]
    record = store.describe_candle(ticker, TimeFrame.D)
    if record is None:
        needs_bootstrap = True
        break

if needs_bootstrap:
    manager.bootstrap_source_candles(
        tickers=[Ticker[t] for t in required_tickers],
        timeframes=[TimeFrame.D, TimeFrame.M],
    )
```

### Step 3: Connect to TWS, fetch daily bars
Fetch bars for all required tickers (same as current script). Only fetch recent data -- enough to cover any gap since the last cache entry.

```python
# Determine how many days to fetch per ticker
for ticker_str in required_tickers:
    record = store.describe_candle(Ticker[ticker_str], TimeFrame.D)
    if record and record.coverage.end:
        # Only fetch from last cached date to today
        days_gap = (datetime.now() - record.coverage.end).days + 5  # small buffer
        lookback = max(days_gap, 10)  # minimum 10 days
    else:
        lookback = config["data"]["lookback_days"]
```

### Step 4: Upsert TWS bars into central cache
```python
for ticker_str, candles_df in fetched_candles.items():
    store.upsert_candles(
        Ticker[ticker_str], TimeFrame.D, candles_df
    )
    # This automatically marks dependent bias artifacts as STALE
```

Also resample daily to monthly and upsert monthly candles:
```python
monthly = resample_daily_to_monthly(daily_candles_from_cache)
for ticker_str, group in monthly.groupby("ticker"):
    store.upsert_candles(Ticker[ticker_str], TimeFrame.M, group)
```

### Step 5: Refresh stale bias node artifacts
```python
vault_dirs = [list of vault ensemble directories]
coverage_end = datetime.now()
coverage_start = coverage_end - timedelta(days=365 * 6)  # match bootstrap range

manager.ensure_vault_cache_coverage(
    vault_ensemble_dirs=vault_dirs,
    start_date=coverage_start,
    end_date=coverage_end,
    refresh_mode="missing_stale_only",
)
```

### Step 6: Build PortfolioCacheQuery, fit + predict
```python
from ensemble.portfolio import PortfolioCacheQuery
from utils.cache.central_cache_models import ArtifactScope

query = PortfolioCacheQuery(
    tickers=tuple(sorted(required_tickers)),
    start=coverage_start,
    end=coverage_end,
    timeframes=(TimeFrame.D, TimeFrame.M),
    scope=ArtifactScope.LIVE,
)

# Compute instrument returns from cached candles
daily_candles = store.query_candles(...)  # for returns calculation
instrument_returns = daily_candles.pivot_table(...).pct_change()

portfolio.fit_from_cache(query, instrument_returns)
positions_df = portfolio.predict_from_cache(query)
```

### Step 7: Position sizing + Telegram (unchanged)
Same as current: fetch ETF prices from TWS, calculate shares, format output, send notification.

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/tws_live_forecast.py` | Rewrite main() to use cache-native flow. Add auto-bootstrap, upsert, cache query logic. |
| `configs/live_forecast_config.json` | Remove `lookback_days` (dynamic now). Add optional `bootstrap_start_date`. |

## Files for Reference (read-only)

| File | Purpose |
|------|---------|
| `utils/cache/runtime/central_cache.py` | `CentralCacheStore.upsert_candles()`, `query_candles()`, `describe_candle()` |
| `utils/cache/runtime/cache_manager.py` | `bootstrap_source_candles()`, `ensure_vault_cache_coverage()` |
| `utils/cache/runtime/central_cache_models.py` | `ArtifactScope`, `PortfolioCacheQuery` model |
| `ensemble/portfolio.py` | `GlobalPortfolio.fit_from_cache()`, `predict_from_cache()` |

## What We Gain

1. **Bias node artifacts persist** across runs -- future vectorized backtests can read them directly
2. **Incremental data accumulation** -- each run only fetches new bars, upserts into cache
3. **Norgate independence** -- after subscription expires, TWS bars extend the cache
4. **Automatic staleness management** -- upsert marks artifacts stale, refresh rebuilds only what changed
5. **Clean separation** -- `data/ohlc_data/` stays immutable, cache is mutable runtime state

## Open Questions

- Should monthly candles be resampled from the full cached daily series (more accurate) or just from the TWS fetch window (simpler)?
  **Decision:** Resample from full cached daily series for accuracy.
- Should the script fail hard if EWSD volatility can't be computed, or fall back to external computation?
  **Decision:** Let `predict_from_cache()` handle it -- it raises `ArtifactMissingError` if EWSD is missing, and `ensure_bias_cache_coverage` includes EWSD as auxiliary.
