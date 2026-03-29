# Cache-Native TWS Live Forecast Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the TWS live forecast script to use the central cache as single source of truth -- bootstrap from repo parquets, upsert TWS bars, refresh bias artifacts, fit+predict via cache-native APIs.

**Architecture:** The script auto-bootstraps `data/ohlc_data/` into `.cache/central_cache/` on first run, then each subsequent run upserts fresh TWS bars via `upsert_candles()`, refreshes stale bias artifacts via `ensure_vault_cache_coverage()`, and runs `GlobalPortfolio.fit_from_cache()` / `predict_from_cache()` with a `PortfolioCacheQuery`.

**Tech Stack:** Python, ibapi, pandas, CentralCacheStore, CacheManager, GlobalPortfolio

**Design doc:** `docs/plans/2026-03-28-tws-cache-native-live-forecast-design.md`

---

### Task 1: Add `ensure_cache_ready()` helper

**Files:**
- Modify: `scripts/tws_live_forecast.py`

This function checks if the central cache has candles for all required tickers. If not, bootstraps from `data/ohlc_data/`. Returns a dict of coverage info.

**Step 1: Write the function**

Add after the existing `discover_required_tickers()` function:

```python
def ensure_cache_ready(required_tickers: Set[str]) -> Dict[str, Any]:
    """Bootstrap central cache from repo parquets if not already populated.

    Returns dict with 'bootstrapped' bool and coverage info per ticker.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.cache_manager import CacheManager

    store = CentralCacheStore.get_instance()
    manager = CacheManager()
    coverage_info = {}
    needs_bootstrap = []

    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            needs_bootstrap.append(ticker_enum)
        else:
            coverage_info[ticker_str] = {
                "start": record.coverage.start,
                "end": record.coverage.end,
                "revision": record.revision,
            }

    bootstrapped = False
    if needs_bootstrap:
        print(f"  Bootstrapping {len(needs_bootstrap)} ticker(s) from repo parquets...")
        result = manager.bootstrap_source_candles(
            tickers=[Ticker[t] for t in sorted(required_tickers)],
            timeframes=[TimeFrame.D, TimeFrame.M],
        )
        bootstrapped = True
        print(f"  Bootstrap: {result['success']} success, {result['failed']} failed")
        # Refresh coverage info after bootstrap
        for ticker_str in sorted(required_tickers):
            try:
                ticker_enum = Ticker[ticker_str]
            except KeyError:
                continue
            record = store.describe_candle(ticker_enum, TimeFrame.D)
            if record:
                coverage_info[ticker_str] = {
                    "start": record.coverage.start,
                    "end": record.coverage.end,
                    "revision": record.revision,
                }

    return {"bootstrapped": bootstrapped, "coverage": coverage_info}
```

**Step 2: Verify import works**

Run: `python -c "from scripts.tws_live_forecast import ensure_cache_ready; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/tws_live_forecast.py
git commit -m "feat: add ensure_cache_ready() helper for auto-bootstrap"
```

---

### Task 2: Add `upsert_tws_candles()` helper

**Files:**
- Modify: `scripts/tws_live_forecast.py`

This function takes the fetched TWS candle DataFrames and upserts them into the central cache, plus resamples daily to monthly.

**Step 1: Write the function**

Add after `ensure_cache_ready()`:

```python
def upsert_tws_candles(
    daily_candles: pd.DataFrame,
    required_tickers: Set[str],
) -> None:
    """Upsert fetched TWS daily bars into central cache and resample to monthly.

    Parameters
    ----------
    daily_candles : pd.DataFrame
        All fetched daily candles with 'ticker' column.
    required_tickers : Set[str]
        Ticker names to upsert.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()

    # Upsert daily candles per ticker
    for ticker_str in sorted(required_tickers):
        ticker_mask = daily_candles["ticker"].astype(str) == ticker_str
        ticker_candles = daily_candles.loc[ticker_mask].copy()
        if ticker_candles.empty:
            continue
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        store.upsert_candles(ticker_enum, TimeFrame.D, ticker_candles)
        print(f"    {ticker_str} D: upserted {len(ticker_candles)} bars")

    # Resample full cached daily series to monthly and upsert
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            continue
        full_daily = store.query_candles(
            ticker_enum, TimeFrame.D,
            start=record.coverage.start,
            end=record.coverage.end,
        ).reset_index()
        if full_daily.empty:
            continue
        full_daily["ticker"] = ticker_str
        monthly = resample_daily_to_monthly(full_daily)
        if not monthly.empty:
            store.upsert_candles(ticker_enum, TimeFrame.M, monthly)
```

**Step 2: Verify import works**

Run: `python -c "from scripts.tws_live_forecast import upsert_tws_candles; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/tws_live_forecast.py
git commit -m "feat: add upsert_tws_candles() for incremental cache ingest"
```

---

### Task 3: Add `refresh_bias_caches()` helper

**Files:**
- Modify: `scripts/tws_live_forecast.py`

This function calls `ensure_vault_cache_coverage()` to rebuild stale bias node artifacts after candle upsert.

**Step 1: Write the function**

```python
def refresh_bias_caches(
    vault_root: str,
    required_tickers: Set[str],
) -> Dict[str, Any]:
    """Refresh stale bias node artifacts for all vault ensembles.

    Returns the summary dict from ensure_vault_cache_coverage.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.cache_manager import CacheManager

    store = CentralCacheStore.get_instance()
    manager = CacheManager()

    # Determine coverage window from cached candles
    earliest_start = None
    latest_end = None
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            continue
        if record.coverage.start and (earliest_start is None or record.coverage.start < earliest_start):
            earliest_start = record.coverage.start
        if record.coverage.end and (latest_end is None or record.coverage.end > latest_end):
            latest_end = record.coverage.end

    if earliest_start is None or latest_end is None:
        raise ValueError("No candle coverage found in cache. Run bootstrap first.")

    # Collect vault ensemble directories
    vault_dirs = []
    for tf_name in ["D", "M"]:
        tf_dir = Path(vault_root) / tf_name
        if not tf_dir.exists():
            continue
        for ens_dir in sorted(tf_dir.iterdir()):
            if ens_dir.is_dir():
                vault_dirs.append(str(ens_dir))

    if not vault_dirs:
        raise ValueError(f"No vault ensembles found in {vault_root}")

    print(f"  Refreshing bias caches for {len(vault_dirs)} ensembles...")
    print(f"  Coverage window: {earliest_start.date()} to {latest_end.date()}")

    summary = manager.ensure_vault_cache_coverage(
        vault_ensemble_dirs=vault_dirs,
        start_date=earliest_start,
        end_date=latest_end,
        refresh_mode="missing_stale_only",
    )

    rebuilt = summary.get("rebuilt", 0)
    failed = summary.get("failed", 0)
    validated = summary.get("validated", 0)
    print(f"  Result: {rebuilt} rebuilt, {validated} already fresh, {failed} failed")

    if failed > 0:
        for d in summary.get("details", []):
            if d.get("status") == "failed":
                print(f"    FAILED: {d.get('module_name')}/{d.get('ticker')}: {d.get('message','')}")

    return summary
```

**Step 2: Verify import works**

Run: `python -c "from scripts.tws_live_forecast import refresh_bias_caches; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/tws_live_forecast.py
git commit -m "feat: add refresh_bias_caches() for stale artifact rebuild"
```

---

### Task 4: Add `build_cache_query()` helper

**Files:**
- Modify: `scripts/tws_live_forecast.py`

Builds a `PortfolioCacheQuery` from the current cache coverage, and computes `instrument_returns` from cached daily candles.

**Step 1: Write the function**

```python
def build_cache_query(
    required_tickers: Set[str],
) -> tuple:
    """Build PortfolioCacheQuery and instrument_returns from cached candles.

    Returns (query, instrument_returns) tuple.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.central_cache_models import ArtifactScope
    from ensemble.portfolio import PortfolioCacheQuery

    store = CentralCacheStore.get_instance()

    # Determine shared coverage window across all tickers
    starts = []
    ends = []
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record and record.coverage.start and record.coverage.end:
            starts.append(record.coverage.start)
            ends.append(record.coverage.end)

    if not starts:
        raise ValueError("No candle coverage in cache")

    # Use the intersection of all coverage windows
    query_start = max(starts)
    query_end = min(ends)

    query = PortfolioCacheQuery(
        tickers=tuple(sorted(required_tickers)),
        start=query_start,
        end=query_end,
        timeframes=(TimeFrame.D, TimeFrame.M),
        scope=ArtifactScope.LIVE,
    )

    # Compute instrument returns from cached daily candles
    frames = []
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        candles = store.query_candles(
            ticker_enum, TimeFrame.D,
            start=query_start, end=query_end,
        ).reset_index()
        if not candles.empty:
            candles = candles.set_index("datetime")["close"].rename(ticker_str)
            frames.append(candles)

    if not frames:
        raise ValueError("No daily candle data in cache for returns computation")

    prices = pd.concat(frames, axis=1).sort_index()
    instrument_returns = prices.pct_change(fill_method=None).dropna(how="all")

    return query, instrument_returns
```

**Step 2: Verify import works**

Run: `python -c "from scripts.tws_live_forecast import build_cache_query; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/tws_live_forecast.py
git commit -m "feat: add build_cache_query() for cache-native portfolio API"
```

---

### Task 5: Rewrite `main()` to use cache-native flow

**Files:**
- Modify: `scripts/tws_live_forecast.py` (the `main()` function)

Replace the current data flow (steps 3-8) with the new cache-native pipeline. Keep the TWS connection, argument parsing, portfolio building, position sizing, and Telegram output unchanged.

**Step 1: Rewrite the main() function body**

The new flow inside the `try` block after TWS connection:

```
1. Fetch TWS daily bars for all required tickers (existing fetch logic)
2. ensure_cache_ready(required_tickers)  -- auto-bootstrap if needed
3. upsert_tws_candles(daily_candles, required_tickers)  -- persist to cache
4. Populate cross-ticker store from cache (for bias nodes that need it)
5. refresh_bias_caches(vault_root, required_tickers)  -- rebuild stale artifacts
6. query, instrument_returns = build_cache_query(required_tickers)
7. portfolio.fit_from_cache(query, instrument_returns)
8. positions_df = portfolio.predict_from_cache(query)
9. Fetch ETF prices, calculate shares, Telegram (unchanged)
```

Key changes to `main()`:
- Remove `candles_per_tf` dict construction
- Remove `compute_daily_ewsd_volatility()` call (EWSD now comes from cache)
- Remove `portfolio.fit(candles_per_tf, ...)` / `portfolio.predict(candles_per_tf, ...)`
- Replace with `fit_from_cache(query, instrument_returns)` / `predict_from_cache(query)`
- Remove the `from utils.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility` import

Also update the dynamic lookback: instead of always fetching `config["data"]["lookback_days"]`, check the cache coverage and only fetch enough to fill the gap.

**Step 2: Remove stale config fields**

Modify `configs/live_forecast_config.json`:
- Remove `data.lookback_days` (now dynamic based on cache gap)
- Remove `data.bar_size` (always "1 day")
- Add `data.min_lookback_days: 10` (minimum fetch window)
- Add `data.max_lookback_days: 365` (fallback when cache is empty)

**Step 3: Update imports at top of file**

Remove:
```python
from utils.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility
```

Add:
```python
from utils.cache.runtime.central_cache import CentralCacheStore
from utils.cache.cache_manager import CacheManager
from utils.cache.central_cache_models import ArtifactScope
from ensemble.portfolio import PortfolioCacheQuery
```

**Step 4: Run the full script in dry-run mode**

Run: `python scripts/tws_live_forecast.py --dry-run --port 7497`
Expected: Full pipeline completes with forecasts for all tickers. Output should show bootstrap (first run) or upsert (subsequent runs), bias cache refresh, and cache-native fit/predict.

**Step 5: Run existing unit tests**

Run: `python -m pytest tests/unit-tests/deployment/test_tws_live_forecast.py -v`
Expected: All 7 tests pass (they test contract creation, position sizing, and Telegram formatting -- none depend on the data flow).

**Step 6: Commit**

```bash
git add scripts/tws_live_forecast.py configs/live_forecast_config.json
git commit -m "feat: rewrite main() to use cache-native fit/predict pipeline"
```

---

### Task 6: Smart lookback -- only fetch what the cache is missing

**Files:**
- Modify: `scripts/tws_live_forecast.py`

Instead of fetching a full year from TWS every run, calculate the gap between the cache's last date and today, and only fetch that many days (plus a small buffer).

**Step 1: Add `compute_fetch_lookback()` helper**

```python
def compute_fetch_lookback(
    ticker_str: str,
    max_lookback: int = 365,
    min_lookback: int = 10,
    buffer_days: int = 5,
) -> int:
    """Determine how many days to fetch from TWS based on cache gap.

    If cache has recent data, only fetch the gap. If cache is empty or stale,
    fetch max_lookback.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()
    try:
        ticker_enum = Ticker[ticker_str]
    except KeyError:
        return max_lookback

    record = store.describe_candle(ticker_enum, TimeFrame.D)
    if record is None or record.coverage.end is None:
        return max_lookback

    gap_days = (datetime.now() - record.coverage.end).days + buffer_days
    return max(min(gap_days, max_lookback), min_lookback)
```

**Step 2: Update the TWS fetch loop in main() to use it**

Replace the fixed `lookback_days` with per-ticker dynamic lookback:

```python
for ticker in sorted(required_tickers):
    if ticker not in instruments:
        continue
    lookback = compute_fetch_lookback(
        ticker,
        max_lookback=config["data"].get("max_lookback_days", 365),
        min_lookback=config["data"].get("min_lookback_days", 10),
    )
    candles = fetch_historical_candles(
        client, ticker, instruments[ticker], lookback_days=lookback,
    )
    ...
```

**Step 3: Run dry-run and verify smaller fetch windows on second run**

Run twice:
1. First run: should bootstrap + fetch 365 days
2. Second run: should show small lookback (10-15 days) since cache is fresh

**Step 4: Commit**

```bash
git add scripts/tws_live_forecast.py
git commit -m "feat: smart lookback -- only fetch TWS bars missing from cache"
```

---

### Summary of new data flow

```
main()
  ├─ build_portfolio(config)              # Load vault ensembles
  ├─ discover_required_tickers(portfolio)  # ES, NQ, RTY, YM, GC, TLT
  ├─ ensure_cache_ready(tickers)           # Bootstrap from data/ohlc_data/ if empty
  ├─ connect TWS, fetch bars              # Smart lookback per ticker
  ├─ upsert_tws_candles(daily, tickers)   # Persist to cache + resample monthly
  ├─ populate cross-ticker store           # For rebalancing bias nodes
  ├─ refresh_bias_caches(vault, tickers)   # Rebuild stale artifacts
  ├─ build_cache_query(tickers)            # PortfolioCacheQuery + instrument_returns
  ├─ portfolio.fit_from_cache(query, returns)
  ├─ portfolio.predict_from_cache(query)
  ├─ fetch ETF prices, calculate shares
  └─ Telegram notification
```
