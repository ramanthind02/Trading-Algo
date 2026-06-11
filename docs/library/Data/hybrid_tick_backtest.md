# Hybrid bars-and-ticks backtesting

> 🕒 **Read [[mt5_timezones]] first.** All `data/mt5_data/**` timestamps (bars and
> ticks) are **broker EET time mislabelled as UTC**; the rollover is at **00:00
> broker time**, and execution windows must be placed in broker wall-clock.

How we get realistic execution cost in backtests **without** storing the full
tick history of the whole universe. The rule of thumb:

> **Backtest on bars; drop to ticks only inside an order's lifetime; cache every
> tick window you touch so the second run is free.**

---

## Why hybrid

Tick data (bid/ask on every quote) is the only thing that contains the spread —
M1 bars are mid/bid OHLC and have none. But you only *need* the spread at the
moments an order is live (placement → fill/cancel). Everywhere else, bars are
enough to drive the signal and mark the book.

So the universe splits cleanly:

| Layer | Resolution | Covers | Purpose |
|-------|-----------|--------|---------|
| **Spine** | D1 (daily models) / M1 (intraday) | full backtest period, all traded instruments | signals, order schedule, coarse marking |
| **Lens** | bid/ask ticks | only `[order placed → filled/cancelled]` windows | real spread at the fill, real "did my limit get hit" |

Storing the lens for the whole universe across all history would be ~80 GB+ and
mostly wasted (you'd archive the spread of 692 stock CFDs you never trade). The
lens is fetched **on demand** and **cached**, so it only ever holds the windows a
backtest actually reached.

---

## The keystone: `ensure_ticks`

`data_platform/providers/mt5/tick_cache.py`

```python
from data_platform.providers.mt5.tick_cache import ensure_ticks

df = ensure_ticks("EURUSD", start, end)   # tz-aware UTC window
```

- Returns bid/ask ticks for `[start, end)` as a DataFrame.
- Fetches from MT5 `copy_ticks_range` **only** the sub-ranges not already cached.
- Caches each fetched gap as an **immutable append-only chunk** —
  `data/mt5_data/{SYM}/ticks_cache/{start_ns}-{end_ns}.parquet`.
- A fully-cached window needs **no MT5 connection** → warm backtests run offline.

### Coverage tracking

A per-symbol JSON manifest records which UTC ranges have been *requested from
MT5* (so a legitimately-empty closed-market window is never re-fetched):

```
data/mt5_data/{SYM}/_ticks_coverage.json   # merged [start,end] intervals
```

On each call, `missing_ranges(start, end, covered)` computes exactly the gaps to
fetch; everything already covered is read straight from the chunk store. The
interval math (`merge_intervals`, `missing_ranges`) is pure and unit-tested in
`tests/data_platform/test_tick_cache_coverage.py`.

### Why append-only chunks (not the bulk `ticks/` store)

The bulk scraper's `ticks/year=YYYY/part.parquet` files can hold tens of millions
of rows. Appending a 2-hour window by read-merge-rewrite is O(file) memory and
will OOM. Because coverage guarantees fetched ranges are **disjoint**, the cache
never needs to merge: each gap is one new file, named by its `start_ns-end_ns`
range, and reads only open the chunk files whose name-range overlaps the request.
Writes are O(chunk); reads are O(touched-chunks).

### Failure semantics

- `symbol_select` fails (terminal down / unknown symbol) → **raise**, range stays
  uncovered and is retried next call. A failed fetch must never poison the cache.
- `copy_ticks_range` returns empty but `symbol_select` succeeded → genuine
  closed-market window; marked covered so it isn't re-fetched.
- Coverage is saved **per gap as it succeeds**, so a mid-batch failure keeps prior
  progress.

---

## The Nautilus bridge

`data_platform/nautilus/ingest.py` → `ingest_mt5_quotes_window(symbol, catalog, start, end)`

Calls `ensure_ticks`, builds `QuoteTick` records (`_build_quotes`), writes them to
the `ParquetDataCatalog`. This is the **on-demand counterpart** to the existing
`ingest_mt5_intraday_windowed` (which filters a *pre-scraped* full-tick store) —
here nothing is pre-scraped; the catalog receives exactly the execution windows
the backtest touches.

Nautilus then does the realism for free: feed it **bars** for the full period and
**quote ticks** only in the execution windows, and its matching engine fills
resting limits / market crosses against the real bid/ask whenever quotes are
present, falling back to bar marking elsewhere. No custom slippage model needed —
the spread comes from the data.

```
Catalog for a backtest =
    M1 bars      (full period)          ← ingest_mt5_intraday / research candles
  + QuoteTicks   (execution windows)    ← ingest_mt5_quotes_window  (on demand)
```

---

## How each model uses it

**Daily models (50+ instruments).** Execution times are deterministic (session
close/open/rollover). Enumerate the windows up front and `ensure_ticks` each one
— no discovery pass needed. A few years of windows is plenty to characterise
cost; you are not optimising the signal on ticks, only deducting realistic
spread.

**Intraday models (<6 instruments).** Order times are not known a priori, so use
two passes: (1) run on M1 only, record each order's active window; (2)
`ensure_ticks` those windows and re-run with tick-accurate fills. With so few
symbols the tick footprint stays tiny.

---

## What MT5 vs Nautilus provide

| | MT5 | Nautilus |
|---|---|---|
| Bars | `copy_rates_range(sym, TF, …)` | replays them as the spine |
| Ticks | `copy_ticks_range(sym, …, COPY_TICKS_ALL)` (cold ~12 s/chunk, warm after) | consumes `QuoteTick` for realistic fills |
| Store | `data/mt5_data/` parquet | `ParquetDataCatalog` (`data/nautilus_catalog/`) |
| Live | 100 ms poll (`fetch_tick`), no true stream | `subscribe_quote_ticks` (true push when live on IB) |

**Live execution** stays on-demand (`fetch_tick` right before placing an order) —
no storage at all. **Backtesting** persists once via `ensure_ticks` because MT5's
cold fetch is ~12 s/chunk and re-streaming every run would be unworkable; the
warm cache then serves every later run at parquet speed.

---

## What this replaced

The earlier plan was an overnight full-universe (844-symbol), full-day,
full-history tick scrape (~80 GB, multiple nights). It was dropped: it archived
spreads for instruments that would never be traded. The demand-driven cache
gives the same backtest realism while only ever holding the windows a backtest
actually reaches.

> The bulk scraper (`scraper.py --ticks`) and the rollover scraper still exist for
> the cases where you deliberately want a full or windowed historical tick archive
> for a small, chosen instrument set. `ensure_ticks` is the default path for
> backtest cost modelling.

---

## Files

| File | Role |
|------|------|
| `data_platform/providers/mt5/tick_cache.py` | `ensure_ticks` keystone + coverage math |
| `data_platform/nautilus/ingest.py` | `ingest_mt5_quotes_window` catalog bridge |
| `tests/data_platform/test_tick_cache_coverage.py` | pure interval-math unit tests |
| `data_platform/providers/mt5/scraper.py` | bulk M1 + full-tick scraper (cold side reused) |
| `research/portfolio/pnl/nautilus_engine.py` | backtest engine that ingests on demand |

> _Verified against current code via CodeGraph on 2026-06-07._
