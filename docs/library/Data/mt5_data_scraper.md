# MT5 Data Scraper

> 🕒 **Timezone trap:** every timestamp this scraper writes (bars **and** ticks) is
> in **broker server time (EET/EEST), mislabelled `tz=UTC`** — *not* real UTC. The
> daily rollover sits at **00:00 broker time = 17:00 New York**. Read
> [[mt5_timezones]] before doing anything window- or session-relative.

## Overview

`data_platform/providers/mt5/scraper.py` is an incremental tick + bar scraper that attaches to the
running Darwinex MT5 terminal and writes partitioned parquet files to `data/mt5_data/`.
It is separate from the live-forecast MT5 connector (`deployment/mt5_data_connector.py`),
which only fetches single candles for signal generation.

---

## Broker: Darwinex Live

| Field           | Value                          |
|-----------------|--------------------------------|
| Account login   | `4000093084`                   |
| Server          | `liveUK-mt5.darwinex.com`      |
| Terminal build  | `5836` (as of 2026-06-04)      |
| Total symbols   | **844** (stocks, ETFs, FX, indices, commodities) |
| Terminal path   | `C:\Program Files\MetaTrader 5\terminal64.exe` |

---

## Symbol categories

| Category       | Count | Example symbols                              |
|----------------|-------|----------------------------------------------|
| US Stocks      | ~700  | AAPL, MSFT, AMZN, NVDA (NYSE / Nasdaq / DOW) |
| US ETFs        | ~30   | ARKK, DIA, EFA, INDA                         |
| Forex          | ~30   | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF       |
| Indices        | 10    | SP500, NDX, GDAXI, WS30, UK100, AUS200, NI225 |
| Commodities    | 2     | XAUUSD (gold), XTIUSD (WTI crude)            |

Notable **not found** in this terminal: NAS100/US100 (use `NDX`), GER40/DAX40 (use `GDAXI`),
BTCUSD (no crypto).

---

## Tick history depth (per broker)

Fetched via `copy_ticks_from(symbol, datetime(2000,1,1), 200_000, COPY_TICKS_ALL)`.
The 200k count cap means the oldest date shown is where that cap's window starts —
true history may extend further for symbols with fewer tick events per day.

| Instrument   | MT5 symbol | Oldest tick reachable | Newest tick    |
|--------------|------------|-----------------------|----------------|
| EURUSD       | EURUSD     | 2011-12-19            | 2026-05-29     |
| GBPUSD       | GBPUSD     | 2011-12-19            | 2026-05-29     |
| USDJPY       | USDJPY     | 2011-12-19            | 2026-05-29     |
| AUDUSD       | AUDUSD     | 2011-12-19            | 2026-05-29     |
| USDCHF       | USDCHF     | 2011-12-19            | 2026-05-29     |
| Gold         | XAUUSD     | 2018-01-25            | 2026-05-28     |
| WTI Crude    | XTIUSD     | 2020-01-02            | 2026-06-01     |
| S&P 500      | SP500      | 2023-03-27            | 2026-06-01     |
| DAX 40       | GDAXI      | 2018-01-25            | 2026-06-02     |
| Dow Jones    | WS30       | 2018-01-31            | 2026-05-29     |
| FTSE 100     | UK100      | 2018-01-25            | 2026-06-01     |
| AAPL         | AAPL       | 2020-01-02            | 2026-06-03     |
| MSFT         | MSFT       | 2018-04-12            | 2026-06-03     |
| AMZN         | AMZN       | 2018-10-02            | 2026-06-03     |
| NVDA         | NVDA       | 2018-10-02            | 2026-06-01     |
| NDX (Nasdaq) | NDX        | N/A (no tick history) | 2026-05-28     |

---

## Bar history — deep M1 IS available (corrected 2026-06-05)

An earlier version of this doc claimed bar history was shallow / required opening a
chart. That was wrong; the real picture, verified empirically:

- **Bar depth is gated by the terminal's `Max bars in chart` setting** (default
  100,000 ≈ 70 days of M1). Set it to **Unlimited** (Tools → Options → Charts) and
  **restart the terminal**, and full M1 history becomes downloadable on demand.
- **Trigger the deep download with `copy_rates_from(symbol, TF, now, big_count)`**,
  NOT `copy_rates_range` over old dates. On a cold base `copy_rates_range(old, old)`
  returns empty; `copy_rates_from` from *now* with a large count (e.g. 30M) triggers
  the progressive broker download. The first call returns the recent ~100k buffer
  instantly and starts the background download — **poll until the returned count
  stabilises** (download landed).
- Measured depth after the cap lift + restart: **GBPUSD 9.78M M1 bars back to 1993**,
  XAUUSD → 1998, NDX/AAPL → 2008. M1 reaches the **same depth as D1**.

`data_platform/providers/mt5/m1_backfill.py` implements this (poll-until-stable,
resumable, parallel, terminal-login guard). Caveat: the terminal **serialises**
deep-history downloads, so `--workers` gives little speed-up on the deep core
(~10 min/deep symbol) — a full-universe M1 backfill is a multi-night job.

> Resampling ticks → bars is still valid for sub-minute aggregation, but M1 (and
> coarser) bars are now fetched directly; no tick detour is needed for them.

---

## Storage layout

```
data/mt5_data/
  _discovery.json          ← broker capability snapshot
  _probe_results.json      ← last TF + tick depth probe output
  <SYMBOL>/
    bars_D1/
      part.parquet         ← daily bars: time(server-tz, see ⚠️ below), open, high, low, close, tick_volume, spread, real_volume
    bars_M1/
      year=YYYY/
        part.parquet       ← minute bars (same columns)
    ticks/
      year=YYYY/
        part.parquet       ← columns: time_msc(ms), bid, ask, last, volume, time(server-tz), flags
```

Files use **zstd compression** and are deduplicated on `time` (bars) / `time_msc` (ticks)
before writing. Each run appends only rows newer than the last stored timestamp.

---

## ⚠️ Timestamp timezone — stored `time` is **broker-server time, NOT UTC**

> [!warning] The `time` column is tagged `utc=True` but actually carries Darwinex
> **server time (EET/EEST = UTC+2 winter / UTC+3 summer)**. `mt5.copy_rates*` /
> `copy_ticks*` return the broker-server clock as an epoch; the scraper wraps it with
> `pd.to_datetime(..., utc=True)`, which **labels** server-local seconds as UTC without
> converting. So the stored tz tag is wrong by +2/+3h. Treating it as real UTC silently
> shifts every bar and **decorrelates daily snapshots** against true-UTC/ET sources.

**Practical conversion (verified empirically, `research/feed_comparison/feed_tz_calibrate.py`, 2026-06-05):**

| From stored `time` | To get… | Do |
|---|---|---|
| stored (EET/EEST) | true UTC | subtract 2h (Nov–Mar) / 3h (Mar–Nov) |
| stored (EET/EEST) | US Eastern (ET) | **subtract 7h, year-round** (EET−EST = ET, both shift with DST) |

So the **16:00 ET US cash close = stored time-of-day 23:00**, year-round. Snapshotting
the last M1 close at/just before **stored 23:00** maximises daily-return correlation to
the matching Norgate close:

| MT5 symbol | Norgate ref | corr at stored-23:00 snapshot | corr if mis-snapped at "16:00 UTC" |
|---|---|---|---|
| SPY (ETF CFD)   | SPY (Norgate)     | **0.995** | 0.71 (compares ETF *open* to index *close*) |
| QQQ (ETF CFD)   | NDX cash / QQQ    | **0.998 / 0.996** | 0.73 |
| SP500 (idx CFD) | SPX cash          | **0.988** | 0.72 |
| XAUUSD (cmdty)  | GLD ETF           | **0.996** | — |
| XAGUSD (cmdty)  | SLV ETF           | **0.995** | — |

How the offset shows up directly in the bars (stored-tz times):

- **ETF CFDs** (SPY/QQQ/GLD/SLV) trade stored **16:31–22:58** = US regular session
  **09:30–16:00 ET** (subtract 7h). One bar per US trading day only.
- **Index/commodity CFDs** (SP500/WS30/XAUUSD/XTIUSD) trade ~23h with a 1-hour gap at
  stored **00:00–00:59** = the **17:00–17:59 ET** CME/financing maintenance break.

When comparing MT5 bars to any other source, **always reconcile the timezone first** —
either snapshot at stored 23:00 (= 16:00 ET) or subtract the EET offset to recover true
UTC. See `research/feed_comparison/feed_compare.py` for the reference implementation.

For the **rollover-window** consequences (dead-zone at stored 00:00–01:00, where the
`rollover_tick_scraper` / `build_symbol_sessions` / `nautilus.ingest` windowing is
mis-placed by the offset, and the correct broker-time exit/entry windows), see the
dedicated page [[mt5_timezones]].

---

## Running the scraper

### Prerequisites

1. MT5 terminal must be **running and logged in** on this machine.
2. `.env` at repo root should contain `MT5_PATH` (path to `terminal64.exe`). When set,
   the scraper binds to that specific terminal — important on multi-terminal machines
   (Darwinex + FTMO demo). Without it, `mt5.initialize()` attaches to whichever
   terminal responds first (nondeterministic). `MT5_USERNAME` / `MT5_PASSWORD` /
   `MT5_SERVER` are used by the live forecast connector, not the scraper.
3. The scraper calls `mt5.initialize(path)` when `MT5_PATH` is set, else `mt5.initialize()`
   — do **not** pass login credentials to the scraper; the terminal handles authentication
   via its own session.

```powershell
# Daily (D1) bars — full history, all terminal symbols (the simplest full backfill)
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.daily_scraper
# Daily, specific symbols
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.daily_scraper --symbols EURUSD XAUUSD

# M1 bars for all symbols
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper

# Include raw ticks (large — ~200k ticks per symbol per chunk)
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks

# Bootstrap M1 from a start date / specific symbols
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --from 2020-01-01
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --symbols EURUSD XAUUSD
```

### Daily vs M1

`daily_scraper.py` is the simplest full backfill — one `copy_rates_range(TIMEFRAME_D1)`
per symbol over full history (FX reaches back to ~1971), written to
`{SYMBOL}/bars_D1/part.parquet`. No tick cap or chunking. Use it to populate daily
bars across the whole universe. `scraper.py` (M1) is for intraday work and the
scheduled task; daily bars can also be resampled from M1, but `daily_scraper.py`
gets full daily history directly and faster.

It enumerates symbols via `mt5.symbols_get()` (the terminal's real names), so it
covers every tradeable symbol without a hand-maintained list — symbol-name
mismatches (e.g. `US500.cash` vs the terminal's actual index name) are avoided.

Logs are written to `logs/mt5_scrape.log` when run via the scheduled task.

---

## Scheduled task

Registered via `deployment/ops/setup_scheduled_task.ps1` as `\TradingAlgo\MT5DataScrape`.
Fires daily at **5:00 PM PT** (8:00 PM ET) — after US equity close and CME settlement.

```powershell
# Register / re-register (run as admin once):
powershell -ExecutionPolicy Bypass -File deployment\ops\setup_scheduled_task.ps1
```

---

## Tick scraping performance — benchmarked findings

All numbers below are measured on a Darwinex Live account (build 5836, Windows 11).

### The two-tier cache — why cold vs warm matters

MT5 stores tick history in two distinct layers:

```
Tier 1 — broker server
  The canonical record. Darwinex streams ticks back to 2011 (FX) / 2018 (metals).
  Never changes. A cold fetch hits this tier.

Tier 2 — terminal local cache (AppData\Roaming\MetaQuotes\Terminal\...)
  Populated on first fetch per symbol/timerange. Lives on disk permanently.
  A warm fetch reads from here without a broker round-trip.
```

The performance difference between the two tiers is dramatic:

| Fetch type | Measured speed | What happens |
|-----------|---------------|-------------|
| **Cold** (first fetch, broker download) | **1.7 M ticks/min** | 10–18 s per chunk regardless of chunk size |
| **Warm** (local cache hit) | **500 M ticks/min** | 0.03–0.08 s per chunk |
| **Ratio** | **~300×** | Same data, 300× faster second time |

Key implication: the bootstrap download is a one-time cost. Every incremental daily run and every subsequent backtest read hits the warm cache at 500 M ticks/min.

### Broker round-trip latency is fixed per call, not per tick

This is the non-obvious fact that drives the chunking strategy:

```
1-day chunk  (~100k ticks):   5.5 s  → 1.1 M ticks/min
7-day chunk  (~625k ticks):   0.07 s → 498 M ticks/min  (cached)
7-day chunk  (~625k ticks):  11–15 s → 2.5 M ticks/min  (cold)
```

A single `copy_ticks_range` call pays one fixed broker round-trip (~12 s cold). The
number of ticks returned barely affects the wall time. Therefore **maximise ticks per
call** by using the largest window that stays under the 200k cap.

### Adaptive chunk sizing

`data_platform/providers/mt5/scraper.py` computes the optimal window per symbol:

```python
chunk_days = max(1, 180_000 // ticks_per_day)   # target ~180k ticks/call (closer to 200k cap)
```

Applied to the target symbols:

| Symbol | Ticks/day (measured) | Chunk | Cold calls for full history | Cold hours |
|--------|---------------------|-------|----------------------------|-----------|
| EURUSD | ~120,000 | 1 day | ~3,770 | ~12.6 h |
| USDJPY | ~120,000 | 1 day | ~3,770 | ~12.6 h |
| AUDNZD | ~130,000 | 1 day | ~3,770 | ~12.6 h |
| EURCHF | ~93,000 | **2 days** | ~1,885 | ~6.3 h |
| XAUUSD (2018→) | ~144,000 | 1 day | ~2,117 | ~7.1 h |
| XAGUSD (2018→) | ~158,000 | 1 day | ~2,117 | ~7.1 h |
| NDX (2018→) | ~327,000 | 1 day | ~2,117 | ~7.1 h |
| XTIUSD (2020→) | ~81,000 | **2 days** | ~807 | ~2.7 h |
| SP500 / SPY (2021→) | ~84,000 | **2 days** | ~403 | ~1.3 h |

> Raising the target from 150k → 180k gives EURCHF, XTIUSD, and SP500 larger chunks,
> cutting their call counts roughly in half. FX majors stay at 1-day chunks because
> ~120-160k ticks/day × 2 = 240-320k which would exceed the 200k cap.

If a call returns exactly 200k ticks the cap was hit — the scraper halves the
chunk size for that symbol going forward.

### Parallel scraping — `--workers N`

The scraper supports concurrent symbol fetching via `--workers N`. All threads share
the same `mt5.initialize()` IPC connection (one process — MT5 only allows one).
MT5's `copy_ticks_range` and `copy_rates_range` are thread-safe for concurrent reads.
Parquet writes are serialised per output file via an internal `threading.Lock`.

**Recommended workers for bootstrap runs: 3–5.**  
With 4 workers the effective throughput is roughly 3–4× (broker-side rate limiting
and IPC serialisation prevent linear scaling, but 3–4× is consistently observed).

```powershell
# 4 parallel symbols — recommended for overnight bootstrap
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --workers 4 --symbols EURUSD USDJPY AUDNZD EURCHF

# All 9 symbols in one run, 4 workers — ~20-30h instead of 81h
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --workers 4 `
    --symbols NDX XAUUSD XAGUSD EURUSD USDJPY AUDNZD EURCHF XTIUSD SP500
```

### Bootstrap schedule (with --workers 4)

Total estimated time: **~20–30 hours** (down from ~91h sequential).
Can be done in 3–4 overnight sessions instead of 10.

```powershell
# Night 1: all 9 symbols, 4 workers — runs ~25h, spans into next morning
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --workers 4 `
    --symbols NDX XAUUSD XAGUSD EURUSD USDJPY AUDNZD EURCHF XTIUSD SP500

# OR split across 2 nights if you want to keep nights under 8h:
# Night 1 (short histories, ~8h at 4 workers):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --workers 4 `
    --symbols SP500 XTIUSD XAGUSD XAUUSD

# Night 2 (FX + NDX, ~12h at 4 workers):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --workers 4 `
    --symbols NDX EURUSD USDJPY AUDNZD EURCHF
```

After bootstrap, daily incremental adds 1–3 chunks per symbol (<1 min total with --workers 4).

### API call that actually works for ticks

```python
# CORRECT — triggers broker download on first call, cache hit thereafter
ticks = mt5.copy_ticks_range(symbol, from_dt, to_dt, mt5.COPY_TICKS_ALL)

# ALSO WORKS for recent data already in cache
ticks = mt5.copy_ticks_from(symbol, from_dt, 200_000, mt5.COPY_TICKS_ALL)
```

Unlike bars (which need a chart open), `copy_ticks_range` fetches from the broker
directly — no GUI pre-loading required.

### Storage: compressed bytes per tick

Measured on real Darwinex tick data (8-column schema: `time_msc, bid, ask, last, volume, time, flags, volume_real`):

| Format | Bytes/tick | Notes |
|--------|-----------|-------|
| Raw numpy struct | 60 B | As returned by MT5 |
| Parquet uncompressed | ~50 B | Column encoding saves ~17% |
| Parquet zstd (level 3) | **~25 B** | **Used by scraper — ~2.4× compression** |

Total for 9 target symbols at 25 B/tick: **~80 GB** compressed. Fits on your 500 GB SSD.

---

## MT5 IPC behaviour notes

- Only **one Python process** can hold the MT5 IPC channel at a time.
- After `mt5.shutdown()` in a prior process, a new process must wait ~5 seconds
  before `mt5.initialize()` will return a connected session with data.
- The scraper and the live forecast server (`deployment/forecast_server.py`) must
  **not** run simultaneously — stagger them by at least 10 seconds.
- Never hardcode credentials in source files — use `.env` (gitignored) exclusively.

---

## Discovery and diagnostics

Probes now live under `data_platform/providers/mt5/probes/`:

```powershell
# Re-run full symbol list + tick history probe (writes _discovery.json):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.mt5_discovery

# Per-symbol TF + tick depth table (representative instruments):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.mt5_probe_tf

# Minimal connectivity sanity check:
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.mt5_minimal
```

---

## Trading hours / session schedule

> [!note] Session hours are not in the MT5 Python API — but you can **infer them from the bars**
> The `MetaTrader5` Python package (5.0.5735) does **not** expose
> `symbol_info_session_quote` / `symbol_info_session_trade` (MQL5-only). The
> `session_*` fields on `symbol_info` are intraday price/volume stats (today's
> first/last price), **not** the weekly schedule; `start_time`/`expiration_time`
> are `0` for perpetual CFDs.
>
> **The robust approach is to derive the schedule empirically from the stored M1
> bars** — within a session bars are ~1 min apart; a larger gap marks a session
> boundary (daily maintenance break, weekend, holiday). Aggregating those gaps by
> weekday gives the recurring open/close pattern, broker-quirks and all.

### Inferring sessions from M1 data (recommended)

```powershell
# After scraping M1 bars, derive the schedule for a symbol (any tz):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.infer_sessions --symbol EURUSD
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.infer_sessions --symbol AAPL --tz America/New_York
```

`infer_sessions.py` reads `data/mt5_data/{SYMBOL}/bars_M1/`, splits the timestamp
series on gaps > `--gap-minutes` (default 5), and reports the modal open/close per
weekday plus a consistency score. This is *descriptive* (what actually traded), so
it captures Darwinex-specific maintenance windows and early closes automatically —
more reliable than any static spec sheet.

### Static reference (typical Darwinex schedule)

The table below is the expected pattern as a sanity check against the inferred
output. Times are the underlying exchange's local time; the server clock is roughly
GMT+2/+3 (EET, DST-adjusted).

| Asset class | Symbols | Trading window (exchange local) | Daily break |
|---|---|---|---|
| FX majors/crosses | EURUSD, GBPUSD, USDJPY, … | Sun 17:00 ET → Fri 17:00 ET, continuous | 17:00–17:05 ET daily rollover |
| Spot metals | XAUUSD, XAGUSD | Sun 18:00 ET → Fri 17:00 ET | 17:00–18:00 ET daily |
| US index CFDs | US500.cash, US100.cash, US30/WS30 | Sun 18:00 ET → Fri 17:00 ET (nearly 24h) | 17:00–18:00 ET daily |
| EU index CFDs | GDAXI, FCHI40, STOXX50E, UK100 | ~01:00–23:00 CET (Mon–Fri) | overnight |
| Energy CFDs | XTIUSD (WTI), XNGUSD (NatGas) | Sun 18:00 ET → Fri 17:00 ET | 17:00–18:00 ET daily |
| US stocks (CFD) | AAPL, MSFT, NVDA, … (~700) | 09:30–16:00 ET, Mon–Fri (regular session) | — (closed nights/weekends) |
| US ETFs (CFD) | SPY-likes (~100) | 09:30–16:00 ET, Mon–Fri | — |

Notes:
- **Equity/ETF CFDs only trade during the US cash session** (09:30–16:00 ET) — this
  is why daily-bar scraping for stocks yields one bar per US trading day, while FX /
  index / metal CFDs run nearly 24×5.
- The daily M1-bar scrape runs at 5 PM PT (8 PM ET), after the US equity close and
  the futures/FX daily rollover, so all of that day's sessions are settled.
- Exchange holidays follow the underlying venue calendar; for US instruments this is
  the NYSE calendar already captured in `data/events/calendar/nyse_holiday_events.json`.

### What you *can* check at runtime

`symbol_info(symbol).trade_mode` reports current tradeability (not the schedule):

| `trade_mode` | Meaning |
|---|---|
| 0 | SYMBOL_TRADE_MODE_DISABLED — not tradeable |
| 1 | LONGONLY |
| 2 | SHORTONLY |
| 3 | CLOSEONLY |
| 4 | FULL — normal two-way trading |

A symbol with `trade_mode == 4` whose market is currently closed will still return
4 (mode is a capability, not a live open/closed flag). To detect "is the market
open right now", compare `symbol_info(symbol).time` (last quote epoch) against the
current time — a stale `time` means the session is closed.

> _Verified against current code via CodeGraph on 2026-06-07._
