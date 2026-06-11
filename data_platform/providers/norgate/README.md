# Norgate Data Provider

Self-contained adapter for fetching, archiving, and migrating every Norgate
database we subscribe to: futures (continuous + individual contracts + specs),
US stocks (survivorship-bias-free), and flat market series (indices, cash
commodities, forex spot).

## Requirements

- Active Norgate subscription (futures coverage)
- Norgate Data Updater (NDU) running locally on Windows
- `norgatedata` package (in repo venv)

NDU health check: `norgatedata.status()` must return `True` before any fetch.

---

## Directory layout

```
data_platform/providers/norgate/
  _constants.py      ticker maps, Norgate symbols, parquet compression settings
  _paths.py          all path functions (single source of truth for data layout)
  fetch_continuous.py  fetch CCB + unadjusted continuous series
  fetch_contracts.py   archive adjusted continuous + all individual expiries
  fetch_specs.py       fetch + persist contract metadata (point values, tick sizes, etc.)
  stocks.py            survivorship-bias-free US stock scraper (TR/CAP/UNADJ + membership)
  market_series.py     indices / cash commodities / forex spot (flat single-series DBs)
  migrate.py           build ohlc_data from working continuous files
  rebuild.py           end-to-end orchestrator
  README.md            this file

data/norgate/
  working/
    continuous/
      adjusted/      {TICKER}.parquet   CCB series — ephemeral, rebuilt on each full run
      unadjusted/    {TICKER}.parquet   raw continuous — ephemeral
  archive/
    continuous/      {TICKER}.parquet   CCB series — permanent, never purged
    contracts/
      {TICKER}/      {NORGATE_SYMBOL}.parquet   per-expiry OHLCV — permanent, incremental
    contract_specs.json     contract metadata for all 23 tickers — permanent
    contract_specs.parquet  same, columnar form for DataFrame access

data/norgate/market_series/      flat single-series DBs (no adjustment/expiries)
  us_indices/        {SAFE_SYMBOL}.parquet   ~1,615 ($SPX, sector/industry indices)
  world_indices/     {SAFE_SYMBOL}.parquet   ~31    ($DAX, $N225, $FT100)
  cash_commodities/  {SAFE_SYMBOL}.parquet   ~100   ($BCOM family, spot ratios)
  forex_spot/        {SAFE_SYMBOL}.parquet   ~57    (EURUSD, $USDX, XAUUSD)

data/stock_data/                 survivorship-bias-free US equities (separate store)
  {BUCKET}/{SAFE_SYMBOL}/
    {D,W,M}_{TR,CAP,UNADJ}_{SYMBOL}.parquet   3 adjustments x 3 timeframes
    membership_{SYMBOL}.parquet               per-index 0/1 constituent history

data/ohlc_data/
  {TICKER}/
    D_{TICKER}.parquet          daily, back-adjusted
    W_{TICKER}.parquet          weekly (W-SUN)
    M_{TICKER}.parquet          monthly (ME)
    D_{TICKER}_unadj.parquet    daily, unadjusted (used as σ denominator in EWSD)
```

### Market series schema

`us_indices` / `world_indices` carry `Volume` (int64) + `Turnover` (float64)
where Norgate provides them; `cash_commodities` / `forex_spot` are OHLC-only.
The original Norgate symbol (`$SPX`, `$BCOMAG`) is stamped into parquet metadata
under `norgate_raw_symbol` since the filename is the safe-rendered form (`_SPX`).

### Working vs archive vs ohlc_data

| Store | Purged on rebuild? | Purpose |
|---|---|---|
| `working/continuous/` | Yes | Intermediate fetch output; input to `migrate.py` |
| `archive/continuous/` | Never | Permanent CCB reference; overwrites on each archive run |
| `archive/contracts/` | Never | Permanent per-expiry OHLCV; incremental (skips cached) |
| `data/ohlc_data/` | No | Canonical D/W/M store consumed by pipeline and EWSD |

---

## Parquet schema

All files written by this adapter use the same canonical schema:

| Column | Type | Notes |
|---|---|---|
| `date` | `date32` (index) | Parquet native DATE; no time component |
| `open` | `float32` | |
| `high` | `float32` | |
| `low` | `float32` | |
| `close` | `float32` | |
| `volume` | `int32` | |

**Compression:** zstd level 3 (~27% smaller than snappy, negligible read overhead).

`load_data()` in `data_platform/loaders.py` handles both this schema and the legacy
schema (`datetime` string column, `float64` OHLC, `int64` volume + timestamp).

---

## Usage

### Full rebuild (normal cadence — while Norgate subscription is active)

```powershell
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.rebuild
```

This runs all five steps: purge working dirs → fetch continuous → archive → migrate → bootstrap cache.

#### Faster dev rebuild (skip archive + cache bootstrap)

```powershell
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.rebuild --skip-archive --skip-cache
```

### Individual steps

```powershell
# Fetch continuous series only (adjusted + unadjusted) into working dirs
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.fetch_continuous

# Update permanent archive (adjusted continuous + all contract expiries)
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.fetch_contracts
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.fetch_contracts --continuous-only
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.fetch_contracts --contracts-only

# Fetch and persist contract specs (point values, tick sizes, margins, exchanges)
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.fetch_specs

# Migrate working dirs to ohlc_data (all tickers or subset)
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.migrate
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.migrate --tickers ES NQ TY
```

### US stocks (survivorship-bias-free; separate store)

```powershell
# Full universe: 14,223 active + 20,988 delisted. Parallel for the big run.
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.stocks --universe both --workers 8

# A single index's members only
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.stocks --watchlist "S&P 500"
```

Incremental: a symbol whose `D_TR` file exists is skipped. Each worker holds its
own NDU connection (`norgatedata` is multiprocessing-safe per the package docs);
writes target distinct per-symbol files so there is no contention.

### Market series (indices / cash commodities / forex spot)

```powershell
# All four flat databases (~1,803 series)
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.market_series

# One category
.\.venv\Scripts\python.exe -m data_platform.providers.norgate.market_series --category forex_spot
```

### Programmatic use

```python
from data_platform.providers.norgate import rebuild, migrate_all, fetch_continuous
from data_platform.providers.norgate import load_contract_specs, TICKER_TO_CCB, archive_continuous_dir

# Full rebuild
rebuild()

# Migrate only
migrate_all(tickers=["ES", "NQ"])

# Load contract specs (works without Norgate running)
specs = load_contract_specs()
es = specs[specs.ticker == "ES"].iloc[0]
print(es.point_value)   # 50.0
print(es.tick_value)    # 12.5

# Paths
print(archive_continuous_dir())   # data/norgate/archive/continuous/
```

---

## Back-adjustment and the σ fix

Norgate continuous futures use **additive back-adjustment** (`_CCB`): each roll
gap is subtracted from all prior prices so that point moves (dollar P&L) are
preserved, but price *levels* are inflated relative to real prices.

This distorts **percentage returns** used in volatility estimation:

```
r_adj = ΔP / P_adj = r_true · k,   k = P_true/P_adj < 1
σ_adj = k · σ_true                 (e.g. k ≈ 0.69 for ES)
```

The fix (implemented in `nodes/volatility/ewsd/ewsd.py`): use the **unadjusted
close** as the denominator so `r = ΔP_adj / P_unadj` reflects the true percentage
move. This requires `D_{TICKER}_unadj.parquet`, which `migrate.py` writes from the
unadjusted working continuous series.

**Effect on ES:** median annualized σ 8.9% → 12.8% (ratio 1.44×). Without this
fix, a 15% vol target runs ~21% hot in live trading.

Research Sharpe is unbiased (the k factor cancels in `position_fraction × log_return`);
only live position sizing is affected.

---

## After the Norgate subscription ends

The `archive/` directories are the permanent reference:

- `archive/continuous/{TICKER}.parquet` — full-history CCB series for every
  ticker. Use to reconstruct ohlc_data without re-fetching.
- `archive/contracts/{TICKER}/{SYM}.parquet` — every individual expiry ever
  available. Contains raw unadjusted prices for roll metadata, custom
  ratio-adjustment, and spread analysis.

To migrate from archive after subscription ends:

```python
# Point working dirs at the archive (read-only, no Norgate needed)
import shutil
from data_platform.providers.norgate._paths import archive_continuous_dir, working_adjusted_dir

working_adjusted_dir().mkdir(parents=True, exist_ok=True)
for src in archive_continuous_dir().glob("*.parquet"):
    shutil.copy2(src, working_adjusted_dir() / src.name)

from data_platform.providers.norgate import migrate_all
migrate_all()
```

IB CONTFUT appends (via `cache/runtime/ib_candle_ratio_align.py`) then
extend ohlc_data forward with ratio-splice continuity.

---

## Extending to new tickers

1. Add entries to `_constants.py`:
   - `TICKER_TO_CCB`: `"XX": "&XX_CCB"`
   - `TICKER_TO_RAW`: auto-derived
   - `TICKER_TO_CONTRACT_PREFIX`: `"XX": "XX"` (Norgate Futures DB prefix)
2. Add the ticker to `lib/core/enums.py` `Ticker` enum.
3. Run `rebuild.py` or the individual fetch/migrate steps.

> _Verified against current code via CodeGraph on 2026-06-07._
