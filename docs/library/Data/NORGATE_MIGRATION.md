# Norgate Data Migration

## Overview

The `data/ohlc_data/` directory contains OHLC futures data used by the entire pipeline. Originally this was Kibot data (not properly back-adjusted for contract rolls). Norgate provides professionally back-adjusted continuous futures data.

The migration replaces Kibot data with Norgate for all 23 available tickers, using a **hybrid splice**: Kibot data before Norgate's start date (~2005) is preserved, and Norgate replaces everything from 2005 onward.

**Kibot-only tickers** (no Norgate available): `NG`, `TLT` — these are left untouched.

## Quick Reference (all commands)

For devs who already have Norgate Data Updater installed and running:

```bash
# 1. Fetch latest Norgate data
python scripts/fetch_norgate_data.py

# 2. Migrate (backs up Kibot, splices Norgate, generates W/M)
python scripts/migrate_norgate_to_ohlc.py

# 3. Clear cache
rm -f utils/cache/*.parquet
rm -rf utils/cache/atr/ utils/cache/rsi/

# 4. Validate
python scripts/validate_norgate_migration.py
python -c "from utils.core.helpers import load_data; from utils.core.enums import *; df = load_data(Ticker.ES, TimeFrame.D); print(df.shape)"
pytest tests/ -q

# 5. Visual comparison (optional)
cd frontend && python app.py   # open http://localhost:5001, use "Compare Kibot" button
```

## Starting from Kibot Data Only (New Dev Setup)

If you only have the Kibot data zip and need to set up Norgate from scratch:

### 0a. Extract Kibot Data

Unzip `data/kibot_data.zip` into `data/ohlc_data/`. Each ticker should have its own directory:

```
data/ohlc_data/
  ES/
    D_ES.parquet
    W_ES.parquet
    M_ES.parquet
  NQ/
    ...
```

### 0b. Install Norgate Data Updater (Windows only)

Norgate Data requires their desktop application running as a local data server. It only runs on Windows.

1. Purchase a subscription at [norgatedata.com](https://norgatedata.com/) — you need the **Futures** data package
2. Download and install **Norgate Data Updater** from your account dashboard
3. Launch the application and log in with your credentials
4. Run a full data update (first sync takes ~10-15 minutes)
5. Verify the app is running — it must stay open in the system tray while fetching data

### 0c. Install Python Package

```bash
pip install norgatedata
```

Verify the connection works:

```python
python -c "import norgatedata; print('Connected:', norgatedata.status())"
```

This must print `Connected: True`. If it prints `False`, the Norgate Data Updater is not running.

### 0d. Understand What Changes

- **23 tickers** get Norgate back-adjusted data from 2005 onward, with Kibot data preserved before 2005
- **2 tickers** (NG, TLT) are untouched — Norgate doesn't carry these
- Weekly/Monthly candles are regenerated from daily (not sourced separately)
- The pipeline (`load_data()`, bias nodes, etc.) works unchanged — same file format

## Prerequisites

1. **Norgate Data Updater** must be installed and running on the machine (see 0b above)
2. **norgatedata** Python package: `pip install norgatedata`
3. An active Norgate subscription with futures data access
4. Kibot data already in `data/ohlc_data/` (see 0a above)

## Step-by-Step: Full Migration

### 1. Fetch Norgate Data

This downloads daily back-adjusted continuous futures for all 23 mapped tickers:

```bash
python scripts/fetch_norgate_data.py
```

Output goes to:
- `data/norgate/continuous_futures/adjusted/` — back-adjusted (CCB) parquets
- `data/norgate/continuous_futures/unadjusted/` — raw (for roll detection)

### 2. Run Migration

This backs up existing data, then replaces D/W/M parquets with hybrid-spliced Norgate data:

```bash
python scripts/migrate_norgate_to_ohlc.py
```

What it does:
- Backs up `data/ohlc_data/` to `data/ohlc_data_kibot_backup/` (skips if backup exists)
- For each of the 23 Norgate tickers:
  - Loads Norgate daily data and converts to Kibot schema
  - Loads Kibot backup daily data and keeps rows before Norgate start date
  - Splices: Kibot pre-2005 + Norgate 2005+
  - Generates Weekly (W-SUN resample) and Monthly (ME resample) from the spliced daily
  - Writes D/W/M parquets to `data/ohlc_data/{TICKER}/`
- Skips NG and TLT (Kibot-only)

### 3. Clear Cache

Cached bias node outputs depend on price data, so they must be cleared:

```bash
rm -f utils/cache/*.parquet
rm -rf utils/cache/atr/ utils/cache/rsi/
```

### 4. Validate

#### Programmatic Validation

```bash
python scripts/validate_norgate_migration.py
```

Generates per-ticker reports in `docs/library/Data/comparisons/`:
- `stats.json` — return correlation, mean/max return difference
- `price_overlay.html` — dual-axis price comparison
- `return_scatter.html` — return scatter plot with R^2
- `divergence_timeline.html` — Kibot/Norgate price ratio over time

#### Quick Smoke Test

```bash
python -c "from utils.core.helpers import load_data; from utils.core.enums import *; df = load_data(Ticker.ES, TimeFrame.D); print(df.shape, df.columns.tolist())"
```

#### Frontend Visual Comparison

```bash
cd frontend && python app.py
```

Open http://localhost:5001, then:
1. Select a ticker, choose D/W/M timeframe
2. Click **"Compare Kibot"** button — overlays original Kibot close prices (orange line)
3. Both series share the same price scale for accurate visual comparison
4. Check that prices track closely in recent years, diverge further back (expected due to different adjustment methods)

#### Test Suite

```bash
pytest tests/ -q
```

### 5. Re-running Migration

To re-run from scratch (e.g., after updating Norgate data):

1. Delete the backup to force a fresh backup: `rm -rf data/ohlc_data_kibot_backup/`
2. Re-fetch Norgate data: `python scripts/fetch_norgate_data.py`
3. Re-run migration: `python scripts/migrate_norgate_to_ohlc.py`
4. Clear cache and re-validate

Or to re-migrate without losing the original Kibot backup, just re-run step 2 — the backup step is skipped if it already exists.

## Ticker Mapping

The mapping from Kibot ticker symbols to Norgate back-adjusted symbols is defined in `scripts/fetch_norgate_data.py`:

| Kibot | Norgate | Category |
|-------|---------|----------|
| ES | &ES_CCB | Equity Index |
| NQ | &NQ_CCB | Equity Index |
| YM | &YM_CCB | Equity Index |
| RTY | &RTY_CCB | Equity Index |
| CL | &CL_CCB | Energy |
| HO | &HO_CCB | Energy |
| GC | &GC_CCB | Metal |
| HG | &HG_CCB | Metal |
| SI | &SI_CCB | Metal |
| PL | &PL_CCB | Metal |
| EU | &6E_CCB | FX |
| JY | &6J_CCB | FX |
| BP | &6B_CCB | FX |
| CD | &6C_CCB | FX |
| SF | &6S_CCB | FX |
| C | &ZC_CCB | Agricultural |
| S | &ZS_CCB | Agricultural |
| W | &ZW_CCB | Agricultural |
| GF | &GF_CCB | Agricultural |
| TY | &ZN_CCB | Fixed Income |
| FV | &ZF_CCB | Fixed Income |
| US | &ZB_CCB | Fixed Income |
| TU | &ZT_CCB | Fixed Income |

## File Format

All parquets in `data/ohlc_data/` follow this schema (consumed by `utils/core/helpers.py:load_data()`):

| Column | Type | Description |
|--------|------|-------------|
| datetime | string (YYYY-MM-DD) | Date |
| open | float64 | Open price |
| high | float64 | High price |
| low | float64 | Low price |
| close | float64 | Close price |
| volume | int64 | Volume |
| timestamp | int64 | Unix seconds (UTC midnight) |

Written with `engine='fastparquet'`.
