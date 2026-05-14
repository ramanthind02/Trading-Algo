# Norgate Canonical Candle Rebuild

## Overview

This repository treats Norgate continuous futures as the canonical historical source for daily/weekly/monthly futures candles.

Canonical flow:

1. **Raw ingestion** (source-faithful)
   - `data/norgate/continuous_futures/adjusted/`
   - `data/norgate/continuous_futures/unadjusted/`
2. **Normalized repository candles**
   - `data/ohlc_data/{TICKER}/D_{TICKER}.parquet`
   - `data/ohlc_data/{TICKER}/W_{TICKER}.parquet`
   - `data/ohlc_data/{TICKER}/M_{TICKER}.parquet`
3. **Runtime canonical cache**
   - `.cache/trading_algo/central_cache/`

## Prerequisites

- Norgate Data Updater running on Windows
- Active futures subscription
- `norgatedata` installed in project venv

## One-command rebuild

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_norgate_canonical_store.py
```

This command:

- clears runtime cache
- clears old Norgate raw snapshots
- fetches full-history adjusted + unadjusted Norgate continuous futures
- rebuilds repository D/W/M candles from adjusted data
- bootstraps central cache from rebuilt repository candles

## Manual sequence

```powershell
.\.venv\Scripts\python.exe scripts\fetch_norgate_data.py
.\.venv\Scripts\python.exe scripts\migrate_norgate_to_ohlc.py
.\.venv\Scripts\python.exe -m utils.cache.runtime.bootstrap_source_candles --reset-existing
```

## Full-history fetch behavior

`scripts/fetch_norgate_data.py` uses symbol-specific start detection via `norgatedata.first_quoted_date(...)` and retries transient failures.

## Source priority

Source priority is configuration-driven via:

- `deployment/config/canonical_source_priority.json`
- `utils/data/source_reconciliation.py`

Default futures daily priority:

1. `norgate`
2. `ib` fallback for live gaps

## IB append reconciliation policy

IB daily updates are reconciled before writing to central cache:

- append-only sessions beyond current cache max date
- mandatory ratio alignment at the splice (`anchor_close / first_new_ib_close`)
- monthly regenerated from reconciled daily frame

See:

- `utils/cache/runtime/ib_candle_ratio_align.py`
- `scripts/enigma_live_forecast.py`

## Validation checklist

After rebuild:

1. spot-check candle date ranges in `data/ohlc_data`
2. run a forecast dry-run (`scripts/enigma_personal_forecast.py --dry-run`)
3. verify central cache coverage via forecast logs

## SaaS parity

The SaaS implementation should use the same canonical architecture and reconciliation policy. See `docs/SaaS/data_source.md`.
