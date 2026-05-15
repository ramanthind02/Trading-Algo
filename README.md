## Data Setup - Canonical Norgate Store

Trading-Algo uses a canonical candle stack:

1. **Ingestion (raw, immutable)**
   - `data/norgate/continuous_futures/adjusted/`
   - `data/norgate/continuous_futures/unadjusted/`
2. **Normalization (canonical schema)**
   - `scripts/migrate_norgate_to_ohlc.py` builds `data/ohlc_data/{TICKER}/D|W|M_*.parquet`
3. **Runtime canonical cache**
   - `.cache/trading_algo/central_cache/` (queried by live/research pipeline)

The runtime/live layer should never read vendor-specific raw formats directly.

### Prerequisites

- Windows host with Norgate Data Updater running
- Active Norgate futures subscription
- Python package: `norgatedata`

### One-command canonical rebuild

From repo root:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_norgate_canonical_store.py
```

This will:

- purge runtime cache and prior Norgate snapshots
- fetch full-history Norgate continuous futures (adjusted + unadjusted)
- rebuild repository candles (`data/ohlc_data`) from adjusted series
- bootstrap central cache from rebuilt repository candles

### Manual steps

```powershell
.\.venv\Scripts\python.exe scripts\fetch_norgate_data.py
.\.venv\Scripts\python.exe scripts\migrate_norgate_to_ohlc.py
.\.venv\Scripts\python.exe -m utils.cache.runtime.bootstrap_source_candles --reset-existing
```

### IBKR append policy (live)

When appending IBKR daily bars to central cache:

- append-only (new sessions only)
- apply **ratio adjustment** at the junction to align incoming OHLC to the current canonical level
- resample monthly candles from reconciled daily output

See:

- `utils/cache/runtime/ib_candle_ratio_align.py`
- `scripts/enigma_live_forecast.py`
- `deployment/config/canonical_source_priority.json`
