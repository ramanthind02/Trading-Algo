# IB (Interactive Brokers) adapter

Implements the **individual-contract archive** (futures expiries via the TWS API).
Live trading + the daily CONTFUT append still live in the live-trading and
cache-runtime layers (high blast radius; migrated later — multi-source Phase 4).

## Individual-contract archive

`contracts.py` enumerates each futures root's expiries via `reqContractDetails`
(`includeExpired=True`) and stores daily bars per expiry:

```
data/ib/contracts/{TICKER}/{LOCAL_SYMBOL}.parquet   e.g. ES/ESM4.parquet
```

Schema matches the Norgate archive (date32 index, float32 OHLC, int32 volume,
zstd-3); IB metadata (`local_symbol`, `last_trade`, `con_id`, `multiplier`) is
stamped into each parquet's file metadata. Incremental — skips existing files.

```powershell
# Requires TWS / IB Gateway on 127.0.0.1:7497 with the API enabled
.\.venv\Scripts\python.exe -m data_platform.providers.ib.contracts --tickers ES NQ GC
.\.venv\Scripts\python.exe -m data_platform.providers.ib.contracts   # all catalog futures
```

The IB root + exchange per ticker come from the InstrumentCatalog
`source_symbols` (`ib_contfut`, `ib_exchange`), seeded by `to_catalog`.

**Coverage note:** IB retains only ~2 years of *expired* futures (e.g. 29 ES
expiries from 2024 forward), so this is the **forward** archive that extends the
deep-history Norgate archive (`data/norgate/archive/contracts/`). Together they
form a continuous individual-contract record across the Norgate→IB handover.
Expired contracts return thin tails near expiry (an IB data limitation); the
active/recent contracts have full daily history.

## Still elsewhere (live path, not moved)

## Where IB code lives today

| Concern | Location | Notes |
|---|---|---|
| Live forecast + IB data fetch | `scripts/enigma_live_forecast.py` | `upsert_tws_candles`, TWS historical bar fetch |
| IB→cache junction-ratio splice | `utils/cache/runtime/ib_candle_ratio_align.py` | `prepare_ib_rows_for_central_cache_append` (append-only + ratio) |
| IB demo / contract construction | `scripts/demo_ib_data_fetch.py` | `ContractSpec`, `create_contract` (supports STK, FUT, CONTFUT) |
| MT5 live-cache sync (sibling) | `scripts/mt5_data_fetch.py` | `sync_mt5_dailies_into_central_cache` (stays — live path) |

## What will land here next (per the multi-source plan)

See [[multi_source_update_architecture]]. When the Norgate→IB handover is implemented:

- `daily_update.py` — `scripts/ib_daily_update.py` equivalent: fetch IB CONTFUT on a
  daily cadence, append-only, via the reconciler.
- The IB splice (`ib_candle_ratio_align.py`) is the *runtime* reconciliation primitive;
  it stays in `utils/cache/runtime/` (the engine layer, mapping to Nautilus DataEngine).
  This adapter calls it; it does not own it.
- The individual-contract archive here already provides the raw per-expiry data for the
  ratio-adjusted σ-denominator (Option C in the multi-source doc) going forward, paired
  with the deep-history Norgate archive.

Live trading code (order execution, signal generation) stays in `scripts/` — out of scope
for the data layer; this adapter is data ingestion only.
