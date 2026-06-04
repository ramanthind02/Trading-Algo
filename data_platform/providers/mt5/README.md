# MT5 (Darwinex) adapter

Incremental MT5 data scraper for the Darwinex terminal. Writes partitioned parquet to
`data/mt5_data/`. CFD data is a **separate parallel store** — it is not merged into the
futures `data/ohlc_data/` (see [[multi_source_update_architecture]] §11.4).

## Layout

```
data_platform/providers/mt5/
  scraper.py          incremental M1 bar + tick scraper (the scheduled-task job)
  daily_scraper.py    full-history daily (D1) bars for the whole symbol universe
  probes/             dev/diagnostic scripts
    verify_daily_coverage.py  check D1 scrape coverage vs broker depth probe
    infer_sessions.py         derive session hours empirically from M1 gaps
    mt5_discovery.py  mt5_probe_daily.py  mt5_probe_tf.py  mt5_minimal.py
    mt5_attach_test.py  mt5_tick_depth_check.py  mt5_tick_sizing.py
    mt5_storage_estimate.py  mt5_time_estimate.py
```

`daily_scraper.py` writes `{SYMBOL}/bars_D1/part.parquet` (one `copy_rates_range`
per symbol over full history; enumerates via `mt5.symbols_get()` so it uses the
terminal's real symbol names). `infer_sessions.py` derives trading-session hours
empirically from stored M1 bars (the MT5 Python API has no session-hours function).

The live-cache sync (`sync_mt5_dailies_into_central_cache`) is a **separate** concern that
stays in `scripts/mt5_data_fetch.py` — it is part of the live forecast path
(`scripts/enigma_live_forecast.py`), not the bulk scraper.

## Running

```powershell
# Daily incremental (M1 bars, all symbols) — what the scheduled task runs
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper

# Tick bootstrap for specific symbols
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --symbols EURUSD XAUUSD

# Full-history D1 bars for the whole symbol universe (one-shot or catch-up)
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.daily_scraper

# Verify D1 coverage after scrape completes
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.verify_daily_coverage

# Discovery / diagnostics
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.probes.mt5_discovery
```

## Scheduled task

The Windows task `\TradingAlgo\MT5DataScrape` runs `deploy/run_mt5_scrape.bat` daily at
5 PM PT. The `.bat` now invokes `python -m data_platform.providers.mt5.scraper`.

**No task re-registration needed** — the task targets the `.bat` file path (unchanged);
only the module the `.bat` calls changed. The task keeps working as-is.

## Constraints

- MT5 terminal must be running and logged in. Only one Python process can hold the MT5 IPC
  channel — stagger the scraper, live forecast, and probes (≥10s apart).
- Use `copy_rates_range` (not `copy_rates_from`) — only that call triggers a broker download.

See the full operational reference: [[mt5_data_scraper]].
