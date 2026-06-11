# MT5 (Darwinex) adapter

Incremental MT5 data scraper for the Darwinex terminal. Writes partitioned parquet to
`data/mt5_data/`. CFD data is a **separate parallel store** — it is not merged into the
futures `data/ohlc_data/` (see [[futures_research_data]] §11.4).

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

The Windows task `\TradingAlgo\MT5DataScrape` runs `deployment/ops/run_mt5_scrape.bat` daily at
5 PM PT. The `.bat` now invokes `python -m data_platform.providers.mt5.scraper`.

**No task re-registration needed** — the task targets the `.bat` file path (unchanged);
only the module the `.bat` calls changed. The task keeps working as-is.

## Broker-aware symbols + per-broker data namespace (WP-4)

We trade **multiple MT5 brokers**, each exposing the same instrument under a
**different native symbol** (e.g. Nasdaq-100 is `NDX` on Darwinex but may be
`US100` / `NAS100` on FTMO; S&P-500 is `SP500` on Darwinex but may be `US500`
on FTMO). FTMO is a different broker from Darwinex, so its prices/history also
differ and must not collide on disk.

**Symbol resolution** — `brokers.py` is the single source of truth, backed by
`configs/mt5_brokers.yaml`:

```python
from data_platform.providers.mt5 import brokers
brokers.resolve("darwinex", "NQ")          # -> "NDX"   (canonical -> broker symbol)
brokers.canonical_for("darwinex", "SP500")  # -> "ES"    (broker symbol -> canonical)
brokers.resolve("ftmo", "NQ")               # raises: FTMO symbols are UNKNOWN placeholders
```

Darwinex is seeded from the confirmed mappings (mirrors
`norgate/to_catalog._MT5_SYMBOL` and `mt5/to_catalog._DARWINEX`). FTMO entries
are `UNKNOWN` placeholders until confirmed by `brokers.discover_symbols('ftmo')`
against a **separate FTMO terminal** (the stub enumerates `mt5.symbols_get()`
only when that separate terminal is connected — it does NOT run now). `resolve`
raises on `UNKNOWN` so we never silently trade a guessed symbol.

**Per-broker data namespace** — going forward, MT5 data is broker-scoped:

```
data/mt5_data/{broker}/{SYMBOL}/bars_M1/year=YYYY/part.parquet
data/mt5_data/{broker}/{SYMBOL}/ticks/year=YYYY/part.parquet
```

Path helpers live in `brokers.py`:

```python
brokers.broker_data_dir("darwinex", "NDX")  # data/mt5_data/darwinex/NDX
brokers.broker_data_dir("ftmo", "US100")    # data/mt5_data/ftmo/US100
brokers.legacy_broker_data_dir("NDX")       # data/mt5_data/NDX (pre-namespace, Darwinex-implicit)
```

**Backward-compat / migration (non-destructive).** The existing flat layout
`data/mt5_data/{SYMBOL}/` is treated as belonging to `darwinex` (the implicit
broker so far). Existing data is **NOT moved or renamed** by this change. To
migrate it into the broker-scoped layout later, move each `data/mt5_data/<SYM>/`
under `data/mt5_data/darwinex/<SYM>/` (a separate, explicit, opt-in step). The
scraper continues to write the legacy flat path until it is updated to the
broker-scoped helper; both paths are readable in the meantime.

## Constraints

- MT5 terminal must be running and logged in. Only one Python process can hold the MT5 IPC
  channel — stagger the scraper, live forecast, and probes (≥10s apart).
- Use `copy_rates_range` (not `copy_rates_from`) — only that call triggers a broker download.

See the full operational reference: [[mt5_data_scraper]].

> _Verified against the working tree on 2026-06-10._
