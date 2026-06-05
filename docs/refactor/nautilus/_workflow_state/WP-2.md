# WP-2 — Data Layer → Nautilus ParquetDataCatalog · Workflow Ledger

**Parity expectation:** EXACT (rtol=1e-8). Additive only (new `data_platform/nautilus/` +
flag + equality test). Legacy loaders untouched (deletion = WP-5).
**Baseline commit:** `5044624` (both parity gates green: `pytest tests/parity` = 2 passed).
**Prereq:** `nautilus_trader==1.227.0` installed (was missing; installed this session).
**Mode:** autonomous loop (user away ~1h).

## Acceptance gate (from 02_data_layer.md)
- [ ] `tests/data_platform/test_candles_equivalence.py`: legacy vs Nautilus candles equal for all WP-1 fixture windows.
- [ ] WP-1 parity (`pytest tests/parity`) green with Nautilus adapter as default source.
- [ ] Cache populate/read still works.  ← Unit 2 (blocked on fork)
- [x] **Instrument round-trip:** 67 rows → Nautilus instruments; counts/ids reconcile. ✓ `e662cfe` (6 tests pass)
- [x] NDX 2026 MT5 1-min + tick ingest smoke (Bar ts_init=close + QuoteTick). ✓ `e662cfe`

## Phase 1 — DISCOVERY (done; 3 explorers, read-only)

**Candle contract (byte-exact target):** `load_data` → index `timestamp` (int64 unix sec),
cols `datetime`(datetime64[ns] **tz-naive**, a date-label = midnight for daily), `open/high/low/close`
(**float64**, upcast from on-disk float32), `volume`(int64). `load_data_multi_ticker` → RangeIndex,
cols `timestamp, datetime, open, high, low, close, volume, ticker(Ticker enum obj), timeframe(TimeFrame enum obj)`.
**Insertion boundary (no FROZEN edits):** swap `data_platform.loaders.load_data`/`load_data_multi_ticker`
behind a flag — both the feature path (`features/extraction/feature_extractor.py`, FROZEN) and the
cache path (`lib/cache/runtime/cross_ticker_store.py` → `helpers.load_data` → `CentralCacheStore`)
funnel through these. `ensemble/` consumes candles from the cache, never the loader directly.

**Cache:** `BiasNodeCache` / `CentralCacheStore` key = `(family,ticker,timeframe,module_name,params,scope)` —
**no data-source path in the key**; caches bias-node *outputs*. ⇒ identical candle values ⇒ identical cache.

**Instruments:** catalog BUILT on disk (`data/instruments/catalog.{parquet,json}`) — **67 instruments**:
FUTURE 23, SPOT 33 (EQUITY incl TLT) , CFD 7, WARRANT 4 (and SPOT+FX 7 via MT5). Map by
`instrument_class` (+`asset_class` for SPOT): FUTURE→FuturesContract, SPOT+EQUITY→Equity,
SPOT+FX→CurrencyPair, CFD→Cfd, WARRANT→Equity/skip. `build_*` fns encode precision/increment/multiplier/venue.

**Parity fixtures the equality test must cover:** feature_research = `SI/D`, 2000-01-01→2022-12-31;
portfolio_research = `ES,NQ,GC,CL` (+TLT cross-peer) `D/W/M`, 2000-01-01→2026-05-13.
**Gate command:** `.\.venv\Scripts\python.exe -m pytest tests/parity` (rtol=1e-8). NOTE: `tests/parity/README.md`
says snapshots "not captured" — STALE; they were captured at `5044624` (2 passed). (README needs a Phase-B fix.)

## ⚠ ESCALATION — architecture fork (bars/candle adapter, Unit 2) — needs user decision

Nautilus `Bar` stores `Price` **quantized to the instrument's `price_precision`**. Legacy candles are
**raw float64** and back-adjusted futures prices are **not tick-aligned**. Consequences:
- A native-`Bar` round-trip at the *real* tick precision (e.g. ES=2) breaks parity (rtol ~6e-7 ≫ 1e-8).
- Storing at high precision (≤9) to preserve floats conflicts with the real tick precision that
  execution (WP-3/WP-4) requires — one instrument can't have both.
**Options (pick before Unit 2):** (A) relax the equality test to rtol=1e-8 to match the parity gate +
store research bars at high precision in a **research-only** instrument set; (B) store raw float64 OHLC as
a **custom Nautilus data type** (exact bytes) for research, native `Bar`s at tick precision for execution;
(C) keep the legacy loader as the research candle source; use the Nautilus catalog only for execution.
RECOMMEND (B) — exact research parity + native execution bars, cleanest separation. **Held for ratification.**

## Phase 2 — IMPLEMENT (additive, serial)
- **Unit 1 (SAFE — building now):** `data_platform/nautilus/{__init__,catalog,instruments}.py` +
  `tests/data_platform/test_nautilus_instruments.py` (round-trip: 67 rows → Nautilus instruments →
  catalog → read back → counts/ids reconcile). No parity exposure; uses real tick precisions.
- **Unit 2 (bars ingest + candle adapter + flag + equality test):** BLOCKED on the fork above. Do NOT build blindly.

## Phase log
| ts | phase | unit | status | parity before/after | frozen-touched? | evidence | notes |
|----|-------|------|--------|---------------------|-----------------|----------|-------|
| init | 0 | setup | done | 2 passed / — | no | commit 5044624 | ledger created |
| p1 | 1 | discovery | done | — | no | 3 explorer manifests | stale paths resolved; fork found |
| p2 | 2 | unit1 instruments + 1b intraday | done | 2 passed/2 passed | no | `e662cfe`, 6 tests | data_platform/nautilus/{catalog,instruments,ingest}.py |
| -- | 2 | unit2 candle adapter | BLOCKED | — | — | — | awaiting user precision-fork decision (A/B/C) |
