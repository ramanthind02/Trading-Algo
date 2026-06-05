# WP-2 — Data Layer → Nautilus ParquetDataCatalog

> **Depends on:** WP-1 (parity harness green on `main`).
> **Parity expectation:** EXACT. This WP only changes how candles are *produced*;
> the candle values handed to the alpha core must be byte-identical, so the WP-1
> snapshots must reproduce with `rtol=1e-8`.

## Objective

Make Nautilus the canonical data store: load instruments + bars from a
`ParquetDataCatalog`, and expose them to the existing pipeline through a thin adapter
that yields the **same candle DataFrame shape** consumers already expect. Then cull
the redundant homegrown loaders (guarded — see [05_cull_and_dry_ledger.md](05_cull_and_dry_ledger.md)).

This is the cleanest cull target: `data_platform/core/catalog.py` was **explicitly
built for this migration** — its docstring says each row feeds
`ParquetDataCatalog.write_data([...instruments...])` and it already models
Nautilus-shaped `Instrument` / `InstrumentId` / `InstrumentClass` / `AssetClass`.

## The candle contract (must not change)

Consumers expect a DataFrame with columns:
`datetime, open, high, low, close, volume, ticker, timeframe`
(datetime is bar time; primary key `(datetime, ticker)`). Confirmed consumers:

- `feature_extraction/feature_extractor.py` via `helpers.load_data_multi_ticker`
  (used by `extract_features`, `extract_features_with_forward_returns`).
- `ensemble/portfolio_impl/global_portfolio_impl.py` via `_query_candles_from_cache`
  / `PortfolioCacheQuery`.
- `ensemble/portfolio_impl/portfolio_tester.py` (`calculate_*_from_candles`).

**The adapter's job:** return exactly this shape from the Nautilus catalog so nothing
downstream notices the source changed.

## Current-state map (confirm with codegraph)

Homegrown data layer (candidate for replace/cull):

- `data_platform/loaders.py` — `load_data`, `load_stock_data`, `_normalize_loaded_frame`,
  `ohlc_data_dir` (reads `data/ohlc_data/{TICKER}/`, `data/stock_data/{SYMBOL}/`).
- `data_platform/core/catalog.py` — `InstrumentCatalog`, `load_catalog`,
  `catalog_parquet_path`, `_row_to_instrument` (already Nautilus-shaped).
- `data_platform/core/source_priority.py` — `load` (source priority resolver,
  `configs/source_priority.yaml`).
- `data_platform/core/{enums,identifiers,instruments}.py` — homegrown domain types
  that mirror Nautilus.
- `data_platform/providers/norgate/*` — `market_series.py`, `stocks_catalog.py`,
  `to_catalog.py`, `fetch_continuous.py`, `_paths.py`, `build_*_instruments`.
- `data_platform/providers/mt5/*` — MT5 bar/tick capture (scraper, probes).
- Cache layer: `utils/cache/runtime/bias_node_cache.py::BiasNodeCache`,
  `utils/cache_manager.py` (`CacheManager.populate_cache`).
- Helper entrypoint: `utils/.../helpers.load_data_multi_ticker` (confirm exact module
  via codegraph; it is the single funnel for candle loads in feature extraction).

## Target design

```
data_platform/
  nautilus/                         # NEW — the Nautilus-backed data layer
    catalog.py                      # open/locate the ParquetDataCatalog
    instruments.py                  # InstrumentCatalog rows  -> nautilus Instrument
                                    #   (FuturesContract/Equity/Cfd/CurrencyPair) by
                                    #   instrument_class; reuse existing build_* logic
    ingest.py                       # provider parquet (norgate/mt5) -> Bar via
                                    #   BarDataWrangler (ts_init = close), write_data()
    candles.py                      # adapter: catalog Bars -> the candle DataFrame
                                    #   contract above (datetime,open,...,ticker,timeframe)
```

Key mechanics:

- **Instruments:** map each `InstrumentCatalog` row to the matching Nautilus instrument
  type keyed by `instrument_class`. The existing `build_stock_instrument`,
  `build_futures_instruments`, `build_mt5_instruments`, `build_tlt_instrument` already
  encode precision/increment/multiplier/venue — reuse that knowledge; emit Nautilus
  instruments and `catalog.write_data(instruments)`.
- **Bars:** wrangle provider OHLCV parquet into Nautilus `Bar`s with `BarType`
  `{INSTRUMENT}-{step}-{DAY|MINUTE}-{LAST|MID}-EXTERNAL`. **Set `ts_init` to the bar
  close** (use `BarDataWrangler` `ts_init_delta` if your source timestamps are at the
  open). This matters for WP-3/WP-4 even though WP-2 itself reads bars back to a
  close-indexed DataFrame.
- **Read-back adapter (`candles.py`):** query the catalog for a (instruments, bar_type,
  start, end) window and reshape to the candle contract. The `datetime` returned must
  equal what `data_platform/loaders.py` returns today (verify the open-vs-close
  timestamp convention against the current loader and match it exactly — this is the
  #1 parity risk).
- **Cache compatibility:** `BiasNodeCache` / `CacheManager.populate_cache` must keep
  working — they cache *bias-node outputs*, not raw candles, so they should be
  unaffected as long as candles are identical. Verify the cache keying does not embed a
  data-source path that the swap would change.

## Migration strategy (incremental, reversible)

1. **Build the catalog alongside the existing store** (no deletion yet). Add a
   `data_platform/nautilus/ingest.py` run that writes instruments + bars into a
   `ParquetDataCatalog` from the same provider parquet the loaders use today.
2. **Add `candles.py` adapter** and a feature flag / settings switch that routes
   `load_data_multi_ticker` (and the portfolio cache candle query) through either the
   legacy loader or the Nautilus adapter. Default: legacy.
3. **Equality test:** for every (ticker, timeframe, window) in the WP-1 fixtures,
   assert `legacy_candles.equals(nautilus_candles)` (after column/dtype normalization).
   This is a new test under `tests/data_platform/` — it is the WP-2 gate.
4. **Flip the default** to the Nautilus adapter once equality holds and WP-1 parity is
   green with the flag on.
5. **Cull** the legacy loaders/catalog/source-priority per WP-5, *only after* steps 3–4
   pass and remain green for the agreed bake-in period.

## Acceptance criteria (gate)

- New `tests/data_platform/test_candles_equivalence.py`: legacy vs Nautilus candle
  DataFrames are equal for all WP-1 fixture windows (exact, modulo dtype normalization).
- WP-1 parity harness passes with the Nautilus adapter as the default source
  (`rtol=1e-8`).
- Cache populate/read still works (`CacheManager.populate_cache` on a fixture ticker).
- Instrument round-trip: every `InstrumentCatalog` row materializes a valid Nautilus
  instrument; counts and ids reconcile.
- Intraday ingest smoke test: the **NDX 2026 MT5 1-min + tick** data ingests into the
  catalog as `Bar` (1-min, `ts_init=close`) and `QuoteTick` (bid/ask), reads back
  correctly, and is available for the WP-3 realism lane (see
  [03_backtest_validation_lane.md](03_backtest_validation_lane.md) "Reference test dataset").

## Risks / notes

- **Timestamp convention (open vs close).** The current loaders index bars at a
  specific time; Nautilus wants `ts_init=close` for execution. Keep the *read-back*
  DataFrame's `datetime` identical to legacy for parity, while storing `ts_init=close`
  in the catalog for WP-3/WP-4. Document both explicitly.
- **Float dtypes.** Providers store float32 (see `market_series.py`); ensure the adapter
  doesn't silently upcast/downcast in a way that perturbs returns. Match legacy dtype.
- **Back-adjustment / continuous futures.** Norgate continuous + unadjusted series and
  the EWSD σ handling (see repo memory `project_data_infra_norgate_migration`) must map
  to the right Nautilus series. Do not change adjustment math — only the storage/IO.
  Nautilus has a **native continuous-future engine** (`BACKWARD/FORWARD_SPREAD/RATIO`
  with caller-supplied roll transitions) — see
  [06_nautilus_feature_leverage.md](06_nautilus_feature_leverage.md) §A. Treat it as
  **ADOPT (parity-gated)**: the safe default is to keep your existing back-adjusted series
  as stored bars (exact parity); only use the Nautilus engine for futures if its adjusted
  output matches your series within tolerance, or for the live node's roll handling.
- **Calendar/holidays.** If legacy loaders apply any session/holiday filtering, the
  adapter must reproduce it (or the upstream ingest must), or candle rows will differ.

## Subagent instructions

- `codegraph_explore "load_data load_stock_data ohlc_data_dir load_data_multi_ticker _query_candles_from_cache InstrumentCatalog"` to pin the exact funnel and call sites.
- Treat the alpha core (§3 in [00_overview.md](00_overview.md)) as read-only.
- Do **not** delete legacy code in this WP — only add the Nautilus path + flag + equality
  test. Deletion is WP-5, guarded.
- Hand back: the catalog ingest, the adapter, the equality test results, and confirmation
  that WP-1 parity is green with the flag flipped on.
