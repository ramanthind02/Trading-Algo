# data_platform

The repo's consolidated data stack: instrument model, provider adapters, event
data, loaders, and the multi-source reconciliation layer. Everything data-related
lives here so there is one obvious home for it, shaped for a future NautilusTrader
migration.

## Layout

```
data_platform/
  core/                 Nautilus-aligned model (see core/README.md)
    identifiers.py        Venue, Symbol, InstrumentId  ({symbol}.{venue})
    enums.py              AssetClass, InstrumentClass, PriceType, BarAggregation,
                          PriceAdjustment, venue_mic_for_exchange
    instruments.py        Instrument (frozen, all asset classes)
    bar_type.py           BarType / BarSpecification (Nautilus string form)
    catalog.py            InstrumentCatalog -> data/instruments/catalog.parquet
    symbol_map.py         source_symbol() / instrument_from_source_symbol()
    source_priority.py    SourcePriorityConfig (config-driven source routing)
    bar_record.py         CanonicalBarRecord (silver-layer schema)
    reconciler.py         SourcePriorityReconciler (wraps the IB ratio-splice)
    provenance.py         provenance + conflict persistence

  providers/            source adapters (one folder per source)
    norgate/              futures + US stocks + contract archive + specs
      backadjust/           roll detection + additive back-adjustment
    yahoo/                TLT ETF scraper
    mt5/                  Darwinex CFD scraper + probes

  events/               market calendar + economic releases (see events/README.md)
    calendar_loader.py, nyse_holidays.py, fed_fomc.py, trading_day_index.py
    econ_releases.py      ~148 Norgate Economic series

  loaders.py            load_data / load_data_multi_ticker / load_stock_data
```

## Data on disk

```
data/
  ohlc_data/{TICKER}/            futures + TLT canonical D/W/M (+ unadj)
  stock_data/{BUCKET}/{SYMBOL}/  Norgate US stocks (3 adjustments + membership)
  norgate/working|archive/       Norgate working dirs + permanent archive
  norgate/market_series/{cat}/   indices / cash commodities / forex spot (~1,803)
  mt5_data/{SYMBOL}/
    bars_M1/year=YYYY/part.parquet   incremental M1 bars (scheduled task)
    bars_D1/part.parquet             full-history daily bars (daily_scraper)
  ib/contracts/{TICKER}/         per-expiry daily OHLCV from TWS API (~2yr forward)
  events/calendar|econ/          holiday/FOMC JSON + econ series parquet
  instruments/catalog.parquet    InstrumentCatalog (69 instruments seeded)
  provenance/{INSTRUMENT}/       per-bar source provenance + conflict logs
  raw_data/                      staging CSVs (TLT Yahoo + prop-firm ES.txt)
```

## Reading data

```python
from data_platform.loaders import load_data, load_stock_data
from data_platform.core import load_catalog, source_symbol
from data_platform.events import load_calendar_bundle, load_econ_series
```

Back-compat: `lib.core.helpers` exposes `load_data` and `load_data_multi_ticker` via a lazy
module `__getattr__` (PEP 562), so existing imports keep working without an eager data-platform
import.

## Layer boundary (Nautilus mapping)

`data_platform` is the **model + ingestion** layer (Nautilus `model` crate + the
bronze→silver→gold pipeline). The **runtime/engine** layer
(`cache/runtime/` — central cache, IB ratio-splice) maps to Nautilus's
DataEngine/Cache and stays where it is; the reconciler *calls* it. Live trading
(`scripts/enigma_live_forecast.py`) is untouched.

## Metadata & registry

Two companion packages were added alongside the providers:

- **`data_platform/storage/`** — PyArrow schema contracts and single-writer helpers (bronze → silver
  schema enforcement at ingest time).
- **`data_platform/registry/`** — `data/registry.db`: a SQLite metadata index (instruments,
  source symbols, blob manifest, lineage FKs, live-trading accounts + command queue). Price rows
  never live in the DB — it is a rebuildable index on top of the parquet stores. CLI:
  `python -m data_platform.registry [rebuild|backup|ingest|accounts|coverage|freshness|runs|lineage|fills|report|costs]`.
  Full operational reference: [docs/library/Data/data_platform_migration_plan.md](../docs/library/Data/data_platform_migration_plan.md).

## Docs

- [core/README.md](core/README.md) — instrument model + Nautilus migration mapping
- [providers/norgate/README.md](providers/norgate/README.md) — Norgate adapter
- [providers/mt5/README.md](providers/mt5/README.md) — MT5 scraper
- [events/README.md](events/README.md) — calendar + econ
- [docs/library/Data/futures_research_data.md](../docs/library/Data/futures_research_data.md) — which series to use when (three-series rule, additive distortion, ratio fix, Norgate→IB handover)
- [docs/library/Data/data_platform_migration_plan.md](../docs/library/Data/data_platform_migration_plan.md) — full migration plan (storage, registry, live ingest)

> _Verified against the working tree on 2026-06-10._
