# data_platform/core — Nautilus-aligned instrument model

`data_platform/core` is the repo's domain model for instruments and bar identity. It
mirrors [NautilusTrader](https://nautilustrader.io)'s `model` crate semantics
without depending on `nautilus_trader`, so that a future migration to Nautilus
is a mechanical mapping rather than a rewrite.

It is the **single source of truth** for: what an instrument is, how it is
identified, its precision/increments/multiplier, and how a bar series is keyed.
Every data adapter (Norgate, IB, MT5) targets this model.

---

## What this is (and is not)

NautilusTrader separates the **domain model** from the **runtime engines**
(DataEngine, ExecutionEngine, RiskEngine, Cache, MessageBus). See
[[architecture]] notes in `docs/`.

`data_platform/core` corresponds to the **model layer only**:

| Nautilus layer | This repo today | Built here? |
|---|---|---|
| `model` (InstrumentId, Instrument, BarType) | `data_platform/core/` | ✅ yes |
| `persistence` (ParquetDataCatalog) | `data/` parquet + `InstrumentCatalog` | partial — our catalog is metadata-only |
| `data` engine (DataEngine, routing) | `utils/cache/runtime/` | not migrated |
| `execution`, `risk`, `portfolio` | `execution/`, `ensemble/` | not migrated |

We are **not** building DataEngine / MessageBus / Cache here — Nautilus provides
those when we migrate. We build the model + catalog so our data is already
shaped correctly for that day.

---

## Components

```
data_platform/core/
  identifiers.py    Venue, Symbol, InstrumentId  ({symbol}.{venue})
  enums.py          AssetClass, InstrumentClass, PriceType, BarAggregation,
                    AggregationSource, PriceAdjustment, NORGATE_EXCHANGE_TO_MIC
  instruments.py    Instrument (frozen dataclass; all asset classes)
  bar_type.py       BarSpecification, BarType (Nautilus string convention)
  catalog.py        InstrumentCatalog (registry -> data/instruments/catalog.parquet)
  symbol_map.py     source_symbol() / instrument_from_source_symbol()
  source_priority.py  SourcePriorityConfig (config-driven source routing)
  bar_record.py     CanonicalBarRecord (silver-layer schema)
  reconciler.py     SourcePriorityReconciler (wraps the IB ratio-splice)
  provenance.py     provenance + conflict persistence
```

### InstrumentId — `{symbol}.{venue}`

```python
from data_platform.core import InstrumentId
InstrumentId.from_str("AAPL.XNAS")   # equity
InstrumentId.from_str("ES.XCME")     # futures continuous root
InstrumentId.from_str("BRK.A.XNYS")  # dotted symbol (split on last dot)
```

Venues are MIC-style codes (`XNAS`, `XNYS`, `XCME`, `XCBT`, `XNYM`, `ARCX`),
identical to Nautilus venue identifiers. The `NORGATE_EXCHANGE_TO_MIC` map in
`enums.py` translates Norgate exchange names to these codes.

### BarType — Nautilus string convention

```python
from data_platform.core import BarType
BarType.from_str("AAPL.XNAS-1-DAY-LAST-EXTERNAL")
BarType.from_str("ES.XCME-1-WEEK-LAST-INTERNAL@1-DAY-EXTERNAL")  # composite
```

`BarSpecification.from_timeframe("D")` / `.to_timeframe()` bridge the repo's
existing `TimeFrame` enum to/from `BarType`, so existing pipeline code keeps
working during migration.

### Instrument — one frozen dataclass for all asset classes

Rather than a deep class tree, a single `Instrument` carries an
`instrument_class` discriminator (FUTURE, SPOT, CFD, OPTION, …). Precision and
increments are first-class (Nautilus enforces them strictly):

- `price_precision` / `price_increment` (tick size)
- `size_precision` / `size_increment`
- `multiplier` (point value $/point), `lot_size`, `margin_init/maint`, fees
- `activation` / `expiration` (futures lifecycle; `None` for equities)
- `price_adjustments` — which adjusted series exist for this instrument
  (e.g. `{"D": "BACK_ADJUSTED", "D_unadj": "NONE"}`)
- `info` — raw provider metadata (Nautilus's `.info` dict equivalent)

### InstrumentCatalog — the registry

```python
from data_platform.core import load_catalog, AssetClass, InstrumentClass
cat = load_catalog()
es = cat.find("ES.XCME")
print(es.multiplier, es.tick_value)         # 50.0  12.5
futures = cat.filter(instrument_class=InstrumentClass.FUTURE)
stocks  = cat.filter(asset_class=AssetClass.EQUITY)
```

Persisted to `data/instruments/catalog.parquet` (+ `catalog.json`). Bar data
itself stays in the existing parquet stores — the catalog is the metadata layer
on top. This is the "keep parquet store, add instrument catalog" approach.

---

## Where data lives

| Asset | Bar store | Keyed by | Catalog InstrumentId |
|---|---|---|---|
| Futures (continuous) | `data/ohlc_data/{TICKER}/` | repo ticker | `ES.XCME` (symbol = ticker) |
| Futures (individual) | `data/norgate/archive/contracts/{TICKER}/` | Norgate symbol | (archive, not in catalog) |
| US stocks | `data/stock_data/{bucket}/{SYMBOL}/` | symbol string | `AAPL.XNAS` |
| CFD (MT5) | `data/mt5_data/{symbol}/` | symbol string | (mapped per adapter) |

Stocks deliberately do **not** go through the futures `Ticker` enum,
`data/ohlc_data/`, or `bootstrap_source_candles` — that path is hardwired for
~25 contracts and cannot scale to 35k symbols. Stocks use a string-keyed
loader (`load_stock_data(symbol, ...)`) instead.

---

## Migration to NautilusTrader — the mapping

When we adopt `nautilus_trader`, each `data_core` type maps 1:1:

| data_core | nautilus_trader |
|---|---|
| `InstrumentId.from_str("AAPL.XNAS")` | `InstrumentId.from_str("AAPL.XNAS")` (identical string) |
| `Venue("XNAS")`, `Symbol("AAPL")` | `Venue`, `Symbol` |
| `BarType.from_str("...-1-DAY-LAST-EXTERNAL")` | `BarType.from_str(...)` (identical string) |
| `Instrument(instrument_class=FUTURE, ...)` | `FuturesContract(...)` |
| `Instrument(instrument_class=SPOT, asset_class=EQUITY)` | `Equity(...)` |
| `Instrument(instrument_class=CFD)` | `Cfd(...)` |
| `InstrumentCatalog` | `ParquetDataCatalog.write_data([instruments])` |

Because the `InstrumentId` and `BarType` **string forms are identical** to
Nautilus, serialized catalog rows and any persisted bar keys carry over
unchanged. The migration is: (1) `pip install nautilus_trader`, (2) write a
thin `to_nautilus_instrument(inst)` that constructs the matching nautilus type
from each catalog row, (3) load bars into a `ParquetDataCatalog`.

### What we adopt from Nautilus principles now

- **Precision as first-class** — every instrument declares price/size precision
  and increments; downstream sizing should use them (`instrument.make_price`
  equivalent) rather than free floats.
- **Fail-fast on invalid data** — adapters reject NaN/Inf/negative prices at
  ingest rather than letting them propagate (corrupt data is worse than no data).
- **Symbology** — `{symbol}.{venue}` unique per system; one canonical
  InstrumentId per economic instrument, with per-source native-symbol mapping.

---

## Related

- [[multi_source_update_architecture]] — continuous Norgate/IB/MT5 update +
  source-priority reconciliation + Norgate→IB futures stitching
- [[futures_backtesting_data_guide]] — which adjusted series to use when
- [[norgate]] — the Norgate adapter
- NautilusTrader docs: Instruments, Continuous Futures, Data, Architecture
