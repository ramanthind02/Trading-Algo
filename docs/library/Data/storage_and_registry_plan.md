# Storage & registry plan — schemas, invariants, and a local metadata DB

> **Status:** proposal / decision-of-record draft (2026-06-08). Grounded in a full
> sweep of every on-disk store — see the companion [[data_store_inventory]] for the
> per-store schema/key/writer/reader/invariant breakdown the recommendation rests on.

## TL;DR (the one-paragraph version)

The data is **not** actually schema-less — there is already a clean canonical model
(`InstrumentId`, the frozen `Instrument` dataclass, `InstrumentCatalog`,
`CanonicalBarRecord`, the reconciler, provenance records). What is missing is an
**enforced, indexed, queryable metadata layer**. Today every "schema" is a docstring
re-implemented per writer and only *coerced* (never *validated*) at read time, and
every "index" is a directory glob over hundreds of thousands of files. The fix is **not**
to put price data in a database. The fix is: **(1) one enforced schema contract per store,
and (2) a small local SQLite registry that indexes the files and holds the lineage the
filesystem cannot.** Bars, ticks, and cache payloads stay exactly where they are as
parquet blobs — the DB never owns a single price row, and is rebuildable from disk with
one command. **`del data\registry.db && registry rebuild` is the entire rollback.**

## The problem, in numbers

| | Count | Size | Note |
|---|---|---|---|
| **All data files** | **382,299** | **~28 GB** | across `data/`, `.cache/`, `vault*/`, `research/` |
| `data/stock_data` | **329,163** | 8.4 GB | 86 % of the file count; ~33k Norgate symbols × (D/W/M × TR/CAP/UNADJ + membership) |
| `data/mt5_data` | ~14,910 | 18 GB | 64 % of the bytes; 847 symbols, hive `year=` M1 partitions |
| `.cache/.../central_cache` | 1,097 pq + 99 json | — | bias-node artifacts; **tripled** by `_feeds/{cfd,spliced,futures_ratio}/` |
| `data/ohlc_data` | 165 | 24 MB | futures + TLT, one file per (ticker, tf, adjustment) |
| `data/nautilus_catalog` | ~19 series | 7 MB | a **third parallel copy** of bar data (regenerable) |
| `data/events/econ` | 148 pq + manifest | 21 MB | one opaque-named file per macro series |
| `vault/` + `vault_personal/` | ~42 json | — | strategy control files, two parallel layouts |
| `research/**` results | 97 csv + 34 pq | — | per-run dirs, no content registry |

Two problems, both real, both fixable:

1. **File sprawl.** 380k loose parquet files with **path-encoded keys** (ticker / timeframe /
   adjustment / params live in directory names and filename suffixes, not in columns).
   Discovery means globbing — e.g. enumerating the stock universe reads **~33,000 parquet
   footers** (`_scraped_symbols`); a single MT5→daily read globs+concats *every* `year=`
   file for a symbol. On Windows this also makes `git status` and indexing crawl.

2. **Unenforced schemas.** The `date32 / float32 OHLC / int32 volume` contract is
   copy-pasted across `migrate.py`, `stocks.py`, `ratio_driver.py`, `market_series.py`
   and only *coerced* at read time in `loaders._normalize_loaded_frame`. Nothing validates
   it. The sweep found the bugs this lets through:

   - **D1 schema drift** — `daily_scraper._D1_SCHEMA` (`int64/int32`) vs
     `scraper._BARS_SCHEMA` (`int32/int16`) write the **same on-disk path** with different
     dtypes; readers coerce silently, masking it.
   - **`date` is stored as `timestamp[ns]`** despite every docstring claiming `date32`.
   - **Engine split** — `ohlc_data` is read with `fastparquet`, `stock_data` with the
     default engine, for byte-identical schemas.
   - **Lossy cache keys** — bias-artifact filenames truncate params to 60 chars; the
     `sma_regime_signal` enum collapses `PositionMode.LONG_ONLY` and `LONG_SHORT` both to
     `PositionMo` (a real collision the code comment at `bias_node_cache.py:233` already
     flags). The `.meta.json` sidecars meant to track this are **dead**: written by no live
     code, 98 on disk, 8 orphaned, 133 parquet have none.
   - **O(n) reverse lookups** — `instrument_from_source_symbol` linear-scans the whole
     catalog because the source→symbol map is buried in a JSON blob column.
   - **Unwired lineage** — `provenance` / `conflicts` are fully defined but written only
     from tests (0 rows in production); there is no `spec → run → result → vault` link at all.

## The reframe that decides the architecture

> The pain is not "we need a database for our data." The pain is "our metadata is
> scattered, unenforced, and unqueryable." Those are different problems with different fixes.

**Price/tick/feature data is bulk columnar time-series read whole into pandas or
Nautilus.** Putting it in a relational row store would be slower and would fight the
vectorized pipeline. It stays parquet. **Forever.**

**The metadata about that data — what exists, what it covers, where it came from, which
spec/run produced it — is small, keyed, joined, and constraint-worthy.** That is a
textbook relational workload, and it is exactly where the filesystem fails us.

So the split is clean:

| Goes **relational** (small, keyed, joined) | Stays a **parquet/file blob** (bulk, columnar, read-whole) |
|---|---|
| Instrument catalog (69 rows) + `source_symbols` + adjustments | All OHLCV bars: `ohlc_data`, `stock_data`, `mt5_data` M1/D1 |
| Provenance + conflict logs (append-only audit) | All tick payloads (bulk `ticks/` + demand-driven `tick_cache` chunks) |
| **Blob manifest** — one row per payload file (path, coverage, schema_hash, rows) | `.cache` bias-node / EWSD / candle / materialized parquet |
| Tick-chunk coverage index (replaces filename-range glob + coverage JSON) | Nautilus `ParquetDataCatalog` (engine-mandated layout) |
| `spec → run → result → vault_entry` lineage (real FKs) | Econ per-series parquet; research result CSV/parquet bodies |
| Calendar: FOMC dates, NYSE holidays (date-keyed lookups) | StrategySpec / vault-feature JSON bodies (stored verbatim in a blob column) |
| **Live node↔dashboard command queue** (atomic claim) | `source_priority.yaml`, broker/runtime configs (git-reviewed safety config) |

The DB **indexes** the blobs; it never stores their rows.

## The plan (three phases, each shippable and reversible)

### Phase 0 — Schema contracts (do this regardless of the DB; ~3–4 days)

This is prerequisite either way and it fixes the actual bugs.

1. `data_platform/storage/contracts.py`: one `pyarrow.Schema` per store (`BAR_SCHEMA`
   with a **real `date32`**), plus invariant asserts (monotonic + unique date, OHLC
   sanity `low ≤ open,close ≤ high`, no-NaN close, adjustment matches file metadata).
2. `data_platform/storage/__init__.py`: a single `write_bars` / `write_artifact`
   entrypoint that wraps the existing atomic write, runs the validate-on-write gate, and
   is the **only** writer. Replace the four duplicated `_write`/`_normalize` bodies with
   calls into it — output must stay byte-identical (the schema is already what they emit).
3. One unit test asserting all writers produce `table.schema.equals(BAR_SCHEMA)`. **This
   single test catches the D1 drift immediately.**
4. Read-side tidy: delete the dead legacy-`datetime`-column branch in
   `_normalize_loaded_frame` (0 of 365 sampled files use it); standardize on one parquet
   engine.

Outcome: the "no schemas/invariants" complaint is resolved in code, not prose — with zero
bytes moved and a trivial rollback.

### Phase 1 — Consolidate the worst sprawl (~1 week; optional but high-value)

5. **Stock store:** re-emit the 329k per-symbol files as a hive dataset
   `data/stock_data_ds/adjustment=*/timeframe=*/` with `raw_symbol` promoted to a real
   column (kills the lossy `safe_symbol` encoding and the 33k-footer glob). Gate behind a
   `frame.equals` parity test on a sampled universe; keep the old tree until it passes.
6. **Central cache:** rename artifact files by a collision-free `params_hash`, store full
   `params_json` in the manifest, and **delete the 98 dead `.meta.json` sidecars**.

### Phase 2 — `data/registry.db`, the metadata index (~2 weeks)

A single embedded **SQLite** file holding **only** the relational contents from the table
above. Structure:

- `data_platform/registry/` — `schema.sql`, `db.py` (connect, `PRAGMA user_version`
  gate, `foreign_keys=ON`, WAL), `writer.py` (the **only** module that mutates), `reader.py`
  (query API).
- A `registry rebuild` CLI that walks the on-disk stores and repopulates every table from
  files — `InstrumentCatalog.load()` → instruments; glob footers **once** → `blob_manifest`
  + `tick_chunks`; `_runs_index.json` → runs; `research/specs/*.json` → specs; vault JSON →
  `vault_entries`; calendar JSON → fomc/holiday. **This command *is* the rebuildable-from-files
  invariant**; a CI test asserts it is idempotent and `PRAGMA foreign_key_check` is empty.
- Cut over the **highest-pain reads first**, behind a flag, files still authoritative:
  the O(n) `instrument_from_source_symbol` scan → indexed lookup; the 33k-footer glob → an
  O(1) manifest query.
- Wire the **live command queue** (atomic `UPDATE … WHERE status='pending'`) to retire the
  `command.json` overwrite race — the one store where a DB is clearly *more correct*, not
  just faster.
- Add the `spec → run → result → vault_entry` FKs last: the run registry already stores
  `spec_id` as a string, so backfill is a copy; the payoff is orphan detection and a
  `producing_run_id` on every promoted strategy.

See the [DDL appendix](#appendix-a--sqlite-schema-grounded-ddl) for the concrete schema.

## Why SQLite (and where DuckDB fits)

**SQLite** is the system of record for the registry: single-file, zero-server, ships with
Python, and its **single-writer lock matches the one-writer-per-broker live design exactly**.
It is the Windows-friendly, low-operational-cost choice for a solo developer.

**DuckDB** is complementary, not the system of record: you can point it at the **same
parquet files** for ad-hoc analytical queries (`SELECT … FROM 'data/stock_data_ds/**/*.parquet'`)
without it owning anything. Use SQLite for the durable index + lineage + command queue;
reach for DuckDB when you want SQL over the blobs themselves. Do **not** make DuckDB the
source of truth — it lacks SQLite's mature single-writer concurrency story.

## The safety invariant (why this is low-stakes)

The DB is a **pure derived index over unchanged files.** Every table except the ephemeral
`live_commands` queue is reconstructable by `registry rebuild`. Consequences:

- **Rollback is `del data\registry.db`** — zero data loss, because no price data ever lived
  there. The parquet stores remain readable by the current pandas/Nautilus paths whether or
  not the DB exists.
- **No bulk data migration** in Phase 2 — no bar file is moved or reformatted, so there is
  no risky re-encode (that risk lives only in the optional Phase 1 stock consolidation,
  which is gated behind a parity test).
- **It cannot corrupt price data** — it doesn't own any.

## Risks & honest trade-offs

- **Drift / second source of truth.** A derived index can desync from disk if a write
  bypasses `writer.py`. Mitigated by the single-writer rule + the rebuildable invariant +
  a CI `manifest == scan` reconciliation check. This is a genuine discipline cost for a
  solo dev — it is the main reason *not* to over-build here.
- **SQLite write contention.** A research run, the live node, and a `rebuild` can contend
  on the write lock. Use WAL, keep writes short, **never `rebuild` during live trading.**
- **Partial coverage is intentional.** `source_priority.yaml` and broker/runtime configs
  stay as git-reviewed files. "Everything in one place" is explicitly *not* the goal — the
  goal is everything *queryable that should be*.
- **The files-only alternative is viable too.** A disciplined-files variant (manifest
  `parquet` per store instead of SQLite) fixes 100 % of the *stated* pain with zero new
  infrastructure and keeps one storage paradigm end-to-end. Its weak spots are exactly the
  things a DB does best: cross-store lineage **JOINs**, **FK enforcement** of the
  `(source, native_symbol)` uniqueness the code only assumes, and an **atomic command
  queue**. The recommendation below splits the difference deliberately.

## Recommendation

**Do Phase 0 now** — it is pure upside, fixes real bugs, and is prerequisite to everything
else. **Do Phase 1 when the stock-universe glob or the dead sidecars actually bite.**
**Adopt the SQLite registry (Phase 2) incrementally**, highest-pain read first, with the
rebuildable-from-files invariant as the escape hatch. Start the registry with the three
pieces that are pure win and have no files-only equivalent:

1. `instruments` + `instrument_source_symbols` (kills the O(n) reverse scan, **enforces**
   the uniqueness invariant),
2. the `blob_manifest` (kills the 33k-footer glob, subsumes the dead `.meta.json` sidecars),
3. the `live_commands` queue (atomic claim, no lost commands).

Add the `spec → run → result → vault` lineage once those prove out. Keep DuckDB in the back
pocket for ad-hoc SQL over the parquet. **Never let the DB own a price row.**

---

## Appendix A — SQLite schema (grounded DDL)

Grounded in the real fields from `data_platform/core/catalog.py`, `reconciler.py`, the
StrategySpec, and the vault control files. Abridged to the load-bearing tables; the live
schema lives in `data_platform/registry/schema.sql`.

```sql
PRAGMA user_version = 1;          -- schema gate, checked by the writer on open
PRAGMA foreign_keys = ON;

-- INSTRUMENT REGISTRY (grounds: Instrument dataclass)
CREATE TABLE instruments (
  id               TEXT PRIMARY KEY,                 -- '{symbol}.{venue}' e.g. 'ES.XCME'
  raw_symbol       TEXT NOT NULL,
  asset_class      TEXT NOT NULL CHECK (asset_class IN
                     ('FX','EQUITY','COMMODITY','DEBT','INDEX','CRYPTOCURRENCY','ALTERNATIVE')),
  instrument_class TEXT NOT NULL CHECK (instrument_class IN
                     ('SPOT','FUTURE','FUTURES_SPREAD','FORWARD','CFD','OPTION','WARRANT','INDEX')),
  price_precision  INTEGER NOT NULL CHECK (price_precision >= 0),
  price_increment  REAL    NOT NULL CHECK (price_increment > 0),
  multiplier       REAL    NOT NULL DEFAULT 1.0,
  quote_currency   TEXT    NOT NULL DEFAULT 'USD',
  activation       DATE, expiration DATE,            -- real DATE, not ISO string
  data_source      TEXT CHECK (data_source IN ('norgate','ib','mt5') OR data_source IS NULL),
  info_json        TEXT NOT NULL DEFAULT '{}'        -- residual provider blob (round-trip)
);

-- normalizes info['source_symbols'] out of the JSON blob — the strongest relational fit
CREATE TABLE instrument_source_symbols (
  instrument_id TEXT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
  source        TEXT NOT NULL,                       -- 'norgate_adj','ib_contfut','mt5'
  native_symbol TEXT NOT NULL,
  PRIMARY KEY (instrument_id, source),
  UNIQUE (source, native_symbol)                     -- ENFORCES the assumed-unique invariant
);

-- LINEAGE: provenance + conflicts (grounds: reconciler.py records)
CREATE TABLE provenance (
  id INTEGER PRIMARY KEY,
  instrument_id TEXT NOT NULL REFERENCES instruments(id),
  resolution TEXT NOT NULL CHECK (resolution IN ('D','W','M','M1')),
  source     TEXT NOT NULL CHECK (source IN ('norgate','ib','mt5','derived_from_daily')),
  ts_ingest  TEXT NOT NULL,
  ratio_applied INTEGER NOT NULL CHECK (ratio_applied IN (0,1)),
  ratio_value REAL, rows_written INTEGER NOT NULL CHECK (rows_written >= 0)
);
CREATE INDEX ix_prov_inst_ts ON provenance(instrument_id, ts_ingest);

CREATE TABLE conflicts (
  id INTEGER PRIMARY KEY,
  instrument_id TEXT NOT NULL REFERENCES instruments(id),
  bar_date DATE NOT NULL,
  source_a TEXT NOT NULL, source_b TEXT NOT NULL,
  close_a REAL NOT NULL, close_b REAL NOT NULL,
  deviation REAL NOT NULL,
  threshold REAL NOT NULL                            -- per-row → fixes interpretability
);

-- BLOB MANIFEST: one row per payload file across ALL stores (the central index)
CREATE TABLE blob_manifest (
  id INTEGER PRIMARY KEY,
  store          TEXT NOT NULL,    -- 'ohlc_data','stock_data','mt5_m1','tick_cache',
                                   -- 'bias_artifact','candle_cache','econ','nautilus_series'...
  key_json       TEXT NOT NULL,    -- composite logical key as JSON (ticker/tf/adj/symbol/params)
  relative_path  TEXT NOT NULL UNIQUE,
  schema_hash    TEXT,             -- declares legacy-vs-current column schema
  engine         TEXT,             -- 'fastparquet'|'pyarrow' (declares per-store engine)
  rows           INTEGER,
  coverage_start DATE, coverage_end DATE,
  written_at     TEXT NOT NULL,
  valid          INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
  validation_error TEXT,
  UNIQUE (store, key_json)
);
CREATE INDEX ix_blob_store_cov ON blob_manifest(store, coverage_end);
-- replaces: _scraped_symbols 33k-footer glob, _ticks_coverage.json, the .meta.json sidecars,
-- econ _manifest.json, central-cache describe_*-by-file-read, filename-as-key parsing.

-- tick-chunk coverage (replaces per-symbol coverage JSON + filename-range glob)
CREATE TABLE tick_chunks (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL,
  file_path TEXT NOT NULL UNIQUE,
  n_ticks INTEGER NOT NULL,
  CHECK (end_ns > start_ns)
);
CREATE INDEX ix_tick_overlap ON tick_chunks(symbol, start_ns, end_ns);

-- RESEARCH LINEAGE: spec -> run -> result -> vault (the FKs the filesystem cannot give)
CREATE TABLE specs (
  id TEXT PRIMARY KEY,                                -- stable slug/uuid, decoupled from name
  name TEXT NOT NULL,
  name_slug TEXT NOT NULL UNIQUE,                     -- collision = error, not silent overwrite
  content_hash TEXT NOT NULL,                         -- pins exactly-which-spec
  spec_version TEXT NOT NULL DEFAULT '1.0',
  valid INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)), error TEXT,
  spec_json TEXT NOT NULL,                            -- full StrategySpec blob (round-trip)
  file_path TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,                            -- uuid4 == output folder
  spec_id TEXT REFERENCES specs(id),                 -- real FK → orphans become detectable
  spec_content_hash TEXT,
  phase TEXT NOT NULL CHECK (phase IN ('exploration','validation')),
  status TEXT NOT NULL CHECK (status IN ('queued','running','completed','failed')),
  feed_lane TEXT, num_combos INTEGER, reports_dir TEXT, viz_dir TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error_text TEXT
);
CREATE INDEX ix_runs_spec ON runs(spec_id);

CREATE TABLE results (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  artifact_type TEXT NOT NULL,                        -- 'equity_curve','param_sensitivity',...
  file_path TEXT NOT NULL,                            -- CSV/parquet body stays a file
  sharpe REAL, t_stat REAL, sortino REAL, n_obs INTEGER  -- headline metrics lifted for query
);

CREATE TABLE sleeves (name TEXT PRIMARY KEY);          -- subsumes custom_sleeves.json

CREATE TABLE vault_entries (
  id INTEGER PRIMARY KEY,
  vault_profile TEXT NOT NULL CHECK (vault_profile IN ('prop','personal','cfd_prop')),
  timeframe TEXT NOT NULL CHECK (timeframe IN ('D','W','M')),
  weight_hierarchy_group TEXT REFERENCES sleeves(name),
  ensemble_leaf TEXT NOT NULL, feature_name TEXT NOT NULL, feature_column TEXT NOT NULL,
  module_name TEXT NOT NULL, direction TEXT,
  producing_run_id TEXT REFERENCES runs(run_id),       -- promoted strategy → its run+spec
  config_json TEXT NOT NULL,                            -- bias_node_spec + base_models blob
  file_path TEXT NOT NULL,
  valid INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
  validation_error TEXT,                                -- surfaces SKIPPED malformed files
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE (vault_profile, timeframe, ensemble_leaf, feature_name)
);

-- CALENDAR (small date-keyed lookups; grounds: data/events/calendar/*.json)
CREATE TABLE fomc_decision (decision_date DATE PRIMARY KEY, scraped_at TEXT);
CREATE TABLE nyse_holiday (
  holiday_id TEXT NOT NULL, closure_date DATE NOT NULL, d0 DATE NOT NULL,
  asset_bucket TEXT NOT NULL CHECK (asset_bucket IN ('equity','gold')),
  PRIMARY KEY (holiday_id, closure_date)
);

-- LIVE node↔dashboard command queue (the ONE genuinely ephemeral, non-rebuildable table)
CREATE TABLE live_commands (
  id TEXT PRIMARY KEY, broker TEXT NOT NULL, action TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending','claimed','done','failed')),
  issued_at TEXT NOT NULL, consumed_at TEXT, result_text TEXT
);
-- atomic claim: UPDATE live_commands SET status='claimed' WHERE id=? AND status='pending';

-- Orphan detection — impossible on the filesystem today
CREATE VIEW v_instruments_without_bars AS
  SELECT i.id FROM instruments i
  LEFT JOIN blob_manifest b
    ON b.store IN ('ohlc_data','stock_data','mt5_m1') AND b.key_json LIKE '%'||i.raw_symbol||'%'
  WHERE b.id IS NULL;
```

## Appendix B — the rebuildable invariant, stated precisely

> For every table T except `live_commands`: the rows of T are a pure function of the files
> on disk. `registry rebuild` recomputes them. A CI test asserts
> `rebuild(); foreign_key_check() == ∅` and that row counts/keys reconcile against a fresh
> footer scan. `live_commands` is ephemeral and self-heals on node restart exactly as
> `command.json` does today.

This is the property that makes the whole thing safe to adopt and trivial to abandon.

> _Survey performed by a 6-agent store sweep + design panel on 2026-06-08; verify file
> counts with `find data .cache vault* research -type f | wc -l` before acting._
