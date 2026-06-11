# Data store inventory — every on-disk store, with schema/key/enforcement/DB-fit

> **Counts in this document are a 2026-06-08 snapshot.** The live equivalent is generated:
> `python -m data_platform.registry report`

> Reference companion to [[storage_and_registry_plan]]. A full sweep of every physical
> data store in the repo (2026-06-08): where it lives, what its schema and logical key are,
> how (if at all) invariants are enforced, and whether the bytes belong in a relational DB
> or should stay a parquet/file blob. **DB-fit verdict** is the column that drives the plan.

**Legend — DB-fit:** 🟥 strong relational fit · 🟧 index/manifest only (payload stays a blob)
· 🟦 stay a file (no DB).

## 1. Canonical bar stores (futures, TLT, stocks)

| Store | Path | Logical key | Enforcement | ~Files | DB-fit |
|---|---|---|---|---|---|
| Futures + TLT | `data/ohlc_data/{TICKER}/{TF}_{TICKER}[_unadj/_ratio].parquet` | (ticker, tf, **adjustment**, date) — adj is filename-only | write-time (`migrate._normalize`); read only *coerces* | 165 | 🟧 manifest row per file (adj/coverage/engine/schema_version) |
| Norgate stocks | `data/stock_data/{BUCKET}/{SAFE_SYMBOL}/{TF}_{ADJ}_{SAFE}.parquet` | (raw_symbol, tf, adj, date); raw_symbol only in parquet **metadata** (lossy `safe_symbol`) | write-time only; universe = **33k-footer glob** | **329,163** | 🟧 manifest badly needed (universe/survivorship/coverage); bars stay parquet |

**Bugs found here:** `date` stored `timestamp[ns]` not `date32`; engine split
(`fastparquet` for futures, default for stocks); dead legacy-`datetime`-column branch on
the read path; adjustment never a column.

## 2. MT5 / live / tick stores

| Store | Path | Key | Enforcement | ~Files | DB-fit |
|---|---|---|---|---|---|
| MT5 M1 bars | `data/mt5_data/{SYM}/bars_M1/year=YYYY/part.parquet` | (symbol, time) | `_BARS_SCHEMA` on write; read-merge-rewrite on append | ~13,848 | 🟧 manifest (symbol,year,min/max time) → kills glob-concat-all |
| MT5 D1 bars | `data/mt5_data/{SYM}/bars_D1/part.parquet` | (symbol, time) | **two conflicting schemas** for one path (drift) | 214 | 🟦 fix the schema, not a DB; 1 manifest row |
| Bulk ticks | `data/mt5_data/{SYM}/ticks/year=YYYY/part.parquet` | (symbol, time_msc) | `_TICKS_SCHEMA`; memory-aware merge | ~1 | 🟧 coverage index only; ticks stay parquet |
| **Tick cache** (hybrid backtest) | `…/{SYM}/ticks_cache/{start_ns}-{end_ns}.parquet` + `_ticks_coverage.json` | (symbol, start_ns, end_ns) in filename | coverage JSON is a **separate source of truth** from the chunks | ~829 + 4 json | 🟧 `tick_chunks` table folds coverage+index into one indexed overlap query |
| Live signal cache | `data/broker_cache/darwinex/central_cache/{candles,artifacts}/…` | (ticker,tf) / (module,ticker,tf,params) | read-time coerce; atomic write; `.meta.json` defined but **0 on disk** | ~85 | 🟧 manifest replaces per-file `describe_*` open |
| **Live IPC / decision state** | `data/broker_cache/{broker}/live_state/{snapshot,command,baseline,halt}.json`, `equity_history.jsonl`, `decision_state.json` | singleton-per-broker; `command` by id | atomic-file + `read_or_quarantine`; one-writer **by convention** | ~12 max | 🟥 **command queue** is the best DB fit in the repo (atomic claim, no lost/overwritten commands) |
| MT5 catalog + backfill progress | catalog (16 hand-coded rows) + `_m1_backfill_progress.json` | InstrumentId; singleton | atomic heartbeat; resume driven by on-disk M1 not the json | 1 | 🟦 progress=JSON fine; 16-row catalog fine as-is |

## 3. Central cache (`.cache/trading_algo/central_cache/`)

| Store | Path | Key | Enforcement | ~Files | DB-fit |
|---|---|---|---|---|---|
| Bias-node / EWSD artifacts | `artifacts/{live,research}/{module}/{TICKER}_{TF}[_{params}].parquet` (+ `_feeds/{cfd,spliced,futures_ratio}/…`) | (feed, scope, module, ticker, tf, params) — **path+lossy-filename** | filename-as-key (**truncated→hash**, `PositionMo` collision); coverage re-read per file | 1,097 | 🟥/🟧 manifest table with `params_json` + collision-free hash (this is what the dead sidecars tried to be) |
| **Dead `.meta.json` sidecars** | `…/{file}.parquet.meta.json` | same tuple as the parquet | **none** — written/read by no live code | 98 (8 orphan, 133 parquet have none) | 🟥 replace entirely with the manifest table |
| Source candle cache | `candles/{TICKER}/{TF}.parquet` (+ `_feeds/…`) | (feed, ticker, tf) | read-time normalize; coverage by reading frame | ~30–40 | 🟧 small manifest for O(1) freshness |
| Materialized predictions | `materialized/{scope}/{base_models/{TF}/{ens}/{hash}.parquet \| portfolio/{name}.parquet}` | content-hash / name | self-validating hash; discovery by `rglob` | 3 | 🟧 manifest maps opaque hash→config |
| Live-refresh status | `live_refresh/last_run.json` | singleton | atomic, last-write-wins | 1 | 🟦 (audit-log table only if history wanted) |

`_feeds/{cfd,spliced,futures_ratio}/` (859 files) are **parallel copies** of the cache
with no cross-feed coherency link.

## 4. Metadata / identity layer (already half a schema)

| Store | Path | Key | Enforcement | DB-fit |
|---|---|---|---|---|
| **InstrumentCatalog** | `data/instruments/catalog.parquet` (+ redundant `catalog.json`) | InstrumentId `{symbol}.{venue}` (69 rows) | frozen dataclass + enums at construct; storage enforces **nothing**; dual-write non-atomic | 🟥 textbook master table (PK + CHECK enums + NOT NULL + real DATE) |
| **Symbol map** | *none* — buried in catalog `info['source_symbols']` JSON | (instrument, source)↔native_symbol | **O(n) reverse scan**; uniqueness only assumed | 🟥 **strongest fit** — `instrument_source_symbols` with UNIQUE index |
| Provenance log | `data/provenance/{INST}/{RES}_provenance.parquet` | (inst, res, ts, source) | frozen dataclass; **unwired** (0 rows, tests only) | 🟥 append-only audit table |
| Conflict log | `data/provenance/{INST}/{RES}_conflicts.parquet` | (inst, date, a, b) | 0.5% threshold gate; **unwired**; threshold not stored | 🟥 audit table + threshold per row |
| SourcePriorityConfig | `configs/source_priority.yaml` | (instrument_class, resolution)→ordered rules | enum coerce + 3-case condition DSL | 🟦 git-versioned config, keep as file |
| Nautilus catalog | `data/nautilus_catalog/data/{bar,custom_research_candle,quote_tick}/…` | BarType / InstrumentId | NautilusTrader owns layout/schema | 🟦 engine-mandated; **3rd parallel copy** of bars (regenerable) |

## 5. Strategy / research outputs

| Store | Path | Key | Enforcement | ~Files | DB-fit |
|---|---|---|---|---|---|
| Vault feature control files | `vault[_personal]/<TF>/<sleeve>/<leaf>/features/*.json` (+ legacy flat) | (profile, tf, leaf, feature_name); long names→`feat_<md5>` | read-time validate; **invalid files silently SKIPPED**; two layouts | ~42 | 🟥 `vault_entries` registry; spec JSON stays a blob column |
| Research specs (StrategySpec) | `research/specs/*.json` | id = name-slug (**collision-overwrite risk**) | strong `__post_init__` validation; no stable id, no content hash | 4 | 🟥 `specs` table (uuid + UNIQUE slug + content_hash) |
| Feature run results | `research/feature/**/results/<run_uuid>/…csv` + `feature_research/shared_results/_runs_index.json` | run_id; `spec_id` is an **unenforced string** | atomic index; CSVs **no schema**; **split-brain output roots** | ~97 csv | 🟥 `runs`+`results` tables (FK to specs); CSV bodies stay files |
| Portfolio / rollover outputs | `research/portfolio/results/…csv`, `research/rollover_cost/outputs/…` | composite natural keys; giant `stream_or_model_id` string | none; fixed paths **overwrite** (history lost) | ~10 csv + ~10 pq | 🟧 summaries→tables; events parquet + Nautilus catalogs stay blobs |
| Runtime configs | `configs/{mt5_brokers.yaml, live_forecast_config*.json, …}` | filename = identity | `brokers.resolve` **raises on UNKNOWN** (safety) | 7 | 🟦 git-reviewed safety config; keep as files |

## 6. Event data

| Store | Path | Key | Enforcement | ~Files | DB-fit |
|---|---|---|---|---|---|
| FOMC + NYSE holidays | `data/events/calendar/{fomc_decision_dates,nyse_holiday_events}.json` | decision date / (holiday_id, closure_date) | read-time validate; coverage window baked (2005–2027), no gap detection | 2 | 🟥 two small date-keyed tables, join to trading_day_index |
| Norgate econ series | `data/events/econ/{SAFE_SYMBOL}.parquet` + `_manifest.json` | (safe_symbol, date); name in parquet metadata | write-time normalize; manifest drifts; near-orphan store | 148 + 1 | 🟥 long table `econ_release(symbol,date,…)` + meta table (these are time series, but date-keyed-join is the access pattern) |

---

## Cross-cutting observations

- **Three parallel copies of bar data:** provider parquet (`ohlc_data`/`mt5_data`) → the
  Nautilus `ResearchCandle` catalog → native Nautilus `Bar`. Kept in sync only by
  re-running ingest, guarded by one byte-exact equivalence test. The Nautilus copy is
  **derived/regenerable**, not a source of truth.
- **Path-encoded keys everywhere:** ticker/tf/adjustment/params/feed/scope live in
  directory and filename structure, never as columns — so every "query" is a glob and every
  key is a brittle string parse.
- **Enforcement is uniformly read-time coercion, not a storage contract.** Schemas are
  docstrings duplicated per writer. The one consistent invariant that *is* enforced —
  `brokers.resolve()` raising on an UNCONFIRMED symbol — is the model to copy.
- **The stores screaming loudest for relational treatment** are the ones already *trying*
  to be tables and failing: the source-symbol map (buried in JSON, O(n) scan), the
  `.meta.json` sidecars (a dead per-artifact manifest), the run index (`spec_id` string with
  no FK), and the live `command.json` (no atomic claim).

> _Survey by a 6-agent store sweep on 2026-06-08. Re-verify counts with
> `find data .cache vault* research -type f | wc -l` before acting on any number here._

> _Verified against the working tree on 2026-06-10._
