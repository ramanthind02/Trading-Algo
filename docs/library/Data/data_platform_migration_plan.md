# Data platform migration plan — market data, research registry, live trading records

> **Status:** proposal v1.1 (2026-06-09, revised after a 3-lens adversarial review).
> **Extends** [[storage_and_registry_plan]] (the market-data registry ADR, which stands
> except where ADR-3 below explicitly supersedes its `live_commands` decision) and
> **supersedes** the canonical-store section of `docs/refactor/nautilus/02_data_layer.md`
> (see ADR-1). Grounded in a 7-subsystem agent survey of the working tree on 2026-06-09
> (market data, research runs, live persistence, Nautilus capabilities, ops/scheduling,
> docs, agent interfaces) plus the per-store [[data_store_inventory]].

## 0. Executive summary

One SQLite registry (`data/registry.db`) becomes the queryable metadata index over three
data domains — **market data**, **research runs**, **live trading records** — while bulk
payloads stay parquet/JSON files exactly as the existing storage plan prescribes. The new
work this plan adds on top of that baseline:

1. **Live trading records are net-new DDL** (accounts, orders, deals, equity snapshots,
   forecast history). Today a prop-firm account reissue silently inherits the old
   account's state, fills survive only in a manually-run `slippage.csv`, and *no
   historical record exists of what we forecast on any past day*. Broker history beyond
   the `history_deals_get` retention window is the most perishable data in the system —
   capture ships early.
2. **A persistent Nautilus `ParquetDataCatalog`** becomes the canonical *consumption* tier
   for backtests/research (today every realistic-lane backtest rebuilds a throwaway tempdir
   catalog from raw parquet). Provider parquet stays the immutable source of record.
3. **Research lineage becomes enforced**: spec snapshot + content hash per run, headline
   metrics lifted into the registry, `producing_run_id` written into vault features at
   promotion, portfolio/validation runs persisted (today they evaporate).
4. **A shared experiment toolkit** replaces the 5 copies of the M1 loader, ~20 inline
   Sharpe implementations, 5 scattered POINT constants, and 2 hand-rolled 20–32 KB Nautilus
   lane scaffolds that agents keep rewriting. It has no registry dependency and starts
   immediately, in parallel.
5. **Automation actually runs**: one declarative Windows Task Scheduler registrar, an
   at-logon + daily catch-up scrape that *chains* freshness → signal-cache → registry
   ingest (today **zero** tasks are registered on this machine and every step is manual),
   unified Telegram alerting, and registry/irreplaceable-store backups.
6. **Agent-facing query surfaces**: a typed read-only `registry` Python API + CLI, then
   `/api/data/*` endpoints, so agents stop globbing 380k files and stop re-deriving
   infrastructure.
7. **Docs**: an immediate stale-fix PR + `docs/index.md`, then a reorganization in the
   NautilusTrader shape (`concepts/ integrations/ operations/ decisions/`) once the
   architecture it documents has landed.

## 1. Decisions of record

| # | Decision | Rationale (short) |
|---|---|---|
| **ADR-1** | **Two-tier canonical store.** Provider parquet (`data/ohlc_data`, `data/mt5_data`, `data/norgate`, `data/stock_data`) is the immutable **source of record**. A single persistent **Nautilus `ParquetDataCatalog`** (`data/nautilus_catalog`) is the canonical **consumption tier** for Nautilus lanes, refreshed incrementally post-scrape. The registry indexes both. | Resolves the WP-2 vs storage-plan conflict. Nautilus parquet schemas migrate between engine versions via Rust binaries — wrong property for a system of record; "wipe and re-ingest" stays the upgrade path. But per-run tempdir catalogs (today's practice) are the slowest, most-repeated conversion in the stack. The catalog is exactly what the Nautilus docs designate as the backtest/live consumption tier (`BacktestDataConfig`, `DataCatalogConfig`). |
| **ADR-2** | **MT5 timestamps stay broker-wallclock-mislabelled-UTC on disk**, but the contract becomes machine-readable: a `timezone='broker_eet_as_utc'` stamp in parquet metadata + manifest rows, and one owning accessor for conversions. No bulk rewrite. | Rewriting ~18 GB + every experiment loader that assumes broker wallclock is high-risk for zero research benefit. The pain was *undocumented-ness*, not the convention. (Live path already converts correctly at connect time.) |
| **ADR-3** | **SQLite topology: one `data/registry.db`, WAL mode, single writer module per domain. Live nodes never touch the DB** — they keep writing files under `data/broker_cache/<broker>/` (atomic JSON for state, plain append-only JSONL for records) and a scheduled **ingester** loads them. Ingest is an **idempotent full-file re-read + PK/UNIQUE upsert**, tolerant of a torn last line — never byte-offset tailing. JSONLs are retained permanently (archived on account rotation, never pruned), so every live registry table is recoverable by re-running ingest. **Supersedes the baseline plan's `live_commands` DB queue**: the node will not claim commands from SQLite; `command.json` file-IPC stays, and the baseline's `live_commands` table is dropped. | Live trading loop must never block on a DB lock. The node is the sole owner of its MT5 terminal; file ingestion avoids the unresolved second-attach contention question entirely. One decision of record, not two. |
| **ADR-4** | **Fills come from the node, not a second terminal attach.** The exec client already polls `history_deals_get` (250 ms, magic-filtered, deduped); it will additionally append **full deal records** (all fields incl. `swap/profit/fee/entry/position_id`) to `live_state/deals.jsonl`. A catch-up CLI (read-only attach, quiet window) backfills gaps when the node was down. *Rejected alternatives:* Nautilus `StreamingConfig`/`convert_stream_to_data` captures events only while the node runs and the emitted `OrderFilled` carries no swap/fee (the exec client persists commission only); `generate_account_report` spans only the node's in-memory cache lifetime. For a CFD book where swap is a dominant cost, the MT5 deal record is the ground truth. | MT5 deal history is richer than Nautilus fill events. Whether two Python processes can share one terminal is asserted in three contradictory directions across the repo — don't build the primary path on it. |
| **ADR-5** | **Accounts are first-class**: `accounts` table keyed by surrogate id, `UNIQUE(broker, login, valid_from)` (brokers reuse logins on demo resets), with exec tier, program phase (challenge/verification/funded/retail/personal/scraper), validity window, and `predecessor_account_id` chaining challenge→funded reissues. An `accounts rotate` CLI archives broker-keyed live state on reissue. **All order/deal keys are account-scoped** — MT5 deal tickets are per-server sequences and Nautilus client order ids are not unique across nodes (no `trader_id` is configured today). | Prop-firm churn is routine; today a new login silently inherits the old account's baseline anchors, equity history, and slippage records — and unscoped tickets would silently collide across accounts. |
| **ADR-6** | **Bar-schema unification: `_BARS_SCHEMA` (int32 tick_volume / int16 spread) wins.** Rewrite the small D1 store (214 files) to match the large M1 store; contract validation adds an int16-overflow assert on spread. **Amended 2026-06-09 during M0.3:** D1 gets its own contract (`MT5_D1_BARS_SCHEMA`, int64 tick_volume / int16 spread) — 10 US-stock D1 files carry daily volumes up to 20.3B (BAC) which can never fit int32; M1 keeps int32/int16 (per-minute volumes verified ~100× headroom). The drift fix is the contract being explicit, not the widths being identical. | Cheapest rewrite by ~4 orders of magnitude; int16 = 32,767 points of spread is ample for confirmed symbols, and the validator turns a silent overflow into a loud error. |
| **ADR-7** | **MT5 stores gain a broker dimension.** `data/mt5_data/<SYM>/` is declared the **darwinex** store (matches reality); future non-Darwinex scrapes write under broker-keyed roots (`brokers.broker_data_dir` already computes them); every manifest row carries `broker`; the scraper **asserts `account_info().server`** matches the intended broker before writing. | Native symbols collide across brokers; today a scrape attached to the wrong terminal silently merges another broker's bars into the Darwinex store (`connect()` falls back to "attach to whatever terminal is running"). |
| **ADR-8** | **Cost data becomes a queryable surface**: a `cost_observations` job materializes per-(instrument, broker, time-of-day bucket) spread stats from M1 `spread`/rollover ticks, joined with realized slippage distributions from the deals table. **On-demand** (`registry costs --refresh`), not scheduled. The experiment toolkit's cost model reads this + `instruments.price_increment` (fixed with live-probed POINT values). | This is the actual deliverable for cost-sensitive intraday research; today spread truth, POINT constants, and slippage live in three disconnected places. On-demand keeps the daily chain short for one operator. |
| **ADR-9** | **Agent access order: typed reader API → `registry` CLI → `/api/data/*` → (maybe) MCP later.** Read-only enforced via `file:...?mode=ro`. Single-writer enforced by detection (`rebuild`/reconcile CI check) + a PreToolUse nudge hook on `to_parquet` writes targeting `data/` — not by trying to sandbox Bash. | Matches how agents actually work here (venv python + Bash + allowlisted CLIs); codegraph proves the CLI/MCP pattern but MCP is new infra with no call volume yet. |
| **ADR-10** | **Non-rebuildable tables get real migration discipline.** Derived tables keep the `rebuild` invariant; live-record tables get `schema_version` + forward-only migration scripts + **nightly backup** (SQLite backup API → `data/backups/`, plus the irreplaceable Norgate archive mirrored off-box once). Because JSONLs are retained (ADR-3), live registry tables are re-ingestable; the *truly* unrecoverable data is broker history beyond each account's `history_deals_get` retention window — hence the M0 retention probe. | "Delete and rebuild" stops being the universal rollback the moment we store records the filesystem never had. |

## 2. Target architecture

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                 data/registry.db (SQLite, WAL)           │
                          │  market: instruments · source_symbols · blob_manifest    │
                          │          tick_chunks · provenance · conflicts · calendar │
                          │  research: specs · runs · results · vault_entries        │
                          │  live: accounts · orders · deals · equity_snapshots      │
                          │        forecast_history · job_runs                       │
                          │  derived: cost_observations                              │
                          └───────▲──────────────▲──────────────────▲────────────────┘
                                  │ rebuild/index │ ingest (sched)   │ ingest (sched)
        SOURCE OF RECORD          │               │                  │
  ┌───────────────────────────┐   │   ┌───────────┴───────────┐  ┌───┴──────────────────────┐
  │ provider parquet          │───┘   │ research artifacts    │  │ live JSONL/JSON          │
  │ ohlc_data · mt5_data      │       │ per-run dirs (CSV/pq) │  │ broker_cache/<broker>/   │
  │ norgate · stock_data      │       │ research/specs/*.json │  │ live_state/{snapshot,    │
  │ events · instruments      │       │ vault*/ control files │  │  deals,submits,forecasts,│
  └────────────┬──────────────┘       └───────────────────────┘  │  equity}.jsonl           │
               │ post-scrape incremental ingest                  └──────────▲───────────────┘
  ┌────────────▼──────────────┐    CONSUMPTION TIER                         │ node writes only
  │ Nautilus ParquetDataCatalog│  Bar · QuoteTick · ResearchCandle ·        │ (files only,
  │ data/nautilus_catalog      │  custom Data (forecasts, spreads, econ)   ┌┴────────────────┐
  └────────────┬──────────────┘                                            │ TradingNode      │
               └── BacktestDataConfig / catalog.query → research lanes,    │ (one per broker) │
                   validation lane, experiment toolkit                     └─────────────────┘
```

Payload rule is unchanged from the baseline plan: **the DB never owns a price row, an
equity-curve row, or a research CSV body** — it indexes files and holds the small, keyed,
joined records (plus the live records that have no file-native home).

## 3. Workstream A — market-data foundation (Phase 0 of the baseline plan, plus fixes)

*Everything in the baseline plan's Phase 0 stands; the survey added items marked NEW.*

1. `data_platform/storage/contracts.py` — one `pyarrow.Schema` per store + invariant
   asserts (real `date32`, OHLC sanity, monotonic unique dates, **NEW:** spread int16
   overflow check, **NEW:** `timezone` metadata stamp per ADR-2).
2. `data_platform/storage/__init__.py` — single `write_bars`/`write_artifact` writer used
   by every Norgate/MT5 writer; byte-identical output gated by parity tests.
3. Unify the M1/D1 schema drift per ADR-6 (rewrite D1, one-time).
4. **NEW:** scraper broker guard per ADR-7 (`assert account_info().server` matches the
   configured broker before any write; refuse "attach to whatever terminal is running").
5. Read-side tidy: delete the dead legacy-`datetime` branch in `_normalize_loaded_frame`,
   standardize on one parquet engine, keep `_normalize_loaded_frame` as the single output
   contract (all four feed paths already funnel through it).
6. **NEW:** resolve the IB ghost — `data_platform/providers/ib/` is deleted but `ib` still
   survives as a source slot in `SourcePriorityConfig.default()`, the reconciler, and
   catalog `source_symbols`. Either delete those branches (declared path: Norgate archive +
   MT5) or re-home a thin adapter around `execution/ib_data_client.py` — **do not leave a
   config-selectable source with no adapter** (default: delete; see §14).
7. **NEW:** probe `history_deals_get` retention per broker account (bounds the M2 fills
   backfill; records outside the window are permanently lost on account churn — the one
   genuinely unrecoverable dataset, per ADR-10).
8. Housekeeping: move the stray committed fixture `data_platform/providers/data/` to
   `tests/fixtures/`; finish deleting `deploy/` (5 unstaged deletions + orphan logs).

**Verify:** schema-equality unit test per writer (catches the D1 drift immediately);
existing byte-equivalence parity tests stay green; a wrong-terminal scrape attempt fails
loudly in a dry-run test.

## 4. Workstream B — registry core

As specified in the baseline plan (structure `data_platform/registry/{schema.sql, db.py,
writer.py, reader.py}` + `registry rebuild`), with these changes:

- **Drop `live_commands`** from the baseline DDL (superseded by ADR-3; file-IPC stays).
- **`job_runs` table** (NEW): one row per scrape/refresh/ingest/backup execution —
  `(job_run_id, job_name, args_json, started_at, finished_at, broker_time_anchor,
  exit_code, rows_written, coverage_json, error_text)`. Hook points already exist and are
  structured: `scraper.update_symbol`'s per-symbol result dict, `refresh_signal_daily`'s
  `SignalRefreshReport`, `check_data_freshness`'s status dict.
- **Freshness becomes a registry query** over `(store, broker, symbol, coverage_end)`
  manifest rows instead of re-reading parquet footers — and trivially extends beyond the
  current 5-symbol bars-only tripwire to ticks, rollover windows, ohlc_data, broker_cache,
  and the catalog. Ages computed against `rollover_market.broker_now()` (host clock is
  untrusted).
- **`blob_manifest` rows carry `broker`** (ADR-7) and the `timezone` stamp (ADR-2).
- Wire the reconciler's `provenance`/`conflicts` writes into the real write paths (the
  records and semantics already exist and are tested; only call sites are missing).
- `registry report` emits the store/path/key/count table → [[data_store_inventory]]
  becomes generated output (kills its drift problem).
- Durability posture: WAL + `PRAGMA synchronous=NORMAL`. A hard crash losing
  un-checkpointed transactions is healed by re-running the idempotent ingests (ADR-3) or
  `rebuild` (derived tables).
- Backups per ADR-10: `registry backup` nightly (SQLite backup API → `data/backups/`,
  rotation), plus a one-time scripted mirror of `data/norgate/archive` + `data/ohlc_data`
  off-box before the subscription lapses.

**Verify:** `rebuild` idempotency + `foreign_key_check == ∅` CI test (baseline);
`manifest == scan` reconciliation check; freshness query result matches the legacy
footer-read check on the same day.

## 5. Workstream C — Nautilus catalog promotion (ADR-1)

*Order matters: instruments are fixed before any series is materialized, because catalog
instrument rows are append-only data files with no in-place update.*

1. **Fix `data/instruments/catalog.parquet` first**: NDX/SP500 `price_increment` 0.01 →
   0.1 (the genuine errors; USDJPY is already correct at 0.001) using the live-probed
   POINT values scattered across experiments — single source for ADR-8. Any *later*
   instrument change requires deleting + rewriting the instrument rows in the Nautilus
   catalog (and re-ingesting affected series only if `price_precision` shrinks).
2. Materialize the persistent catalog once: M1 bars + synth/real QuoteTicks + instruments
   via the existing `ingest.py` functions and `write_instruments_to_catalog` (the proven
   single instrument-seeding path). **Pin the catalog class**: the PyArrow-backed
   `nautilus_trader.persistence.catalog.ParquetDataCatalog` — our `@customdataclass`
   types (ResearchCandle, planned forecast/spread/econ series) are incompatible with the
   PyO3 catalog.
3. **Post-scrape incremental append** job. Append safety comes from strictly-forward
   coverage (the scraper resumes after the last stored bar; the manifest knows the
   catalog's `coverage_end`) — **never** pass `skip_disjoint_check=True` in the scheduled
   job. Scheduled `consolidate_data_by_period(1 day)` is anti-fragmentation only, and runs
   *before* the registry's catalog scan so manifest rows don't dangle.
   **Repair primitive** for corrected/backfilled provider data (D1 rewrite, Norgate
   rebuilds, scraper overlap-merge branches): `catalog.delete_data_range(data_cls,
   identifier, start, end)` for the affected window, then re-ingest from provider parquet
   — triggered when the registry's source-mtime staleness check (item 7) detects a
   rewritten provider span.
4. Research lanes read the catalog instead of re-ingesting: `research/portfolio/pnl/
   nautilus_engine.py` first (it builds a tempdir catalog *per backtest call* today, and
   already accepts `catalog_path` — this is wiring, not building), then the
   experiment-toolkit Nautilus lane. Migrate lanes to declarative
   `BacktestDataConfig`/`BacktestRunConfig` where the data lives in the catalog; keep
   low-level `engine.add_data` only for in-memory synthetic streams (validation lane's
   synth quotes).
5. Promote `_read_partitions`/`_filter_by_time`/`_resolve_instrument` in
   `data_platform/nautilus/ingest.py` to public API — experiments already import the
   underscore names, so the module's public boundary is wrong for what lanes need
   (coverage exists in `tests/data_platform/test_nautilus_intraday_ingest.py`; extend it).
6. Model derived/research series as `@customdataclass` Data in the same catalog (the
   byte-exact `ResearchCandle` pattern is proven): forecasts, spread series, econ series.
7. Registry records every catalog ingest (function, symbols, span, source mtimes,
   nautilus version, precision mode) so catalog staleness vs provider parquet is
   detectable; catalog rows in `blob_manifest` map to the `{start}_{end}.parquet`
   interval files and are refreshed after each consolidation.

**Verify:** existing `test_candles_equivalence` byte-parity stays green; one realistic-lane
backtest produces identical results reading from the persistent catalog vs the old
tempdir path; ingest → append → consolidate → repair round-trip test.

## 6. Workstream D — research registry, lineage, and the experiment toolkit

**Registry (modifies the baseline `specs/runs/results` DDL — see Appendix):**

- `runs.kind` **replaces** the baseline's `phase` column (`exploration|validation|
  portfolio|final_validation|experiment`). SQLite cannot alter a CHECK constraint, so this
  is a table-rebuild migration, not an ALTER — acceptable because the runs table ships
  empty and is backfilled.
- `runs` gains: **`spec_snapshot_json` + `spec_hash`** (immutable copy of the launch
  payload — the single biggest lineage fix), environment provenance
  (`research_feed_used`, EWSD blend, realistic phases, `git_sha`), `log_path` (full
  stdout moves OUT of `_runs_index.json`), and a `headline_metrics_json` column (sharpe,
  t_stat, nw_sharpe, dsr, gate_passed) extracted once at completion so cross-run
  comparison stops re-parsing CSVs.
- `specs` get content-hash versioning (edits create versions; runs pin `spec_hash`).
- **Close the gate-identity hole:** vault-save eligibility resolves the gate report from
  the *specific* validation run's per-run dir (registry lookup), not latest-wins manifest
  matching of the canonical shared folder (legacy fallback only).
- **Promotion writes lineage:** `vault_entries.producing_run_id` + `gate_report_path` at
  commit time, and `run_id`/`spec_hash` keys added INTO the vault feature JSON
  (`hierarchy_spec.py` reads keys tolerantly — backward compatible).
- **Persist what evaporates:** `PortfolioWorkspaceJobManager` gets the same registry +
  per-run-dir treatment as `SpecRunManager` (today jobs vanish on restart and overwrite
  shared canonical dirs); `scripts/validate_candidate.py` writes its `ValidationResult`
  (metrics JSON, equity parquet, fills/positions parquet) into a registered run dir — it
  is the promotion-deciding artifact and today leaves zero record.
- **Cutover protocol for `_runs_index.json`** (the live writer rewrites the whole index
  on every status change and force-fails in-flight runs on restart): migrate only with
  the frontend server stopped and zero queued/running runs; any run found
  queued/running imports as `status='failed'` (matching `_load()` semantics); after
  cutover **the registry is the sole runs store** (`SpecRunManager` persists to it) and
  `_runs_index.json` is frozen as the schema-version-0 rebuild source.

**Experiment toolkit (`research/toolkit/`)** — *no registry dependency; starts in
parallel with M1/M2.* Promote the best existing copies, function-level:

| Toolkit module | Promoted from / built on |
|---|---|
| `bars.py` — M1 loader / de-stale / resample / bars-per-year | `vault_intraday/data_io.py` (cleanest copy, has caching) |
| `sessions.py` — broker/ET session windows, `_bsec` constants | `lafo_kama_mr/engine.py` |
| `metrics.py` — daily-grid Sharpe/PF/maxDD/by-year **for the vectorized (non-Nautilus) POC lanes** | `lafo`/`gold_digger` (≈20 inline copies today) |
| `costs.py` — recorded-spread cost model | POINT + spread from registry (ADR-8) once it lands; falls back to the current probed constants until then |
| `verify.py` — two-way PnL lookahead assert | `lafo.assert_no_lookahead` |
| `sizing.py` — vol-target size-at-entry overlay | `lafo.apply_vol_target` |
| `nautilus_lane.py` — engine+venue+FillModel scaffold, synth-quote ingest, POC-reconciliation report (corr > 0.99 pass rule) | extracted from `orb_ibs`/`peter` lanes. **The trade ledger comes from `engine.trader.generate_order_fills_report()` / `generate_positions_report()`** (per-fill and per-position DataFrames with realized PnL, avg prices, commissions, timestamps — Nautilus already ships this); headline stats can register as custom `PortfolioStatistic`s on the trader's `PortfolioAnalyzer`. A custom recorder mixin survives only for data Nautilus genuinely lacks. |

A new experiment then supplies only its `on_bar` signal logic (~150–300 lines — the
genuinely novel part). Experiments register in `runs` with `kind='experiment'`, slug,
FINDINGS.md path, and an outputs manifest, so agent work stops being invisible.
`.claude/skills/research-infra/SKILL.md` is updated to mandate toolkit + registry reader
as the data-access path (highest-leverage place to change agent behavior).

**Verify:** toolkit reproduces one published experiment's headline numbers exactly
(e.g. ORB net Sharpe from FINDINGS.md); a spec-driven run round-trips
spec → run → gate → vault with FKs intact; `_runs_index.json` backfill row-count parity.

## 7. Workstream E — live trading records (most perishable; ships early)

**Node-side (small, append-only, off the critical path):**

1. `deals.jsonl` — the exec client's existing deal-polling loop additionally appends every
   new deal with ALL fields (`ticket, order, position_id, time, type, entry, volume,
   price, commission, swap, fee, profit, symbol, comment, magic`). Today the emitted
   Nautilus fill keeps price/volume/commission/time/tickets and discards
   swap/profit/fee/entry/position_id.
2. Submit-side capture is **two writes joined by `client_order_id`**: the strategy-side
   `_record_slip_submit` (already has canonical/side/qty/bid/ask/broker_time) gains
   intent (EXIT|ENTRY window, target_fraction); the `order_send` response
   (`retcode, result.price, result.bid/ask`) is only observable inside the vendored exec
   client, so it appends a small `submit_results.jsonl` there — currently those fields
   are received and discarded.
3. **Comment-key normalization**: the plain `comment == client_order_id` join breaks on
   two paths — dashboard flatten stamps `'vault-flatten'` and the exec client's
   close-position path stamps `'close:{client_order_id}'`. Ingest strips the `close:`
   prefix and assigns synthetic order ids for flatten deals so every deal attributes.
4. `forecasts.jsonl` — write once per decision at `_open_entry_window` (and
   `manual_trade` compute-targets): `(as_of, canonical, forecast_score, target_fraction,
   target_qty, engine_config_hash, warmup_ready)`. **The single biggest observability gap;
   ~10 lines at an existing call site.**
5. `equity.jsonl` — durable uncompacted copy of the slow-tick equity sample
   (`balance, equity, floating_pnl, gross_notional, marks_fresh`); the capped dashboard
   JSONL stays as-is for the UI.

**Registry-side:**

6. `registry ingest live` (manual until M3, then scheduled): full-file re-read of the
   JSONLs, upsert into `orders/deals/forecast_history/equity_snapshots`, attributing rows
   to `account_id` via `(broker, login)` from the snapshot. Slippage
   (`spread_bps, slip_vs_touch_bps, slip_vs_mid_bps`) becomes a **SQL view** joining
   deals↔orders on `(account_id, client_order_id)` — LEFT JOIN from deals so no-submit
   deals are kept (matching today's `track_slippage` semantics, including the
   mid-denominator convention); `slippage.csv` becomes a derived artifact.
7. **Backfill with explicit attribution** (the legacy stores are broker-keyed, not
   login-keyed): (1) operator seeds `accounts` (with validity windows) via
   `registry accounts add` *before* backfill; (2) legacy rows attribute by broker + row
   timestamp ∈ validity window; (3) rows matching no window land on an explicit
   per-broker `legacy-unknown` account row (`status='retired'`) rather than failing or
   guessing. Sources: `slippage/slippage.csv`, `submits.jsonl`, `equity_history.jsonl`,
   and the legacy `logs/cfd_prop_audit/*.json` (it already contains the same entities and
   proves the join keys).
8. Catch-up CLI for node-down gaps: read-only terminal attach in a quiet window
   (manual_trade pattern), explicitly NOT the primary path (ADR-4).
9. `accounts` lifecycle CLI: `registry accounts add|rotate|retire` — `rotate` creates the
   successor row (predecessor FK), archives `live_state/` + `slippage/` into an
   account-keyed folder, resets `baseline.json`, and prompts the `.env` edit. Also set a
   per-broker `trader_id` in the node config so client order ids are globally
   distinguishable going forward (today both nodes default to `TRADER-001`).
10. Cost surface (ADR-8): `registry costs --refresh` aggregates M1 `spread` +
    rollover-window ticks + realized slippage into `cost_observations`; the toolkit cost
    model and validation lanes consume it.
11. Cleanup: delete orphaned `deployment/live/runtime/decision_state.py` **plus its unit
    test** (`tests/unit-tests/deployment/live_nautilus/test_decision_state.py`) and the
    stale references (deployment/live/README.md lines 63-64/98/101, the
    `live_state.py:39` docstring); remove `ftmo/decision_state.json` from disk.

*Deferred (reviewed out of v1 scope for a single operator): a `command_audit` table —
command/halt history beyond `last_command_result` has near-zero query value while one
person issues all commands; `halt.json` is already durable. Revisit if multi-operator.*

**Verify:** a full demo rollover cycle (EXIT → ENTRY) lands orders+deals+forecast rows
with FKs intact and the slippage view reconciles with `track_slippage.py` output on the
same window (row counts will match only with the LEFT-JOIN semantics above);
account-rotate leaves the old account's history queryable and the new account's gauges
clean.

## 8. Workstream F — automation & scheduling (Windows)

1. **One declarative registrar** (`deployment/ops/register_tasks.ps1`): a task table
   (name, trigger, wrapper, machine-role `dev-box|vps`) merging the two existing
   installers' best parts (Kind=Local boundary + `$PSScriptRoot` root resolution +
   `StartWhenAvailable` from `setup_scheduled_task.ps1`; `MultipleInstances IgnoreNew` +
   optional `-Credential` from `install_cfd_prop_tasks.ps1`). Default to a
   **non-elevated current-user principal** — nothing in the chain needs admin, and
   re-registering after edits must never hit a UAC wall. Delete the stale
   `setup_scheduled_task.bat`, the `C:\Users\adabla` forecast bats (pending the IB
   decision, §14), and fix the hardcoded path in `run_rollover_tick_scrape.bat`.
2. **At-logon scrape** (the user requirement): `-AtLogOn` trigger + daily catch-up
   trigger, both running one **chained wrapper** that grows with the milestones —
   at M3: MT5 scrape → freshness check (registry-backed) → Darwinex signal-cache
   `refresh_signal_daily` → `registry ingest` (job_runs + manifest + live);
   appended at M5: catalog incremental append + consolidation.
   The chain starts with a **clock-drift check** (host now vs
   `rollover_market.broker_now()`, recorded in `job_runs.broker_time_anchor`,
   Telegram-alert past a threshold) because the daily trigger fires on the untrusted
   host clock.
3. **Terminal-contention resolution (required before M3):** the scrape must not race the
   armed Darwinex node for a terminal. Default: **scrape from the dedicated scraper
   terminal/login** (the Darwinex tick-scraper terminal is already a separate,
   do-not-disturb install; the `accounts.program_phase='scraper'` row models it). If the
   configured paths ever collide, first settle read-only second-attach empirically with a
   gated live test (`tests/live_mt5` pattern) before scheduling. Also define
   launch-vs-wait semantics: `mt5.initialize(path)` LAUNCHES a terminal if none is
   running — the wrapper must wait/retry for the autostarted, logged-in terminal rather
   than spawning a second un-logged-in instance.
4. Rollover tick scrape and the freshness tripwire keep their independent schedules (the
   tripwire's independence is deliberate — it detects a broken schedule).
5. **Unified alerting** on `lib/core/notify.TelegramNotifier` (give the PS wrappers a tiny
   `python -m` entrypoint); document the three env-var families in one place; **finish the
   post-leak token rotation before wiring more automation**.
6. Nightly `registry backup` task (ADR-10).
7. Norgate: freeze as archive store post-subscription; script the documented
   archive→working→`migrate_all` fallback as a runnable command; no scheduled Norgate job.

**Verify:** reboot test — logon fires the chain, `job_runs` shows every step with exit
codes, freshness query goes green, and a deliberately-broken step produces a Telegram
alert.

## 9. Workstream G — agent interface

In priority order (ADR-9):

1. `data_platform/registry/reader.py` — typed read-only Python API (the same import path
   research code uses; `file:...?mode=ro`).
2. **`registry` CLI** — `coverage`, `freshness`, `runs`, `lineage`, `accounts`, `fills`,
   `costs`, `report`, `rebuild`, `backup`, `ingest` subcommands; allowlisted in
   `.claude/settings.json`. Matches the Bash-first agent pattern and the graphify-CLI
   precedent.
3. `/api/data/*` FastAPI routers for UI+agent parity (coverage, accounts, fills,
   forecasts, job status).
4. Repo-level `.mcp.json` for codegraph (today user-level only — a fresh clone gets no
   agent tooling); an MCP server over the registry only if call volume later justifies it.
5. Guardrails: keep the human-gated vault write exactly as-is; extend the existing
   PreToolUse Bash nudge hook to warn on `to_parquet` targeting `data/` outside
   `data_platform/storage`; the honest enforcement is detection — the
   `manifest == scan` CI reconciliation catches bypass writes.
6. Update `.claude/skills/research-infra/SKILL.md` (toolkit + registry mandated); archive
   the superseded `/research` orchestrator skill + its 3 subagents to stop convention
   drift.

## 10. Workstream H — UI / dashboard

For a single operator the `registry` CLI answers most questions; UI ships **one** page in
v1 and defers the rest:

| Page | Backed by | v1? |
|---|---|---|
| **Data Coverage** — store × symbol × coverage_end heat-grid, freshness, job_runs history | `blob_manifest`, `job_runs` | **yes** |
| Accounts & Fills — account cards, orders/deals tables, slippage distributions | `accounts`, `orders`, `deals`, slippage view | deferred (CLI covers it) |
| Forecast History — targets per day per account vs fills | `forecast_history` ⋈ `deals` | deferred |
| Costs — spread/slippage surfaces per instrument/broker/ToD | `cost_observations` | deferred |
| LiveMonitor (existing) gains account identity + durable equity | `equity_snapshots` | small add-on |

The live dashboard's existing file-IPC (snapshot/command/halt) is untouched — UI reads
registry for history, files for real-time.

## 11. Workstream I — docs & cleanup

**Ships immediately (any time from M0):** the stale-fix PR — `utils/` references in
`docs/SaaS/position_sizing.md` + `data_platform/README.md`; IB mention in
`data_platform/core/README.md`; 53-vs-69 instrument count; `[[deployment_live]]` +
`[[multi_source_update_architecture]]` broken wikilinks; git-add the untracked
`deployment/live/README.md` + `frontend/README.md` — plus a `docs/index.md` map (called
for by WP-7, never created).

**Ships after the architecture lands (M7):**

1. Adopt the NautilusTrader shape: `getting_started/` + `concepts/` (data model, storage &
   registry, feeds & adjustment, timezones, pipeline/ensemble, vault, research lifecycle,
   live runtime) + `integrations/` (norgate, mt5_darwinex, nautilus_catalog, ib tombstone)
   + `operations/` (scraper/scheduler runbooks, live arming, accounts rotation, backups) +
   `decisions/` (ADR-numbered; this plan's §1 seeds it).
2. **Single-home rule:** package README = operational truth for that package; `docs/` =
   concepts/decisions/workflows; central docs link to READMEs instead of restating (the
   data layout is currently described in ≥3 places).
3. Promote `docs/SaaS/robustness_tests/` (the declared source of truth for the research
   phase model) out of the SaaS namespace → `concepts/research_lifecycle/`; archive the
   rest of SaaS + `docs/refactor/nautilus/` once the migration lands.
4. Keep the WP-7 conventions that worked: "Verified against commit <sha>" footers, docs
   ship with the code change; add a wikilink/relative-link checker (broken links were the
   most common defect).
5. [[data_store_inventory]] becomes `registry report` output.
6. New operational docs the migration creates: the research-run registry contract and the
   live-state file contract (JSONL shapes + ingest semantics).

## 12. Sequencing

Ordered so each milestone is shippable and the most perishable data is captured earliest.
The experiment toolkit runs in parallel — it has no registry dependency.

| # | Milestone | Contents | Depends on | ~Effort |
|---|---|---|---|---|
| M0 | **Foundation & fixes** | Workstream A (contracts, single writer, D1 rewrite, broker guard, IB-ghost resolution, **retention probe**, housekeeping) | — | 1 wk |
| — | **Toolkit (parallel)** | Workstream D toolkit half (`research/toolkit/`, costs.py on fallback constants) | — | 1 wk, overlaps M0–M2 |
| M1 | **Registry core** | Workstream B (schema, rebuild, job_runs, freshness, backups) | M0 | 1 wk |
| M2 | **Live records** | Workstream E (node JSONLs, ingest, accounts, backfill; verify is calendar-gated on real rollover cycles) | M1 | 1–1.5 wk |
| M3 | **Automation** | Workstream F (registrar, at-logon chain v1, terminal-contention resolution, alerting, backup task) | M1 (M2 for the live-ingest step) | 3–5 d |
| M4 | **Research registry** | Workstream D registry half (runs/specs/lineage, cutover protocol, portfolio + validate_candidate persistence) | M1 | 1–1.5 wk |
| M5 | **Catalog promotion** | Workstream C; extends the M3 chain with append + consolidation; cost surface v1 (ADR-8) | M1 | 1 wk |
| M6 | **Agent surfaces + UI** | Workstream G + the Data Coverage page (rest of H deferred) | M2, M4 | 3–5 d |
| M7 | **Docs reorg** | Workstream I part 2 (stale-fix PR ships at M0) | all | 3–5 d |

**Critical path (M0 → M3 + toolkit): ~3–4 weeks part-time.** Full scope ≈ 6–8 weeks
calendar with agent fan-out per milestone. Each milestone ends with its **Verify** block
green and a docs delta.

## 13. Risks & invariants

- **Rebuildable invariant (unchanged)** for all derived tables: `registry rebuild`
  recomputes them from files; CI asserts idempotency + clean FK check. **Amended (ADR-10):**
  live-record tables are excluded from the invariant but remain re-ingestable from the
  retained JSONLs; backups + forward-only migrations cover the rest.
- **Single-writer discipline** remains the main behavioral cost; mitigated by detection
  (reconcile check) not prevention. Never `rebuild` during live trading; live nodes never
  write the DB at all (ADR-3).
- **Parity harness for the migration itself:** frozen-hash comparison of `load_data`
  output per (ticker, tf, feed, adjustment) before/after schema unification and registry
  cutover; slippage.csv vs derived-view reconciliation; catalog-vs-tempdir backtest
  parity. No cutover without its parity test green.
- **Topology assumption:** single dev box. The cfd_prop VPS installer suggests a second
  machine may exist/return; if so, remote writers ship JSONLs for central ingest (or go
  through the FastAPI) — **never** file-sync SQLite. Note: catalog files are **not
  portable across Nautilus precision modes** (Windows standard vs Linux high-precision) —
  another reason the catalog stays a per-machine derived artifact; the registry records
  nautilus version + precision mode per ingest.
- **Nautilus version upgrades** can change catalog schemas → catalog stays wipe-and-
  re-ingest.
- **Token rotation** (post-leak) is a precondition for wiring more Telegram automation.

## 14. Open decisions (defaults chosen; flip any of these)

1. **IB / "personal" scope** — default: register the IB account in `accounts` for
   identity/equity tracking, but **retire** the IB forecast pipeline (stale adabla-path
   bats, TWS dependency) and delete the `ib` source slots. Alternative: re-home a thin IB
   adapter for post-Norgate futures forward-extension.
2. **Deployment topology** — default: single dev box hosts registry + nodes + scheduler.
   If the FTMO VPS (or a funded-account VPS) is active or planned, say so — it changes the
   ingest transport (per-machine JSONL ship → central ingest), not the schema.
3. **Tick-store policy** — default: stay rollover-window-only + demand-driven tick cache;
   cost surface uses M1 spread + rollover ticks. Alternative: expand to full-session tick
   capture for richer intraday cost modeling (storage + scrape-time cost).
4. **Forecast-server / enigma EWSD paths** — default: capture-or-retire decided per path at
   M2 (cfd_prop audit JSONs are ingested either way); the old `forecast_server.py` is a
   candidate for retirement rather than registry integration.
5. **True-UTC catalog tier** — ADR-2 keeps broker wallclock everywhere today. If we later
   want the *catalog* tier in true UTC, that is a separate ADR with its own parity
   re-baseline; not in this plan's scope.

---

## Appendix — net-new DDL (modifies the baseline plan's Appendix A)

Baseline deltas: **drop `live_commands`** (ADR-3); `runs.phase` → `runs.kind` via table
rebuild (§6); `vault_entries` + `gate_report_path`, `promoted_by`.

```sql
-- ACCOUNTS (ADR-5) — NOT rebuildable; schema_version + migrations + backup
CREATE TABLE accounts (
  account_id        INTEGER PRIMARY KEY,
  broker            TEXT NOT NULL,                  -- 'darwinex','ftmo','fundednext',...
  venue             TEXT NOT NULL DEFAULT 'MT5' CHECK (venue IN ('MT5','IB')),
  login             TEXT NOT NULL,
  server            TEXT NOT NULL,
  exec_tier         TEXT NOT NULL CHECK (exec_tier IN ('sandbox','demo','live')),
  program_phase     TEXT NOT NULL CHECK (program_phase IN
                      ('challenge','verification','funded','retail','personal','scraper')),
  currency          TEXT NOT NULL DEFAULT 'USD',
  initial_balance   REAL,
  magic_number      INTEGER,
  trader_id         TEXT,                            -- per-broker Nautilus trader id (§7.9)
  risk_rules_json   TEXT,                            -- overrides mt5_brokers.yaml defaults
  status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN
                      ('active','retired','breached','passed')),
  valid_from        TEXT NOT NULL,
  valid_to          TEXT,
  predecessor_account_id INTEGER REFERENCES accounts(account_id),  -- challenge→funded chain
  notes             TEXT,
  UNIQUE (broker, login, valid_from)                 -- brokers reuse logins on demo resets
);

-- ORDERS (submit-time intent + venue response; account-scoped — client order ids are
-- NOT globally unique across nodes until trader_id is set per broker)
CREATE TABLE orders (
  id               INTEGER PRIMARY KEY,
  account_id       INTEGER NOT NULL REFERENCES accounts(account_id),
  client_order_id  TEXT NOT NULL,                    -- == MT5 comment (≤31 chars);
                                                     -- synthetic for flatten deals (§7.3)
  instrument_id    TEXT REFERENCES instruments(id),
  canonical        TEXT NOT NULL, symbol TEXT NOT NULL,
  side             TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  qty              REAL NOT NULL,
  intent           TEXT CHECK (intent IN ('EXIT','ENTRY','MANUAL','FLATTEN')),
  target_fraction  REAL,
  bid_at_submit    REAL, ask_at_submit REAL,
  broker_time_submit TEXT NOT NULL,
  retcode          INTEGER, result_price REAL, result_deal INTEGER,
  source           TEXT NOT NULL DEFAULT 'node',     -- 'node','manual_trade','legacy_audit'
  UNIQUE (account_id, client_order_id)
);

-- DEALS (full MT5 deal record; ground truth for fills). Tickets are per-server
-- sequences — NEVER unique across accounts.
CREATE TABLE deals (
  id            INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL REFERENCES accounts(account_id),
  ticket        INTEGER NOT NULL,                    -- MT5 deal ticket
  order_ticket  INTEGER, position_id INTEGER,
  client_order_id TEXT,                              -- deal.comment, normalized (§7.3)
  symbol        TEXT NOT NULL, canonical TEXT,
  deal_type     INTEGER NOT NULL,                    -- mt5 DEAL_TYPE_* (BUY=0, SELL=1)
  entry         TEXT CHECK (entry IN ('IN','OUT','INOUT','OUT_BY')),
  volume        REAL NOT NULL, price REAL NOT NULL,
  commission    REAL NOT NULL DEFAULT 0, swap REAL NOT NULL DEFAULT 0,
  fee           REAL NOT NULL DEFAULT 0, profit REAL NOT NULL DEFAULT 0,
  broker_time   TEXT NOT NULL, magic INTEGER,
  UNIQUE (account_id, ticket)
);
CREATE INDEX ix_deals_acct_time ON deals(account_id, broker_time);

-- derived, not stored: slippage view. LEFT JOIN keeps no-submit deals (matches
-- track_slippage semantics); denominators use MID per slippage.py convention.
CREATE VIEW v_slippage AS
  SELECT d.account_id, d.ticket, d.canonical, d.entry, d.volume, d.price AS fill_px,
         o.bid_at_submit, o.ask_at_submit,
         (o.bid_at_submit + o.ask_at_submit) / 2.0 AS mid,
         (o.ask_at_submit - o.bid_at_submit)
           / NULLIF((o.bid_at_submit + o.ask_at_submit) / 2.0, 0) * 10000 AS spread_bps,
         CASE WHEN d.deal_type = 0  -- BUY (incl. OUT deals closing shorts)
              THEN (d.price - o.ask_at_submit)
              ELSE (o.bid_at_submit - d.price)
         END / NULLIF((o.bid_at_submit + o.ask_at_submit) / 2.0, 0) * 10000
           AS slip_vs_touch_bps,
         CASE WHEN d.deal_type = 0
              THEN (d.price - (o.bid_at_submit + o.ask_at_submit) / 2.0)
              ELSE ((o.bid_at_submit + o.ask_at_submit) / 2.0 - d.price)
         END / NULLIF((o.bid_at_submit + o.ask_at_submit) / 2.0, 0) * 10000
           AS slip_vs_mid_bps,
         d.commission, d.swap, d.broker_time
  FROM deals d
  LEFT JOIN orders o
    ON o.account_id = d.account_id AND o.client_order_id = d.client_order_id
  WHERE d.deal_type IN (0, 1);

-- FORECAST HISTORY (one row per decision per instrument; today this evaporates)
CREATE TABLE forecast_history (
  id INTEGER PRIMARY KEY,
  account_id   INTEGER NOT NULL REFERENCES accounts(account_id),
  as_of        TEXT NOT NULL,                        -- broker date of the decision
  canonical    TEXT NOT NULL,
  forecast_score REAL, target_fraction REAL, target_qty REAL,
  engine_config_hash TEXT, vault_root TEXT,
  warmup_ready INTEGER NOT NULL CHECK (warmup_ready IN (0,1)),
  source       TEXT NOT NULL DEFAULT 'node',
  UNIQUE (account_id, as_of, canonical, source)
);

-- EQUITY SNAPSHOTS (durable, uncompacted; the JSONL stays as the dashboard cache)
CREATE TABLE equity_snapshots (
  id INTEGER PRIMARY KEY,
  account_id  INTEGER NOT NULL REFERENCES accounts(account_id),
  ts          TEXT NOT NULL,
  balance REAL, equity REAL, floating_pnl REAL, gross_notional REAL,
  marks_fresh INTEGER CHECK (marks_fresh IN (0,1)),
  UNIQUE (account_id, ts)
);

-- JOB RUNS (every scrape/refresh/ingest/backup execution)
CREATE TABLE job_runs (
  job_run_id  INTEGER PRIMARY KEY,
  job_name    TEXT NOT NULL,                          -- 'mt5_scrape','signal_refresh',...
  args_json   TEXT, started_at TEXT NOT NULL, finished_at TEXT,
  broker_time_anchor TEXT,                            -- broker_now() at start (host untrusted)
  exit_code   INTEGER, rows_written INTEGER, coverage_json TEXT, error_text TEXT
);
CREATE INDEX ix_job_runs_name_time ON job_runs(job_name, started_at);

-- COST SURFACE (ADR-8; derived/rebuildable; refreshed on demand)
CREATE TABLE cost_observations (
  id INTEGER PRIMARY KEY,
  instrument_id TEXT NOT NULL REFERENCES instruments(id),
  broker        TEXT NOT NULL,
  tod_bucket    TEXT NOT NULL,                        -- e.g. '09:30-10:00 ET'
  window_start  DATE NOT NULL, window_end DATE NOT NULL,
  spread_points_p25 REAL, spread_points_p50 REAL, spread_points_p75 REAL,
  spread_bps_p50 REAL, slip_vs_touch_bps_p50 REAL, slip_vs_touch_bps_p90 REAL,
  n_spread_obs INTEGER, n_fill_obs INTEGER, computed_at TEXT NOT NULL,
  UNIQUE (instrument_id, broker, tod_bucket, window_start, window_end)
);

-- RUNS table (REPLACES the baseline definition via table rebuild — SQLite cannot
-- alter the baseline's CHECK(phase IN ('exploration','validation')))
-- kind TEXT NOT NULL CHECK (kind IN ('exploration','validation','portfolio',
--                                    'final_validation','experiment'))  -- replaces phase
-- + spec_snapshot_json TEXT, spec_hash TEXT
-- + research_feed_used TEXT, ewsd_blend TEXT, realistic_phases TEXT, git_sha TEXT
-- + log_path TEXT, headline_metrics_json TEXT
-- vault_entries: + gate_report_path TEXT, promoted_by TEXT  (producing_run_id exists)
```

> Survey provenance: 7-agent subsystem sweep + completeness critic + 3-lens adversarial
> review, 2026-06-09, working tree at branch `mt5_data` (post-`c3bd8a4` Nautilus
> refactor). Key stale-memory corrections found along the way: the Nautilus 1.227
> fill-emission bug is FIXED in the working tree; `scripts/fetch_mt5_data.py` no longer
> exists (scrape entrypoint is `python -m data_platform.providers.mt5.scraper`); zero
> scheduled tasks are currently registered on this machine; `ingest.py` does have test
> coverage (`tests/data_platform/test_nautilus_intraday_ingest.py`).
