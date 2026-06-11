-- ============================================================================
-- data/registry.db — schema v1
--
-- The queryable metadata index over three domains: market data, research
-- runs, live trading records. Payloads (bars, ticks, research CSVs, equity
-- curves) stay parquet/JSONL files — THE DB NEVER OWNS A PRICE ROW.
--
-- Sources of record for this DDL:
--   docs/library/Data/storage_and_registry_plan.md  (baseline, Appendix A)
--   docs/library/Data/data_platform_migration_plan.md (extension + appendix)
-- Amendments applied relative to the baseline:
--   * live_commands DROPPED — ADR-3 supersedes (command.json file-IPC stays;
--     live nodes never touch this DB).
--   * 'ib' removed from every source CHECK — the IB data pipeline is retired
--     (M0.6); do not re-add without an adapter behind it.
--   * runs.kind replaces the baseline's runs.phase (wider lifecycle).
--   * blob_manifest carries broker (ADR-7) and timezone (ADR-2) columns.
--
-- Rebuildability (ADR-10): every table in the MARKET DATA and RESEARCH
-- domains is a pure function of files on disk — `registry rebuild`
-- recomputes them. LIVE-domain tables (accounts, orders, deals,
-- forecast_history, equity_snapshots) are re-ingestable from the retained
-- broker_cache JSONLs but NOT from-scratch rebuildable once broker history
-- expires — they are covered by `registry backup` + forward-only migrations
-- gated on PRAGMA user_version.
--
-- Connection discipline (db.py): WAL, synchronous=NORMAL, foreign_keys=ON
-- (per-connection), single writer module per domain. Never `rebuild` while
-- a live node is trading.
-- ============================================================================

PRAGMA user_version = 1;

-- ============================================================================
-- DOMAIN: MARKET DATA (rebuildable from files)
-- ============================================================================

-- Master instrument registry (grounds: data_platform/core/catalog.py rows).
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
  activation       DATE,
  expiration       DATE,
  data_source      TEXT CHECK (data_source IN ('norgate','mt5') OR data_source IS NULL),
  info_json        TEXT NOT NULL DEFAULT '{}'        -- residual provider blob (round-trip)
);

-- Normalizes info['source_symbols'] out of the JSON blob (kills the O(n)
-- reverse scan; ENFORCES the previously-assumed uniqueness invariant).
CREATE TABLE instrument_source_symbols (
  instrument_id TEXT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
  source        TEXT NOT NULL,                       -- 'norgate_adj','mt5','etf_proxy',...
  native_symbol TEXT NOT NULL,
  PRIMARY KEY (instrument_id, source),
  UNIQUE (source, native_symbol)
);

-- Append-only ingest lineage (grounds: data_platform/core/provenance.py).
CREATE TABLE provenance (
  id            INTEGER PRIMARY KEY,
  instrument_id TEXT NOT NULL REFERENCES instruments(id),
  resolution    TEXT NOT NULL CHECK (resolution IN ('D','W','M','M1')),
  source        TEXT NOT NULL CHECK (source IN ('norgate','mt5','derived_from_daily')),
  ts_ingest     TEXT NOT NULL,
  ratio_applied INTEGER NOT NULL CHECK (ratio_applied IN (0,1)),
  ratio_value   REAL,
  rows_written  INTEGER NOT NULL CHECK (rows_written >= 0)
);
CREATE INDEX ix_prov_inst_ts ON provenance(instrument_id, ts_ingest);

-- Cross-source disagreement log (grounds: reconciler.py ConflictRecord).
CREATE TABLE conflicts (
  id            INTEGER PRIMARY KEY,
  instrument_id TEXT NOT NULL REFERENCES instruments(id),
  bar_date      DATE NOT NULL,
  source_a      TEXT NOT NULL,
  source_b      TEXT NOT NULL,
  close_a       REAL NOT NULL,
  close_b       REAL NOT NULL,
  deviation     REAL NOT NULL,
  threshold     REAL NOT NULL                        -- stored per-row for interpretability
);

-- One row per payload file across ALL stores — the central index that
-- replaces directory globs, footer scans, and the dead .meta.json sidecars.
CREATE TABLE blob_manifest (
  id             INTEGER PRIMARY KEY,
  store          TEXT NOT NULL,    -- 'ohlc_data','stock_data','mt5_m1','mt5_d1','mt5_ticks',
                                   -- 'tick_cache','rollover_ticks','bias_artifact',
                                   -- 'candle_cache','econ','market_series','nautilus_catalog',...
  key_json       TEXT NOT NULL,    -- composite logical key as JSON (ticker/tf/adj/feed/params)
  broker         TEXT,             -- ADR-7: MT5 stores are broker-keyed ('darwinex' today)
  timezone       TEXT,             -- ADR-2: 'broker_eet_as_utc' for MT5 stores, NULL otherwise
  relative_path  TEXT NOT NULL UNIQUE,
  schema_hash    TEXT,
  engine         TEXT,             -- parquet engine the store is read with
  rows           INTEGER,
  coverage_start DATE,
  coverage_end   DATE,
  written_at     TEXT NOT NULL,
  valid          INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
  validation_error TEXT,
  UNIQUE (store, key_json)
);
CREATE INDEX ix_blob_store_cov ON blob_manifest(store, coverage_end);

-- Tick-chunk coverage (folds ticks_cache filename ranges + coverage JSON
-- into one indexed overlap query).
CREATE TABLE tick_chunks (
  id        INTEGER PRIMARY KEY,
  symbol    TEXT NOT NULL,
  start_ns  INTEGER NOT NULL,
  end_ns    INTEGER NOT NULL,
  file_path TEXT NOT NULL UNIQUE,
  n_ticks   INTEGER NOT NULL,
  CHECK (end_ns > start_ns)
);
CREATE INDEX ix_tick_overlap ON tick_chunks(symbol, start_ns, end_ns);

-- Calendar lookups (grounds: data/events/calendar/*.json).
CREATE TABLE fomc_decision (
  decision_date DATE PRIMARY KEY,
  scraped_at    TEXT
);
CREATE TABLE nyse_holiday (
  holiday_id   TEXT NOT NULL,
  closure_date DATE NOT NULL,
  d0           DATE NOT NULL,
  asset_bucket TEXT NOT NULL CHECK (asset_bucket IN ('equity','gold')),
  PRIMARY KEY (holiday_id, closure_date)
);

-- ============================================================================
-- DOMAIN: RESEARCH (rebuildable from files: specs JSON, run dirs, vault JSON)
-- ============================================================================

CREATE TABLE specs (
  id           TEXT PRIMARY KEY,                     -- stable id, decoupled from name
  name         TEXT NOT NULL,
  name_slug    TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,                        -- pins exactly-which-spec; edits = new version
  spec_version TEXT NOT NULL DEFAULT '1.0',
  valid        INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
  error        TEXT,
  spec_json    TEXT NOT NULL,                        -- full StrategySpec blob (round-trip)
  file_path    TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

-- runs.kind REPLACES the baseline's phase column (SQLite cannot widen a
-- CHECK in place; this table ships fresh and is backfilled from
-- _runs_index.json, where legacy 'phase' values map 1:1 into kind).
CREATE TABLE runs (
  run_id            TEXT PRIMARY KEY,                -- uuid4 == output folder name
  kind              TEXT NOT NULL CHECK (kind IN
                      ('exploration','validation','portfolio','final_validation','experiment')),
  spec_id           TEXT REFERENCES specs(id),
  spec_hash         TEXT,                            -- content hash of the spec AS LAUNCHED
  spec_snapshot_json TEXT,                           -- immutable copy of the launch payload
  status            TEXT NOT NULL CHECK (status IN ('queued','running','completed','failed')),
  -- environment provenance (set process-globally today, recorded nowhere):
  research_feed_used TEXT,                           -- 'futures_ratio' | 'cfd' | ...
  ewsd_blend        TEXT,
  realistic_phases  TEXT,
  git_sha           TEXT,
  num_combos        INTEGER,
  reports_dir       TEXT,
  viz_dir           TEXT,
  log_path          TEXT,                            -- stdout moves OUT of the index JSON
  headline_metrics_json TEXT,                        -- {sharpe, t_stat, nw_sharpe, dsr, gate_passed}
  created_at        TEXT NOT NULL,
  started_at        TEXT,
  finished_at       TEXT,
  error_text        TEXT
);
CREATE INDEX ix_runs_spec ON runs(spec_id);
CREATE INDEX ix_runs_kind_created ON runs(kind, created_at);

CREATE TABLE results (
  id            INTEGER PRIMARY KEY,
  run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  artifact_type TEXT NOT NULL,                       -- 'equity_curve','param_sensitivity',...
  file_path     TEXT NOT NULL,                       -- CSV/parquet body stays a file
  sharpe        REAL,
  t_stat        REAL,
  sortino       REAL,
  n_obs         INTEGER
);
CREATE INDEX ix_results_run ON results(run_id);

CREATE TABLE sleeves (name TEXT PRIMARY KEY);        -- subsumes custom_sleeves.json

CREATE TABLE vault_entries (
  id                     INTEGER PRIMARY KEY,
  vault_profile          TEXT NOT NULL CHECK (vault_profile IN ('prop','personal','cfd_prop')),
  timeframe              TEXT NOT NULL CHECK (timeframe IN ('D','W','M')),
  weight_hierarchy_group TEXT REFERENCES sleeves(name),
  ensemble_leaf          TEXT NOT NULL,
  feature_name           TEXT NOT NULL,
  feature_column         TEXT NOT NULL,
  module_name            TEXT NOT NULL,
  direction              TEXT,
  producing_run_id       TEXT REFERENCES runs(run_id),  -- the lineage FK the filesystem never had
  gate_report_path       TEXT,                          -- the SPECIFIC gate report that authorized
  promoted_by            TEXT,
  config_json            TEXT NOT NULL,
  file_path              TEXT NOT NULL,
  valid                  INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
  validation_error       TEXT,
  created_at             TEXT NOT NULL,
  updated_at             TEXT NOT NULL,
  UNIQUE (vault_profile, timeframe, ensemble_leaf, feature_name)
);

-- ============================================================================
-- DOMAIN: LIVE TRADING RECORDS (re-ingestable from JSONLs; backed up; NOT
-- from-scratch rebuildable once broker history expires)
-- ============================================================================

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
  trader_id         TEXT,                            -- per-broker Nautilus trader id (plan §7.9)
  risk_rules_json   TEXT,                            -- overrides mt5_brokers.yaml defaults
  status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN
                      ('active','retired','breached','passed')),
  valid_from        TEXT NOT NULL,
  valid_to          TEXT,
  predecessor_account_id INTEGER REFERENCES accounts(account_id),  -- challenge→funded chain
  notes             TEXT,
  UNIQUE (broker, login, valid_from)                 -- brokers reuse logins on demo resets
);

-- Submit-time intent + venue response. Account-scoped: client order ids are
-- NOT globally unique across nodes until trader_id is set per broker.
CREATE TABLE orders (
  id                 INTEGER PRIMARY KEY,
  account_id         INTEGER NOT NULL REFERENCES accounts(account_id),
  client_order_id    TEXT NOT NULL,                  -- == MT5 comment (≤31 chars);
                                                     -- synthetic for flatten deals (plan §7.3)
  instrument_id      TEXT REFERENCES instruments(id),
  canonical          TEXT NOT NULL,
  symbol             TEXT NOT NULL,
  side               TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  qty                REAL NOT NULL,
  intent             TEXT CHECK (intent IN ('EXIT','ENTRY','MANUAL','FLATTEN')),
  target_fraction    REAL,
  bid_at_submit      REAL,
  ask_at_submit      REAL,
  broker_time_submit TEXT NOT NULL,
  retcode            INTEGER,
  result_price       REAL,
  result_deal        INTEGER,
  source             TEXT NOT NULL DEFAULT 'node',   -- 'node','manual_trade','legacy_audit'
  UNIQUE (account_id, client_order_id)
);

-- Full MT5 deal record — ground truth for fills. Tickets are per-server
-- sequences: NEVER unique across accounts (hence the composite key).
CREATE TABLE deals (
  id              INTEGER PRIMARY KEY,
  account_id      INTEGER NOT NULL REFERENCES accounts(account_id),
  ticket          INTEGER NOT NULL,                  -- MT5 deal ticket
  order_ticket    INTEGER,
  position_id     INTEGER,
  client_order_id TEXT,                              -- deal.comment, normalized (plan §7.3)
  symbol          TEXT NOT NULL,
  canonical       TEXT,
  deal_type       INTEGER NOT NULL,                  -- mt5 DEAL_TYPE_* (BUY=0, SELL=1)
  entry           TEXT CHECK (entry IN ('IN','OUT','INOUT','OUT_BY')),
  volume          REAL NOT NULL,
  price           REAL NOT NULL,
  commission      REAL NOT NULL DEFAULT 0,
  swap            REAL NOT NULL DEFAULT 0,
  fee             REAL NOT NULL DEFAULT 0,
  profit          REAL NOT NULL DEFAULT 0,
  broker_time     TEXT NOT NULL,
  magic           INTEGER,
  UNIQUE (account_id, ticket)
);
CREATE INDEX ix_deals_acct_time ON deals(account_id, broker_time);

-- Derived, never stored: slippage. LEFT JOIN keeps no-submit deals (matches
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

-- One row per decision per instrument — today this evaporates in a 5s snapshot.
CREATE TABLE forecast_history (
  id                 INTEGER PRIMARY KEY,
  account_id         INTEGER NOT NULL REFERENCES accounts(account_id),
  as_of              TEXT NOT NULL,                  -- broker date of the decision
  canonical          TEXT NOT NULL,
  forecast_score     REAL,
  target_fraction    REAL,
  target_qty         REAL,
  engine_config_hash TEXT,
  vault_root         TEXT,
  warmup_ready       INTEGER NOT NULL CHECK (warmup_ready IN (0,1)),
  source             TEXT NOT NULL DEFAULT 'node',
  UNIQUE (account_id, as_of, canonical, source)
);

-- Durable, uncompacted equity record (the capped JSONL stays a dashboard cache).
CREATE TABLE equity_snapshots (
  id             INTEGER PRIMARY KEY,
  account_id     INTEGER NOT NULL REFERENCES accounts(account_id),
  ts             TEXT NOT NULL,
  balance        REAL,
  equity         REAL,
  floating_pnl   REAL,
  gross_notional REAL,
  marks_fresh    INTEGER CHECK (marks_fresh IN (0,1)),
  UNIQUE (account_id, ts)
);

-- ============================================================================
-- DOMAIN: OPS (job executions; derived cost surface)
-- ============================================================================

CREATE TABLE job_runs (
  job_run_id         INTEGER PRIMARY KEY,
  job_name           TEXT NOT NULL,                  -- 'mt5_scrape','signal_refresh','registry_ingest',...
  args_json          TEXT,
  started_at         TEXT NOT NULL,
  finished_at        TEXT,
  broker_time_anchor TEXT,                           -- broker_now() at start (host clock untrusted)
  exit_code          INTEGER,
  rows_written       INTEGER,
  coverage_json      TEXT,
  error_text         TEXT
);
CREATE INDEX ix_job_runs_name_time ON job_runs(job_name, started_at);

-- ADR-8 cost surface — derived/rebuildable; refreshed on demand
-- (`registry costs --refresh`), consumed by toolkit costs.py + validation lanes.
CREATE TABLE cost_observations (
  id                     INTEGER PRIMARY KEY,
  instrument_id          TEXT NOT NULL REFERENCES instruments(id),
  broker                 TEXT NOT NULL,
  tod_bucket             TEXT NOT NULL,              -- e.g. '09:30-10:00 ET'
  window_start           DATE NOT NULL,
  window_end             DATE NOT NULL,
  spread_points_p25      REAL,
  spread_points_p50      REAL,
  spread_points_p75      REAL,
  spread_bps_p50         REAL,
  slip_vs_touch_bps_p50  REAL,
  slip_vs_touch_bps_p90  REAL,
  n_spread_obs           INTEGER,
  n_fill_obs             INTEGER,
  computed_at            TEXT NOT NULL,
  UNIQUE (instrument_id, broker, tod_bucket, window_start, window_end)
);

-- ============================================================================
-- Diagnostics
-- ============================================================================

-- Orphan detection — impossible on the filesystem today.
CREATE VIEW v_instruments_without_bars AS
  SELECT i.id FROM instruments i
  LEFT JOIN blob_manifest b
    ON b.store IN ('ohlc_data','stock_data','mt5_m1','mt5_d1')
   AND b.key_json LIKE '%' || i.raw_symbol || '%'
  WHERE b.id IS NULL;
