"""Registry writer — the ONLY module that mutates the DB (single-writer rule, ADR-3).

No function auto-commits.  Callers MUST wrap mutations in db.transaction(conn)::

    from data_platform.registry import db, writer
    with db.transaction(conn):
        writer.upsert_instrument(conn, ...)

All public functions accept an open sqlite3.Connection plus a frozen dataclass
or simple typed parameters.  They follow repo conventions: frozen dataclasses,
full type hints, no booleans in public APIs (int flags for SQLite CHECK columns).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


# ── Domain dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Instrument:
    id: str                          # '{symbol}.{venue}', e.g. 'ES.XCME'
    raw_symbol: str
    asset_class: str
    instrument_class: str
    price_precision: int
    price_increment: float
    multiplier: float = 1.0
    quote_currency: str = "USD"
    activation: str | None = None
    expiration: str | None = None
    data_source: str | None = None
    info_json: str = "{}"


@dataclass(frozen=True)
class BlobRecord:
    store: str
    key_json: str
    relative_path: str
    written_at: str
    broker: str | None = None
    timezone: str | None = None
    schema_hash: str | None = None
    engine: str | None = None
    rows: int | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    valid: int = 1
    validation_error: str | None = None


@dataclass(frozen=True)
class JobRun:
    job_name: str
    started_at: str
    args_json: str | None = None
    broker_time_anchor: str | None = None


@dataclass(frozen=True)
class SpecRecord:
    id: str
    name: str
    name_slug: str
    content_hash: str
    spec_json: str
    file_path: str
    created_at: str
    updated_at: str
    spec_version: str = "1.0"
    valid: int = 1
    error: str | None = None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: str
    status: str
    created_at: str
    spec_id: str | None = None
    spec_hash: str | None = None
    spec_snapshot_json: str | None = None
    research_feed_used: str | None = None
    ewsd_blend: str | None = None
    realistic_phases: str | None = None
    git_sha: str | None = None
    num_combos: int | None = None
    reports_dir: str | None = None
    viz_dir: str | None = None
    log_path: str | None = None
    headline_metrics_json: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class ResultRecord:
    run_id: str
    artifact_type: str
    file_path: str
    sharpe: float | None = None
    t_stat: float | None = None
    sortino: float | None = None
    n_obs: int | None = None


@dataclass(frozen=True)
class VaultEntry:
    vault_profile: str
    timeframe: str
    ensemble_leaf: str
    feature_name: str
    feature_column: str
    module_name: str
    config_json: str
    file_path: str
    created_at: str
    updated_at: str
    weight_hierarchy_group: str | None = None
    direction: str | None = None
    producing_run_id: str | None = None
    gate_report_path: str | None = None
    promoted_by: str | None = None
    valid: int = 1
    validation_error: str | None = None


@dataclass(frozen=True)
class Account:
    account_id: int
    broker: str
    login: str
    server: str
    exec_tier: str
    program_phase: str
    valid_from: str
    venue: str = "MT5"
    currency: str = "USD"
    initial_balance: float | None = None
    magic_number: int | None = None
    trader_id: str | None = None
    risk_rules_json: str | None = None
    status: str = "active"
    valid_to: str | None = None
    predecessor_account_id: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Order:
    account_id: int
    client_order_id: str
    canonical: str
    symbol: str
    side: str
    qty: float
    broker_time_submit: str
    instrument_id: str | None = None
    intent: str | None = None
    target_fraction: float | None = None
    bid_at_submit: float | None = None
    ask_at_submit: float | None = None
    retcode: int | None = None
    result_price: float | None = None
    result_deal: int | None = None
    source: str = "node"


@dataclass(frozen=True)
class Deal:
    account_id: int
    ticket: int
    symbol: str
    deal_type: int              # MT5 DEAL_TYPE_*: BUY=0, SELL=1
    volume: float
    price: float
    broker_time: str
    commission: float = 0.0
    swap: float = 0.0
    fee: float = 0.0
    profit: float = 0.0
    order_ticket: int | None = None
    position_id: int | None = None
    client_order_id: str | None = None
    canonical: str | None = None
    entry: str | None = None    # 'IN','OUT','INOUT','OUT_BY'
    magic: int | None = None


@dataclass(frozen=True)
class Forecast:
    account_id: int
    as_of: str                  # broker date of decision
    canonical: str
    warmup_ready: int           # 0 or 1
    source: str = "node"
    forecast_score: float | None = None
    target_fraction: float | None = None
    target_qty: float | None = None
    engine_config_hash: str | None = None
    vault_root: str | None = None


@dataclass(frozen=True)
class EquitySnapshot:
    account_id: int
    ts: str
    balance: float | None = None
    equity: float | None = None
    floating_pnl: float | None = None
    gross_notional: float | None = None
    marks_fresh: int | None = None  # 0 or 1


# ── Market-data domain ────────────────────────────────────────────────────────

def upsert_instrument(conn: sqlite3.Connection, instrument: Instrument) -> None:
    """Upsert on PRIMARY KEY (id); updates all mutable fields on conflict."""
    conn.execute(
        """
        INSERT INTO instruments
          (id, raw_symbol, asset_class, instrument_class, price_precision,
           price_increment, multiplier, quote_currency, activation, expiration,
           data_source, info_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          raw_symbol       = excluded.raw_symbol,
          asset_class      = excluded.asset_class,
          instrument_class = excluded.instrument_class,
          price_precision  = excluded.price_precision,
          price_increment  = excluded.price_increment,
          multiplier       = excluded.multiplier,
          quote_currency   = excluded.quote_currency,
          activation       = excluded.activation,
          expiration       = excluded.expiration,
          data_source      = excluded.data_source,
          info_json        = excluded.info_json
        """,
        (
            instrument.id, instrument.raw_symbol, instrument.asset_class,
            instrument.instrument_class, instrument.price_precision,
            instrument.price_increment, instrument.multiplier,
            instrument.quote_currency, instrument.activation,
            instrument.expiration, instrument.data_source, instrument.info_json,
        ),
    )


def replace_source_symbols(
    conn: sqlite3.Connection,
    instrument_id: str,
    mapping: dict[str, str],
) -> None:
    """Replace all source-symbol rows for instrument_id with mapping {source: native_symbol}.

    Raises IntegrityError if any (source, native_symbol) pair is already held by a
    different instrument (UNIQUE constraint on instrument_source_symbols).
    """
    conn.execute(
        "DELETE FROM instrument_source_symbols WHERE instrument_id = ?",
        (instrument_id,),
    )
    conn.executemany(
        "INSERT INTO instrument_source_symbols (instrument_id, source, native_symbol) "
        "VALUES (?, ?, ?)",
        [(instrument_id, source, sym) for source, sym in mapping.items()],
    )


def record_blob(conn: sqlite3.Connection, blob: BlobRecord) -> None:
    """INSERT OR REPLACE into blob_manifest keyed on UNIQUE(store, key_json)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO blob_manifest
          (store, key_json, broker, timezone, relative_path, schema_hash,
           engine, rows, coverage_start, coverage_end, written_at, valid,
           validation_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            blob.store, blob.key_json, blob.broker, blob.timezone,
            blob.relative_path, blob.schema_hash, blob.engine,
            blob.rows, blob.coverage_start, blob.coverage_end,
            blob.written_at, blob.valid, blob.validation_error,
        ),
    )


# ── Ops domain ────────────────────────────────────────────────────────────────

def record_job_run(conn: sqlite3.Connection, job_run: JobRun) -> int:
    """INSERT a new job_runs row; returns the generated job_run_id."""
    cur = conn.execute(
        """
        INSERT INTO job_runs (job_name, args_json, started_at, broker_time_anchor)
        VALUES (?, ?, ?, ?)
        """,
        (
            job_run.job_name, job_run.args_json,
            job_run.started_at, job_run.broker_time_anchor,
        ),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def finish_job_run(
    conn: sqlite3.Connection,
    job_run_id: int,
    exit_code: int,
    rows_written: int,
    coverage_json: str | None,
    error_text: str | None,
) -> None:
    """Stamp finished_at and write outcome fields for an in-progress job_run."""
    conn.execute(
        """
        UPDATE job_runs
        SET finished_at    = ?,
            exit_code      = ?,
            rows_written   = ?,
            coverage_json  = ?,
            error_text     = ?
        WHERE job_run_id = ?
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            exit_code,
            rows_written,
            coverage_json,
            error_text,
            job_run_id,
        ),
    )


# ── Research domain ───────────────────────────────────────────────────────────

def upsert_spec(conn: sqlite3.Connection, spec: SpecRecord) -> None:
    """Upsert on PRIMARY KEY (id)."""
    conn.execute(
        """
        INSERT INTO specs
          (id, name, name_slug, content_hash, spec_version, valid, error,
           spec_json, file_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name         = excluded.name,
          name_slug    = excluded.name_slug,
          content_hash = excluded.content_hash,
          spec_version = excluded.spec_version,
          valid        = excluded.valid,
          error        = excluded.error,
          spec_json    = excluded.spec_json,
          file_path    = excluded.file_path,
          updated_at   = excluded.updated_at
        """,
        (
            spec.id, spec.name, spec.name_slug, spec.content_hash,
            spec.spec_version, spec.valid, spec.error,
            spec.spec_json, spec.file_path, spec.created_at, spec.updated_at,
        ),
    )


def upsert_run(conn: sqlite3.Connection, run: RunRecord) -> None:
    """Upsert on PRIMARY KEY (run_id)."""
    conn.execute(
        """
        INSERT INTO runs
          (run_id, kind, spec_id, spec_hash, spec_snapshot_json, status,
           research_feed_used, ewsd_blend, realistic_phases, git_sha,
           num_combos, reports_dir, viz_dir, log_path, headline_metrics_json,
           created_at, started_at, finished_at, error_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
          kind                 = excluded.kind,
          spec_id              = excluded.spec_id,
          spec_hash            = excluded.spec_hash,
          spec_snapshot_json   = excluded.spec_snapshot_json,
          status               = excluded.status,
          research_feed_used   = excluded.research_feed_used,
          ewsd_blend           = excluded.ewsd_blend,
          realistic_phases     = excluded.realistic_phases,
          git_sha              = excluded.git_sha,
          num_combos           = excluded.num_combos,
          reports_dir          = excluded.reports_dir,
          viz_dir              = excluded.viz_dir,
          log_path             = excluded.log_path,
          headline_metrics_json = excluded.headline_metrics_json,
          started_at           = excluded.started_at,
          finished_at          = excluded.finished_at,
          error_text           = excluded.error_text
        """,
        (
            run.run_id, run.kind, run.spec_id, run.spec_hash,
            run.spec_snapshot_json, run.status,
            run.research_feed_used, run.ewsd_blend, run.realistic_phases,
            run.git_sha, run.num_combos, run.reports_dir, run.viz_dir,
            run.log_path, run.headline_metrics_json,
            run.created_at, run.started_at, run.finished_at, run.error_text,
        ),
    )


def add_result(conn: sqlite3.Connection, result: ResultRecord) -> None:
    """INSERT a result row.  No UNIQUE constraint; multiple rows per run are valid."""
    conn.execute(
        """
        INSERT INTO results
          (run_id, artifact_type, file_path, sharpe, t_stat, sortino, n_obs)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.run_id, result.artifact_type, result.file_path,
            result.sharpe, result.t_stat, result.sortino, result.n_obs,
        ),
    )


def upsert_vault_entry(conn: sqlite3.Connection, entry: VaultEntry) -> None:
    """Upsert on UNIQUE(vault_profile, timeframe, ensemble_leaf, feature_name)."""
    conn.execute(
        """
        INSERT INTO vault_entries
          (vault_profile, timeframe, weight_hierarchy_group, ensemble_leaf,
           feature_name, feature_column, module_name, direction,
           producing_run_id, gate_report_path, promoted_by,
           config_json, file_path, valid, validation_error,
           created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vault_profile, timeframe, ensemble_leaf, feature_name) DO UPDATE SET
          weight_hierarchy_group = excluded.weight_hierarchy_group,
          feature_column         = excluded.feature_column,
          module_name            = excluded.module_name,
          direction              = excluded.direction,
          producing_run_id       = excluded.producing_run_id,
          gate_report_path       = excluded.gate_report_path,
          promoted_by            = excluded.promoted_by,
          config_json            = excluded.config_json,
          file_path              = excluded.file_path,
          valid                  = excluded.valid,
          validation_error       = excluded.validation_error,
          updated_at             = excluded.updated_at
        """,
        (
            entry.vault_profile, entry.timeframe, entry.weight_hierarchy_group,
            entry.ensemble_leaf, entry.feature_name, entry.feature_column,
            entry.module_name, entry.direction, entry.producing_run_id,
            entry.gate_report_path, entry.promoted_by,
            entry.config_json, entry.file_path, entry.valid,
            entry.validation_error, entry.created_at, entry.updated_at,
        ),
    )


# ── Live-trading domain ───────────────────────────────────────────────────────

def upsert_account(conn: sqlite3.Connection, account: Account) -> None:
    """Upsert on PRIMARY KEY (account_id)."""
    conn.execute(
        """
        INSERT INTO accounts
          (account_id, broker, venue, login, server, exec_tier, program_phase,
           currency, initial_balance, magic_number, trader_id, risk_rules_json,
           status, valid_from, valid_to, predecessor_account_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
          broker                 = excluded.broker,
          venue                  = excluded.venue,
          login                  = excluded.login,
          server                 = excluded.server,
          exec_tier              = excluded.exec_tier,
          program_phase          = excluded.program_phase,
          currency               = excluded.currency,
          initial_balance        = excluded.initial_balance,
          magic_number           = excluded.magic_number,
          trader_id              = excluded.trader_id,
          risk_rules_json        = excluded.risk_rules_json,
          status                 = excluded.status,
          valid_from             = excluded.valid_from,
          valid_to               = excluded.valid_to,
          predecessor_account_id = excluded.predecessor_account_id,
          notes                  = excluded.notes
        """,
        (
            account.account_id, account.broker, account.venue, account.login,
            account.server, account.exec_tier, account.program_phase,
            account.currency, account.initial_balance, account.magic_number,
            account.trader_id, account.risk_rules_json, account.status,
            account.valid_from, account.valid_to,
            account.predecessor_account_id, account.notes,
        ),
    )


def insert_order(conn: sqlite3.Connection, order: Order) -> None:
    """Upsert on UNIQUE(account_id, client_order_id); latest values win."""
    conn.execute(
        """
        INSERT INTO orders
          (account_id, client_order_id, instrument_id, canonical, symbol,
           side, qty, intent, target_fraction, bid_at_submit, ask_at_submit,
           broker_time_submit, retcode, result_price, result_deal, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, client_order_id) DO UPDATE SET
          instrument_id      = excluded.instrument_id,
          canonical          = excluded.canonical,
          symbol             = excluded.symbol,
          side               = excluded.side,
          qty                = excluded.qty,
          intent             = excluded.intent,
          target_fraction    = excluded.target_fraction,
          bid_at_submit      = excluded.bid_at_submit,
          ask_at_submit      = excluded.ask_at_submit,
          broker_time_submit = excluded.broker_time_submit,
          retcode            = excluded.retcode,
          result_price       = excluded.result_price,
          result_deal        = excluded.result_deal,
          source             = excluded.source
        """,
        (
            order.account_id, order.client_order_id, order.instrument_id,
            order.canonical, order.symbol, order.side, order.qty,
            order.intent, order.target_fraction,
            order.bid_at_submit, order.ask_at_submit,
            order.broker_time_submit, order.retcode,
            order.result_price, order.result_deal, order.source,
        ),
    )


def insert_deal(conn: sqlite3.Connection, deal: Deal) -> None:
    """Upsert on UNIQUE(account_id, ticket); latest values win."""
    conn.execute(
        """
        INSERT INTO deals
          (account_id, ticket, order_ticket, position_id, client_order_id,
           symbol, canonical, deal_type, entry, volume, price,
           commission, swap, fee, profit, broker_time, magic)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, ticket) DO UPDATE SET
          order_ticket     = excluded.order_ticket,
          position_id      = excluded.position_id,
          client_order_id  = excluded.client_order_id,
          symbol           = excluded.symbol,
          canonical        = excluded.canonical,
          deal_type        = excluded.deal_type,
          entry            = excluded.entry,
          volume           = excluded.volume,
          price            = excluded.price,
          commission       = excluded.commission,
          swap             = excluded.swap,
          fee              = excluded.fee,
          profit           = excluded.profit,
          broker_time      = excluded.broker_time,
          magic            = excluded.magic
        """,
        (
            deal.account_id, deal.ticket, deal.order_ticket, deal.position_id,
            deal.client_order_id, deal.symbol, deal.canonical,
            deal.deal_type, deal.entry, deal.volume, deal.price,
            deal.commission, deal.swap, deal.fee, deal.profit,
            deal.broker_time, deal.magic,
        ),
    )


def insert_forecast(conn: sqlite3.Connection, forecast: Forecast) -> None:
    """Upsert on UNIQUE(account_id, as_of, canonical, source); latest values win."""
    conn.execute(
        """
        INSERT INTO forecast_history
          (account_id, as_of, canonical, forecast_score, target_fraction,
           target_qty, engine_config_hash, vault_root, warmup_ready, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, as_of, canonical, source) DO UPDATE SET
          forecast_score     = excluded.forecast_score,
          target_fraction    = excluded.target_fraction,
          target_qty         = excluded.target_qty,
          engine_config_hash = excluded.engine_config_hash,
          vault_root         = excluded.vault_root,
          warmup_ready       = excluded.warmup_ready
        """,
        (
            forecast.account_id, forecast.as_of, forecast.canonical,
            forecast.forecast_score, forecast.target_fraction, forecast.target_qty,
            forecast.engine_config_hash, forecast.vault_root,
            forecast.warmup_ready, forecast.source,
        ),
    )


def insert_equity_snapshot(conn: sqlite3.Connection, snap: EquitySnapshot) -> None:
    """Upsert on UNIQUE(account_id, ts); latest values win."""
    conn.execute(
        """
        INSERT INTO equity_snapshots
          (account_id, ts, balance, equity, floating_pnl, gross_notional, marks_fresh)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, ts) DO UPDATE SET
          balance        = excluded.balance,
          equity         = excluded.equity,
          floating_pnl   = excluded.floating_pnl,
          gross_notional = excluded.gross_notional,
          marks_fresh    = excluded.marks_fresh
        """,
        (
            snap.account_id, snap.ts, snap.balance, snap.equity,
            snap.floating_pnl, snap.gross_notional, snap.marks_fresh,
        ),
    )
