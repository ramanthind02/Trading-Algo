"""Registry reader — read-only query API (ADR-9).

All functions accept an open sqlite3.Connection (preferably from
db.connect_readonly()) and return typed results.  No writes occur here.
"""
from __future__ import annotations

import sqlite3


def instrument_by_source_symbol(
    conn: sqlite3.Connection,
    source: str,
    native_symbol: str,
) -> sqlite3.Row | None:
    """Return the instruments row for (source, native_symbol).

    O(1) replacement for the O(n) reverse-scan over info_json blobs.
    Returns None when the pair is not registered.
    """
    return conn.execute(
        """
        SELECT i.*
        FROM instruments i
        JOIN instrument_source_symbols s ON s.instrument_id = i.id
        WHERE s.source = ? AND s.native_symbol = ?
        """,
        (source, native_symbol),
    ).fetchone()


def coverage(
    conn: sqlite3.Connection,
    store: str,
    key_filter: str | None = None,
) -> list[sqlite3.Row]:
    """Return (key_json, coverage_start, coverage_end, rows) for valid rows in a store.

    key_filter: optional SQL LIKE pattern applied to key_json.
    """
    if key_filter is not None:
        return conn.execute(
            """
            SELECT key_json, coverage_start, coverage_end, rows
            FROM blob_manifest
            WHERE store = ? AND key_json LIKE ? AND valid = 1
            ORDER BY key_json
            """,
            (store, key_filter),
        ).fetchall()
    return conn.execute(
        """
        SELECT key_json, coverage_start, coverage_end, rows
        FROM blob_manifest
        WHERE store = ? AND valid = 1
        ORDER BY key_json
        """,
        (store,),
    ).fetchall()


def freshness(
    conn: sqlite3.Connection,
    store: str,
) -> list[sqlite3.Row]:
    """Return (key_json, newest coverage_end) per logical key for a store."""
    return conn.execute(
        """
        SELECT key_json, MAX(coverage_end) AS coverage_end
        FROM blob_manifest
        WHERE store = ? AND valid = 1
        GROUP BY key_json
        ORDER BY key_json
        """,
        (store,),
    ).fetchall()


def runs_by_spec(
    conn: sqlite3.Connection,
    spec_id: str,
) -> list[sqlite3.Row]:
    """Return all runs for a spec_id, newest first."""
    return conn.execute(
        "SELECT * FROM runs WHERE spec_id = ? ORDER BY created_at DESC",
        (spec_id,),
    ).fetchall()


def runs_for_spec_hash(
    conn: sqlite3.Connection,
    spec_hash: str,
    kind: str | None = None,
) -> list[sqlite3.Row]:
    """Return runs matching spec_hash, newest first; optionally filtered by kind."""
    if kind is not None:
        return conn.execute(
            """
            SELECT * FROM runs
            WHERE spec_hash = ? AND kind = ?
            ORDER BY created_at DESC
            """,
            (spec_hash, kind),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM runs WHERE spec_hash = ? ORDER BY created_at DESC",
        (spec_hash,),
    ).fetchall()


def run(
    conn: sqlite3.Connection,
    run_id: str,
) -> sqlite3.Row | None:
    """Return a single run row or None."""
    return conn.execute(
        "SELECT * FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()


def headline_metrics(
    conn: sqlite3.Connection,
    kind: str | None = None,
) -> list[sqlite3.Row]:
    """Return (run_id, kind, status, headline_metrics_json, created_at) rows.

    Optionally filtered by run kind.  Only rows where headline_metrics_json is set.
    """
    if kind is not None:
        return conn.execute(
            """
            SELECT run_id, kind, status, headline_metrics_json, created_at
            FROM runs
            WHERE kind = ? AND headline_metrics_json IS NOT NULL
            ORDER BY created_at DESC
            """,
            (kind,),
        ).fetchall()
    return conn.execute(
        """
        SELECT run_id, kind, status, headline_metrics_json, created_at
        FROM runs
        WHERE headline_metrics_json IS NOT NULL
        ORDER BY created_at DESC
        """,
    ).fetchall()


def vault_lineage(
    conn: sqlite3.Connection,
    feature_name: str,
) -> sqlite3.Row | None:
    """Return a joined vault_entries+runs+specs row for a feature_name (most-recent vault)."""
    return conn.execute(
        """
        SELECT
          v.*,
          r.kind              AS run_kind,
          r.status            AS run_status,
          r.headline_metrics_json,
          r.created_at        AS run_created_at,
          s.name              AS spec_name,
          s.name_slug,
          s.content_hash      AS spec_content_hash,
          s.spec_json
        FROM vault_entries v
        LEFT JOIN runs  r ON r.run_id = v.producing_run_id
        LEFT JOIN specs s ON s.id     = r.spec_id
        WHERE v.feature_name = ?
        ORDER BY v.updated_at DESC
        LIMIT 1
        """,
        (feature_name,),
    ).fetchone()


def accounts_active(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all active accounts."""
    return conn.execute(
        "SELECT * FROM accounts WHERE status = 'active' ORDER BY broker, login",
    ).fetchall()


def deals_for_account(
    conn: sqlite3.Connection,
    account_id: int,
    since: str | None = None,
) -> list[sqlite3.Row]:
    """Return deals for account_id, optionally from broker_time >= since."""
    if since is not None:
        return conn.execute(
            """
            SELECT * FROM deals
            WHERE account_id = ? AND broker_time >= ?
            ORDER BY broker_time
            """,
            (account_id, since),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM deals WHERE account_id = ? ORDER BY broker_time",
        (account_id,),
    ).fetchall()


def slippage(
    conn: sqlite3.Connection,
    account_id: int,
) -> list[sqlite3.Row]:
    """Return all v_slippage rows for an account."""
    return conn.execute(
        "SELECT * FROM v_slippage WHERE account_id = ?",
        (account_id,),
    ).fetchall()


def job_history(
    conn: sqlite3.Connection,
    job_name: str,
    limit: int = 20,
) -> list[sqlite3.Row]:
    """Return the most recent job_runs rows for job_name."""
    return conn.execute(
        """
        SELECT * FROM job_runs
        WHERE job_name = ?
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (job_name, limit),
    ).fetchall()
