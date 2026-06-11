"""Data registry API endpoints — /api/data/* (M6.1, ADR-9).

All queries open read-only connections via db.connect_readonly(); returns
empty collections cleanly when the registry DB is missing so the frontend
degrades gracefully.

Routes
------
GET /api/data/coverage   -- blob_manifest per-store summary (or per-key with ?store=)
GET /api/data/freshness  -- newest coverage_end per store (or per key with ?store=)
GET /api/data/jobs       -- job_runs rows (optional ?name=&limit=)
GET /api/data/accounts   -- accounts table rows
GET /api/data/fills      -- v_slippage rows joined to accounts (optional ?broker=&limit=)
GET /api/data/forecasts  -- forecast_history rows (optional ?broker=)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter()

# Override in tests via monkeypatch (mirrors the pattern in frontend.api.runs).
_REGISTRY_DB_PATH: Path | None = None


def _open_readonly():
    from data_platform.registry import db as _db

    return _db.connect_readonly(_REGISTRY_DB_PATH)


# ── coverage ──────────────────────────────────────────────────────────────────


@router.get("/coverage")
def data_coverage(store: str | None = Query(default=None)) -> dict[str, Any]:
    """Per-store summary, or per-key listing when ?store= is provided."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"stores": []}
    try:
        if store:
            rows = conn.execute(
                """
                SELECT key_json, coverage_start, coverage_end, rows
                FROM blob_manifest
                WHERE store = ? AND valid = 1
                ORDER BY key_json
                """,
                (store,),
            ).fetchall()
            return {"store": store, "keys": [dict(r) for r in rows]}
        else:
            rows = conn.execute(
                """
                SELECT store,
                       count(*)            AS key_count,
                       MIN(coverage_start) AS earliest,
                       MAX(coverage_end)   AS latest,
                       SUM(rows)           AS total_rows
                FROM blob_manifest
                WHERE valid = 1
                GROUP BY store
                ORDER BY store
                """
            ).fetchall()
            return {"stores": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── freshness ─────────────────────────────────────────────────────────────────


@router.get("/freshness")
def data_freshness(store: str | None = Query(default=None)) -> dict[str, Any]:
    """Newest coverage_end per store, or per key when ?store= is provided."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"stores": []}
    try:
        if store:
            rows = conn.execute(
                """
                SELECT key_json, MAX(coverage_end) AS coverage_end
                FROM blob_manifest
                WHERE store = ? AND valid = 1
                GROUP BY key_json
                ORDER BY key_json
                """,
                (store,),
            ).fetchall()
            return {"store": store, "keys": [dict(r) for r in rows]}
        else:
            rows = conn.execute(
                """
                SELECT store, MAX(coverage_end) AS coverage_end
                FROM blob_manifest
                WHERE valid = 1
                GROUP BY store
                ORDER BY store
                """
            ).fetchall()
            return {"stores": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── jobs ──────────────────────────────────────────────────────────────────────


@router.get("/jobs")
def data_jobs(
    name: str | None = Query(default=None),
    limit: int = Query(default=20),
) -> dict[str, Any]:
    """Recent job_runs rows; ?name= to filter by job_name."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"jobs": []}
    try:
        if name:
            rows = conn.execute(
                "SELECT * FROM job_runs WHERE job_name = ? ORDER BY started_at DESC LIMIT ?",
                (name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM job_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"jobs": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── accounts ──────────────────────────────────────────────────────────────────


@router.get("/accounts")
def data_accounts() -> dict[str, Any]:
    """All accounts (active + retired) ordered by broker, valid_from."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"accounts": []}
    try:
        rows = conn.execute(
            """
            SELECT account_id, broker, login, exec_tier, program_phase,
                   status, valid_from, valid_to, currency, initial_balance
            FROM accounts
            ORDER BY broker, valid_from
            """
        ).fetchall()
        return {"accounts": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── fills ─────────────────────────────────────────────────────────────────────


@router.get("/fills")
def data_fills(
    broker: str | None = Query(default=None),
    limit: int = Query(default=100),
) -> dict[str, Any]:
    """v_slippage rows joined to accounts; ?broker= to filter."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"fills": []}
    try:
        if broker:
            rows = conn.execute(
                """
                SELECT v.ticket, v.canonical, v.entry, v.volume,
                       v.fill_px, v.spread_bps,
                       v.slip_vs_touch_bps, v.slip_vs_mid_bps,
                       v.commission, v.swap, v.broker_time,
                       a.broker
                FROM v_slippage v
                JOIN accounts a ON a.account_id = v.account_id
                WHERE a.broker = ?
                ORDER BY v.broker_time DESC
                LIMIT ?
                """,
                (broker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT v.ticket, v.canonical, v.entry, v.volume,
                       v.fill_px, v.spread_bps,
                       v.slip_vs_touch_bps, v.slip_vs_mid_bps,
                       v.commission, v.swap, v.broker_time,
                       a.broker
                FROM v_slippage v
                JOIN accounts a ON a.account_id = v.account_id
                ORDER BY v.broker_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {"fills": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── forecasts ─────────────────────────────────────────────────────────────────


@router.get("/forecasts")
def data_forecasts(broker: str | None = Query(default=None)) -> dict[str, Any]:
    """forecast_history rows; ?broker= to filter by broker."""
    try:
        conn = _open_readonly()
    except FileNotFoundError:
        return {"forecasts": []}
    try:
        if broker:
            rows = conn.execute(
                """
                SELECT f.id, f.as_of, f.canonical,
                       f.forecast_score, f.target_fraction, f.target_qty,
                       f.warmup_ready, f.source, a.broker
                FROM forecast_history f
                JOIN accounts a ON a.account_id = f.account_id
                WHERE a.broker = ?
                ORDER BY f.as_of DESC, f.canonical
                """,
                (broker,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT f.*, a.broker
                FROM forecast_history f
                JOIN accounts a ON a.account_id = f.account_id
                ORDER BY f.as_of DESC, f.canonical
                """
            ).fetchall()
        return {"forecasts": [dict(r) for r in rows]}
    finally:
        conn.close()
