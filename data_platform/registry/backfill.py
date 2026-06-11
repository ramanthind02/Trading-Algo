"""Registry ingest backfill — migration plan §7.7.

Imports LEGACY live records as schema-version-0 rows from two sources:

1. ``data/broker_cache/<broker>/slippage/slippage.csv``
   track_slippage output (ticket, order, side, entry, fill_px, …).
   Maps to Deal rows; UPSERT on (account_id, ticket) is safe against
   overlap with subsequent ``ingest live`` runs.
   Attribution: the single active account for that broker (2+ actives →
   loud error; rows predating valid_from are still attributed — these
   accounts have held the book since 2026-06-08 and validity windows
   only disambiguate once rotations exist).

2. ``logs/cfd_prop_audit/*.json``
   Enigma execution audits.  Each file has:
     * ``plans``   — per-account snapshots  → equity_snapshots rows
     * ``reports`` — per-action outcomes    → Deal rows (sparse)
   Logins NOT in the accounts table get one synthetic *retired* account
   per (broker, login).  Failed reports (success=false or no deal_ticket)
   are skipped.

Design rules
------------
* Deals have no ``source`` column; provenance is tracked in
  ``job_runs.coverage_json`` as ``slippage_csv_deals`` / ``audit_deals``.
* All mutations inside ``db.transaction()``.
* Re-runs are idempotent: UPSERT on (account_id, ticket) / (account_id, ts).
"""
from __future__ import annotations

import csv
import json
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

from data_platform.registry import writer
from data_platform.registry.db import transaction


# ── constants ─────────────────────────────────────────────────────────────────

_SIDE_TO_DEAL_TYPE: dict[str, int] = {"BUY": 0, "SELL": 1}
_VALID_ENTRY_VALUES: frozenset[str] = frozenset({"IN", "OUT", "INOUT", "OUT_BY"})
_BACKFILL_MAGIC = 510
# Brokers whose names we can recognise in label / server strings
_KNOWN_BROKERS: tuple[str, ...] = ("ftmo", "fundednext", "darwinex")


# ── path helpers ───────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


# ── account resolution ─────────────────────────────────────────────────────────

def _resolve_single_active_account(conn: sqlite3.Connection, broker: str) -> int:
    """Return the account_id for the one active account of *broker*.

    Raises ``ValueError`` if there are 0 or 2+ active accounts.
    """
    rows = conn.execute(
        "SELECT account_id FROM accounts WHERE broker = ? AND status = 'active'",
        (broker,),
    ).fetchall()
    if len(rows) == 0:
        raise ValueError(
            f"[backfill] No active account for broker={broker!r}. "
            "Cannot attribute slippage CSV rows. "
            f"Run: python -m data_platform.registry accounts add --broker {broker} ..."
        )
    if len(rows) > 1:
        raise ValueError(
            f"[backfill] Broker {broker!r} has {len(rows)} active accounts — "
            "attribution is ambiguous. Rotate to a single active account first."
        )
    return int(rows[0]["account_id"])


def _broker_from_label_or_server(label: str, server: str) -> str:
    """Best-effort broker name extracted from an audit plan's label or server."""
    for field in (label.lower(), server.lower()):
        for name in _KNOWN_BROKERS:
            if name in field:
                return name
    # Fall back: first word of label, lower-cased
    first = (label.strip().split()[0] if label.strip() else None) or server.strip().split("-")[0]
    return first.lower() if first else "unknown"


def _get_or_create_synthetic_account(
    conn: sqlite3.Connection,
    broker: str,
    login: str,
    server: str,
    valid_from: str,
) -> int:
    """Return account_id for (broker, login); INSERT a synthetic retired row if absent.

    Existing accounts (any status) are returned as-is.  New synthetic rows
    carry ``status='retired'`` and ``notes='legacy cfd_prop audit backfill'``.
    """
    rows = conn.execute(
        "SELECT account_id FROM accounts WHERE broker = ? AND login = ? "
        "ORDER BY valid_from DESC",
        (broker, login),
    ).fetchall()
    if rows:
        return int(rows[0]["account_id"])

    max_id = conn.execute("SELECT MAX(account_id) FROM accounts").fetchone()[0]
    account_id = (max_id or 0) + 1
    acct = writer.Account(
        account_id=account_id,
        broker=broker,
        login=login,
        server=server,
        exec_tier="demo",
        program_phase="challenge",
        valid_from=valid_from,
        status="retired",
        notes="legacy cfd_prop audit backfill",
    )
    with transaction(conn):
        writer.upsert_account(conn, acct)
    print(
        f"[backfill audit] created synthetic retired account "
        f"id={account_id} broker={broker!r} login={login!r}"
    )
    return account_id


# ── source 1: slippage CSV ─────────────────────────────────────────────────────

def _backfill_slippage(
    conn: sqlite3.Connection,
    repo_root: Path,
) -> dict[str, int]:
    """Import ``slippage/slippage.csv`` for every broker under broker_cache.

    Returns ``{broker: deals_inserted}``.
    """
    broker_cache = repo_root / "data" / "broker_cache"
    if not broker_cache.exists():
        print("[backfill slippage] data/broker_cache/ not found — skipping.")
        return {}

    results: dict[str, int] = {}

    for broker_dir in sorted(broker_cache.iterdir()):
        if not broker_dir.is_dir() or broker_dir.name.startswith("_"):
            continue
        broker = broker_dir.name
        slip_csv = broker_dir / "slippage" / "slippage.csv"
        if not slip_csv.is_file():
            continue

        try:
            account_id = _resolve_single_active_account(conn, broker)
        except ValueError as exc:
            print(f"[backfill slippage] SKIP {broker!r}: {exc}")
            continue

        inserted = 0
        skipped = 0

        with open(slip_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    side = str(row.get("side", "")).strip().upper()
                    deal_type = _SIDE_TO_DEAL_TYPE.get(side)
                    if deal_type is None:
                        skipped += 1
                        continue

                    entry_raw = str(row.get("entry", "")).strip().upper()
                    entry = entry_raw if entry_raw in _VALID_ENTRY_VALUES else None

                    order_raw = str(row.get("order", "")).strip()
                    order_ticket = int(order_raw) if order_raw else None

                    coid_raw = str(row.get("client_order_id", "")).strip()
                    coid = coid_raw or None

                    canonical_raw = str(row.get("canonical", "")).strip()
                    canonical = canonical_raw or None

                    deal = writer.Deal(
                        account_id=account_id,
                        ticket=int(row["ticket"]),
                        order_ticket=order_ticket,
                        position_id=None,
                        client_order_id=coid,
                        symbol=str(row["symbol"]),
                        canonical=canonical,
                        deal_type=deal_type,
                        entry=entry,
                        volume=float(row["volume"]),
                        price=float(row["fill_px"]),
                        commission=float(row.get("commission") or 0.0),
                        swap=float(row.get("swap") or 0.0),
                        fee=0.0,
                        profit=0.0,
                        broker_time=str(row["broker_time"]),
                        magic=_BACKFILL_MAGIC,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    skipped += 1
                    warnings.warn(
                        f"[backfill slippage] {broker}: skipping row ({exc}): {dict(row)}",
                        stacklevel=2,
                    )
                    continue

                with transaction(conn):
                    writer.insert_deal(conn, deal)
                inserted += 1

        print(
            f"[backfill slippage] {broker}: {inserted} deals inserted"
            + (f", {skipped} skipped" if skipped else "")
        )
        results[broker] = inserted

    return results


# ── source 2: cfd_prop audit JSONs ────────────────────────────────────────────

def _backfill_audit_jsons(
    conn: sqlite3.Connection,
    repo_root: Path,
) -> dict[str, int]:
    """Import ``logs/cfd_prop_audit/*.json``.

    Returns ``{"equity_snapshots": N, "deals": N}``.
    """
    audit_dir = repo_root / "logs" / "cfd_prop_audit"
    if not audit_dir.is_dir():
        print("[backfill audit] logs/cfd_prop_audit/ not found — skipping.")
        return {"equity_snapshots": 0, "deals": 0}

    audit_files = sorted(audit_dir.glob("*.json"))
    if not audit_files:
        print("[backfill audit] logs/cfd_prop_audit/ is empty — skipping.")
        return {"equity_snapshots": 0, "deals": 0}

    total_equity = 0
    total_deals = 0

    for audit_file in audit_files:
        try:
            data = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(
                f"[backfill audit] cannot parse {audit_file.name} ({exc}) — skipping",
                stacklevel=2,
            )
            continue

        ts = str(data.get("timestamp_utc", "")).strip()
        valid_from = ts or "2026-06-08"
        plans: list[dict] = data.get("plans", [])
        reports: list[dict] = data.get("reports", [])

        # Build account_ids for this file; equity snapshots per plan.
        file_account_ids: list[int] = []
        for plan in plans:
            login = str(plan.get("login", "")).strip()
            if not login or login == "0":
                continue
            server = str(plan.get("server", "")).strip()
            label = str(plan.get("label", "")).strip()
            broker = _broker_from_label_or_server(label, server)

            acct_id = _get_or_create_synthetic_account(
                conn, broker, login, server, valid_from
            )
            file_account_ids.append(acct_id)

            # Equity snapshot
            balance = plan.get("balance")
            equity = plan.get("equity")
            if ts and (balance is not None or equity is not None):
                snap = writer.EquitySnapshot(
                    account_id=acct_id,
                    ts=ts,
                    balance=float(balance) if balance is not None else None,
                    equity=float(equity) if equity is not None else None,
                )
                with transaction(conn):
                    writer.insert_equity_snapshot(conn, snap)
                total_equity += 1

        if not file_account_ids:
            continue

        # Attribute reports to the first (primary) account in the file.
        primary_account_id = file_account_ids[0]

        for rpt in reports:
            # Skip failed reports or those without a valid deal ticket
            if not rpt.get("success"):
                continue
            raw_ticket = rpt.get("deal_ticket")
            if raw_ticket is None:
                continue
            try:
                ticket = int(raw_ticket)
            except (TypeError, ValueError):
                continue
            if ticket == 0:
                continue

            # Sparse Deal row — symbol / price / volume unknown
            deal = writer.Deal(
                account_id=primary_account_id,
                ticket=ticket,
                symbol="UNKNOWN",
                deal_type=0,        # BUY placeholder (actual direction unknown)
                volume=0.0,
                price=0.0,
                broker_time=ts or "1970-01-01T00:00:00",
                magic=None,
            )
            with transaction(conn):
                writer.insert_deal(conn, deal)
            total_deals += 1

    print(
        f"[backfill audit] equity_snapshots={total_equity} deals={total_deals}"
    )
    return {"equity_snapshots": total_equity, "deals": total_deals}


# ── orchestrator ───────────────────────────────────────────────────────────────

def run_backfill(
    conn: sqlite3.Connection,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Run the full legacy backfill.

    Records one ``job_runs`` row (name='registry_ingest_backfill').
    Returns a coverage dict:
    ``{"slippage_csv_deals": {broker: N}, "audit": {"equity_snapshots": N, "deals": N}}``.

    Provenance note: deals have no ``source`` column; counts are tracked in
    ``job_runs.coverage_json``.  No orders or forecasts are inserted by this
    backfill (the ``source='legacy_audit'`` convention on ``orders.source``
    is therefore not applicable here).
    """
    import time

    rr = repo_root or _repo_root()
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    with transaction(conn):
        job_run_id = writer.record_job_run(
            conn,
            writer.JobRun(
                job_name="registry_ingest_backfill",
                started_at=started_at,
            ),
        )

    error_text: str | None = None
    slippage_counts: dict[str, int] = {}
    audit_counts: dict[str, int] = {}

    try:
        slippage_counts = _backfill_slippage(conn, rr)
        audit_counts = _backfill_audit_jsons(conn, rr)
    except Exception as exc:
        error_text = str(exc)
        raise
    finally:
        duration = time.monotonic() - t0
        total_rows = sum(slippage_counts.values()) + sum(audit_counts.values())
        coverage_json = json.dumps(
            {
                "duration_seconds": round(duration, 2),
                "slippage_csv_deals": slippage_counts,
                "audit": audit_counts,
            }
        )
        with transaction(conn):
            writer.finish_job_run(
                conn,
                job_run_id,
                exit_code=1 if error_text else 0,
                rows_written=total_rows,
                coverage_json=coverage_json,
                error_text=error_text,
            )

    return {
        "slippage_csv_deals": slippage_counts,
        "audit": audit_counts,
    }
