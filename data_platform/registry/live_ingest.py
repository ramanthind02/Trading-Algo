"""Live-trading registry ingest (migration plan §7.6).

Reads JSONL / JSON files written by the running node (deals.jsonl,
slippage/submits.jsonl, live_state/forecasts.jsonl, equity_history.jsonl,
equity.jsonl) and inserts them into the registry DB via writer.py.

Design rules
------------
* ADR-3: idempotent full-file re-read — never tail by byte offset; tolerate a
  torn/malformed last line (skip with a warning count, never raise).
* Mutations ONLY inside db.transaction().
* account resolution: snapshot.json → (broker, login) → accounts row;
  NO account row → skip the broker with a loud message, no guessing.
* broker_time is stored as-is from the JSONL (broker wall-clock ISO string) —
  no timezone conversion (docs/library/Data/mt5_timezones.md).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ── path helpers (kept import-free from deployment.live so the module is usable
#    in tests without the live runtime on the path) ─────────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


def _broker_cache_root(repo_root: Path, broker: str) -> Path:
    """Mirrors deployment.live.broker_data.broker_cache_root without the import."""
    return repo_root / "data" / "broker_cache" / broker / "central_cache"


def _live_state_dir(repo_root: Path, broker: str) -> Path:
    """Mirrors deployment.live.monitoring.live_state.live_state_dir."""
    return _broker_cache_root(repo_root, broker).parent / "live_state"


def _slippage_dir(repo_root: Path, broker: str) -> Path:
    return _broker_cache_root(repo_root, broker).parent / "slippage"


def _archive_dir(repo_root: Path, broker: str) -> Path:
    return _broker_cache_root(repo_root, broker).parent / "_archive"


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[tuple[dict, bool]]:
    """Yield (record_dict, is_last_line) for every parseable line in a JSONL file.

    Silently skips blank lines; yields ``is_last_line=True`` only for the very
    last non-blank line so callers can detect a potentially-torn last record
    (ADR-3: the last line of an append-only file may be incomplete if the node
    crashed mid-write).
    """
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    # Collect non-blank lines with their original positions
    non_blank = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    last_idx = non_blank[-1][0] if non_blank else -1
    for orig_idx, line in non_blank:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj, (orig_idx == last_idx)
        except json.JSONDecodeError:
            if orig_idx == last_idx:
                # Tolerate torn last line (ADR-3) — caller counts the skip
                pass
            # else: mid-file corruption is a real warning but we still skip


def _read_jsonl(
    path: Path,
    *,
    warn_prefix: str,
) -> tuple[list[dict], int]:
    """Return (records, skip_count) where skip_count counts torn/malformed lines."""
    records: list[dict] = []
    skipped = 0
    if not path.is_file():
        return records, skipped
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                skipped += 1
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        warnings.warn(
            f"{warn_prefix}: skipped {skipped} malformed/torn line(s) in {path}",
            stacklevel=2,
        )
    return records, skipped


# ── account resolution ─────────────────────────────────────────────────────────

def _resolve_account_id(
    conn: sqlite3.Connection,
    broker: str,
    snapshot_path: Path,
) -> int | None:
    """Read snapshot.json and match accounts table; return account_id or None."""
    if not snapshot_path.is_file():
        print(
            f"[registry ingest live] SKIP broker={broker!r}: "
            "snapshot.json not found — node has never published live state."
        )
        return None
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        # Snapshot is disposable (fsync=False, rewritten every ~5s) — a hard
        # crash can leave it NUL-filled until the node next runs. Attribution
        # stays DETERMINISTIC: fall back only when the broker has exactly one
        # active account; multiple actives would be guessing, so skip loudly.
        fallback = conn.execute(
            "SELECT account_id FROM accounts WHERE broker = ? AND status = 'active'",
            (broker,),
        ).fetchall()
        if len(fallback) == 1:
            print(
                f"[registry ingest live] broker={broker!r}: snapshot.json unreadable "
                f"({exc}) — attributing to the broker's single active account "
                f"id={fallback[0][0]}."
            )
            return int(fallback[0][0])
        print(
            f"[registry ingest live] SKIP broker={broker!r}: "
            f"cannot read snapshot.json ({exc}) and broker has "
            f"{len(fallback)} active accounts — need exactly 1 to fall back."
        )
        return None

    acct_block = raw.get("account", {})
    login = str(acct_block.get("login", "")).strip()
    if not login or login == "0":
        print(
            f"[registry ingest live] SKIP broker={broker!r}: "
            "snapshot.json has no usable account.login field."
        )
        return None

    rows = conn.execute(
        """
        SELECT account_id, valid_from, valid_to, status
        FROM accounts
        WHERE broker = ? AND login = ?
        ORDER BY valid_from DESC
        """,
        (broker, login),
    ).fetchall()

    if not rows:
        print(
            f"[registry ingest live] SKIP broker={broker!r} login={login!r}: "
            "no matching row in accounts table. "
            "Run: python -m data_platform.registry accounts add "
            f"--broker {broker} --login {login} --server <S> --exec-tier demo --phase <P>"
        )
        return None

    # Prefer active account; fall back to most-recent by valid_from
    active = [r for r in rows if r["status"] == "active"]
    chosen = active[0] if active else rows[0]
    return int(chosen["account_id"])


# ── entry-int → text map ───────────────────────────────────────────────────────

_ENTRY_MAP: dict[int, str] = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}


def _entry_text(val) -> str | None:
    try:
        return _ENTRY_MAP.get(int(val))
    except (TypeError, ValueError):
        return None


# ── comment normalisation ──────────────────────────────────────────────────────

def _normalize_comment(comment: str | None) -> str | None:
    """Strip 'close:' prefix; 'vault-flatten' or empty → NULL."""
    if not comment:
        return None
    s = str(comment).strip()
    if not s or s.lower() == "vault-flatten":
        return None
    if s.lower().startswith("close:"):
        s = s[6:].strip()
    return s or None


# ── canonical reverse-map (best effort, NULL on miss) ─────────────────────────

def _safe_canonical(broker: str, symbol: str) -> str | None:
    try:
        from data_platform.providers.mt5.brokers import canonical_for
        return canonical_for(broker, symbol)
    except (KeyError, Exception):
        return None


# ── individual file ingesters ─────────────────────────────────────────────────

def _ingest_deals(
    conn: sqlite3.Connection,
    account_id: int,
    broker: str,
    deals_path: Path,
) -> tuple[int, int]:
    """Insert deals from deals.jsonl. Returns (rows_inserted, rows_skipped)."""
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    records, skipped = _read_jsonl(deals_path, warn_prefix=f"deals [{broker}]")
    inserted = 0
    for rec in records:
        try:
            deal = writer.Deal(
                account_id=account_id,
                ticket=int(rec["ticket"]),
                order_ticket=int(rec["order"]) if rec.get("order") else None,
                position_id=int(rec["position_id"]) if rec.get("position_id") else None,
                client_order_id=_normalize_comment(rec.get("comment")),
                symbol=str(rec["symbol"]),
                canonical=_safe_canonical(broker, str(rec["symbol"])),
                deal_type=int(rec["type"]),
                entry=_entry_text(rec.get("entry")),
                volume=float(rec["volume"]),
                price=float(rec["price"]),
                commission=float(rec.get("commission") or 0.0),
                swap=float(rec.get("swap") or 0.0),
                fee=float(rec.get("fee") or 0.0),
                profit=float(rec.get("profit") or 0.0),
                # broker_time: stored as-is (broker wall-clock ISO, no tz conversion)
                broker_time=datetime.fromtimestamp(int(rec["time"])).isoformat()
                    if isinstance(rec.get("time"), (int, float))
                    else str(rec.get("broker_time") or rec.get("time") or ""),
                magic=int(rec["magic"]) if rec.get("magic") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            warnings.warn(
                f"deals [{broker}]: skipping malformed record ({exc}): {rec}",
                stacklevel=2,
            )
            continue
        with transaction(conn):
            writer.insert_deal(conn, deal)
        inserted += 1
    return inserted, skipped


def _ingest_orders(
    conn: sqlite3.Connection,
    account_id: int,
    broker: str,
    submits_path: Path,
    results_path: Path | None,
) -> tuple[int, int]:
    """Insert order rows from slippage/submits.jsonl. Returns (rows_inserted, rows_skipped)."""
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    records, skipped = _read_jsonl(submits_path, warn_prefix=f"submits [{broker}]")

    # Optional results file (submit_results.jsonl) — join by client_order_id
    results_by_coid: dict[str, dict] = {}
    if results_path is not None and results_path.is_file():
        res_records, _ = _read_jsonl(results_path, warn_prefix=f"submit_results [{broker}]")
        for r in res_records:
            coid = r.get("client_order_id")
            if coid:
                results_by_coid[str(coid)] = r

    inserted = 0
    for rec in records:
        coid = str(rec.get("client_order_id") or "")
        if not coid:
            skipped += 1
            continue
        canonical = str(rec.get("canonical") or "")
        symbol_raw = rec.get("symbol") or canonical
        # resolve broker symbol from canonical if symbol not stored directly
        if not symbol_raw:
            skipped += 1
            continue
        # Try to resolve the native MT5 symbol from canonical, fall back to stored value
        try:
            from data_platform.providers.mt5.brokers import resolve as _resolve
            native_symbol = _resolve(broker, canonical) if canonical else symbol_raw
        except Exception:
            native_symbol = symbol_raw or canonical

        res = results_by_coid.get(coid, {})
        side_raw = str(rec.get("side") or "BUY").upper()
        side = side_raw if side_raw in ("BUY", "SELL") else "BUY"
        try:
            order = writer.Order(
                account_id=account_id,
                client_order_id=coid,
                canonical=canonical or native_symbol,
                symbol=native_symbol,
                side=side,
                qty=float(rec.get("qty") or 0.0),
                broker_time_submit=str(rec.get("broker_time") or ""),
                intent=str(rec["intent"]).upper() if rec.get("intent") else None,
                target_fraction=float(rec["target_fraction"])
                    if rec.get("target_fraction") is not None else None,
                bid_at_submit=float(rec["bid"]) if rec.get("bid") is not None else None,
                ask_at_submit=float(rec["ask"]) if rec.get("ask") is not None else None,
                retcode=int(res["retcode"]) if res.get("retcode") is not None else None,
                result_price=float(res["result_price"])
                    if res.get("result_price") is not None else None,
                result_deal=int(res["result_deal"])
                    if res.get("result_deal") is not None else None,
                source="node",
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            warnings.warn(
                f"submits [{broker}]: skipping malformed record ({exc}): {rec}",
                stacklevel=2,
            )
            continue
        with transaction(conn):
            writer.insert_order(conn, order)
        inserted += 1
    return inserted, skipped


def _ingest_forecasts(
    conn: sqlite3.Connection,
    account_id: int,
    broker: str,
    forecasts_path: Path,
) -> tuple[int, int]:
    """Insert forecast rows from forecasts.jsonl. Returns (rows_inserted, rows_skipped)."""
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    records, skipped = _read_jsonl(forecasts_path, warn_prefix=f"forecasts [{broker}]")
    inserted = 0
    for rec in records:
        try:
            forecast = writer.Forecast(
                account_id=account_id,
                as_of=str(rec["as_of"]),
                canonical=str(rec["canonical"]),
                warmup_ready=int(rec.get("warmup_ready", 0)),
                source="node",
                forecast_score=float(rec["forecast_score"])
                    if rec.get("forecast_score") is not None else None,
                target_fraction=float(rec["target_fraction"])
                    if rec.get("target_fraction") is not None else None,
                target_qty=float(rec["target_qty"])
                    if rec.get("target_qty") is not None else None,
                engine_config_hash=str(rec["engine_config_hash"])
                    if rec.get("engine_config_hash") else None,
                vault_root=str(rec["vault_root"])
                    if rec.get("vault_root") else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            warnings.warn(
                f"forecasts [{broker}]: skipping malformed record ({exc}): {rec}",
                stacklevel=2,
            )
            continue
        with transaction(conn):
            writer.insert_forecast(conn, forecast)
        inserted += 1
    return inserted, skipped


def _ingest_equity(
    conn: sqlite3.Connection,
    account_id: int,
    broker: str,
    *paths: Path,
) -> tuple[int, int]:
    """Insert equity snapshot rows from one or more JSONL paths.

    Handles both equity_history.jsonl ({ts, equity}) and durable equity.jsonl
    which may have more fields.  UNIQUE(account_id, ts) dedupes overlap.
    """
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    inserted = 0
    skipped = 0
    seen_ts: set[str] = set()
    for path in paths:
        records, path_skipped = _read_jsonl(path, warn_prefix=f"equity [{broker}]")
        skipped += path_skipped
        for rec in records:
            ts = str(rec.get("ts") or "")
            if not ts or ts in seen_ts:
                continue
            try:
                snap = writer.EquitySnapshot(
                    account_id=account_id,
                    ts=ts,
                    balance=float(rec["balance"]) if rec.get("balance") is not None else None,
                    equity=float(rec["equity"]) if rec.get("equity") is not None else None,
                    floating_pnl=float(rec["floating_pnl"])
                        if rec.get("floating_pnl") is not None else None,
                    gross_notional=float(rec["gross_notional"])
                        if rec.get("gross_notional") is not None else None,
                    marks_fresh=int(rec["marks_fresh"])
                        if rec.get("marks_fresh") is not None else None,
                )
            except (KeyError, TypeError, ValueError) as exc:
                skipped += 1
                warnings.warn(
                    f"equity [{broker}]: skipping malformed record ({exc}): {rec}",
                    stacklevel=2,
                )
                continue
            seen_ts.add(ts)
            with transaction(conn):
                writer.insert_equity_snapshot(conn, snap)
            inserted += 1
    return inserted, skipped


# ── top-level ingest ───────────────────────────────────────────────────────────

def ingest_broker(
    conn: sqlite3.Connection,
    broker: str,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Ingest all live-state files for one broker.

    Returns a coverage dict: {file: {rows, skipped}}.
    Skips the broker (with a loud print) if no accounts row matches.
    """
    rr = repo_root or _repo_root()
    state_dir = _live_state_dir(rr, broker)
    slip_dir = _slippage_dir(rr, broker)

    snap_p = state_dir / "snapshot.json"
    account_id = _resolve_account_id(conn, broker, snap_p)
    if account_id is None:
        return {}

    coverage: dict[str, dict] = {}

    # ── deals.jsonl ───────────────────────────────────────────────────────────
    deals_p = state_dir / "deals.jsonl"
    ins, skp = _ingest_deals(conn, account_id, broker, deals_p)
    coverage["deals"] = {"rows": ins, "skipped": skp}

    # ── slippage/submits.jsonl + optional submit_results.jsonl ────────────────
    submits_p = slip_dir / "submits.jsonl"
    results_p = state_dir / "submit_results.jsonl"
    ins, skp = _ingest_orders(
        conn, account_id, broker, submits_p,
        results_p if results_p.is_file() else None,
    )
    coverage["orders"] = {"rows": ins, "skipped": skp}

    # ── forecasts.jsonl ───────────────────────────────────────────────────────
    forecasts_p = state_dir / "forecasts.jsonl"
    ins, skp = _ingest_forecasts(conn, account_id, broker, forecasts_p)
    coverage["forecasts"] = {"rows": ins, "skipped": skp}

    # ── equity: durable equity.jsonl (may not exist) + equity_history.jsonl ──
    durable_equity_p = _broker_cache_root(rr, broker).parent / "equity.jsonl"
    legacy_equity_p = state_dir / "equity_history.jsonl"
    equity_paths = [p for p in (durable_equity_p, legacy_equity_p) if p.is_file()]
    if equity_paths:
        ins, skp = _ingest_equity(conn, account_id, broker, *equity_paths)
    else:
        ins, skp = 0, 0
    coverage["equity"] = {"rows": ins, "skipped": skp}

    # ── slippage view sanity check ────────────────────────────────────────────
    conn.execute("SELECT count(*) FROM v_slippage WHERE account_id = ?", (account_id,)).fetchone()

    return coverage


def ingest_all(
    conn: sqlite3.Connection,
    *,
    broker: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, dict]:
    """Ingest live data for all (or one) broker(s) that have a live_state directory.

    Returns {broker: coverage_dict}.
    """
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    rr = repo_root or _repo_root()
    broker_cache = rr / "data" / "broker_cache"

    brokers: list[str]
    if broker is not None:
        brokers = [broker]
    else:
        if not broker_cache.exists():
            print("[registry ingest live] data/broker_cache/ does not exist — nothing to ingest.")
            return {}
        brokers = [
            d.name for d in sorted(broker_cache.iterdir())
            if d.is_dir() and not d.name.startswith("_")
            and (_live_state_dir(rr, d.name)).exists()
        ]

    if not brokers:
        print("[registry ingest live] No broker live_state/ dirs found.")
        return {}

    import time
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    all_coverage: dict[str, dict] = {}
    for b in brokers:
        print(f"[registry ingest live] ingesting broker={b!r} ...")
        cov = ingest_broker(conn, b, repo_root=rr)
        if cov:
            all_coverage[b] = cov
            total_rows = sum(v["rows"] for v in cov.values())
            total_skip = sum(v["skipped"] for v in cov.values())
            print(
                f"  {b}: "
                + ", ".join(f"{k}={v['rows']}" for k, v in cov.items())
                + (f"  (skipped {total_skip})" if total_skip else "")
            )

    # Record job_runs row
    duration = time.monotonic() - t0
    import json as _json
    coverage_json = _json.dumps({
        "duration_seconds": round(duration, 2),
        "brokers": all_coverage,
    })
    total_rows = sum(
        v["rows"] for cov in all_coverage.values() for v in cov.values()
    )
    with transaction(conn):
        jid = writer.record_job_run(
            conn,
            writer.JobRun(
                job_name="registry_ingest_live",
                started_at=started_at,
                args_json=_json.dumps({"broker": broker}),
            ),
        )
    with transaction(conn):
        writer.finish_job_run(
            conn,
            jid,
            exit_code=0,
            rows_written=total_rows,
            coverage_json=coverage_json,
            error_text=None,
        )

    return all_coverage


# ── accounts CLI helpers ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def accounts_add(
    conn: sqlite3.Connection,
    *,
    broker: str,
    login: str,
    server: str,
    exec_tier: str,
    phase: str,
    currency: str = "USD",
    initial_balance: float | None = None,
    magic: int | None = None,
    notes: str | None = None,
) -> int:
    """Insert a new active account row. Returns the new account_id."""
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    # Next auto-increment id
    max_id = conn.execute("SELECT MAX(account_id) FROM accounts").fetchone()[0]
    account_id = (max_id or 0) + 1

    acct = writer.Account(
        account_id=account_id,
        broker=broker,
        login=login,
        server=server,
        exec_tier=exec_tier,
        program_phase=phase,
        valid_from=_now_iso(),
        currency=currency,
        initial_balance=initial_balance,
        magic_number=magic,
        notes=notes,
        status="active",
    )
    with transaction(conn):
        writer.upsert_account(conn, acct)
    return account_id


def accounts_rotate(
    conn: sqlite3.Connection,
    broker: str,
    *,
    new_login: str,
    new_server: str,
    phase: str | None = None,
    repo_root: Path | None = None,
) -> tuple[int, int]:
    """Retire the active account for broker, insert successor, archive JSONLs.

    Returns (old_account_id, new_account_id).
    """
    from data_platform.registry import writer
    from data_platform.registry.db import transaction

    # Find active account for this broker
    rows = conn.execute(
        "SELECT * FROM accounts WHERE broker = ? AND status = 'active' ORDER BY valid_from DESC",
        (broker,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No active account found for broker={broker!r}")
    old = rows[0]
    old_id = int(old["account_id"])
    now = _now_iso()

    # Retire old account
    with transaction(conn):
        conn.execute(
            "UPDATE accounts SET status='retired', valid_to=? WHERE account_id=?",
            (now, old_id),
        )

    # Insert successor
    new_phase = phase or str(old["program_phase"])
    new_id = accounts_add(
        conn,
        broker=broker,
        login=new_login,
        server=new_server,
        exec_tier=str(old["exec_tier"]),
        phase=new_phase,
        currency=str(old["currency"]),
        initial_balance=old["initial_balance"],
        magic=old["magic_number"],
        notes=old["notes"],
    )

    # Set predecessor FK
    with transaction(conn):
        conn.execute(
            "UPDATE accounts SET predecessor_account_id=? WHERE account_id=?",
            (old_id, new_id),
        )

    # Archive JSONLs: move *.jsonl + slippage/ into _archive/account_<old_id>/
    # The snapshot/command/halt .json files stay in place.
    rr = repo_root or _repo_root()
    state_dir = _live_state_dir(rr, broker)
    slip_dir = _slippage_dir(rr, broker)
    archive_base = _archive_dir(rr, broker) / f"account_{old_id}"
    archive_base.mkdir(parents=True, exist_ok=True)

    # Move JSONL files from live_state/
    for jsonl_file in state_dir.glob("*.jsonl"):
        dest = archive_base / jsonl_file.name
        shutil.move(str(jsonl_file), str(dest))

    # Move slippage/ directory contents
    if slip_dir.exists():
        archive_slip = archive_base / "slippage"
        archive_slip.mkdir(parents=True, exist_ok=True)
        for f in slip_dir.glob("*.jsonl"):
            shutil.move(str(f), str(archive_slip / f.name))
        for f in slip_dir.glob("*.csv"):
            shutil.move(str(f), str(archive_slip / f.name))

    print(
        f"[accounts rotate] Archived {broker!r} account_{old_id} JSONLs → {archive_base}"
    )
    print(
        f"[accounts rotate] REMINDER: update .env credentials for {broker!r} "
        f"and reset {state_dir / 'baseline.json'} (delete it) before restarting the node."
    )

    return old_id, new_id


def accounts_retire(
    conn: sqlite3.Connection,
    broker: str,
) -> int:
    """Set valid_to=now + status=retired for the active account of broker.

    Returns the retired account_id.
    """
    from data_platform.registry.db import transaction

    rows = conn.execute(
        "SELECT account_id FROM accounts WHERE broker = ? AND status = 'active' "
        "ORDER BY valid_from DESC",
        (broker,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No active account found for broker={broker!r}")
    account_id = int(rows[0]["account_id"])
    with transaction(conn):
        conn.execute(
            "UPDATE accounts SET status='retired', valid_to=? WHERE account_id=?",
            (_now_iso(), account_id),
        )
    return account_id
