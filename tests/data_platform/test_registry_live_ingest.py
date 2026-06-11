"""Unit tests for data_platform.registry.live_ingest + accounts CLI.

All tests use tmp_path DBs and synthetic broker_cache trees.
No real broker terminals or MT5 connections are opened.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data_platform.registry import db, writer
from data_platform.registry.db import transaction
from data_platform.registry.live_ingest import (
    _live_state_dir,
    _slippage_dir,
    accounts_add,
    accounts_retire,
    accounts_rotate,
    ingest_all,
    ingest_broker,
    _normalize_comment,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _open(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "registry.db")


def _make_broker_tree(root: Path, broker: str) -> Path:
    """Create the minimal directory structure for a broker."""
    state_dir = root / "data" / "broker_cache" / broker / "live_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    slip_dir = root / "data" / "broker_cache" / broker / "slippage"
    slip_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _write_snapshot(state_dir: Path, login: str, server: str) -> None:
    snap = {
        "account": {"login": login, "server": server, "currency": "USD",
                    "balance": 10000.0, "equity": 10100.0},
        "schema_version": 1,
    }
    (state_dir / "snapshot.json").write_text(json.dumps(snap), encoding="utf-8")


def _seed_account(
    conn: sqlite3.Connection,
    *,
    broker: str,
    login: str = "12345",
    server: str = "test-demo",
) -> int:
    return accounts_add(
        conn,
        broker=broker,
        login=login,
        server=server,
        exec_tier="demo",
        phase="challenge",
    )


# ── comment normalisation ─────────────────────────────────────────────────────

def test_normalize_comment_strips_close_prefix() -> None:
    assert _normalize_comment("close:X2") == "X2"
    assert _normalize_comment("close: X2 ") == "X2"


def test_normalize_comment_vault_flatten_to_null() -> None:
    assert _normalize_comment("vault-flatten") is None
    assert _normalize_comment("VAULT-FLATTEN") is None


def test_normalize_comment_empty_to_null() -> None:
    assert _normalize_comment("") is None
    assert _normalize_comment(None) is None


def test_normalize_comment_plain_passthrough() -> None:
    assert _normalize_comment("X2") == "X2"


# ── accounts add / list / rotate / retire ─────────────────────────────────────

def test_accounts_add_creates_active_row(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    acct_id = accounts_add(
        conn,
        broker="ftmo",
        login="55555",
        server="ftmo-demo",
        exec_tier="demo",
        phase="challenge",
        initial_balance=10000.0,
        magic=510,
    )
    row = conn.execute(
        "SELECT * FROM accounts WHERE account_id=?", (acct_id,)
    ).fetchone()
    assert row is not None
    assert row["broker"] == "ftmo"
    assert row["login"] == "55555"
    assert row["status"] == "active"
    assert row["exec_tier"] == "demo"
    assert row["program_phase"] == "challenge"
    assert row["initial_balance"] == 10000.0
    assert row["magic_number"] == 510
    conn.close()


def test_accounts_list_shows_all(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    id1 = accounts_add(conn, broker="ftmo",     login="11111", server="s1",
                       exec_tier="demo", phase="challenge")
    id2 = accounts_add(conn, broker="darwinex", login="22222", server="s2",
                       exec_tier="demo", phase="retail")
    rows = conn.execute(
        "SELECT account_id, broker, status FROM accounts ORDER BY account_id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["broker"] == "ftmo"
    assert rows[1]["broker"] == "darwinex"
    conn.close()


def test_accounts_retire_sets_status(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    acct_id = _seed_account(conn, broker="ftmo")
    retired_id = accounts_retire(conn, "ftmo")
    assert retired_id == acct_id
    row = conn.execute(
        "SELECT status, valid_to FROM accounts WHERE account_id=?", (acct_id,)
    ).fetchone()
    assert row["status"] == "retired"
    assert row["valid_to"] is not None
    conn.close()


def test_accounts_rotate_chains_predecessor_fk(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    # Seed account + live_state dir (for archival)
    _make_broker_tree(tmp_path, "ftmo")
    state_dir = _live_state_dir(tmp_path, "ftmo")
    _write_snapshot(state_dir, login="11111", server="ftmo-demo")
    old_id = _seed_account(conn, broker="ftmo", login="11111", server="ftmo-demo")

    # Write a JSONL file to archive
    (state_dir / "deals.jsonl").write_text('{"ticket":1}\n', encoding="utf-8")
    (state_dir / "forecasts.jsonl").write_text('{"as_of":"2026-01-01"}\n', encoding="utf-8")

    ret_old, new_id = accounts_rotate(
        conn,
        "ftmo",
        new_login="22222",
        new_server="ftmo-demo2",
        phase="funded",
        repo_root=tmp_path,
    )

    assert ret_old == old_id

    # Old account is retired
    old_row = conn.execute(
        "SELECT status, valid_to FROM accounts WHERE account_id=?", (old_id,)
    ).fetchone()
    assert old_row["status"] == "retired"
    assert old_row["valid_to"] is not None

    # New account has correct phase + predecessor FK
    new_row = conn.execute(
        "SELECT * FROM accounts WHERE account_id=?", (new_id,)
    ).fetchone()
    assert new_row["broker"] == "ftmo"
    assert new_row["login"] == "22222"
    assert new_row["program_phase"] == "funded"
    assert new_row["status"] == "active"
    assert new_row["predecessor_account_id"] == old_id

    conn.close()


def test_accounts_rotate_archives_jsonls(tmp_path: Path) -> None:
    """JSONLs move to _archive/account_<id>/; snapshot.json stays."""
    conn = _open(tmp_path)
    _make_broker_tree(tmp_path, "darwinex")
    state_dir = _live_state_dir(tmp_path, "darwinex")
    slip_dir = _slippage_dir(tmp_path, "darwinex")
    _write_snapshot(state_dir, login="99", server="dw-demo")
    old_id = _seed_account(conn, broker="darwinex", login="99", server="dw-demo")

    # Create files to be archived
    (state_dir / "deals.jsonl").write_text("", encoding="utf-8")
    (state_dir / "forecasts.jsonl").write_text("", encoding="utf-8")
    (slip_dir / "submits.jsonl").write_text("", encoding="utf-8")

    accounts_rotate(
        conn,
        "darwinex",
        new_login="100",
        new_server="dw-demo2",
        repo_root=tmp_path,
    )

    archive = tmp_path / "data" / "broker_cache" / "darwinex" / "_archive" / f"account_{old_id}"
    assert (archive / "deals.jsonl").exists()
    assert (archive / "forecasts.jsonl").exists()
    assert (archive / "slippage" / "submits.jsonl").exists()
    # snapshot.json should stay (not moved)
    assert (state_dir / "snapshot.json").exists()

    conn.close()


def test_accounts_retire_no_active_raises(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with pytest.raises(ValueError, match="No active account"):
        accounts_retire(conn, "nonexistent_broker")
    conn.close()


# ── ingest: unknown broker skips loudly ───────────────────────────────────────

def test_ingest_unknown_broker_skips(tmp_path: Path, capsys) -> None:
    """Broker with live_state/ but no accounts row → skip with message."""
    conn = _open(tmp_path)
    state_dir = _make_broker_tree(tmp_path, "mystery")
    _write_snapshot(state_dir, login="9999", server="mystery-demo")

    cov = ingest_broker(conn, "mystery", repo_root=tmp_path)
    assert cov == {}  # nothing ingested

    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "accounts add" in out
    conn.close()


def test_ingest_no_snapshot_skips(tmp_path: Path, capsys) -> None:
    """Broker live_state/ without snapshot.json → skip."""
    conn = _open(tmp_path)
    _make_broker_tree(tmp_path, "ghost")
    # Do NOT write snapshot.json

    cov = ingest_broker(conn, "ghost", repo_root=tmp_path)
    assert cov == {}

    out = capsys.readouterr().out
    assert "SKIP" in out
    conn.close()


# ── ingest: full synthetic fixture ───────────────────────────────────────────

def _build_synthetic_fixture(root: Path, broker: str, login: str) -> None:
    """Populate a full synthetic broker_cache tree."""
    state_dir = _make_broker_tree(root, broker)
    slip_dir = _slippage_dir(root, broker)
    _write_snapshot(state_dir, login=login, server="test-server")

    # deals.jsonl: 4 lines:
    #   1) normal deal, comment='X2'
    #   2) comment='close:MyOrder'  → normalized to 'MyOrder'
    #   3) comment='vault-flatten'  → NULL client_order_id
    #   4) torn last line (invalid JSON)
    deals = [
        json.dumps({"ticket": 1001, "order": 201, "time": 1748736000,
                    "type": 0, "entry": 0, "magic": 510, "position_id": 501,
                    "volume": 1.0, "price": 19000.0, "commission": -1.5,
                    "swap": 0.0, "fee": 0.0, "profit": 50.0,
                    "symbol": "NAS100", "comment": "X2", "external_id": ""}),
        json.dumps({"ticket": 1002, "order": 202, "time": 1748736001,
                    "type": 1, "entry": 1, "magic": 510, "position_id": 502,
                    "volume": 0.5, "price": 18950.0, "commission": -0.75,
                    "swap": -0.10, "fee": 0.0, "profit": -20.0,
                    "symbol": "NAS100", "comment": "close:MyOrder", "external_id": ""}),
        json.dumps({"ticket": 1003, "order": 203, "time": 1748736002,
                    "type": 0, "entry": 0, "magic": 510, "position_id": 503,
                    "volume": 2.0, "price": 3200.0, "commission": -2.0,
                    "swap": 0.0, "fee": 0.0, "profit": 0.0,
                    "symbol": "GOLD", "comment": "vault-flatten", "external_id": ""}),
        "{torn line",  # last line: malformed JSON (torn write)
    ]
    (state_dir / "deals.jsonl").write_text("\n".join(deals) + "\n", encoding="utf-8")

    # submits.jsonl: 2 rows incl. intent
    submits = [
        json.dumps({"client_order_id": "X2", "canonical": "NQ", "side": "BUY",
                    "qty": 1.0, "bid": 18990.0, "ask": 19010.0,
                    "broker_time": "2026-06-01T17:00:00",
                    "intent": "ENTRY", "target_fraction": 0.15}),
        json.dumps({"client_order_id": "MyOrder", "canonical": "NQ", "side": "SELL",
                    "qty": 0.5, "bid": 18940.0, "ask": 18960.0,
                    "broker_time": "2026-06-01T17:05:00",
                    "intent": "EXIT", "target_fraction": 0.0}),
    ]
    (slip_dir / "submits.jsonl").write_text("\n".join(submits) + "\n", encoding="utf-8")

    # forecasts.jsonl: 2 rows
    forecasts = [
        json.dumps({"schema_version": 1, "broker": broker, "as_of": "2026-06-01",
                    "canonical": "NQ", "forecast_score": 1.23, "target_fraction": 0.15,
                    "engine_config_hash": "abc123", "vault_root": "/vault",
                    "warmup_ready": 1}),
        json.dumps({"schema_version": 1, "broker": broker, "as_of": "2026-06-01",
                    "canonical": "GC", "forecast_score": -0.5, "target_fraction": 0.05,
                    "engine_config_hash": "abc123", "vault_root": "/vault",
                    "warmup_ready": 1}),
    ]
    (state_dir / "forecasts.jsonl").write_text("\n".join(forecasts) + "\n", encoding="utf-8")

    # equity_history.jsonl: 2 rows
    equity = [
        json.dumps({"ts": "2026-06-01T17:00:00", "equity": 10000.0}),
        json.dumps({"ts": "2026-06-01T18:00:00", "equity": 10050.0}),
    ]
    (state_dir / "equity_history.jsonl").write_text("\n".join(equity) + "\n", encoding="utf-8")


def test_ingest_row_counts(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    _seed_account(conn, broker="darwinex", login="12345", server="test-server")

    cov = ingest_broker(conn, "darwinex", repo_root=tmp_path)

    # 3 valid deals (1 torn line skipped)
    assert cov["deals"]["rows"] == 3
    assert cov["deals"]["skipped"] == 1

    # 2 orders
    assert cov["orders"]["rows"] == 2

    # 2 forecasts
    assert cov["forecasts"]["rows"] == 2

    # 2 equity snapshots
    assert cov["equity"]["rows"] == 2

    conn.close()


def test_ingest_comment_normalization(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    _seed_account(conn, broker="darwinex", login="12345", server="test-server")
    ingest_broker(conn, "darwinex", repo_root=tmp_path)

    deals = conn.execute(
        "SELECT ticket, client_order_id FROM deals ORDER BY ticket"
    ).fetchall()
    by_ticket = {r["ticket"]: r["client_order_id"] for r in deals}

    # ticket 1001: comment 'X2' → kept as-is
    assert by_ticket[1001] == "X2"
    # ticket 1002: comment 'close:MyOrder' → 'MyOrder'
    assert by_ticket[1002] == "MyOrder"
    # ticket 1003: comment 'vault-flatten' → NULL
    assert by_ticket[1003] is None

    conn.close()


def test_ingest_torn_last_line_not_fatal(tmp_path: Path) -> None:
    """The torn last line in deals.jsonl must be skipped, not raise."""
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    _seed_account(conn, broker="darwinex", login="12345", server="test-server")

    # Should not raise
    import warnings
    with warnings.catch_warnings(record=True):
        cov = ingest_broker(conn, "darwinex", repo_root=tmp_path)

    # 4 lines in deals.jsonl, 1 is torn → 3 inserted, 1 skipped
    assert cov["deals"]["rows"] == 3
    assert cov["deals"]["skipped"] == 1

    conn.close()


def test_ingest_idempotent(tmp_path: Path) -> None:
    """Running ingest twice must not add duplicate rows."""
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    _seed_account(conn, broker="darwinex", login="12345", server="test-server")

    ingest_broker(conn, "darwinex", repo_root=tmp_path)
    count_after_1 = conn.execute("SELECT count(*) FROM deals").fetchone()[0]

    ingest_broker(conn, "darwinex", repo_root=tmp_path)
    count_after_2 = conn.execute("SELECT count(*) FROM deals").fetchone()[0]

    assert count_after_1 == count_after_2 == 3

    conn.close()


def test_ingest_submits_intent_stored(tmp_path: Path) -> None:
    """Intent and target_fraction from submits.jsonl reach the orders table."""
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    _seed_account(conn, broker="darwinex", login="12345", server="test-server")
    ingest_broker(conn, "darwinex", repo_root=tmp_path)

    rows = conn.execute(
        "SELECT client_order_id, intent, target_fraction FROM orders ORDER BY client_order_id"
    ).fetchall()
    by_coid = {r["client_order_id"]: r for r in rows}

    assert by_coid["X2"]["intent"] == "ENTRY"
    assert abs(by_coid["X2"]["target_fraction"] - 0.15) < 1e-9
    assert by_coid["MyOrder"]["intent"] == "EXIT"

    conn.close()


def test_ingest_v_slippage_queryable(tmp_path: Path) -> None:
    """v_slippage must be queryable and compute spread_bps for the joined deal."""
    conn = _open(tmp_path)
    _build_synthetic_fixture(tmp_path, "darwinex", login="12345")
    acct_id = _seed_account(conn, broker="darwinex", login="12345", server="test-server")
    ingest_broker(conn, "darwinex", repo_root=tmp_path)

    rows = conn.execute(
        "SELECT * FROM v_slippage WHERE account_id=?", (acct_id,)
    ).fetchall()
    # At least one row should have spread_bps computed (ticket 1001 joined with X2 submit)
    rows_with_spread = [r for r in rows if r["spread_bps"] is not None]
    assert len(rows_with_spread) >= 1

    r = rows_with_spread[0]
    assert r["spread_bps"] > 0

    conn.close()


def test_ingest_all_multi_broker(tmp_path: Path) -> None:
    """ingest_all processes multiple brokers from broker_cache/."""
    conn = _open(tmp_path)

    for broker in ("darwinex", "ftmo"):
        _build_synthetic_fixture(tmp_path, broker, login=f"{broker[:3]}111")
        _seed_account(conn, broker=broker, login=f"{broker[:3]}111",
                      server=f"{broker}-demo")

    # Patch live_state_dir so ingest_all finds both brokers via directory scan
    cov = ingest_all(conn, repo_root=tmp_path)
    assert "darwinex" in cov
    assert "ftmo" in cov

    conn.close()


def test_ingest_equity_deduplicates_overlap(tmp_path: Path) -> None:
    """Identical (account_id, ts) from two paths must not create duplicate rows."""
    conn = _open(tmp_path)
    state_dir = _make_broker_tree(tmp_path, "darwinex")
    _write_snapshot(state_dir, login="12345", server="dw-demo")
    acct_id = _seed_account(conn, broker="darwinex", login="12345", server="dw-demo")

    # Write the same ts to both equity.jsonl (durable) and equity_history.jsonl
    eq_row = json.dumps({"ts": "2026-06-01T17:00:00", "equity": 10000.0})
    (state_dir / "equity_history.jsonl").write_text(eq_row + "\n", encoding="utf-8")
    durable_path = tmp_path / "data" / "broker_cache" / "darwinex" / "equity.jsonl"
    durable_path.write_text(eq_row + "\n", encoding="utf-8")

    # Also write a unique row in durable file
    unique_row = json.dumps({"ts": "2026-06-01T18:00:00", "equity": 10100.0})
    with durable_path.open("a", encoding="utf-8") as f:
        f.write(unique_row + "\n")

    # Minimal deals/orders/forecasts
    (state_dir / "deals.jsonl").write_text("", encoding="utf-8")
    (_slippage_dir(tmp_path, "darwinex") / "submits.jsonl").write_text(
        "", encoding="utf-8"
    )
    (state_dir / "forecasts.jsonl").write_text("", encoding="utf-8")

    ingest_broker(conn, "darwinex", repo_root=tmp_path)

    count = conn.execute(
        "SELECT count(*) FROM equity_snapshots WHERE account_id=?", (acct_id,)
    ).fetchone()[0]
    # 2 unique timestamps total (not 3 from overlap)
    assert count == 2

    conn.close()
