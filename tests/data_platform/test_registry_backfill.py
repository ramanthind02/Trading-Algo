"""Unit tests for data_platform.registry.backfill (migration plan §7.7).

Uses synthetic fixtures in tmp_path — no real broker connections.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from data_platform.registry import db
from data_platform.registry.live_ingest import accounts_add
from data_platform.registry.backfill import (
    _broker_from_label_or_server,
    _resolve_single_active_account,
    run_backfill,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _open(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "registry.db")


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


def _write_slippage_csv(path: Path, rows: list[dict]) -> None:
    """Write a slippage.csv with the standard header."""
    fieldnames = [
        "ticket", "broker", "canonical", "symbol", "side", "entry",
        "volume", "broker_time", "fill_px", "bid", "ask", "mid",
        "spread_bps", "expected_touch_px", "slip_vs_touch_bps",
        "slip_vs_mid_bps", "commission", "swap", "order", "client_order_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_audit_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── synthetic fixtures ────────────────────────────────────────────────────────

_SLIPPAGE_ROWS = [
    {
        "ticket": 1001, "broker": "darwinex", "canonical": "ES", "symbol": "SP500",
        "side": "BUY", "entry": "IN", "volume": 0.11,
        "broker_time": "2026-06-08T23:45:16", "fill_px": 7400.0,
        "bid": "", "ask": "", "mid": "", "spread_bps": "", "expected_touch_px": "",
        "slip_vs_touch_bps": "", "slip_vs_mid_bps": "",
        "commission": -0.03, "swap": 0.0, "order": 2001, "client_order_id": "",
    },
    {
        "ticket": 1002, "broker": "darwinex", "canonical": "NQ", "symbol": "NDX",
        "side": "SELL", "entry": "OUT", "volume": 0.03,
        "broker_time": "2026-06-08T23:52:36", "fill_px": 29000.0,
        "bid": "", "ask": "", "mid": "", "spread_bps": "", "expected_touch_px": "",
        "slip_vs_touch_bps": "", "slip_vs_mid_bps": "",
        "commission": -0.08, "swap": 0.0, "order": 2002, "client_order_id": "",
    },
    {
        # INOUT entry — must be preserved as-is
        "ticket": 1003, "broker": "darwinex", "canonical": "GC", "symbol": "XAUUSD",
        "side": "BUY", "entry": "INOUT", "volume": 0.02,
        "broker_time": "2026-06-08T23:52:37", "fill_px": 3200.0,
        "bid": "", "ask": "", "mid": "", "spread_bps": "", "expected_touch_px": "",
        "slip_vs_touch_bps": "", "slip_vs_mid_bps": "",
        "commission": -0.22, "swap": 0.0, "order": 2003, "client_order_id": "",
    },
]

_AUDIT_DATA = {
    "run_id": "test-run-001",
    "timestamp_utc": "2026-06-01T17:00:00",
    "plans": [
        {
            "label": "ftmo_challenge",
            "login": "99999",            # NOT in accounts → synthetic retired account
            "server": "ftmo-demo",
            "currency": "USD",
            "balance": 50000.0,
            "equity": 50100.0,
            "margin_free": 45000.0,
            "actions_by_symbol": {},
        }
    ],
    "reports": [
        # Successful report with a real deal_ticket → should become a Deal row
        {"action": "close", "success": True, "deal_ticket": 5001, "retcode": 10009, "comment": "ok"},
        # Failed report → must be skipped
        {"action": "open", "success": False, "deal_ticket": None, "retcode": 10006, "comment": "no funds"},
    ],
}


# ── helpers / unit tests ──────────────────────────────────────────────────────

def test_broker_from_label_known() -> None:
    assert _broker_from_label_or_server("ftmo_challenge", "ftmo-demo") == "ftmo"
    assert _broker_from_label_or_server("fundednext_plan", "") == "fundednext"
    assert _broker_from_label_or_server("", "darwinex-real") == "darwinex"


def test_broker_from_label_fallback() -> None:
    """Unknown broker → first word of label, lower-cased."""
    broker = _broker_from_label_or_server("MyBroker plan", "mybroker-demo")
    assert broker == "mybroker"


def test_resolve_single_active_account_two_actives_raises(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    _seed_account(conn, broker="ftmo", login="A", server="s1")
    _seed_account(conn, broker="ftmo", login="B", server="s1")
    with pytest.raises(ValueError, match="2 active accounts"):
        _resolve_single_active_account(conn, "ftmo")
    conn.close()


def test_resolve_single_active_account_no_account_raises(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with pytest.raises(ValueError, match="No active account"):
        _resolve_single_active_account(conn, "ghost")
    conn.close()


# ── backfill: slippage CSV ────────────────────────────────────────────────────

def _setup_slippage_fixture(tmp_path: Path) -> sqlite3.Connection:
    """Seed a darwinex active account + write synthetic slippage.csv."""
    conn = _open(tmp_path)
    _seed_account(conn, broker="darwinex", login="12345", server="dw-demo")
    slip_csv = tmp_path / "data" / "broker_cache" / "darwinex" / "slippage" / "slippage.csv"
    _write_slippage_csv(slip_csv, _SLIPPAGE_ROWS)
    return conn


def test_slippage_row_count(tmp_path: Path) -> None:
    """All 3 slippage rows are inserted as deals."""
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    count = conn.execute("SELECT count(*) FROM deals").fetchone()[0]
    assert count == 3
    conn.close()


def test_slippage_buy_deal_type(tmp_path: Path) -> None:
    """BUY → deal_type=0."""
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    row = conn.execute("SELECT deal_type FROM deals WHERE ticket=1001").fetchone()
    assert row is not None
    assert row["deal_type"] == 0
    conn.close()


def test_slippage_sell_deal_type(tmp_path: Path) -> None:
    """SELL → deal_type=1."""
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    row = conn.execute("SELECT deal_type FROM deals WHERE ticket=1002").fetchone()
    assert row is not None
    assert row["deal_type"] == 1
    conn.close()


def test_slippage_inout_entry_preserved(tmp_path: Path) -> None:
    """INOUT entry value is stored unchanged."""
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    row = conn.execute("SELECT entry FROM deals WHERE ticket=1003").fetchone()
    assert row is not None
    assert row["entry"] == "INOUT"
    conn.close()


def test_slippage_order_ticket_mapped(tmp_path: Path) -> None:
    """``order`` CSV column maps to ``order_ticket``."""
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    row = conn.execute("SELECT order_ticket FROM deals WHERE ticket=1001").fetchone()
    assert row["order_ticket"] == 2001
    conn.close()


def test_slippage_magic_is_510(tmp_path: Path) -> None:
    conn = _setup_slippage_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    rows = conn.execute("SELECT magic FROM deals").fetchall()
    assert all(r["magic"] == 510 for r in rows)
    conn.close()


# ── backfill: audit JSONs ─────────────────────────────────────────────────────

def _setup_audit_fixture(tmp_path: Path, seed_darwinex: bool = True) -> sqlite3.Connection:
    """Optionally seed a darwinex active account; write synthetic audit JSON."""
    conn = _open(tmp_path)
    if seed_darwinex:
        _seed_account(conn, broker="darwinex", login="12345", server="dw-demo")
    audit_file = tmp_path / "logs" / "cfd_prop_audit" / "audit_001.json"
    _write_audit_json(audit_file, _AUDIT_DATA)
    return conn


def test_audit_synthetic_retired_account_created(tmp_path: Path) -> None:
    """Login 99999 is not in accounts → a synthetic retired account is created."""
    conn = _setup_audit_fixture(tmp_path, seed_darwinex=False)
    run_backfill(conn, repo_root=tmp_path)

    rows = conn.execute(
        "SELECT status, notes, broker FROM accounts WHERE login='99999'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "retired"
    assert rows[0]["notes"] == "legacy cfd_prop audit backfill"
    assert rows[0]["broker"] == "ftmo"
    conn.close()


def test_audit_successful_report_creates_deal(tmp_path: Path) -> None:
    """One successful report with deal_ticket=5001 → one Deal row."""
    conn = _setup_audit_fixture(tmp_path, seed_darwinex=False)
    run_backfill(conn, repo_root=tmp_path)

    row = conn.execute("SELECT ticket FROM deals WHERE ticket=5001").fetchone()
    assert row is not None
    conn.close()


def test_audit_failed_report_skipped(tmp_path: Path) -> None:
    """The failed report (success=False, deal_ticket=None) must NOT be in deals."""
    conn = _setup_audit_fixture(tmp_path, seed_darwinex=False)
    run_backfill(conn, repo_root=tmp_path)

    # Only one deal from reports (the successful one); the failed one has no ticket
    deal_count = conn.execute("SELECT count(*) FROM deals").fetchone()[0]
    assert deal_count == 1
    conn.close()


def test_audit_equity_snapshot_inserted(tmp_path: Path) -> None:
    """Plan balance/equity at timestamp_utc becomes an equity_snapshot row."""
    conn = _setup_audit_fixture(tmp_path, seed_darwinex=False)
    run_backfill(conn, repo_root=tmp_path)

    rows = conn.execute("SELECT balance, equity FROM equity_snapshots").fetchall()
    assert len(rows) == 1
    assert abs(rows[0]["balance"] - 50000.0) < 1e-6
    assert abs(rows[0]["equity"] - 50100.0) < 1e-6
    conn.close()


# ── combined fixture: slippage + audit ────────────────────────────────────────

def _setup_combined_fixture(tmp_path: Path) -> sqlite3.Connection:
    conn = _open(tmp_path)
    _seed_account(conn, broker="darwinex", login="12345", server="dw-demo")
    slip_csv = tmp_path / "data" / "broker_cache" / "darwinex" / "slippage" / "slippage.csv"
    _write_slippage_csv(slip_csv, _SLIPPAGE_ROWS)
    audit_file = tmp_path / "logs" / "cfd_prop_audit" / "audit_001.json"
    _write_audit_json(audit_file, _AUDIT_DATA)
    return conn


def test_combined_total_deals(tmp_path: Path) -> None:
    """3 slippage deals + 1 audit deal = 4 total deals."""
    conn = _setup_combined_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    count = conn.execute("SELECT count(*) FROM deals").fetchone()[0]
    assert count == 4
    conn.close()


def test_idempotent_no_new_rows_on_rerun(tmp_path: Path) -> None:
    """Re-running backfill must not insert duplicate rows."""
    conn = _setup_combined_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    deals_after_1 = conn.execute("SELECT count(*) FROM deals").fetchone()[0]
    equity_after_1 = conn.execute("SELECT count(*) FROM equity_snapshots").fetchone()[0]
    accounts_after_1 = conn.execute("SELECT count(*) FROM accounts").fetchone()[0]

    run_backfill(conn, repo_root=tmp_path)
    assert conn.execute("SELECT count(*) FROM deals").fetchone()[0] == deals_after_1
    assert conn.execute("SELECT count(*) FROM equity_snapshots").fetchone()[0] == equity_after_1
    assert conn.execute("SELECT count(*) FROM accounts").fetchone()[0] == accounts_after_1
    conn.close()


def test_job_run_row_recorded(tmp_path: Path) -> None:
    """run_backfill always records exactly one job_runs row per call."""
    conn = _setup_combined_fixture(tmp_path)
    run_backfill(conn, repo_root=tmp_path)
    rows = conn.execute(
        "SELECT job_name, exit_code FROM job_runs WHERE job_name='registry_ingest_backfill'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["exit_code"] == 0
    conn.close()


def test_missing_audit_dir_does_not_raise(tmp_path: Path) -> None:
    """Missing logs/cfd_prop_audit/ is silently skipped (no exception)."""
    conn = _open(tmp_path)
    _seed_account(conn, broker="darwinex", login="12345", server="dw-demo")
    slip_csv = tmp_path / "data" / "broker_cache" / "darwinex" / "slippage" / "slippage.csv"
    _write_slippage_csv(slip_csv, _SLIPPAGE_ROWS)
    # No audit dir created
    result = run_backfill(conn, repo_root=tmp_path)
    assert result["audit"]["deals"] == 0
    assert result["audit"]["equity_snapshots"] == 0
    conn.close()


def test_missing_broker_cache_does_not_raise(tmp_path: Path) -> None:
    """Missing data/broker_cache/ is silently skipped (no exception)."""
    conn = _open(tmp_path)
    # No broker_cache directory
    audit_file = tmp_path / "logs" / "cfd_prop_audit" / "audit_001.json"
    _write_audit_json(audit_file, _AUDIT_DATA)
    result = run_backfill(conn, repo_root=tmp_path)
    assert result["slippage_csv_deals"] == {}
    conn.close()
