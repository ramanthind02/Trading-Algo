"""Unit tests for data_platform.registry core (db / writer / reader).

All tests run on tmp_path DBs — no prod data touched.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from data_platform.registry import db, reader, writer
from data_platform.registry.db import RegistryVersionError, transaction
from data_platform.registry.writer import (
    Account,
    Deal,
    EquitySnapshot,
    Forecast,
    Instrument,
    Order,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _open(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "registry.db")


def _account(account_id: int = 1) -> Account:
    return Account(
        account_id=account_id,
        broker="darwinex",
        login="12345",
        server="darwinex-demo",
        exec_tier="demo",
        program_phase="retail",
        valid_from="2026-01-01",
    )


# ── schema creation ───────────────────────────────────────────────────────────

def test_connect_creates_schema_and_version(tmp_path: Path) -> None:
    """Fresh connect: 20 tables, 2 views, user_version=1."""
    conn = _open(tmp_path)

    rows = conn.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table','view') ORDER BY type, name"
    ).fetchall()
    tables = [r["name"] for r in rows if r["type"] == "table"]
    views  = [r["name"] for r in rows if r["type"] == "view"]

    assert len(tables) == 20, f"Expected 20 tables, got {len(tables)}: {tables}"
    assert len(views)  == 2,  f"Expected 2 views, got {len(views)}: {views}"

    user_version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == db.SCHEMA_VERSION == 1

    conn.close()


def test_connect_existing_correct_version(tmp_path: Path) -> None:
    """Re-opening an already-initialised DB does not re-apply the schema."""
    path = tmp_path / "registry.db"
    c1 = db.connect(path)
    c1.close()
    c2 = db.connect(path)          # must not raise
    count: int = c2.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    assert count == 20
    c2.close()


# ── version mismatch ──────────────────────────────────────────────────────────

def test_connect_version_mismatch_raises(tmp_path: Path) -> None:
    """A DB whose user_version != SCHEMA_VERSION raises RegistryVersionError."""
    path = tmp_path / "stale.db"
    boot = sqlite3.connect(str(path))
    boot.execute("PRAGMA user_version = 99")
    boot.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")
    boot.commit()
    boot.close()

    with pytest.raises(RegistryVersionError, match="99"):
        db.connect(path)


# ── read-only connection ──────────────────────────────────────────────────────

def test_connect_readonly_select_works(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    db.connect(path).close()

    ro = db.connect_readonly(path)
    count: int = ro.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    assert count == 20
    ro.close()


def test_connect_readonly_insert_raises(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    db.connect(path).close()

    ro = db.connect_readonly(path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute("INSERT INTO sleeves (name) VALUES ('should-fail')")
    ro.close()


def test_connect_readonly_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        db.connect_readonly(tmp_path / "nonexistent.db")


# ── transaction context manager ───────────────────────────────────────────────

def test_transaction_commits_on_success(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with transaction(conn):
        conn.execute("INSERT INTO sleeves (name) VALUES ('test-ok')")
    rows = conn.execute("SELECT name FROM sleeves").fetchall()
    assert any(r["name"] == "test-ok" for r in rows)
    conn.close()


def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with pytest.raises(ValueError):
        with transaction(conn):
            conn.execute("INSERT INTO sleeves (name) VALUES ('should-roll-back')")
            raise ValueError("intentional rollback")
    rows = conn.execute("SELECT * FROM sleeves WHERE name='should-roll-back'").fetchall()
    assert len(rows) == 0
    conn.close()


# ── FK enforcement ────────────────────────────────────────────────────────────

def test_fk_deal_without_account_raises(tmp_path: Path) -> None:
    """foreign_keys=ON: deal with unknown account_id must raise IntegrityError."""
    conn = _open(tmp_path)
    bad_deal = Deal(
        account_id=9999, ticket=1, symbol="EURUSD",
        deal_type=0, volume=1.0, price=1.10,
        broker_time="2026-06-01T00:00:00",
    )
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(conn):
            writer.insert_deal(conn, bad_deal)
    conn.close()


# ── writer round-trip: account → order → deal → v_slippage ───────────────────

def test_slippage_view_computed_values(tmp_path: Path) -> None:
    """BUY fill at ask: spread=2/101×1e4, slip_vs_touch=0, slip_vs_mid=1/101×1e4."""
    conn = _open(tmp_path)

    acct  = _account()
    order = Order(
        account_id=1, client_order_id="ord-001",
        canonical="NQ", symbol="NAS100",
        side="BUY", qty=1.0,
        broker_time_submit="2026-06-01T17:00:00",
        bid_at_submit=100.0, ask_at_submit=102.0,
    )
    deal = Deal(
        account_id=1, ticket=9001, symbol="NAS100",
        deal_type=0, volume=1.0, price=102.0,
        broker_time="2026-06-01T17:00:01",
        client_order_id="ord-001",
    )

    with transaction(conn):
        writer.upsert_account(conn, acct)
        writer.insert_order(conn, order)
        writer.insert_deal(conn, deal)

    rows = conn.execute(
        "SELECT * FROM v_slippage WHERE account_id=1 AND ticket=9001"
    ).fetchall()
    assert len(rows) == 1
    r = rows[0]

    mid            = (100.0 + 102.0) / 2.0          # 101.0
    spread_bps     = (102.0 - 100.0) / mid * 10_000  # ~198.02
    slip_vs_touch  = 0.0                             # BUY filled at ask
    slip_vs_mid    = (102.0 - mid) / mid * 10_000    # ~99.01

    assert abs(r["mid"]              - mid)           < 1e-9
    assert abs(r["spread_bps"]       - spread_bps)    < 1e-6
    assert abs(r["slip_vs_touch_bps"]- slip_vs_touch) < 1e-9
    assert abs(r["slip_vs_mid_bps"]  - slip_vs_mid)   < 1e-6

    conn.close()


# ── deal idempotence ──────────────────────────────────────────────────────────

def test_insert_deal_idempotent_latest_wins(tmp_path: Path) -> None:
    """Same (account_id, ticket) inserted twice → 1 row; latest values kept."""
    conn = _open(tmp_path)
    with transaction(conn):
        writer.upsert_account(conn, _account())

    v1 = Deal(account_id=1, ticket=42, symbol="EURUSD",
              deal_type=0, volume=1.0, price=1.1000,
              broker_time="2026-06-01T00:00:00", profit=0.0)
    v2 = Deal(account_id=1, ticket=42, symbol="EURUSD",
              deal_type=0, volume=2.0, price=1.1050,
              broker_time="2026-06-01T00:00:00", profit=5.0)

    with transaction(conn):
        writer.insert_deal(conn, v1)
    with transaction(conn):
        writer.insert_deal(conn, v2)

    rows = conn.execute(
        "SELECT * FROM deals WHERE account_id=1 AND ticket=42"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["volume"] == 2.0
    assert rows[0]["profit"] == 5.0
    conn.close()


# ── forecast + equity unique-key upserts ─────────────────────────────────────

def test_forecast_upsert_deduplicates(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with transaction(conn):
        writer.upsert_account(conn, _account())

    fc1 = Forecast(account_id=1, as_of="2026-06-01", canonical="NQ",
                   warmup_ready=1, forecast_score=1.5)
    fc2 = Forecast(account_id=1, as_of="2026-06-01", canonical="NQ",
                   warmup_ready=1, forecast_score=2.0)

    with transaction(conn):
        writer.insert_forecast(conn, fc1)
    with transaction(conn):
        writer.insert_forecast(conn, fc2)

    rows = conn.execute("SELECT * FROM forecast_history WHERE account_id=1").fetchall()
    assert len(rows) == 1
    assert rows[0]["forecast_score"] == 2.0
    conn.close()


def test_equity_snapshot_upsert_deduplicates(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with transaction(conn):
        writer.upsert_account(conn, _account())

    s1 = EquitySnapshot(account_id=1, ts="2026-06-01T00:00:00", equity=10_000.0)
    s2 = EquitySnapshot(account_id=1, ts="2026-06-01T00:00:00", equity=10_500.0)

    with transaction(conn):
        writer.insert_equity_snapshot(conn, s1)
    with transaction(conn):
        writer.insert_equity_snapshot(conn, s2)

    rows = conn.execute("SELECT * FROM equity_snapshots WHERE account_id=1").fetchall()
    assert len(rows) == 1
    assert rows[0]["equity"] == 10_500.0
    conn.close()


# ── instrument + source symbols ───────────────────────────────────────────────

def test_instrument_source_symbol_roundtrip(tmp_path: Path) -> None:
    """instrument_by_source_symbol returns the correct instruments row."""
    conn = _open(tmp_path)
    inst = Instrument(
        id="ES.XCME", raw_symbol="ES", asset_class="INDEX",
        instrument_class="FUTURE", price_precision=2, price_increment=0.25,
    )
    with transaction(conn):
        writer.upsert_instrument(conn, inst)
        writer.replace_source_symbols(conn, "ES.XCME", {
            "norgate_adj": "ES",
            "mt5": "ES_futures",
        })

    row = reader.instrument_by_source_symbol(conn, "mt5", "ES_futures")
    assert row is not None
    assert row["id"] == "ES.XCME"

    missing = reader.instrument_by_source_symbol(conn, "mt5", "UNKNOWN")
    assert missing is None
    conn.close()


def test_source_symbol_unique_constraint_raises(tmp_path: Path) -> None:
    """UNIQUE(source, native_symbol): two instruments claiming the same pair → IntegrityError."""
    conn = _open(tmp_path)
    inst1 = Instrument(id="ES.XCME",  raw_symbol="ES",  asset_class="INDEX",
                       instrument_class="FUTURE", price_precision=2, price_increment=0.25)
    inst2 = Instrument(id="ES2.XCME", raw_symbol="ES2", asset_class="INDEX",
                       instrument_class="FUTURE", price_precision=2, price_increment=0.25)

    with transaction(conn):
        writer.upsert_instrument(conn, inst1)
        writer.replace_source_symbols(conn, "ES.XCME", {"mt5": "ES_futures"})

    with transaction(conn):
        writer.upsert_instrument(conn, inst2)

    with pytest.raises(sqlite3.IntegrityError):
        with transaction(conn):
            # ("mt5", "ES_futures") already owned by ES.XCME
            writer.replace_source_symbols(conn, "ES2.XCME", {"mt5": "ES_futures"})

    conn.close()


# ── reader queries ────────────────────────────────────────────────────────────

def test_reader_accounts_active(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    with transaction(conn):
        writer.upsert_account(conn, _account(1))
        writer.upsert_account(conn, Account(
            account_id=2, broker="ftmo", login="99999",
            server="ftmo-demo", exec_tier="demo",
            program_phase="challenge", valid_from="2026-01-01",
            status="retired",
        ))
    active = reader.accounts_active(conn)
    assert len(active) == 1
    assert active[0]["account_id"] == 1
    conn.close()


def test_reader_job_history(tmp_path: Path) -> None:
    from data_platform.registry.writer import JobRun

    conn = _open(tmp_path)
    with transaction(conn):
        writer.record_job_run(conn, JobRun(job_name="mt5_scrape", started_at="2026-06-01T00:00:00"))
        writer.record_job_run(conn, JobRun(job_name="mt5_scrape", started_at="2026-06-02T00:00:00"))
        writer.record_job_run(conn, JobRun(job_name="signal_refresh", started_at="2026-06-01T12:00:00"))

    rows = reader.job_history(conn, "mt5_scrape", limit=10)
    assert len(rows) == 2
    # newest first
    assert rows[0]["started_at"] > rows[1]["started_at"]

    other = reader.job_history(conn, "signal_refresh")
    assert len(other) == 1
    conn.close()


def test_reader_coverage_and_freshness(tmp_path: Path) -> None:
    from data_platform.registry.writer import BlobRecord

    conn = _open(tmp_path)
    b1 = BlobRecord(store="ohlc_data", key_json='{"sym":"ES"}',
                    relative_path="ohlc_data/ES/D_ES.parquet",
                    written_at="2026-06-01T00:00:00",
                    coverage_start="2010-01-01", coverage_end="2026-06-01")
    b2 = BlobRecord(store="ohlc_data", key_json='{"sym":"NQ"}',
                    relative_path="ohlc_data/NQ/D_NQ.parquet",
                    written_at="2026-06-01T00:00:00",
                    coverage_start="2010-01-01", coverage_end="2026-05-30")
    with transaction(conn):
        writer.record_blob(conn, b1)
        writer.record_blob(conn, b2)

    cov = reader.coverage(conn, "ohlc_data")
    assert len(cov) == 2

    fresh = reader.freshness(conn, "ohlc_data")
    assert len(fresh) == 2
    ends = {r["key_json"]: r["coverage_end"] for r in fresh}
    assert ends['{"sym":"ES"}'] == "2026-06-01"
    assert ends['{"sym":"NQ"}'] == "2026-05-30"
    conn.close()
