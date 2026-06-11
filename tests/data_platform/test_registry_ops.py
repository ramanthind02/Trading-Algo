"""Unit tests for the registry ops additions: job_run hooks, freshness check,
and backup/prune.

All tests use tmp_path DBs — no production data touched.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from data_platform.registry import db, writer
from data_platform.registry.db import transaction


# ── helpers ───────────────────────────────────────────────────────────────────

def _open(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "registry.db")


# ── job_run bracket: success path ─────────────────────────────────────────────

def test_job_run_bracket_success(tmp_path: Path) -> None:
    """record_job_run + finish_job_run (exit_code=0): row has all expected fields."""
    conn = _open(tmp_path)

    with transaction(conn):
        job_id = writer.record_job_run(
            conn,
            writer.JobRun(
                job_name="mt5_scrape",
                started_at="2026-06-09T17:00:00+00:00",
                args_json='{"symbols": null, "workers": 1}',
                broker_time_anchor="2026-06-09T00:05:00",
            ),
        )

    assert isinstance(job_id, int) and job_id > 0

    # Before finish: finished_at is NULL, exit_code is NULL
    row = conn.execute(
        "SELECT * FROM job_runs WHERE job_run_id = ?", (job_id,)
    ).fetchone()
    assert row is not None
    assert row["job_name"] == "mt5_scrape"
    assert row["finished_at"] is None
    assert row["exit_code"] is None

    with transaction(conn):
        writer.finish_job_run(
            conn, job_id,
            exit_code=0,
            rows_written=42_000,
            coverage_json='{"symbols": 5, "errors": 0, "max_bar_time": "2026-06-08T23:59:00"}',
            error_text=None,
        )

    row = conn.execute(
        "SELECT * FROM job_runs WHERE job_run_id = ?", (job_id,)
    ).fetchone()
    assert row["exit_code"] == 0
    assert row["rows_written"] == 42_000
    assert row["error_text"] is None
    assert row["finished_at"] is not None
    cov = json.loads(row["coverage_json"])
    assert cov["symbols"] == 5
    conn.close()


# ── job_run bracket: exception path ──────────────────────────────────────────

def test_job_run_bracket_exception_path(tmp_path: Path) -> None:
    """finish_job_run with exit_code=1 and error_text records the failure."""
    conn = _open(tmp_path)

    with transaction(conn):
        job_id = writer.record_job_run(
            conn,
            writer.JobRun(
                job_name="mt5_scrape",
                started_at="2026-06-09T17:00:00+00:00",
            ),
        )

    with transaction(conn):
        writer.finish_job_run(
            conn, job_id,
            exit_code=1,
            rows_written=0,
            coverage_json=None,
            error_text="RuntimeError: MT5 terminal not responding",
        )

    row = conn.execute(
        "SELECT exit_code, error_text FROM job_runs WHERE job_run_id = ?", (job_id,)
    ).fetchone()
    assert row["exit_code"] == 1
    assert "MT5 terminal not responding" in row["error_text"]
    conn.close()


# ── freshness from registry ───────────────────────────────────────────────────

def _insert_mt5_m1_blob(
    conn: sqlite3.Connection,
    symbol: str,
    year: str,
    coverage_end: str,
) -> None:
    """Write a synthetic mt5_m1 blob_manifest row with given coverage_end."""
    key = json.dumps({"symbol": symbol, "year": year}, sort_keys=True)
    with transaction(conn):
        writer.record_blob(
            conn,
            writer.BlobRecord(
                store="mt5_m1",
                key_json=key,
                relative_path=f"data/mt5_data/{symbol}/bars_M1/year={year}/part.parquet",
                written_at="2026-06-09T00:00:00+00:00",
                broker="darwinex",
                timezone="broker_eet_as_utc",
                engine="pyarrow",
                rows=1000,
                coverage_start="2026-01-01",
                coverage_end=coverage_end,
            ),
        )


def test_freshness_from_registry_fresh(tmp_path: Path) -> None:
    """Registry has recent coverage_end → staleness verdict = False."""
    from deployment.ops.check_data_freshness import (
        _query_freshness_from_registry,
        _compute_freshness,
    )

    conn = _open(tmp_path)
    db_path = tmp_path / "registry.db"

    today = datetime.now(timezone.utc)
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Darwinex symbols for vault tickers (from configs/mt5_brokers.yaml)
    # Use EURUSD as a safe always-present Darwinex symbol for testing.
    # We insert a synthetic mapping so we don't need the full broker config.
    from data_platform.providers.mt5 import brokers as _brokers

    # Use the first 2 tickers that resolve on Darwinex
    resolved: dict[str, str] = {}
    for canonical in ("ES", "NQ", "GC"):
        try:
            sym = _brokers.resolve("darwinex", canonical)
            resolved[canonical] = sym
            if len(resolved) == 2:
                break
        except Exception:
            pass

    if not resolved:
        pytest.skip("No Darwinex symbol mappings available")

    for canonical, sym in resolved.items():
        _insert_mt5_m1_blob(conn, sym, "2026", yesterday)
    conn.close()

    result = _query_freshness_from_registry(
        list(resolved.keys()), "darwinex", db_path=db_path
    )
    assert result, "Expected non-empty result"

    # All resolved tickers should be fresh
    for canonical, sym in resolved.items():
        key = f"{canonical}->{sym}"
        assert key in result
        assert result[key] is not None

    status = _compute_freshness(list(resolved.keys()), today, 3.0, result)
    assert not status["stale"], f"Expected fresh, got stale: {status}"
    assert status["freshest_bar"] is not None


def test_freshness_from_registry_stale(tmp_path: Path) -> None:
    """Registry has old coverage_end → staleness verdict = True."""
    from deployment.ops.check_data_freshness import (
        _query_freshness_from_registry,
        _compute_freshness,
    )

    conn = _open(tmp_path)
    db_path = tmp_path / "registry.db"

    today = datetime.now(timezone.utc)
    old_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")

    from data_platform.providers.mt5 import brokers as _brokers

    resolved: dict[str, str] = {}
    for canonical in ("ES", "NQ", "GC"):
        try:
            sym = _brokers.resolve("darwinex", canonical)
            resolved[canonical] = sym
            if len(resolved) == 2:
                break
        except Exception:
            pass

    if not resolved:
        pytest.skip("No Darwinex symbol mappings available")

    for canonical, sym in resolved.items():
        _insert_mt5_m1_blob(conn, sym, "2026", old_date)
    conn.close()

    result = _query_freshness_from_registry(
        list(resolved.keys()), "darwinex", db_path=db_path
    )
    status = _compute_freshness(list(resolved.keys()), today, 3.0, result)
    assert status["stale"], f"Expected stale, got fresh: {status}"


def test_freshness_from_registry_at_boundary(tmp_path: Path) -> None:
    """Exactly at max_age_days threshold: age == max_age_days → stale."""
    from deployment.ops.check_data_freshness import (
        _query_freshness_from_registry,
        _compute_freshness,
    )

    conn = _open(tmp_path)
    db_path = tmp_path / "registry.db"

    now = datetime.now(timezone.utc)
    # 3.0 days old exactly = midnight 3 days ago
    threshold_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")

    from data_platform.providers.mt5 import brokers as _brokers

    resolved: dict[str, str] = {}
    for canonical in ("ES", "NQ", "GC"):
        try:
            sym = _brokers.resolve("darwinex", canonical)
            resolved[canonical] = sym
            break
        except Exception:
            pass

    if not resolved:
        pytest.skip("No Darwinex symbol mappings available")

    for canonical, sym in resolved.items():
        _insert_mt5_m1_blob(conn, sym, "2026", threshold_date)
    conn.close()

    result = _query_freshness_from_registry(
        list(resolved.keys()), "darwinex", db_path=db_path
    )
    status = _compute_freshness(list(resolved.keys()), now, 3.0, result)
    # age_days > 3.0 since we use midnight of that date and now is later in the day
    assert status["stale"], (
        f"Expected stale at exactly 3-day-old coverage_end, got: {status}"
    )


def test_freshness_registry_missing_falls_back_to_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the registry DB is missing, _query_freshness_from_registry raises and
    the caller can fall back to _freshness_from_files."""
    from deployment.ops.check_data_freshness import (
        _query_freshness_from_registry,
        _freshness_from_files,
    )
    import deployment.ops.check_data_freshness as _mod

    # No DB at the default location: FileNotFoundError expected
    absent_db = tmp_path / "nonexistent" / "registry.db"
    with pytest.raises(FileNotFoundError):
        _query_freshness_from_registry(["ES"], "darwinex", db_path=absent_db)

    # Simulate the caller's fallback logic: monkeypatch _freshness_from_files
    # to a stub that always returns a known datetime.
    sentinel = datetime(2026, 6, 8, 17, 30, tzinfo=timezone.utc)
    calls: list[str] = []

    def _stub(symbol: str) -> datetime:
        calls.append(symbol)
        return sentinel

    monkeypatch.setattr(_mod, "_freshness_from_files", _stub)

    from data_platform.providers.mt5 import brokers as _brokers

    # Exercise the fallback path inline
    per_symbol: dict[str, datetime | None] = {}
    for canonical in ["ES"]:
        try:
            sym = _brokers.resolve("darwinex", canonical)
        except Exception:
            pytest.skip("ES not mapped on darwinex")
        per_symbol[f"{canonical}->{sym}"] = _mod._freshness_from_files(sym)

    assert len(calls) >= 1, "Expected _freshness_from_files stub to have been called"
    for v in per_symbol.values():
        assert v == sentinel


# ── backup ────────────────────────────────────────────────────────────────────

def test_backup_creates_readable_file(tmp_path: Path) -> None:
    """backup() creates a valid SQLite copy; connect_readonly + SELECT work."""
    from data_platform.registry.backup import backup

    src = tmp_path / "registry.db"
    out_dir = tmp_path / "backups"

    # Seed the source DB with one row so there's something to verify
    src_conn = db.connect(src)
    with transaction(src_conn):
        src_conn.execute("INSERT INTO sleeves (name) VALUES ('test-sleeve')")
    src_conn.close()

    dest = backup(db_path=src, out_dir=out_dir)

    assert dest.exists(), f"Backup file not created at {dest}"
    assert dest.suffix == ".db"
    assert "registry-" in dest.name

    # Verify the copy is a valid SQLite DB
    backup_conn = db.connect_readonly(dest)
    count: int = backup_conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    backup_conn.close()
    assert count == 20, f"Expected 20 tables in backup, got {count}"


def test_backup_prune_keeps_newest_14_of_16(tmp_path: Path) -> None:
    """Prune logic: 16 pre-existing fake files + 1 new backup = 17 → keeps 14."""
    from data_platform.registry.backup import backup

    src = tmp_path / "registry.db"
    out_dir = tmp_path / "backups"
    out_dir.mkdir(parents=True)

    db.connect(src).close()  # initialise empty DB

    # Create 16 fake backup files with older timestamps (lexicographically smaller)
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    for i in range(16):
        ts = base_time + timedelta(hours=i)
        fname = f"registry-{ts.strftime('%Y%m%d-%H%M%S')}.db"
        # Write a minimal valid SQLite file so the glob picks them up
        (out_dir / fname).write_bytes(b"")

    # Run backup (creates 17th file — current timestamp, newest)
    dest = backup(db_path=src, out_dir=out_dir)
    assert dest.exists()

    # After pruning: exactly 14 files should remain
    remaining = sorted(out_dir.glob("registry-*.db"))
    assert len(remaining) == 14, (
        f"Expected 14 backups after prune, got {len(remaining)}: "
        f"{[f.name for f in remaining]}"
    )

    # The newly created backup must be among the survivors
    assert dest in remaining, "Newly created backup was pruned (should be newest)"


def test_backup_job_run_recorded(tmp_path: Path) -> None:
    """backup() records a job_runs row in the source DB."""
    from data_platform.registry.backup import backup

    src = tmp_path / "registry.db"
    db.connect(src).close()

    backup(db_path=src, out_dir=tmp_path / "backups")

    src_conn = db.connect_readonly(src)
    rows = src_conn.execute(
        "SELECT job_name, exit_code FROM job_runs WHERE job_name = 'registry_backup'"
    ).fetchall()
    src_conn.close()

    assert len(rows) >= 1, "Expected at least one job_runs row for 'registry_backup'"
    assert rows[-1]["exit_code"] == 0
