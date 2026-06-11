"""M4.1 registry cutover — SpecRunManager persistence tests.

Four behavioural contracts:

1. ``start()`` captures ``spec_hash`` + ``spec_snapshot_json`` in the in-memory run AND the
   registry DB row at enqueue time (before the worker executes).
2. Completing a run writes ``headline_metrics_json`` to the registry DB.
3. A server restart (new ``SpecRunManager`` loading the same DB) marks any run that was
   ``queued`` or ``running`` as ``failed`` with "Interrupted by a server restart."
4. The frozen legacy ``_runs_index.json`` is NEVER written during the full lifecycle
   (start → complete → delete).
"""

from __future__ import annotations

import json

import pytest

from frontend.api.runs import SpecRunManager


# ── Helpers ───────────────────────────────────────────────────────────────────


def _payload(**overrides) -> dict:
    base = {
        "name": "unit_rsi_mr",
        "hypothesis": "test",
        "author": "tester",
        "created": "2026-01-01T00:00:00",
        "tickers": ["ES"],
        "data_feed": "darwinex_cfd",
        "mode": "daily",
        "timeframe": "D",
        "signal": {"module_name": "rsi_signal", "param_grid": {"rsi_period": [2, 3]}},
        "direction": "long_short",
        "vault": {"weight_hierarchy_group": "mean_reversion_indices", "ensemble_name": "x"},
    }
    base.update(overrides)
    return base


@pytest.fixture()
def stub_execute(monkeypatch):
    """No-op the heavy pipeline so start() only exercises path-derivation and registry writes."""
    monkeypatch.setattr(SpecRunManager, "_execute", staticmethod(lambda *a, **k: None))


# ── Test 1: spec_hash + snapshot captured at enqueue time ────────────────────


def test_start_captures_spec_hash_and_snapshot_in_db(stub_execute, tmp_path):
    """``start()`` stores spec_hash + spec_snapshot_json immediately (before the worker runs)."""
    from data_platform.registry import db as _db

    db_path = tmp_path / "registry.db"
    manager = SpecRunManager(db_path=db_path)
    payload = _payload()
    run = manager.start("unit_rsi_mr", payload)

    # In-memory fields are populated.
    assert run["spec_hash"] is not None, "spec_hash must be set"
    assert len(run["spec_hash"]) == 64, "spec_hash must be a sha256 hex string"
    assert run["spec_snapshot_json"] is not None
    snap = json.loads(run["spec_snapshot_json"])
    assert snap["name"] == "unit_rsi_mr"

    # The DB row is written before the worker thread gets to execute.
    conn = _db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT spec_hash, spec_snapshot_json, kind FROM runs WHERE run_id = ?",
            (run["run_id"],),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "run must be present in registry DB immediately after start()"
    assert row["spec_hash"] == run["spec_hash"]
    assert row["spec_snapshot_json"] is not None
    assert json.loads(row["spec_snapshot_json"])["name"] == "unit_rsi_mr"
    assert row["kind"] == "exploration"


def test_start_spec_hash_matches_canonical_recipe(stub_execute, tmp_path):
    """spec_hash is sha256 over json.dumps(sort_keys=True, separators=(',',':'))."""
    import hashlib

    manager = SpecRunManager(db_path=tmp_path / "registry.db")
    payload = _payload()
    run = manager.start("unit_rsi_mr", payload)

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
    assert run["spec_hash"] == expected_hash


# ── Test 2: completion writes headline_metrics_json to registry ───────────────


def test_completion_writes_headline_metrics_to_registry(monkeypatch, tmp_path):
    """After the run completes, ``headline_metrics_json`` is persisted in the DB."""
    import frontend.api.runs as runs_mod
    from data_platform.registry import db as _db

    monkeypatch.setattr(SpecRunManager, "_execute", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(runs_mod, "_extract_headline_metrics", lambda run: '{"sharpe": 1.5}')

    db_path = tmp_path / "registry.db"
    manager = SpecRunManager(db_path=db_path)
    run = manager.start("unit_rsi_mr", _payload())
    run_id = run["run_id"]

    # Drain the executor to ensure the worker has finished.
    manager._executor.shutdown(wait=True)

    # In-memory state.
    r = manager.get(run_id)
    assert r is not None
    assert r["status"] == "completed", r.get("error_text")
    assert r["headline_metrics_json"] == '{"sharpe": 1.5}'

    # Persisted to DB.
    conn = _db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status, headline_metrics_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row["status"] == "completed"
    assert row["headline_metrics_json"] == '{"sharpe": 1.5}'


# ── Test 3: restart marks queued/running runs as failed ───────────────────────


def test_restart_marks_interrupted_runs_as_failed(tmp_path):
    """A new SpecRunManager loading an existing DB marks non-terminal runs as failed."""
    from data_platform.registry import db as _db, writer as _writer

    db_path = tmp_path / "registry.db"

    # Simulate runs left in non-terminal states when the previous server process died.
    conn = _db.connect(db_path)
    try:
        with _db.transaction(conn):
            _writer.upsert_run(
                conn,
                _writer.RunRecord(
                    run_id="run-queued-001",
                    kind="exploration",
                    status="queued",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
            )
            _writer.upsert_run(
                conn,
                _writer.RunRecord(
                    run_id="run-running-002",
                    kind="exploration",
                    status="running",
                    created_at="2026-01-01T00:00:01+00:00",
                ),
            )
            # A completed run must NOT be touched.
            _writer.upsert_run(
                conn,
                _writer.RunRecord(
                    run_id="run-completed-003",
                    kind="exploration",
                    status="completed",
                    created_at="2026-01-01T00:00:02+00:00",
                ),
            )
    finally:
        conn.close()

    # "Restart": create a fresh manager pointing at the same DB.
    manager = SpecRunManager(db_path=db_path)

    # Non-terminal runs are marked failed.
    for run_id in ("run-queued-001", "run-running-002"):
        r = manager.get(run_id)
        assert r is not None, f"{run_id} must be loaded"
        assert r["status"] == "failed", f"{run_id} status should be failed, got {r['status']}"
        assert r["error_text"] == "Interrupted by a server restart."

    # Completed run is untouched.
    r = manager.get("run-completed-003")
    assert r is not None
    assert r["status"] == "completed"

    # DB rows are also updated.
    conn = _db.connect(db_path)
    try:
        rows = {
            row["run_id"]: dict(row)
            for row in conn.execute("SELECT run_id, status FROM runs").fetchall()
        }
    finally:
        conn.close()

    assert rows["run-queued-001"]["status"] == "failed"
    assert rows["run-running-002"]["status"] == "failed"
    assert rows["run-completed-003"]["status"] == "completed"

    # No active run survives a restart.
    assert manager._active_id is None


# ── Test 4: _runs_index.json is NEVER written ─────────────────────────────────


def test_runs_index_never_written_by_full_lifecycle(monkeypatch, tmp_path):
    """The frozen legacy _runs_index.json must NOT be touched during start→complete→delete."""
    import frontend.api.runs as runs_mod

    # Create the legacy file with known content and record its mtime.
    index_path = tmp_path / "_runs_index.json"
    index_path.write_text("[]", encoding="utf-8")
    mtime_before = index_path.stat().st_mtime_ns

    monkeypatch.setattr(runs_mod, "_RUNS_INDEX", index_path)
    monkeypatch.setattr(SpecRunManager, "_execute", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(runs_mod, "_extract_headline_metrics", lambda run: None)

    db_path = tmp_path / "registry.db"
    manager = SpecRunManager(db_path=db_path)
    run = manager.start("unit_rsi_mr", _payload())
    run_id = run["run_id"]

    # Drain the executor (ensures the run is in a terminal state before delete).
    manager._executor.shutdown(wait=True)

    assert manager.get(run_id)["status"] == "completed"
    manager.delete(run_id)

    # The legacy file must NOT have been touched at any point.
    assert index_path.stat().st_mtime_ns == mtime_before, (
        "_runs_index.json was written during the run lifecycle — it must remain frozen"
    )
    # Content is also unchanged (still the original "[]").
    assert index_path.read_text(encoding="utf-8") == "[]"
