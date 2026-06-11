"""Tests for data_platform.registry read subcommands (M6.1, ADR-9).

Each read subcommand is tested against a seeded tmp-path DB.  The costs
``--refresh`` path is tested against a synthetic M1 parquet store so that no
real repo data is needed.  The /api/data/* endpoints are tested via FastAPI
TestClient with the module-level _REGISTRY_DB_PATH monkeypatched.

All tests use tmp_path DBs — no production data touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from data_platform.registry import db, writer
from data_platform.registry.db import transaction


# ── helpers ───────────────────────────────────────────────────────────────────


def _open(tmp_path: Path):
    return db.connect(tmp_path / "registry.db")


def _seed_blobs(conn) -> None:
    """Insert two valid blob_manifest rows for coverage / freshness / report tests."""
    with transaction(conn):
        writer.record_blob(
            conn,
            writer.BlobRecord(
                store="ohlc_data",
                key_json='{"sym":"ES","tf":"D"}',
                relative_path="data/ohlc_data/ES/D_ES.parquet",
                written_at="2026-06-01T00:00:00",
                coverage_start="2010-01-01",
                coverage_end="2026-06-01",
                rows=4_000,
            ),
        )
        writer.record_blob(
            conn,
            writer.BlobRecord(
                store="ohlc_data",
                key_json='{"sym":"NQ","tf":"D"}',
                relative_path="data/ohlc_data/NQ/D_NQ.parquet",
                written_at="2026-06-01T00:00:00",
                coverage_start="2010-01-01",
                coverage_end="2026-05-30",
                rows=3_900,
            ),
        )
        writer.record_blob(
            conn,
            writer.BlobRecord(
                store="mt5_m1",
                key_json='{"sym":"EURUSD","year":"2025"}',
                relative_path="data/mt5_data/EURUSD/bars_M1/year=2025/part.parquet",
                written_at="2026-01-01T00:00:00",
                coverage_start="2025-01-01",
                coverage_end="2025-12-31",
                rows=370_000,
            ),
        )


def _seed_runs(conn) -> None:
    """Insert a spec + two run rows for runs / lineage tests."""
    with transaction(conn):
        conn.execute(
            """INSERT INTO specs (id, name, name_slug, content_hash, spec_json, file_path, created_at, updated_at)
               VALUES ('spec-001', 'rsi_mr', 'rsi_mr', 'aabbcc', '{}', 'research/specs/rsi_mr.json',
                       '2026-01-01T00:00:00', '2026-01-01T00:00:00')"""
        )
        writer.upsert_run(
            conn,
            writer.RunRecord(
                run_id="run-001",
                kind="exploration",
                status="completed",
                created_at="2026-01-02T10:00:00",
                spec_id="spec-001",
                spec_hash="aabbcc",
                headline_metrics_json='{"sharpe": 1.23, "t_stat": 2.5}',
            ),
        )
        writer.upsert_run(
            conn,
            writer.RunRecord(
                run_id="run-002",
                kind="validation",
                status="completed",
                created_at="2026-01-03T10:00:00",
                spec_id="spec-001",
                spec_hash="aabbcc",
                headline_metrics_json='{"sharpe": 0.95}',
            ),
        )


def _seed_vault(conn) -> None:
    """Insert a vault_entry + sleeve for lineage test."""
    with transaction(conn):
        conn.execute("INSERT OR IGNORE INTO sleeves (name) VALUES ('mean_reversion_indices')")
        writer.upsert_vault_entry(
            conn,
            writer.VaultEntry(
                vault_profile="prop",
                timeframe="D",
                ensemble_leaf="rsi_mr_long",
                feature_name="rsi_signal_D_lookback_14",
                feature_column="rsi_signal_D_lookback_14",
                module_name="rsi_signal",
                config_json="{}",
                file_path="vault/D/mean_reversion_indices/rsi_mr_long/features/rsi_signal_D_lookback_14.json",
                created_at="2026-01-03T10:00:00",
                updated_at="2026-01-03T10:00:00",
                weight_hierarchy_group="mean_reversion_indices",
                producing_run_id="run-002",
            ),
        )


def _seed_accounts_and_fills(conn) -> None:
    """Insert an account, order, and deal for fills / forecasts tests."""
    acct = writer.Account(
        account_id=1,
        broker="darwinex",
        login="11111",
        server="darwinex-demo",
        exec_tier="demo",
        program_phase="retail",
        valid_from="2026-01-01",
    )
    order = writer.Order(
        account_id=1,
        client_order_id="ord-test-001",
        canonical="NQ",
        symbol="NDX",
        side="BUY",
        qty=1.0,
        broker_time_submit="2026-06-01T17:00:00",
        bid_at_submit=20_000.0,
        ask_at_submit=20_007.0,
    )
    deal = writer.Deal(
        account_id=1,
        ticket=9001,
        symbol="NDX",
        deal_type=0,
        volume=1.0,
        price=20_007.0,
        broker_time="2026-06-01T17:00:01",
        client_order_id="ord-test-001",
        canonical="NQ",
    )
    with transaction(conn):
        writer.upsert_account(conn, acct)
        writer.insert_order(conn, order)
        writer.insert_deal(conn, deal)


# ── coverage ──────────────────────────────────────────────────────────────────


def test_coverage_all_stores(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_blobs(conn)
    conn.close()

    from data_platform.registry.cli_read import _coverage

    _coverage(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "ohlc_data" in out
    assert "mt5_m1" in out
    # 2 ohlc_data rows + 1 mt5_m1 row
    assert "7900" in out.replace(",", "").replace(" ", "").replace("_", "") or "7900" in out or "7,900" in out


def test_coverage_single_store(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_blobs(conn)
    conn.close()

    from data_platform.registry.cli_read import _coverage

    _coverage(["--store", "ohlc_data", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "ES" in out
    assert "NQ" in out
    # mt5_m1 should not appear
    assert "EURUSD" not in out


def test_coverage_missing_db(tmp_path, capsys):
    from data_platform.registry.cli_read import _coverage

    with pytest.raises(SystemExit):
        _coverage(["--db", str(tmp_path / "nonexistent.db")])


# ── freshness ─────────────────────────────────────────────────────────────────


def test_freshness_all_stores(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_blobs(conn)
    conn.close()

    from data_platform.registry.cli_read import _freshness

    _freshness(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "ohlc_data" in out
    assert "2026-06-01" in out


def test_freshness_single_store(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_blobs(conn)
    conn.close()

    from data_platform.registry.cli_read import _freshness

    _freshness(["--store", "ohlc_data", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    # Both ES and NQ keys appear
    assert "ES" in out
    assert "NQ" in out


# ── runs ──────────────────────────────────────────────────────────────────────


def test_runs_all(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_runs(conn)
    conn.close()

    from data_platform.registry.cli_read import _runs

    _runs(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "run-001" in out
    assert "run-002" in out
    assert "exploration" in out
    assert "1.230" in out  # sharpe from run-001 headline


def test_runs_filter_kind(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_runs(conn)
    conn.close()

    from data_platform.registry.cli_read import _runs

    _runs(["--kind", "validation", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "run-002" in out
    assert "run-001" not in out


def test_runs_empty(tmp_path, capsys):
    conn = _open(tmp_path)
    conn.close()

    from data_platform.registry.cli_read import _runs

    _runs(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "No runs found" in out


# ── lineage ───────────────────────────────────────────────────────────────────


def test_lineage_vault_feature(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_runs(conn)
    _seed_vault(conn)
    conn.close()

    from data_platform.registry.cli_read import _lineage

    _lineage(["rsi_signal_D_lookback_14", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "Vault lineage" in out
    assert "rsi_mr_long" in out          # ensemble_leaf
    assert "run-002" in out              # producing_run_id


def test_lineage_run_id(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_runs(conn)
    conn.close()

    from data_platform.registry.cli_read import _lineage

    _lineage(["run-001", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "Run lineage" in out
    assert "exploration" in out


def test_lineage_not_found(tmp_path, capsys):
    conn = _open(tmp_path)
    conn.close()

    from data_platform.registry.cli_read import _lineage

    with pytest.raises(SystemExit):
        _lineage(["nonexistent_feature", "--db", str(tmp_path / "registry.db")])
    err = capsys.readouterr().err
    assert "No vault entry or run found" in err


# ── fills ─────────────────────────────────────────────────────────────────────


def test_fills_all(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_accounts_and_fills(conn)
    conn.close()

    from data_platform.registry.cli_read import _fills

    _fills(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "darwinex" in out
    assert "9001" in out
    assert "NQ" in out


def test_fills_broker_filter(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_accounts_and_fills(conn)
    conn.close()

    from data_platform.registry.cli_read import _fills

    _fills(["--broker", "darwinex", "--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "9001" in out


def test_fills_empty_no_data(tmp_path, capsys):
    conn = _open(tmp_path)
    conn.close()

    from data_platform.registry.cli_read import _fills

    _fills(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "No fills found" in out


# ── report ────────────────────────────────────────────────────────────────────


def test_report_markdown_table(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_blobs(conn)
    conn.close()

    from data_platform.registry.cli_read import _report

    _report(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    # Markdown table headers
    assert "| store |" in out
    assert "| keys |" in out
    # Store names appear
    assert "ohlc_data" in out
    assert "mt5_m1" in out
    # Footer summary line
    assert "stores" in out
    assert "keys" in out


def test_report_empty_db(tmp_path, capsys):
    conn = _open(tmp_path)
    conn.close()

    from data_platform.registry.cli_read import _report

    _report(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    # Empty case still prints headers
    assert "| store |" in out
    assert "empty" in out.lower() or "rebuild" in out.lower()


# ── costs (read-only path) ────────────────────────────────────────────────────


def _seed_cost_obs(conn) -> None:
    """Insert a synthetic cost_observations row + instrument for read tests."""
    inst = writer.Instrument(
        id="NDX.XNAS",
        raw_symbol="NDX",
        asset_class="INDEX",
        instrument_class="CFD",
        price_precision=1,
        price_increment=0.1,
    )
    with transaction(conn):
        writer.upsert_instrument(conn, inst)
        conn.execute(
            """INSERT INTO cost_observations
               (instrument_id, broker, tod_bucket, window_start, window_end,
                spread_points_p25, spread_points_p50, spread_points_p75,
                spread_bps_p50, n_spread_obs, n_fill_obs, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "NDX.XNAS", "darwinex", "09:00",
                "2025-01-01", "2025-12-31",
                6.0, 7.0, 8.0,
                3.33, 50_000, 0,
                "2026-06-09T00:00:00",
            ),
        )


def test_costs_read_shows_rows(tmp_path, capsys):
    conn = _open(tmp_path)
    _seed_cost_obs(conn)
    conn.close()

    from data_platform.registry.cli_read import _costs

    _costs(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "NDX.XNAS" in out
    assert "darwinex" in out
    assert "09:00" in out
    assert "7.0000" in out  # p50 = 7.0


def test_costs_no_rows_message(tmp_path, capsys):
    conn = _open(tmp_path)
    conn.close()

    from data_platform.registry.cli_read import _costs

    _costs(["--db", str(tmp_path / "registry.db")])
    out = capsys.readouterr().out
    assert "No cost observations" in out


# ── costs --refresh (with synthetic M1 data) ─────────────────────────────────


def _make_synthetic_m1(base_dir: Path, native_sym: str, n_rows: int = 5_000) -> None:
    """Write a synthetic M1 parquet partition for costs --refresh tests."""
    import numpy as np
    import pandas as pd

    year_dir = base_dir / native_sym / "bars_M1" / "year=2025"
    year_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    # Timestamps: 5000 minutes starting 2025-01-02 01:00
    start = pd.Timestamp("2025-01-02 01:00:00", tz="UTC")
    times = pd.date_range(start, periods=n_rows, freq="min")
    closes = 20_000.0 + rng.normal(0, 100, n_rows).cumsum()
    spreads = rng.integers(5, 12, n_rows)

    df = pd.DataFrame({"time": times, "close": closes, "spread": spreads})
    df.to_parquet(str(year_dir / "part.parquet"), index=False)


def _seed_instrument_for_costs(conn, native_sym: str, instrument_id: str) -> None:
    """Insert an instrument + MT5 source symbol into the seeded DB."""
    inst = writer.Instrument(
        id=instrument_id,
        raw_symbol=instrument_id.split(".")[0],
        asset_class="INDEX",
        instrument_class="CFD",
        price_precision=1,
        price_increment=0.1,
    )
    with transaction(conn):
        writer.upsert_instrument(conn, inst)
        conn.execute(
            "INSERT OR IGNORE INTO instrument_source_symbols (instrument_id, source, native_symbol) VALUES (?, ?, ?)",
            (instrument_id, "mt5", native_sym),
        )


def test_costs_refresh_creates_observations(tmp_path, capsys, monkeypatch):
    """costs --refresh reads synthetic M1 data and inserts cost_observations rows."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    mt5_root = tmp_path / "data" / "mt5_data"
    _make_synthetic_m1(mt5_root, "TESTNDX", n_rows=2_880)  # ~2 days × 1440 min

    conn = _open(tmp_path)
    _seed_instrument_for_costs(conn, "TESTNDX", "TESTNDX.XNAS")
    conn.close()

    # Monkeypatch _repo_root() so costs_refresh looks for M1 under tmp_path.
    import data_platform.registry.db as _db_mod

    monkeypatch.setattr(_db_mod, "_repo_root", lambda: tmp_path)

    from data_platform.registry.cli_read import _costs_refresh

    _costs_refresh(db_path=tmp_path / "registry.db", instrument_filter=None)
    out = capsys.readouterr().out
    assert "TESTNDX" in out
    assert "Cost refresh complete" in out

    # Verify rows were written
    conn2 = db.connect_readonly(tmp_path / "registry.db")
    rows = conn2.execute(
        "SELECT * FROM cost_observations WHERE instrument_id = 'TESTNDX.XNAS'"
    ).fetchall()
    conn2.close()

    # Should have ~24 bucket rows (one per hour of the day that has data)
    assert len(rows) > 0, "Expected at least one cost_observations row"

    # Percentiles should be sane: spread was between 5-11 so p50 ~ 7-8
    for r in rows:
        assert r["spread_points_p25"] <= r["spread_points_p50"] <= r["spread_points_p75"]
        assert r["spread_points_p25"] >= 5
        assert r["spread_points_p75"] <= 12
        assert r["spread_bps_p50"] is not None
        assert r["spread_bps_p50"] > 0
        assert r["n_spread_obs"] > 0
        assert r["broker"] == "darwinex"


def test_costs_refresh_skip_no_m1(tmp_path, capsys, monkeypatch):
    """Instrument registered but no M1 data dir for that symbol → gracefully skipped (0 rows)."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    # Create the mt5_data root so the code doesn't abort, but NOSYM has no sub-dir.
    (tmp_path / "data" / "mt5_data").mkdir(parents=True)

    conn = _open(tmp_path)
    _seed_instrument_for_costs(conn, "NOSYM", "NOSYM.XNAS")
    conn.close()

    import data_platform.registry.db as _db_mod

    monkeypatch.setattr(_db_mod, "_repo_root", lambda: tmp_path)

    from data_platform.registry.cli_read import _costs_refresh

    _costs_refresh(db_path=tmp_path / "registry.db", instrument_filter=None)
    out = capsys.readouterr().out
    assert "Cost refresh complete" in out
    assert "0 rows" in out


def test_costs_refresh_records_job_run(tmp_path, capsys, monkeypatch):
    """costs --refresh always records a job_runs row even when no instruments produce rows."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    # Create an empty mt5_data root so the code doesn't abort early.
    (tmp_path / "data" / "mt5_data").mkdir(parents=True)

    conn = _open(tmp_path)
    conn.close()

    import data_platform.registry.db as _db_mod

    monkeypatch.setattr(_db_mod, "_repo_root", lambda: tmp_path)

    from data_platform.registry.cli_read import _costs_refresh

    _costs_refresh(db_path=tmp_path / "registry.db", instrument_filter=None)

    conn2 = db.connect_readonly(tmp_path / "registry.db")
    rows = conn2.execute(
        "SELECT * FROM job_runs WHERE job_name = 'costs_refresh'"
    ).fetchall()
    conn2.close()

    assert len(rows) == 1
    assert rows[0]["finished_at"] is not None


# ── /api/data/* FastAPI endpoints ─────────────────────────────────────────────


@pytest.fixture()
def _data_api_db(tmp_path, monkeypatch):
    """Seed a tmp DB and inject its path into frontend.api.data."""
    import frontend.api.data as data_mod

    db_path = tmp_path / "registry.db"
    conn = db.connect(db_path)
    _seed_blobs(conn)
    _seed_runs(conn)
    _seed_accounts_and_fills(conn)
    _seed_cost_obs(conn)
    conn.close()

    monkeypatch.setattr(data_mod, "_REGISTRY_DB_PATH", db_path)
    return db_path


def _client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from frontend.api.server import app

    return TestClient(app)


def test_api_data_coverage(_data_api_db):
    client = _client()
    resp = client.get("/api/data/coverage")
    assert resp.status_code == 200
    body = resp.json()
    assert "stores" in body
    store_names = {s["store"] for s in body["stores"]}
    assert "ohlc_data" in store_names
    assert "mt5_m1" in store_names


def test_api_data_coverage_store_filter(_data_api_db):
    client = _client()
    resp = client.get("/api/data/coverage", params={"store": "ohlc_data"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "ohlc_data"
    assert len(body["keys"]) == 2


def test_api_data_freshness(_data_api_db):
    client = _client()
    resp = client.get("/api/data/freshness")
    assert resp.status_code == 200
    body = resp.json()
    assert "stores" in body
    assert len(body["stores"]) >= 2


def test_api_data_freshness_store_filter(_data_api_db):
    client = _client()
    resp = client.get("/api/data/freshness", params={"store": "ohlc_data"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "ohlc_data"
    assert len(body["keys"]) == 2


def test_api_data_jobs(_data_api_db):
    client = _client()
    resp = client.get("/api/data/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert "jobs" in body


def test_api_data_accounts(_data_api_db):
    client = _client()
    resp = client.get("/api/data/accounts")
    assert resp.status_code == 200
    body = resp.json()
    assert "accounts" in body
    assert len(body["accounts"]) == 1
    assert body["accounts"][0]["broker"] == "darwinex"


def test_api_data_fills(_data_api_db):
    client = _client()
    resp = client.get("/api/data/fills")
    assert resp.status_code == 200
    body = resp.json()
    assert "fills" in body
    assert len(body["fills"]) == 1
    fill = body["fills"][0]
    assert fill["canonical"] == "NQ"
    assert fill["broker"] == "darwinex"


def test_api_data_fills_broker_filter(_data_api_db):
    client = _client()
    resp = client.get("/api/data/fills", params={"broker": "darwinex"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["fills"]) == 1


def test_api_data_fills_unknown_broker(_data_api_db):
    client = _client()
    resp = client.get("/api/data/fills", params={"broker": "ftmo"})
    assert resp.status_code == 200
    assert resp.json()["fills"] == []


def test_api_data_forecasts(_data_api_db):
    client = _client()
    resp = client.get("/api/data/forecasts")
    assert resp.status_code == 200
    body = resp.json()
    assert "forecasts" in body


def test_api_data_returns_empty_when_db_missing(tmp_path, monkeypatch):
    """All /api/data endpoints return empty collections when DB file does not exist."""
    import frontend.api.data as data_mod

    monkeypatch.setattr(data_mod, "_REGISTRY_DB_PATH", tmp_path / "nonexistent.db")
    client = _client()

    for endpoint in ("/api/data/coverage", "/api/data/freshness", "/api/data/jobs",
                     "/api/data/accounts", "/api/data/fills", "/api/data/forecasts"):
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"{endpoint} returned {resp.status_code}"
        body = resp.json()
        # All endpoints return the empty collection (list)
        assert any(isinstance(v, list) and len(v) == 0 for v in body.values()), (
            f"{endpoint} did not return an empty collection: {body}"
        )
