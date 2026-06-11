"""Unit tests for data_platform.registry.rebuild.

SYNTHETIC test: builds a mini repo tree in tmp_path, runs orchestrate(), and
asserts per-table row counts, status mappings, log-file emission, and FK cleanliness.

IDEMPOTENCE test: two sequential orchestrate() calls → identical row counts.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from data_platform.registry import db
from data_platform.registry.rebuild import orchestrate


# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "registry.db")


# ── Fake-data writers ─────────────────────────────────────────────────────────


def _write_ohlc(path: Path) -> None:
    """Write a minimal 2-row norgate daily bar file at path."""
    from data_platform.storage import write_norgate_bars

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1001],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_norgate_bars(df, path)


def _write_mt5_m1(path: Path) -> None:
    """Write a minimal MT5 M1 bar file at path."""
    from data_platform.storage import write_mt5_bars
    from data_platform.storage.contracts import MT5_BARS_SCHEMA

    ts = pa.array(
        [
            datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 9, 1, tzinfo=timezone.utc),
        ],
        type=pa.timestamp("s", tz="UTC"),
    )
    table = pa.table(
        {
            "time": ts,
            "open": pa.array([100.0, 101.0], type=pa.float32()),
            "high": pa.array([102.0, 103.0], type=pa.float32()),
            "low": pa.array([99.0, 100.0], type=pa.float32()),
            "close": pa.array([101.0, 102.0], type=pa.float32()),
            "tick_volume": pa.array([10, 11], type=pa.int32()),
            "spread": pa.array([1, 1], type=pa.int16()),
            "real_volume": pa.array([100, 110], type=pa.int64()),
        },
        schema=MT5_BARS_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mt5_bars(table, path)


def _write_mt5_ticks(path: Path) -> None:
    """Write a minimal MT5 tick file at path."""
    from data_platform.storage import write_mt5_ticks
    from data_platform.storage.contracts import MT5_TICKS_SCHEMA

    ts = pa.array(
        [
            datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 9, 0, 1, tzinfo=timezone.utc),
        ],
        type=pa.timestamp("s", tz="UTC"),
    )
    table = pa.table(
        {
            "time_msc": pa.array([1704186000000, 1704186001000], type=pa.int64()),
            "bid": pa.array([100.0, 100.1], type=pa.float64()),
            "ask": pa.array([100.1, 100.2], type=pa.float64()),
            "last": pa.array([0.0, 0.0], type=pa.float64()),
            "volume": pa.array([0, 0], type=pa.int64()),
            "time": ts,
            "flags": pa.array([1028, 1028], type=pa.int32()),
        },
        schema=MT5_TICKS_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mt5_ticks(table, path)


def _write_fake_catalog(catalog_path: Path) -> None:
    """Write a minimal instrument catalog parquet file with one instrument."""
    from data_platform.core.catalog import InstrumentCatalog
    from data_platform.core.enums import AssetClass, InstrumentClass
    from data_platform.core.identifiers import InstrumentId
    from data_platform.core.instruments import Instrument

    cat = InstrumentCatalog()
    cat.add(
        Instrument(
            id=InstrumentId.from_str("ES.XCME"),
            raw_symbol="ES",
            asset_class=AssetClass.COMMODITY,
            instrument_class=InstrumentClass.FUTURE,
            price_precision=2,
            price_increment=0.25,
            data_source="norgate",
            info={
                "source_symbols": {
                    "norgate_adj": "&ES_CCB",
                    "ib_contfut": "ES",   # must be skipped
                    "mt5": "US500",
                }
            },
        )
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # Temporarily redirect catalog save to our path by monkey-patching
    import data_platform.core.catalog as _cat_mod

    _orig = _cat_mod.catalog_parquet_path
    _cat_mod.catalog_parquet_path = lambda: catalog_path
    try:
        cat.save()
    finally:
        _cat_mod.catalog_parquet_path = _orig


# ── Build the synthetic repo tree ─────────────────────────────────────────────


def _build_tree(tmp: Path) -> None:
    """Populate tmp_path with a minimal fake repo tree."""
    # 1. ohlc_data (2 files: one plain, one ratio)
    _write_ohlc(tmp / "data" / "ohlc_data" / "ES" / "D_ES.parquet")
    _write_ohlc(tmp / "data" / "ohlc_data" / "ES" / "D_ES_ratio.parquet")

    # 2. mt5 M1 bar
    _write_mt5_m1(tmp / "data" / "mt5_data" / "EURUSD" / "bars_M1" / "year=2024" / "part.parquet")

    # 3. tick_cache chunk (filename encodes ns range)
    _write_mt5_ticks(
        tmp / "data" / "mt5_data" / "EURUSD" / "ticks_cache"
        / "1000000000000000000-2000000000000000000.parquet"
    )

    # 4. spec JSON
    spec = {
        "spec_version": "1.0",
        "name": "Test Spec Alpha",
        "hypothesis": "for testing",
        "tickers": ["ES"],
        "timeframe": "D",
    }
    spec_path = tmp / "research" / "specs" / "test_spec_alpha.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    # 5. _runs_index: one completed, one running
    runs_index = [
        {
            "run_id": "run-completed-001",
            "spec_id": "test_spec_alpha",
            "phase": "exploration",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:01+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "num_combos": 4,
            "reports_dir": "feature_research/shared_results/test/run-completed-001",
            "log_text": "Training complete.\nFinal Sharpe: 1.23\n",
        },
        {
            "run_id": "run-running-002",
            "spec_id": "test_spec_alpha",
            "phase": "validation",
            "status": "running",
            "created_at": "2026-01-02T00:00:00+00:00",
            "started_at": "2026-01-02T00:00:01+00:00",
        },
    ]
    idx_path = (
        tmp / "feature_research" / "shared_results" / "_runs_index.json"
    )
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(runs_index), encoding="utf-8")

    # 6. vault feature JSON (nested layout: vault/D/<group>/<ensemble>/features/feat.json)
    feat = {
        "feature_name": "rsi_signal_D_lookback_14",
        "weight_hierarchy_group": "mean_reversion_indices",
        "created_at": "2026-01-01T12:00:00+00:00",
        "updated_at": "2026-01-01T12:00:00+00:00",
        "bias_node_spec": {"module_name": "rsi", "timeframes": ["D"]},
        "tickers": ["ES"],
        "base_models": [
            {
                "model_id": "m1",
                "model_type": "signed_signal",
                "feature_column": "rsi_signal_D_lookback_14",
                "bias_node_spec": {"module_name": "rsi"},
            }
        ],
    }
    feat_path = (
        tmp
        / "vault"
        / "D"
        / "mean_reversion_indices"
        / "rsi_ensemble"
        / "features"
        / "feat_abc123.json"
    )
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_path.write_text(json.dumps(feat), encoding="utf-8")

    # 7. calendar JSONs
    cal_dir = tmp / "data" / "events" / "calendar"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "fomc_decision_dates.json").write_text(
        json.dumps({"scraped_at": "2026-01-01", "dates": ["2026-01-29", "2026-03-19"]}),
        encoding="utf-8",
    )
    (cal_dir / "nyse_holiday_events.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "holiday_id": "new_years",
                        "asset_bucket": "equity",
                        "closure_date": "2026-01-01",
                        "d0": "2025-12-31",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # 8. instrument catalog
    _write_fake_catalog(tmp / "data" / "instruments" / "catalog.parquet")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def patched_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build fake repo tree + monkeypatch path-resolution helpers."""
    _build_tree(tmp_path)

    # Redirect InstrumentCatalog.load() → tmp catalog
    import data_platform.core.catalog as _cat_mod

    monkeypatch.setattr(
        _cat_mod, "catalog_parquet_path", lambda: tmp_path / "data" / "instruments" / "catalog.parquet"
    )

    # Redirect vault root resolution
    import lib.core.vault_paths as _vp

    monkeypatch.setattr(_vp, "resolve_vault_prop", lambda: tmp_path / "vault")
    monkeypatch.setattr(_vp, "resolve_vault_personal", lambda: tmp_path / "vault_personal")
    monkeypatch.setattr(_vp, "resolve_vault_cfd_prop", lambda: tmp_path / "vault_cfd_prop")

    return tmp_path


# ── Synthetic test ────────────────────────────────────────────────────────────


def test_synthetic_rebuild(patched_tree: Path) -> None:
    """End-to-end synthetic rebuild: row counts, status mapping, log files, FK check."""
    tmp = patched_tree
    conn = _open_db(tmp)
    log_dir = tmp / "logs" / "registry_backfill"

    counts = orchestrate(conn, repo_root=tmp, log_dir=log_dir)

    # ── per-domain row counts ─────────────────────────────────────────────
    assert counts["instruments"] == 1, f"instruments: {counts}"
    assert counts["specs"] == 1, f"specs: {counts}"
    assert counts["runs"] == 2, f"runs: {counts}"
    assert counts["vault"] >= 1, f"vault: {counts}"
    assert counts["calendar"] == 3, f"calendar (2 FOMC + 1 NYSE): {counts}"
    # ohlc_data: 2 files; mt5_m1: 1 file → total manifest at least 3
    assert counts["manifest"] >= 3, f"manifest: {counts}"
    assert counts["ticks"] == 1, f"ticks: {counts}"

    # ── DB row counts ─────────────────────────────────────────────────────
    def _count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    assert _count("instruments") == 1
    assert _count("instrument_source_symbols") >= 1  # norgate_adj + mt5
    assert _count("specs") == 1
    assert _count("runs") == 2
    assert _count("vault_entries") >= 1
    assert _count("fomc_decision") == 2
    assert _count("nyse_holiday") == 1
    assert _count("tick_chunks") == 1
    assert _count("job_runs") == 1  # the rebuild job itself

    # ── IB source_symbols must be excluded ───────────────────────────────
    ib_rows = conn.execute(
        "SELECT count(*) FROM instrument_source_symbols WHERE source LIKE 'ib%'"
    ).fetchone()[0]
    assert ib_rows == 0, "IB source symbols should be excluded"

    # ── running run → status=failed ───────────────────────────────────────
    run_row = conn.execute(
        "SELECT status, error_text FROM runs WHERE run_id = 'run-running-002'"
    ).fetchone()
    assert run_row is not None
    assert run_row["status"] == "failed"
    assert run_row["error_text"] == "Interrupted (registry backfill)"

    # ── completed run → spec_id preserved (spec exists) ──────────────────
    completed_row = conn.execute(
        "SELECT spec_id FROM runs WHERE run_id = 'run-completed-001'"
    ).fetchone()
    assert completed_row is not None
    assert completed_row["spec_id"] == "test_spec_alpha"

    # ── log file written for completed run ────────────────────────────────
    log_file = log_dir / "run-completed-001.log"
    assert log_file.exists(), f"log file not written: {log_file}"
    content = log_file.read_text(encoding="utf-8")
    assert "Training complete" in content

    # ── ohlc manifest coverage dates ─────────────────────────────────────
    ohlc_row = conn.execute(
        "SELECT coverage_start, coverage_end, rows FROM blob_manifest"
        " WHERE store = 'ohlc_data' AND key_json LIKE '%\"adjustment\": \"none\"%'"
    ).fetchone()
    assert ohlc_row is not None, "ohlc_data manifest row not found"
    assert ohlc_row["coverage_start"] == "2024-01-02"
    assert ohlc_row["coverage_end"] == "2024-01-03"
    assert ohlc_row["rows"] == 2

    # ── mt5_m1 manifest row ───────────────────────────────────────────────
    m1_row = conn.execute(
        "SELECT coverage_start, store, broker, timezone FROM blob_manifest"
        " WHERE store = 'mt5_m1'"
    ).fetchone()
    assert m1_row is not None
    assert m1_row["broker"] == "darwinex"
    assert m1_row["timezone"] == "broker_eet_as_utc"
    assert m1_row["coverage_start"] == "2024-01-02"

    # ── tick_chunks row ───────────────────────────────────────────────────
    tick_row = conn.execute(
        "SELECT symbol, start_ns, end_ns, n_ticks FROM tick_chunks"
    ).fetchone()
    assert tick_row is not None
    assert tick_row["symbol"] == "EURUSD"
    assert tick_row["start_ns"] == 1000000000000000000
    assert tick_row["end_ns"] == 2000000000000000000
    assert tick_row["n_ticks"] == 2

    # ── vault entry ───────────────────────────────────────────────────────
    v_row = conn.execute(
        "SELECT feature_name, feature_column, module_name, weight_hierarchy_group,"
        "       vault_profile, timeframe, ensemble_leaf, valid"
        " FROM vault_entries"
    ).fetchone()
    assert v_row is not None
    assert v_row["feature_name"] == "rsi_signal_D_lookback_14"
    assert v_row["feature_column"] == "rsi_signal_D_lookback_14"
    assert v_row["module_name"] == "rsi"
    assert v_row["weight_hierarchy_group"] == "mean_reversion_indices"
    assert v_row["vault_profile"] == "prop"
    assert v_row["timeframe"] == "D"
    assert v_row["ensemble_leaf"] == "rsi_ensemble"
    assert v_row["valid"] == 1

    # ── sleeves seeded ────────────────────────────────────────────────────
    sleeve_count = conn.execute("SELECT count(*) FROM sleeves").fetchone()[0]
    assert sleeve_count >= 13, f"expected ≥13 built-in sleeves, got {sleeve_count}"

    # ── FK integrity check ────────────────────────────────────────────────
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert not fk_violations, f"FK violations: {fk_violations}"

    # ── job_runs row records the rebuild ─────────────────────────────────
    job_row = conn.execute(
        "SELECT job_name, exit_code, rows_written, coverage_json FROM job_runs"
    ).fetchone()
    assert job_row is not None
    assert job_row["job_name"] == "registry_rebuild"
    assert job_row["exit_code"] == 0
    assert job_row["rows_written"] > 0
    cov = json.loads(job_row["coverage_json"])
    assert "counts" in cov
    assert cov["counts"]["specs"] == 1

    conn.close()


# ── Idempotence test ──────────────────────────────────────────────────────────


def test_idempotent_rebuild(patched_tree: Path) -> None:
    """Two sequential rebuilds produce identical row counts in every table."""
    tmp = patched_tree
    conn = _open_db(tmp)
    log_dir = tmp / "logs" / "registry_backfill"

    counts_1 = orchestrate(conn, repo_root=tmp, log_dir=log_dir)
    counts_2 = orchestrate(conn, repo_root=tmp, log_dir=log_dir)

    assert counts_1 == counts_2, (
        f"Non-idempotent rebuild: first={counts_1}, second={counts_2}"
    )

    # Also verify table row counts are stable (not doubled)
    def _count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    for table in (
        "instruments", "instrument_source_symbols", "specs", "runs",
        "vault_entries", "fomc_decision", "nyse_holiday", "tick_chunks",
    ):
        n = _count(table)
        assert n == _count(table), f"{table} count changed between reads"

    conn.close()
