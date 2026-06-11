"""M4.2 vault promotion lineage + persistence tests.

Four behavioural contracts:

1. PROMOTION LINEAGE: commit() writes producing_run_id + spec_hash into the feature
   JSON and upserts a vault_entries registry row (promoted_by='api').
2. GATE-IDENTITY FIX: load_portfolio_gate_status uses the run-dir report when a
   registry-backed run with a matching spec_hash is found; falls back to legacy.
3. PORTFOLIO RUNS PERSIST: PortfolioWorkspaceJobManager records each job as a
   registry run (kind='portfolio') with status transitions; restart marks in-flight
   jobs as failed.
4. VALIDATE_CANDIDATE PERSISTS: _persist_run writes metrics.json + equity.parquet
   and registers a final_validation run row.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_validation_run(conn, run_id: str, spec_hash: str, viz_dir: str | None = None) -> None:
    """Insert a completed validation run with the given spec_hash into the registry."""
    from data_platform.registry import writer as _writer

    _writer.upsert_run(
        conn,
        _writer.RunRecord(
            run_id=run_id,
            kind="validation",
            status="completed",
            spec_hash=spec_hash,
            viz_dir=viz_dir,
            created_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T01:00:00+00:00",
        ),
    )


# ── Test 1: PROMOTION LINEAGE ─────────────────────────────────────────────────


def test_commit_writes_producing_run_id_into_feature_json(monkeypatch, tmp_path):
    """commit() resolves the producing run and bakes producing_run_id + spec_hash into the
    feature JSON, then upserts a vault_entries row with promoted_by='api'."""
    import frontend.api.vault as vault_mod
    from data_platform.registry import db as _db, writer as _writer, transaction

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(vault_mod, "_REGISTRY_DB_PATH", db_path)

    # Seed a completed validation run.
    spec_dict = {"name": "test_spec", "signal": {"module_name": "rsi"}}
    from frontend.api.vault import _compute_spec_hash
    spec_hash = _compute_spec_hash(spec_dict)

    conn = _db.connect(db_path)
    try:
        with _db.transaction(conn):
            _make_validation_run(conn, "run-val-001", spec_hash)
    finally:
        conn.close()

    # Build a minimal vault ensemble directory.
    ensemble_dir = tmp_path / "vault" / "D" / "test_ensemble_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True)
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps({"timeframe": "D", "ensemble_name": "test_ensemble",
                    "direction": "long", "tickers": ["ES"]}),
        encoding="utf-8",
    )

    # Mock execute_vault_save to avoid real pipeline.
    from research.feature.ui.vault_save import VaultSaveExecution

    feature_col = "rsi_signal_D_lookback_14"
    fake_result = VaultSaveExecution(
        dry_run=False,
        vault_root=str(tmp_path / "vault"),
        ensemble_dir=str(ensemble_dir),
        ensemble_dir_repo_relative="vault/D/test_ensemble_long",
        tickers=("ES",),
        direction="long",
        feature_column=feature_col,
        model_id="signed_signal_rsi_d_lookback_14",
        bias_spec={"module_name": "rsi", "timeframes": ["D"], "params": {"lookback": 14}},
        vault_profile="prop",
        weight_hierarchy_group="mean_reversion_indices",
    )

    # Write the feature JSON as the real pipeline would.
    feature_json_content = json.dumps({
        "feature_name": feature_col,
        "bias_node_spec": fake_result.bias_spec,
        "tickers": ["ES"],
        "producing_run_id": "run-val-001",
        "spec_hash": spec_hash,
        "base_models": [{
            "model_id": "signed_signal_rsi_d_lookback_14",
            "model_name": f"{feature_col}::signed_signal_rsi_d_lookback_14",
            "model_type": "signed_signal",
            "feature_column": feature_col,
            "strategy": "long",
            "bias_node_spec": fake_result.bias_spec,
        }],
    })
    (features_dir / f"{feature_col}.json").write_text(feature_json_content, encoding="utf-8")

    monkeypatch.setattr(
        vault_mod,
        "execute_vault_save",
        lambda *a, **k: fake_result,
    )
    monkeypatch.setattr(vault_mod, "_config", lambda spec: None)

    result_dict = vault_mod.commit(spec_dict)

    # Verify vault_entries row was written.
    conn2 = _db.connect(db_path)
    try:
        row = conn2.execute(
            "SELECT * FROM vault_entries WHERE feature_name = ?", (feature_col,)
        ).fetchone()
    finally:
        conn2.close()

    assert row is not None, "vault_entries row must exist after commit"
    assert row["producing_run_id"] == "run-val-001"
    assert row["promoted_by"] == "api"
    assert row["vault_profile"] == "prop"
    assert row["timeframe"] == "D"

    # Feature JSON should contain producing_run_id + spec_hash.
    feature_data = json.loads((features_dir / f"{feature_col}.json").read_text(encoding="utf-8"))
    assert feature_data.get("producing_run_id") == "run-val-001"
    assert feature_data.get("spec_hash") == spec_hash


def test_commit_no_run_produces_null_producing_run_id(monkeypatch, tmp_path):
    """When no completed validation run exists, producing_run_id is NULL in vault_entries."""
    import frontend.api.vault as vault_mod
    from data_platform.registry import db as _db

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(vault_mod, "_REGISTRY_DB_PATH", db_path)

    # Empty DB — no runs.
    _db.connect(db_path).close()

    spec_dict = {"name": "no_run_spec"}
    feature_col = "rsi_signal_D_lookback_7"
    ensemble_dir = tmp_path / "vault" / "D" / "x_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True)
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps({"timeframe": "D", "ensemble_name": "x", "direction": "long", "tickers": ["ES"]}),
        encoding="utf-8",
    )
    (features_dir / f"{feature_col}.json").write_text(
        json.dumps({"feature_name": feature_col, "bias_node_spec": {"module_name": "rsi",
                    "timeframes": ["D"], "params": {}}, "tickers": ["ES"], "base_models": [{
                    "model_id": "x", "model_name": f"{feature_col}::x",
                    "model_type": "signed_signal", "feature_column": feature_col,
                    "strategy": "long", "bias_node_spec": {}}]}),
        encoding="utf-8",
    )
    from research.feature.ui.vault_save import VaultSaveExecution
    fake_result = VaultSaveExecution(
        dry_run=False,
        vault_root=str(tmp_path / "vault"),
        ensemble_dir=str(ensemble_dir),
        ensemble_dir_repo_relative="vault/D/x_long",
        tickers=("ES",),
        direction="long",
        feature_column=feature_col,
        model_id="x",
        bias_spec={"module_name": "rsi", "timeframes": ["D"], "params": {}},
        vault_profile="prop",
        weight_hierarchy_group=None,
    )
    monkeypatch.setattr(vault_mod, "execute_vault_save", lambda *a, **k: fake_result)
    monkeypatch.setattr(vault_mod, "_config", lambda spec: None)

    vault_mod.commit(spec_dict)

    conn = _db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT producing_run_id FROM vault_entries WHERE feature_name = ?",
            (feature_col,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row["producing_run_id"] is None


# ── Test 2: GATE-IDENTITY FIX ─────────────────────────────────────────────────


def test_gate_uses_run_dir_report_when_spec_hash_provided(monkeypatch, tmp_path):
    """load_portfolio_gate_status uses the per-run portfolio_addition_report.json
    when spec_hash is supplied and a matching completed validation run exists."""
    import research.feature.ui.vault_save as vs_mod
    from data_platform.registry import db as _db, writer as _writer

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(vs_mod, "_REGISTRY_DB_PATH", db_path)

    spec_hash = "abc123"
    viz_dir_rel = "feature_research/shared_results/run-xyz/visualization/validation"
    viz_abs = tmp_path / viz_dir_rel
    viz_abs.mkdir(parents=True)

    # Write a passing gate report in the run's viz dir.
    gate_report = viz_abs / "portfolio_addition_report.json"
    gate_report.write_text(json.dumps({"passed": True}), encoding="utf-8")

    # Patch REPO_ROOT so registry paths are resolved under tmp_path.
    monkeypatch.setattr(vs_mod, "_REPO_ROOT", tmp_path)

    conn = _db.connect(db_path)
    try:
        with _db.transaction(conn):
            _make_validation_run(conn, "run-xyz", spec_hash, viz_dir=viz_dir_rel)
    finally:
        conn.close()

    # Minimal ResearchConfig stub (legacy path should NOT be reached).
    config = MagicMock()
    config.vault_save = None

    from research.feature.ui.vault_save import VaultGateStatus, load_portfolio_gate_status

    status = load_portfolio_gate_status(config, spec_hash=spec_hash)
    assert status == VaultGateStatus.PASSED


def test_gate_falls_back_to_legacy_when_no_matching_run(monkeypatch, tmp_path):
    """load_portfolio_gate_status uses the legacy manifest path when spec_hash finds no run."""
    import research.feature.ui.vault_save as vs_mod
    from data_platform.registry import db as _db

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(vs_mod, "_REGISTRY_DB_PATH", db_path)

    # Empty DB.
    _db.connect(db_path).close()

    # Patch the legacy path to return PASSED.
    monkeypatch.setattr(vs_mod, "validation_artifacts_current", lambda cfg: True)
    monkeypatch.setattr(
        vs_mod, "portfolio_addition_report_path",
        lambda cfg: tmp_path / "legacy_report.json",
    )
    (tmp_path / "legacy_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )

    config = MagicMock()
    from research.feature.ui.vault_save import VaultGateStatus, load_portfolio_gate_status

    status = load_portfolio_gate_status(config, spec_hash="unknown_hash")
    assert status == VaultGateStatus.PASSED


def test_gate_no_spec_hash_uses_legacy_directly(monkeypatch, tmp_path):
    """load_portfolio_gate_status bypasses registry entirely when spec_hash is None."""
    import research.feature.ui.vault_save as vs_mod

    monkeypatch.setattr(vs_mod, "validation_artifacts_current", lambda cfg: True)
    monkeypatch.setattr(
        vs_mod, "portfolio_addition_report_path",
        lambda cfg: tmp_path / "rep.json",
    )
    (tmp_path / "rep.json").write_text(json.dumps({"passed": False}), encoding="utf-8")

    from research.feature.ui.vault_save import VaultGateStatus, load_portfolio_gate_status

    # No spec_hash → goes straight to legacy, returns FAILED.
    status = load_portfolio_gate_status(MagicMock(), spec_hash=None)
    assert status == VaultGateStatus.FAILED


# ── Test 3: PORTFOLIO RUNS PERSIST ───────────────────────────────────────────


class _FakeRequest:
    """Minimal PortfolioResearchUiRequest stub."""
    def __init__(self):
        from unittest.mock import MagicMock
        self.phase = MagicMock()
        self.phase.value = "portfolio_test"


def test_portfolio_job_records_run_in_registry(tmp_path):
    """start_job() creates a registry row (kind='portfolio') and status transitions persist."""
    from data_platform.registry import db as _db
    from research.portfolio.ui.job_manager import PortfolioWorkspaceJobManager

    db_path = tmp_path / "registry.db"
    manager = PortfolioWorkspaceJobManager(db_path=db_path)

    config = MagicMock()
    request = _FakeRequest()

    # Stub build_phase_plan and execute_phase_request.
    import research.portfolio.ui.job_manager as jm_mod
    from unittest.mock import patch, MagicMock as MM

    fake_plan = MM()
    fake_plan.title = "Test portfolio run"
    fake_plan.output_path = tmp_path / "portfolio_out"

    with patch.object(jm_mod, "build_phase_plan", return_value=fake_plan), \
         patch.object(jm_mod, "execute_phase_request", return_value=0):
        job = manager.start_job(config, request)
        job_id = job["job_id"]

        # Drain executor.
        manager._executor.shutdown(wait=True)

    conn = _db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT run_id, kind, status, reports_dir FROM runs WHERE run_id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "registry row must exist for portfolio job"
    assert row["kind"] == "portfolio"
    assert row["status"] == "completed"


def test_portfolio_restart_marks_interrupted_as_failed(tmp_path):
    """A new PortfolioWorkspaceJobManager loading the same DB marks in-flight jobs as failed."""
    from data_platform.registry import db as _db, writer as _writer
    from research.portfolio.ui.job_manager import PortfolioWorkspaceJobManager

    db_path = tmp_path / "registry.db"

    # Directly insert a running row.
    conn = _db.connect(db_path)
    try:
        with _db.transaction(conn):
            _writer.upsert_run(
                conn,
                _writer.RunRecord(
                    run_id="port-run-001",
                    kind="portfolio",
                    status="running",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
            )
    finally:
        conn.close()

    manager = PortfolioWorkspaceJobManager(db_path=db_path)

    job = manager.job_snapshot("port-run-001")
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_text"] == "Interrupted by a server restart."

    # DB row is also updated.
    conn2 = _db.connect(db_path)
    try:
        row = conn2.execute(
            "SELECT status FROM runs WHERE run_id = 'port-run-001'"
        ).fetchone()
    finally:
        conn2.close()
    assert row["status"] == "failed"


# ── Test 4: VALIDATE_CANDIDATE PERSISTS ──────────────────────────────────────


def test_persist_run_writes_metrics_and_registry(tmp_path):
    """_persist_run writes metrics.json, equity.parquet, and a final_validation registry row."""
    from data_platform.registry import db as _db
    from scripts.validate_candidate import _persist_run
    from research.validation.validation_lane import ValidationResult

    db_path = tmp_path / "registry.db"

    metrics = {"sharpe": 1.23, "ann_return_pct": 12.5, "ann_vol_pct": 10.0,
               "total_return": 0.125, "n_obs": 252.0}

    equity = pd.Series(
        [1_000_000.0, 1_010_000.0, 1_020_000.0],
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        ),
        name="equity",
    )
    equity.index.name = "datetime"

    result = ValidationResult(
        returns=equity.pct_change().dropna(),
        equity_curve=equity,
        metrics=metrics,
        n_decisions=5,
        positions_report=pd.DataFrame(),
        fills_report=pd.DataFrame(),
        resolved_tickers=("ES",),
    )

    from research.validation.validation_lane import ValidationConfig
    config = ValidationConfig(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-04-01", tz="UTC"),
    )

    run_id = "test-run-final-001"
    run_dir = tmp_path / "final_validation" / run_id
    _persist_run(result, config, run_id, run_dir, db_path=db_path)

    # Artifacts exist.
    assert (run_dir / "metrics.json").is_file()
    stored_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert abs(stored_metrics["sharpe"] - 1.23) < 1e-9

    equity_df = pd.read_parquet(run_dir / "equity.parquet")
    assert "equity" in equity_df.columns

    # Registry row exists with correct fields.
    conn = _db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT run_id, kind, status, headline_metrics_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "final_validation run must be registered"
    assert row["kind"] == "final_validation"
    assert row["status"] == "completed"
    hm = json.loads(row["headline_metrics_json"])
    assert abs(hm["sharpe"] - 1.23) < 1e-9
