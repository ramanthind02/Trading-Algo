"""Shared fixtures for the frontend_api tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_runs_index(monkeypatch, tmp_path):
    """Isolate run-persistence to a per-test throwaway registry DB (and legacy index path).

    ``SpecRunManager`` now persists to ``data/registry.db`` (via ``_REGISTRY_DB_PATH``).
    Without this fixture every test would read/write the real registry DB and could observe
    each other's runs.

    We patch:
    - ``_RUNS_INDEX``                              — frozen legacy JSON
    - ``_REGISTRY_DB_PATH`` (runs, vault, vault_save, portfolio job_manager)
    - ``RUN_MANAGER._db_path``                     — already-constructed singleton
    - ``spec_store._REGISTRY_DB_PATH``             — spec upsert/delete calls
    - ``frontend.api.portfolio._MANAGER``          — reset so next call picks up tmp_db
    """

    import frontend.api.runs as runs_mod
    import frontend.api.spec_store as spec_store_mod
    import frontend.api.vault as vault_mod
    import research.feature.ui.vault_save as vault_save_mod
    import research.portfolio.ui.job_manager as portfolio_jm_mod

    tmp_db = tmp_path / "registry.db"
    monkeypatch.setattr(runs_mod, "_RUNS_INDEX", tmp_path / "_runs_index.json")
    monkeypatch.setattr(runs_mod, "_REGISTRY_DB_PATH", tmp_db)
    monkeypatch.setattr(runs_mod.RUN_MANAGER, "_db_path", tmp_db)
    monkeypatch.setattr(spec_store_mod, "_REGISTRY_DB_PATH", tmp_db)
    monkeypatch.setattr(vault_mod, "_REGISTRY_DB_PATH", tmp_db)
    monkeypatch.setattr(vault_save_mod, "_REGISTRY_DB_PATH", tmp_db)
    monkeypatch.setattr(portfolio_jm_mod, "_REGISTRY_DB_PATH", tmp_db)

    # Reset the lazy portfolio manager singleton so it picks up the patched DB path.
    import frontend.api.portfolio as portfolio_api_mod
    monkeypatch.setattr(portfolio_api_mod, "_MANAGER", None)
