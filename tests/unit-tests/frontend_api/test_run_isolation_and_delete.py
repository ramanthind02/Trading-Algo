"""Tests locking in two recent run-manager changes:

* **Issue 6 — per-run result isolation:** each run's ``reports_dir`` / ``viz_dir`` /
  ``viz_parent_dir`` is keyed by its own ``run_id`` and differs between runs (so two runs never
  read back each other's outputs). Asserted at the ``SpecRunManager.start`` path-derivation level
  (cheap — the real pipeline is stubbed out).
* **Issue 5 — run deletion:** ``DELETE /api/runs/{run_id}`` removes a terminal run (200, gone
  from the listing), 409s on an in-progress run, and 404s on an unknown id. Exercised through the
  FastAPI ``TestClient`` against the same ``RUN_MANAGER`` singleton the routes use.

Style mirrors ``test_api.py`` (the ``_payload`` builder, ``importorskip("httpx")`` for the HTTP
smoke, monkeypatching to keep things off the real pipeline).
"""

from __future__ import annotations

import shutil

import pytest

from frontend.api.runs import SpecRun, SpecRunManager


def _payload(**overrides) -> dict:
    """A valid spec payload (same shape as test_api.py's builder)."""

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


# ── Issue 6: per-run result isolation (path derivation) ───────────────────────


@pytest.fixture()
def stub_execute(monkeypatch):
    """Stub the heavy pipeline so ``start`` only exercises path derivation/registry, not a run."""

    monkeypatch.setattr(SpecRunManager, "_execute", staticmethod(lambda *a, **k: None))


def _start_isolated(manager: SpecRunManager, spec_id: str, payload: dict) -> dict:
    """Start a run, drain its executor, then replace it so the next ``start`` is admitted.

    Draining (``shutdown(wait=True)``) ensures the worker thread has fully finished before the
    helper returns.  This prevents leaked executor jobs from running in a later test's monkeypatch
    context and inadvertently capturing calls that belong to that test's assertions.
    """

    from concurrent.futures import ThreadPoolExecutor

    run = manager.start(spec_id, payload)
    manager._executor.shutdown(wait=True)
    # Replace the executor so subsequent start() calls on the same manager work.
    manager._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-run")
    with manager._lock:
        manager._active_id = None
    return run


def test_two_runs_have_isolated_result_dirs(stub_execute):
    manager = SpecRunManager()

    run_a = _start_isolated(manager, "spec_a", _payload(name="spec_a"))
    run_b = _start_isolated(manager, "spec_b", _payload(name="spec_b"))

    assert run_a["run_id"] != run_b["run_id"]

    for run in (run_a, run_b):
        rid = run["run_id"]
        # Every per-run output location is keyed by this run's id.
        assert rid in run["reports_dir"], run["reports_dir"]
        assert rid in run["viz_dir"], run["viz_dir"]
        assert rid in run["viz_parent_dir"], run["viz_parent_dir"]

    # ...and the two runs share none of their output folders.
    for field in ("reports_dir", "viz_dir", "viz_parent_dir"):
        assert run_a[field] != run_b[field], field


def test_two_runs_of_one_spec_still_isolated(stub_execute):
    """Re-running the *same* spec must not collide — isolation is by run_id, not spec name."""

    manager = SpecRunManager()
    payload = _payload(name="same_spec")

    run_1 = _start_isolated(manager, "same_spec", payload)
    run_2 = _start_isolated(manager, "same_spec", payload)

    assert run_1["run_id"] != run_2["run_id"]
    for field in ("reports_dir", "viz_dir", "viz_parent_dir"):
        assert run_1[field] != run_2[field], field
        # reports_dir is namespaced by spec name then run_id; the run_id is what disambiguates.
        assert run_1["run_id"] in run_1[field]
        assert run_2["run_id"] in run_2[field]


def test_validation_phase_dirs_keyed_by_run_id(stub_execute):
    """The validation phase derives a different viz root but is still per-run isolated."""

    manager = SpecRunManager()
    run = manager.start("spec_v", _payload(name="spec_v"), phase="validation")
    rid = run["run_id"]
    assert run["phase"] == "validation"
    assert rid in run["reports_dir"]
    assert rid in run["viz_dir"]
    assert rid in run["viz_parent_dir"]
    # Validation CSVs land under <viz_parent>/visualization/validation.
    assert run["viz_dir"].endswith("visualization/validation")


# ── Single-feed exploration (one per-run dir, no lane fan-out) ─────────────────


def test_exploration_start_single_per_run_dir(stub_execute):
    """Exploration is single-feed: one per-run dir keyed by run_id, no lane fan-out."""

    manager = SpecRunManager()
    run = _start_isolated(manager, "spec_single", _payload(name="spec_single"))
    rid = run["run_id"]

    # Single-feed research: one per-run dir keyed by run_id.
    assert rid in run["reports_dir"] and run["reports_dir"].endswith(f"/{rid}")
    assert rid in run["viz_dir"] and run["viz_dir"].endswith("visualization")
    assert rid in run["viz_parent_dir"] and run["viz_parent_dir"].endswith(f"/{rid}")


def test_execute_runs_single_exploration_on_ratio_feed(monkeypatch):
    """``_execute`` (DAILY exploration) calls the orchestrator ONCE on the faithful ratio futures
    feed and writes to the single per-run reports dir (no dual-lane fan-out)."""

    import sys
    import types
    from dataclasses import dataclass as _dataclass, field as _field

    import frontend.api.runs as runs_mod

    @_dataclass
    class _StubConfig:
        data_feed: str = "cfd"
        exploration_futures_index_tickers: tuple = ()
        visualization_parent_dir: object = None

    feeds: list = []
    monkeypatch.setattr(
        "lib.core.research_feed.set_research_feed",
        lambda feed=None, *a, **k: feeds.append(feed),
    )

    import research.spec as spec_pkg  # ensure the module object exists to patch

    monkeypatch.setattr(spec_pkg, "to_feature_config", lambda spec: _StubConfig(), raising=False)
    monkeypatch.setattr(spec_pkg, "apply_vol_scaling", lambda spec: None, raising=False)

    calls: list[dict] = []

    @_dataclass
    class _StubResult:
        eda_results: dict = _field(default_factory=dict)

    def _fake_execute_exploration_phase(config, reports_dir):
        reports_dir.mkdir(parents=True, exist_ok=True)
        config.visualization_parent_dir.mkdir(parents=True, exist_ok=True)
        calls.append({"reports_dir": str(reports_dir)})
        return _StubResult(eda_results={"combo": "x"})

    fake_exploration = types.ModuleType("research.feature.exploration")
    fake_exploration.execute_exploration_phase = _fake_execute_exploration_phase
    monkeypatch.setitem(sys.modules, "research.feature.exploration", fake_exploration)

    rid = "deadbeef-0000-0000-0000-00000000face"
    try:
        runs_mod.SpecRunManager._execute(
            _payload(name="spec_single"),  # mode="daily" → ratio futures feed
            "exploration",
            f"_unittest_single_/{rid}",
            f"_unittest_single_/{rid}/reports",
        )

        # Exactly ONE exploration run (no dual-lane).
        assert len(calls) == 1
        # DAILY research runs on the faithful ratio futures feed.
        assert "futures_ratio" in feeds
        assert (runs_mod.REPO_ROOT / "_unittest_single_" / rid / "reports").is_dir()
    finally:
        shutil.rmtree(runs_mod.REPO_ROOT / "_unittest_single_", ignore_errors=True)


def test_build_run_results_single_default_lane():
    """A run with no ``lanes`` (single-feed exploration / validation) yields a ``default`` lane."""

    from frontend.api import results

    run = {
        "phase": "validation",
        "reports_dir": "feature_research/shared_results/signed_signal/_x_/rid",
        "viz_dir": "feature_research/shared_results/_x_/rid/visualization/validation",
        "lanes": {},
    }
    out = results.build_run_results(run)
    assert list(out["lanes"]) == ["default"]
    assert out["primary_lane"] == "default"


# ── Issue 5: DELETE /api/runs/{run_id} (via TestClient) ───────────────────────


@pytest.fixture()
def clean_run_manager(monkeypatch):
    """Use the singleton the routes share, but isolate each test's runs and restore after.

    The server route functions close over ``frontend.api.server.RUN_MANAGER`` (a module-level
    alias of the same singleton), so swapping the registry on the singleton affects both.
    """

    from frontend.api.runs import RUN_MANAGER

    with RUN_MANAGER._lock:
        saved_runs = dict(RUN_MANAGER._runs)
        saved_active = RUN_MANAGER._active_id
        RUN_MANAGER._runs.clear()
        RUN_MANAGER._active_id = None
    yield RUN_MANAGER
    with RUN_MANAGER._lock:
        RUN_MANAGER._runs.clear()
        RUN_MANAGER._runs.update(saved_runs)
        RUN_MANAGER._active_id = saved_active


def _make_run(run_id: str, status: str) -> SpecRun:
    """A SpecRun record with relative output dirs that do NOT exist on disk.

    delete() rmtrees with ``ignore_errors=True``, so pointing at nonexistent (yet repo-relative,
    traversal-free) folders means the test never touches real files.
    """

    base = f"feature_research/shared_results/signed_signal/_unittest_/{run_id}"
    return SpecRun(
        run_id=run_id,
        spec_id="unit_spec",
        spec_name="unit_spec",
        phase="exploration",
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        num_combos=1,
        reports_dir=base,
        viz_dir=f"research/feature/in_sample/results/_unittest_/{run_id}/visualization",
        viz_parent_dir=f"research/feature/in_sample/results/_unittest_/{run_id}",
    )


def _insert(manager: SpecRunManager, run: SpecRun, *, active: bool = False) -> None:
    with manager._lock:
        manager._runs[run.run_id] = run
        if active:
            manager._active_id = run.run_id


# -- via the server route functions directly (no httpx needed) -----------------
# These call the FastAPI handlers exactly as the router would, so they assert the real
# status-code mapping (200 / 409 / 404) without an HTTP client dependency.


def test_route_delete_completed_run_removes_it(clean_run_manager):
    from frontend.api import server

    run = _make_run("00000000-0000-0000-0000-00000000c0de", status="completed")
    _insert(clean_run_manager, run)

    # Listed first...
    assert run.run_id in {r["run_id"] for r in server.list_runs()["runs"]}

    assert server.remove_run(run.run_id) == {"status": "deleted", "id": run.run_id}

    # ...gone afterwards: dropped from the listing and a single-run lookup 404s.
    assert run.run_id not in {r["run_id"] for r in server.list_runs()["runs"]}
    with pytest.raises(server.HTTPException) as exc:
        server.get_run(run.run_id)
    assert exc.value.status_code == 404


@pytest.mark.parametrize("status", ["queued", "running"])
def test_route_delete_in_progress_run_conflicts(clean_run_manager, status):
    from frontend.api import server

    run = _make_run(f"11111111-0000-0000-0000-0000000000{status}", status=status)
    _insert(clean_run_manager, run, active=True)

    with pytest.raises(server.HTTPException) as exc:
        server.remove_run(run.run_id)
    assert exc.value.status_code == 409
    # The record survives a conflicting delete.
    assert clean_run_manager.get(run.run_id) is not None


def test_route_delete_unknown_run_404s(clean_run_manager):
    from frontend.api import server

    with pytest.raises(server.HTTPException) as exc:
        server.remove_run("does-not-exist")
    assert exc.value.status_code == 404


# -- via the FastAPI TestClient (HTTP wiring smoke; skipped without httpx) ------


def test_http_delete_completed_run_round_trip(clean_run_manager):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from frontend.api.server import app

    client = TestClient(app)
    run = _make_run("22222222-0000-0000-0000-00000000beef", status="completed")
    _insert(clean_run_manager, run)

    assert run.run_id in {r["run_id"] for r in client.get("/api/runs").json()["runs"]}

    resp = client.delete(f"/api/runs/{run.run_id}")
    assert resp.status_code in (200, 204), resp.text
    if resp.status_code == 200:
        assert resp.json() == {"status": "deleted", "id": run.run_id}

    assert run.run_id not in {r["run_id"] for r in client.get("/api/runs").json()["runs"]}
    assert client.get(f"/api/runs/{run.run_id}").status_code == 404
    # 409 on in-progress and 404 on unknown, over HTTP.
    busy = _make_run("33333333-0000-0000-0000-000000000bad", status="running")
    _insert(clean_run_manager, busy, active=True)
    assert client.delete(f"/api/runs/{busy.run_id}").status_code == 409
    assert client.delete("/api/runs/nope").status_code == 404
