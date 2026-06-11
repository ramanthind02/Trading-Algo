"""FastAPI app for the research frontend. Run from the repo root::

    .venv/Scripts/python.exe -m uvicorn frontend.api.server:app --reload --port 5057

All data routes are under ``/api``. In production the built React app
(``frontend/web/dist``) is served at ``/``; in dev the Vite server (port 5173) calls the API
cross-origin (CORS is open to localhost).
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from frontend.api import (
    artifacts,
    conditional_returns,
    data as data_api,
    live,
    portfolio,
    results,
    schema,
    spec_store,
    vault,
    vault_browser,
)
from frontend.api.paths import REPO_ROOT, resolve_artifact_path
from frontend.api.runs import RUN_MANAGER

app = FastAPI(title="Trading-Algo Research API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── data registry routes (M6.1) ───────────────────────────────────────────────
app.include_router(data_api.router, prefix="/api/data")


# ── meta ─────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/schema")
def get_schema() -> dict[str, Any]:
    return schema.form_schema()


@app.get("/api/modules")
def get_modules() -> dict[str, Any]:
    return {"modules": schema.module_catalog()}


@app.get("/api/modules/{name}")
def get_module(name: str) -> dict[str, Any]:
    try:
        return schema.module_detail(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # importing the node failed — surface it, don't 500 opaquely
        raise HTTPException(status_code=422, detail=f"Could not introspect {name}: {exc}") from exc


# ── specs ────────────────────────────────────────────────────────────────────


@app.get("/api/specs")
def list_specs() -> dict[str, Any]:
    return {"specs": spec_store.list_specs()}


@app.get("/api/specs/{spec_id}")
def get_spec(spec_id: str) -> dict[str, Any]:
    try:
        return spec_store.get_spec(spec_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/specs/{spec_id}")
def put_spec(spec_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return spec_store.save_spec(spec_id, payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/specs")
def create_spec(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    spec_id = spec_store.derive_id(payload)
    try:
        return spec_store.save_spec(spec_id, payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/specs/{spec_id}")
def remove_spec(spec_id: str) -> dict[str, str]:
    try:
        spec_store.delete_spec(spec_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted", "id": spec_id}


@app.post("/api/specs/validate")
def validate_spec(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return spec_store.validate_payload(payload)


# ── runs ─────────────────────────────────────────────────────────────────────


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    return {"runs": RUN_MANAGER.list()}


@app.post("/api/runs")
def start_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Launch a phase. Body: ``{"spec_id"|"spec", "phase": "exploration"|"validation"}``."""

    spec_dict = payload.get("spec")
    spec_id = payload.get("spec_id")
    phase = str(payload.get("phase", "exploration"))
    if spec_dict is None and spec_id is not None:
        spec_dict = spec_store.get_spec(spec_id)["spec"]
    if spec_dict is None:
        raise HTTPException(status_code=422, detail="Provide 'spec' (inline) or 'spec_id'.")
    try:
        return RUN_MANAGER.start(spec_id or spec_store.derive_id(spec_dict), spec_dict, phase)
    except ValueError as exc:  # invalid spec / unknown phase
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # already running
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUN_MANAGER.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.get("/api/runs/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, Any]:
    run = RUN_MANAGER.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    reports_dir, viz_dir = results.primary_view_dirs(run)
    return artifacts.list_for_run(reports_dir, viz_dir)


@app.get("/api/runs/{run_id}/results")
def run_results(run_id: str) -> dict[str, Any]:
    """Per-lane results: ``{"lanes": {<lane>: {headline, grid, plateau, equity}}, "lane_order",
    "primary_lane"}``. Dual-feed exploration returns both ``futures`` and ``cfd`` lanes; a
    validation/legacy run returns a single ``default`` lane."""

    run = RUN_MANAGER.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return results.build_run_results(run)


@app.post("/api/runs/{run_id}/conditional-returns")
def run_conditional_returns(run_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Bucket the run's best combo's returns by a chosen indicator (+ optional binary regime).

    Body: ``{"indicator": {"module", "params"}, "bin_mode": "quantile"|"fixed", "n_bins": int,
    "edges": [..], "regime": {"module", "params", "threshold", "above"} | null, "per_ticker": bool}``.
    """

    run = RUN_MANAGER.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        return conditional_returns.build_conditional_returns(run, payload)
    except FileNotFoundError as exc:  # spec file gone
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:  # bad indicator / params
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str) -> dict[str, str]:
    try:
        RUN_MANAGER.delete(run_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:  # run in progress
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "deleted", "id": run_id}


# ── artifacts ────────────────────────────────────────────────────────────────


@app.get("/api/artifacts/preview")
def artifact_preview(path: str = Query(...)) -> dict[str, Any]:
    try:
        return artifacts.preview(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/artifacts/raw")
def artifact_raw(path: str = Query(...)) -> FileResponse:
    try:
        resolved = resolve_artifact_path(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(resolved)


# ── vault promotion (human-gated) ─────────────────────────────────────────────


def _resolve_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    spec_dict = payload.get("spec")
    spec_id = payload.get("spec_id")
    if spec_dict is None and spec_id is not None:
        spec_dict = spec_store.get_spec(spec_id)["spec"]
    if spec_dict is None:
        raise HTTPException(status_code=422, detail="Provide 'spec' (inline) or 'spec_id'.")
    return spec_dict


@app.post("/api/vault/preview")
def vault_preview(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return vault.preview(_resolve_spec_payload(payload))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/vault/commit")
def vault_commit(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return vault.commit(_resolve_spec_payload(payload))
    except ValueError as exc:  # not eligible / invalid spec
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── vault browser / features archive ──────────────────────────────────────────


@app.get("/api/vault/features")
def vault_features(profile: str = Query("prop")) -> dict[str, Any]:
    return vault_browser.list_vault(profile)


@app.get("/api/vault/features/detail")
def vault_feature_detail(profile: str = Query(...), path: str = Query(...)) -> dict[str, Any]:
    try:
        return vault_browser.feature_detail(profile, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vault/features/to-spec")
def vault_feature_to_spec(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Reconstruct a spec from a vault feature and save it to research/specs/ for re-research."""

    profile = payload.get("profile", "prop")
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=422, detail="'path' is required.")
    try:
        spec_dict = vault_browser.spec_from_feature(profile, path)
        return spec_store.save_spec(spec_store.derive_id(spec_dict), spec_dict)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── portfolio research ────────────────────────────────────────────────────────


@app.get("/api/portfolio/defaults")
def portfolio_defaults() -> dict[str, Any]:
    return portfolio.defaults()


@app.post("/api/portfolio/runs")
def portfolio_start(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    try:
        return portfolio.start(payload)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/portfolio/runs/{job_id}")
def portfolio_job(job_id: str) -> dict[str, Any]:
    job = portfolio.job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Portfolio run not found: {job_id}")
    return job


@app.get("/api/portfolio/artifacts")
def portfolio_artifacts() -> dict[str, Any]:
    return portfolio.artifacts()


@app.get("/api/portfolio/weight-layer")
def portfolio_weight_layer() -> dict[str, Any]:
    return portfolio.weight_layer()


# ── sleeves (user-defined, additive to the built-ins) ─────────────────────────


@app.get("/api/sleeves")
def list_sleeves() -> dict[str, Any]:
    from ensemble.vault.constants import valid_weight_hierarchy_groups

    return {"sleeves": sorted(valid_weight_hierarchy_groups())}


@app.post("/api/sleeves")
def add_sleeve(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from ensemble.vault.constants import add_custom_sleeve

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=422, detail="'name' is required.")
    try:
        sleeves = add_custom_sleeve(str(name))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"sleeves": sorted(sleeves)}


# ── live monitoring (node-mediated dashboard feed) ────────────────────────────


@app.get("/api/live/brokers")
def live_brokers() -> dict[str, Any]:
    return live.list_brokers()


@app.get("/api/live/snapshot")
def live_snapshot(broker: str = Query(...)) -> dict[str, Any]:
    try:
        return live.get_snapshot(broker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/live/risk")
def live_risk(broker: str = Query(...)) -> dict[str, Any]:
    try:
        return live.get_risk(broker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/live/equity")
def live_equity(broker: str = Query(...), limit: int = Query(2000)) -> dict[str, Any]:
    try:
        return live.get_equity(broker, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/live/flatten")
def live_flatten(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Queue a node-mediated flatten-all kill switch (demo-only, confirm-gated)."""

    broker = payload.get("broker")
    confirm = payload.get("confirm")
    if not broker or confirm is None:
        raise HTTPException(status_code=422, detail="Provide 'broker' and 'confirm'.")
    try:
        return live.flatten(str(broker), confirm=str(confirm))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:  # live/funded account — demo-only gate
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:  # node offline / stale snapshot
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ── static (built React app) + SPA deep-link fallback ─────────────────────────

_DIST = REPO_ROOT / "frontend" / "web" / "dist"
if _DIST.is_dir():
    # Hashed build assets.
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """Serve a real file if it exists, else index.html (client-side routing/refresh)."""

        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (_DIST / full_path).resolve()
        if full_path and candidate.is_file() and _DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
