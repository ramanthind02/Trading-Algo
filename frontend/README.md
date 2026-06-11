# Research frontend

A StrategySpec workbench: build/edit a strategy spec, run its exploration phase, and examine
results — replacing hand-editing `research/feature/config.py`. An agent (in a Claude Code chat)
and this UI share the same `research/specs/*.json` files, so you can have the agent design a
strategy and then run + inspect it here.

## Layout

| Path | What it is |
|---|---|
| `frontend/api/` | FastAPI JSON backend. Wraps `research/spec` (the StrategySpec + adapter), the exploration pipeline, and the artifact files. |
| `frontend/web/` | React + Vite + TypeScript SPA (Mantine UI). Spec Library, Spec Builder, Runs & results. |
| `frontend/app.py`, `*.html`, `static/` | **Legacy** Flask dashboard — superseded by the React app; retained for reference. |

## Run it (dev)

Two processes. From the repo root:

```powershell
# 1. API (port 5057)
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --reload --port 5057

# 2. Web (port 5173, proxies /api -> 5057)
cd frontend\web
npm install        # first time only
npm run dev
```

Open http://localhost:5173.

## Build (prod)

```powershell
cd frontend\web
npm run build      # emits frontend/web/dist
```

When `frontend/web/dist` exists, the FastAPI app serves it at `/` (and still owns `/api`), so a
single `uvicorn frontend.api.server:app` serves the whole thing.

## App sections

| Section | What it does |
|---------|-------------|
| **Spec Library** | List/create/delete specs in `research/specs/`. Each card shows combo count, validation status. |
| **Spec Builder** | Live-validated form for one spec. Saves to `research/specs/<name>.json`. |
| **Runs & results** | Launch exploration or validation; live log; results (plateau/equity/grid/headline via Plotly) once complete. |
| **Portfolio research** | Run portfolio-level research (`research/portfolio/`); view weight-layer results. |
| **Vault archive** | Browse vault features; reconstruct a spec from a vault feature for re-research. |

## API routes (port 5057)

| Method + path | Purpose |
|---------------|---------|
| `GET /api/schema` | Form schema (tickers, enums, max_grid_combos) |
| `GET /api/modules` / `GET /api/modules/{name}` | Bias-node catalog + detail |
| `GET /api/specs` / `GET /api/specs/{id}` | List / get spec |
| `POST /api/specs` / `PUT /api/specs/{id}` | Create / update spec |
| `DELETE /api/specs/{id}` | Delete spec |
| `POST /api/specs/validate` | Live validation (debounced from form) |
| `GET /api/runs` / `POST /api/runs` | List runs / launch phase (`exploration` or `validation`) |
| `GET /api/runs/{id}` / `GET /api/runs/{id}/results` | Run status / results |
| `GET /api/runs/{id}/artifacts` | Artifact listing for a run |
| `GET /api/artifacts/preview` / `GET /api/artifacts/raw` | Preview or download an artifact |
| `POST /api/vault/preview` / `POST /api/vault/commit` | Dry-run / commit vault promotion |
| `GET /api/vault/features` | Vault archive browser |
| `GET /api/portfolio/defaults` / `POST /api/portfolio/runs` | Portfolio research |
| `GET /api/sleeves` / `POST /api/sleeves` | List / add vault sleeves |

## How a spec flows

```
research/specs/<name>.json  ──(adapter.to_feature_config)──►  exploration/validation phase
        ▲  ▲                                                               │
   agent │  │ Spec Builder (this UI)                                       ▼
        └──┴─────────────────────────  artifacts (CSV/JSON/PNG) ──► results views (Plotly)
```

Nothing here writes to the vault automatically — promotion stays a separate, human-gated step
(`POST /api/vault/commit`).
The StrategySpec JSON contract + (de)serialization live in `research/spec/serialization.py`.

> _Verified against current code via CodeGraph on 2026-06-07._
