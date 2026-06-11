# UI / UX

> **Note:** An earlier version of this file redirected to `docs/SaaS/UI-UX/`.
> That folder does not exist in the repo, so those links were dead. This page now
> documents the **current local research UI** that actually ships in
> `Trading-Algo`. The polished hosted "QuantFoundry-Web" product UI
> (dark-shell brand, keyboard-first workspace, streaming results) remains design
> intent — see `product_vision.md` for that vision.

## Current local UI (what exists today)

> **Updated (Nautilus refactor):** the repo now ships a **React + Vite + Mantine**
> app under `frontend/web/`, served by a **FastAPI** backend (`frontend/api/server.py`,
> default port 5057). That React app is the current primary local UI. The **Flask**
> app described below (`frontend/app.py`) is retained as legacy and may be removed.

The legacy Flask app serves two static HTML pages and a
JSON API backed by the `research/feature/ui` and `research/portfolio/ui`
packages. Entry point: `frontend/app.py`.

### Pages

| Route | Page | Served file |
|---|---|---|
| `/` | Redirects to `/feature-research`. | — |
| `/feature-research` | Feature research workspace. | `frontend/feature_research.html` |
| `/portfolio-research` | Portfolio research workspace. | `frontend/portfolio_research.html` |

### What each workspace does

**Feature research** (`research/feature/ui`): configure and plan a research
phase, submit a run, poll job status, preview output artifacts, view parameter-
sensitivity pivot data, and commit validated features to the vault. Backed by
`WorkspaceJobManager`, `research/feature/ui/planner.py`,
`research/feature/ui/workspace.py`, `research/feature/ui/pivot_data.py`, and
`research/feature/ui/vault_save.py`.

**Portfolio research** (`research/portfolio/ui`): configure and plan a portfolio
run, submit it, poll status, and preview artifacts (e.g. the futures-sim
tearsheet/CSV outputs). Backed by `PortfolioWorkspaceJobManager`,
`research/portfolio/ui/planner.py`, and `research/portfolio/ui/workspace.py`.

### API surface (selected)

The Flask app exposes a parallel JSON API for each workspace
(`research/feature/app.py` routes):

- `GET  /api/{feature,portfolio}-research/defaults` — UI defaults from `load_config`.
- `POST /api/{feature,portfolio}-research/plan` — build a phase plan from a UI request.
- `POST /api/{feature,portfolio}-research/run` — submit a job.
- `GET  /api/{feature,portfolio}-research/jobs/<job_id>` — job status (+ `/jobs/latest` for feature research).
- `GET  /api/{feature,portfolio}-research/artifact-preview` and `artifact-raw` — read run outputs.
- `GET  /api/feature-research/pivot-data` — parameter-sensitivity pivot payload.
- `POST /api/feature-research/vault-save[/preview]` — preview and execute a vault commit.

### Running it

Start the Flask app with the shared-venv interpreter (see `CLAUDE.md` for the
exact interpreter path), then open `http://localhost:<port>/feature-research`.

## Hosted product UI (forward-looking)

The hosted SaaS UI — navigation (Dashboard, Research Workspace, Strategy Library,
Portfolio Builder, Deployment), the dark "Ink/Paper/Ember" brand, keyboard-first
interactions, and streaming bar-by-bar results — is described as product vision
in `product_vision.md` and as a frontend structure in `technical_design.md` §13.
It is not built in this repo yet.

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
