# UI / UX

> **Note:** An earlier version of this file redirected to `docs/SaaS/UI-UX/`.
> That folder does not exist in the repo, so those links were dead. This page now
> documents the **current local research UI** that actually ships in
> `Trading-Algo`. The polished hosted "QuantFoundry-Web" product UI
> (dark-shell brand, keyboard-first workspace, streaming results) remains design
> intent — see `product_vision.md` for that vision.

## Current local UI (what exists today)

The repo ships a local research UI as a small **Flask** app, not a React/Vercel
frontend. Entry point: `frontend/app.py`. It serves two static HTML pages and a
JSON API backed by the `feature_research/ui` and `portfolio_research/ui`
packages.

### Pages

| Route | Page | Served file |
|---|---|---|
| `/` | Redirects to `/feature-research`. | — |
| `/feature-research` | Feature research workspace. | `frontend/feature_research.html` |
| `/portfolio-research` | Portfolio research workspace. | `frontend/portfolio_research.html` |

### What each workspace does

**Feature research** (`feature_research/ui`): configure and plan a research
phase, submit a run, poll job status, preview output artifacts, view parameter-
sensitivity pivot data, and commit validated features to the vault. Backed by
`WorkspaceJobManager`, `feature_research/ui/planner.py`,
`feature_research/ui/workspace.py`, `feature_research/ui/pivot_data.py`, and
`feature_research/ui/vault_save.py`.

**Portfolio research** (`portfolio_research/ui`): configure and plan a portfolio
run, submit it, poll status, and preview artifacts (e.g. the futures-sim
tearsheet/CSV outputs). Backed by `PortfolioWorkspaceJobManager`,
`portfolio_research/ui/planner.py`, and `portfolio_research/ui/workspace.py`.

### API surface (selected)

The Flask app exposes a parallel JSON API for each workspace
(`feature_research/app.py` routes):

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

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
