# Dashboard — MVP UI (ship first)

**Authority:** This page overrides richer layouts in sibling specs when building the first frontend. Add UI only when the API field exists and is populated in dev/prod.

**Goal:** A landing page that is trivial to implement — list deployments, show latest signals if available, link into research — without ops, data-pipeline, or quota surfaces.

## MVP principles

1. **No speculative fields** — do not design for `dataset_version`, scheduler health, freshness heuristics, or quota strips until the backend exposes them and product asks for them.
2. **Two deployment statuses only** — `Running` and `Stopped` from `Deployment.status`. No derived pills (`Stale`, `Waiting for data`, `Job failed`) in MVP UI.
3. **No dashboard aggregate endpoint required** — compose from existing CRUD: `GET /api/deployments` + `GET /api/v1/signals/latest` per deployment (acceptable for small N in alpha).
4. **Read-only routing** — links only; no deploy/stop/keys on dashboard.

## What the page contains (MVP)

| Section | Ship? | Contents |
|---------|-------|----------|
| A. Account strip | **No** | Defer — see [account_strip.md](account_strip.md) |
| B. Deployed portfolios | **Yes** | Minimal [deployment card](#minimal-deployment-card) |
| C. Continue research | **Yes** | Minimal [project row](#minimal-project-row) |
| D. Library snapshot | **No** | Defer — see [library_snapshot.md](library_snapshot.md) |

## Full-page wireframe (MVP)

```text
┌──────────┬────────────────────────────────────────────────────────┐
│ SIDEBAR  │  Dashboard                                              │
│ (5 nav)  ├────────────────────────────────────────────────────────┤
│          │                                                         │
│          │  DEPLOYED PORTFOLIOS              View all →            │
│          │  ┌─────────────────────┐  ┌─────────────────────┐      │
│          │  │ Name · v2  Running  │  │ Name · v1  Stopped  │      │
│          │  │ As of 2026-05-16    │  │ No signals yet      │      │
│          │  │ ES  +1.25   42%     │  │                     │      │
│          │  │ [View deployment]   │  │ [View deployment]   │      │
│          │  └─────────────────────┘  └─────────────────────┘      │
│          │                                                         │
│          │  CONTINUE RESEARCH                  New project +        │
│          │  ┌─────────────────────────────────────────────────┐  │
│          │  │ ES momentum v1 · edited 2h ago  Open workspace →│  │
│          │  └─────────────────────────────────────────────────┘  │
│          │                                                         │
└──────────┴────────────────────────────────────────────────────────┘
```

Single column on narrow viewports. Max 6 deployment cards, then link to Deployment list.

## Minimal deployment card

**Component ID:** `DeploymentSummaryCard` (MVP variant)

```text
┌─────────────────────────────────────────┐
│ {Portfolio name} · v{n}     [Running]   │
├─────────────────────────────────────────┤
│ As of {as_of}                           │  ← omit block if no signal yet
│ {ticker}  {values…}                     │  ← up to 5 rows, see signal columns
├─────────────────────────────────────────┤
│ [View deployment]                       │
└─────────────────────────────────────────┘
```

### Fields (MVP only)

| Field | Source | Required |
|-------|--------|----------|
| Portfolio name | `Portfolio.name` via deployment join | Yes |
| Version | `PortfolioVersion.version_number` | Yes if versioning ships |
| Status pill | `Deployment.status` → `Running` / `Stopped` | Yes |
| As of | `latest_signal.as_of` | Only when signals endpoint returns data |
| Signal rows | `latest_signal.signals[]` | Only when present |

### Explicitly not in MVP card

| Field | Add when |
|-------|----------|
| `dataset_version` | Signal API returns it and product wants it shown |
| Freshness (`2h ago`, `Stale`) | `generated_at` + publish SLA defined |
| Scheduler line | Scheduler run history API exists |
| `Waiting for data` / `Job failed` pills | `display_status` or deployment health API exists |
| Output mode label | Optional caption; not required for v0 |
| Deployed relative date | Nice-to-have; defer |
| Expand / scheduler events | [phase_2.md](phase_2.md) |
| Signal API button on card | Defer to Deployment detail (one less CTA on dashboard) |

### Signal table (MVP)

Show columns that exist on each signal row for the deployment’s `output_mode`:

| `output_mode` | Show |
|---------------|------|
| `forecast_score` | Ticker, Forecast |
| `position_fraction` | Ticker, Position % |
| `contracts` | Ticker, Contracts |

Cap at **5 tickers**; if more, `+N more` text (no expand in MVP).

### Empty signal state (deployment running)

```text
No signals yet
```

No spinner unless a fetch is in flight.

## Minimal project row

**Component ID:** `ResearchProjectRow` (MVP variant)

```text
┌─────────────────────────────────────────────────────────┐
│ {Project name} · edited {relative time}  Open workspace → │
└─────────────────────────────────────────────────────────┘
```

### Fields (MVP only)

| Field | Source |
|-------|--------|
| Project name | `ResearchProject.name` |
| Edited | `ResearchProject.updated_at` |
| Link | `/research/{project_id}` (default Zone Manager) |

### Explicitly not in MVP row

| Field | Add when |
|-------|----------|
| Job badge (`Sweep running`) | Job status on `GET /api/projects` or dedicated jobs endpoint |
| Last strategy name | Project detail includes last touched strategy |
| Mini zone bar | `zones_summary` on project list API |
| `New project +` in header | Only if create-project flow ships; else link from empty state |

## Empty states (MVP)

**No deployments:**

```text
No live portfolios yet
[Go to Portfolio Builder]
```

**No projects:** hide Continue research section.

**No deployments and no projects:** same as above; optional second line pointing to Research Workspace (no template flow unless templates ship).

## API (MVP)

No `GET /api/dashboard` required. Frontend:

```text
GET /api/deployments
  → for each running deployment (or all):
GET /api/v1/signals/latest?deployment_id=...
GET /api/projects?limit=5&sort=-updated_at
```

See [api_contract.md](api_contract.md) for optional aggregate shape when N grows.

## When to extend the card

Use this order — update [deployment_summary_card.md](deployment_summary_card.md) and flip fields from deferred → MVP in this doc:

1. Signal API stable → as-of + ticker table on card  
2. Signal API adds `generated_at` → optional “updated … ago”  
3. Signal API adds `dataset_version` → show in meta line  
4. Scheduler / health API → status pills + scheduler line + expand  
5. Quota API + billing → [account_strip.md](account_strip.md)  
6. Library counts endpoint → [library_snapshot.md](library_snapshot.md)
