# MVP Scope — UI Features

The app ships three top-level nav items. All strategy, composition, and deployment management lives inside the portfolio hub.

## Navigation

| Ship in MVP | Defer |
|-------------|-------|
| 3-item sidebar: Dashboard, Portfolios, Research | Collapsed sidebar state |
| Portfolio context sub-label in Research nav item | |

## Dashboard

| Ship in MVP | Defer |
|-------------|-------|
| Portfolio cards: name, status pill, last signal `as_of` | Sparklines, live P&L |
| Quick link: Open portfolio | Per-signal expand panel |
| Empty state → `/portfolios/new` | Account/quota strip |

API: `GET /api/deployments` + `GET /api/deployments/{id}/signals/latest`

## Portfolios — list & create

| Ship in MVP | Defer |
|-------------|-------|
| Portfolio list with cards | Sorting / filtering |
| Create wizard (A → B → C → D) | Clone portfolio with new zone |
| Portfolio zone locked on create | Edit universe after create |
| Status pill on list card | |

## Portfolio hub — Overview tab

| Ship in MVP | Defer |
|-------------|-------|
| Zone + bar count (read-only) | Equity curve |
| Strategy count + research link | Regime analysis |
| Latest signals (if deployed) | |
| Quick links to Research and Deploy | |

## Portfolio hub — Strategies tab

| Ship in MVP | Defer |
|-------------|-------|
| Strategy cards: name, version, universe, val SR, test status | Tagging |
| Strategy detail: version history, metrics, source viewer | Team sharing |
| Add to Compose shortcut | Benchmark comparison |

## Portfolio hub — Compose tab

| Ship in MVP | Defer |
|-------------|-------|
| Add/remove committed strategies | Optimization |
| Manual weight input (equal default) | Auto weights |
| Weight layer selector (equal_signal default) | |
| Run holdout evaluation on portfolio zone | |
| Deploy from Compose | |

## Portfolio hub — Deploy tab

| Ship in MVP | Defer |
|-------------|-------|
| Deployment card: version, status, `as_of` | Scheduler status |
| Latest signals table (ticker + forecast + position) | Dataset version |
| Stop deployment (confirm dialog) | Live performance charts |
| Signal API: create / list / revoke keys + docs snippet | Monitoring suite |
| New deployment CTA | |

## Research pipeline

| Step | Ship in MVP | Defer |
|------|-------------|-------|
| 0 Portfolio picker | Pick portfolio | |
| 1 Setup | Strategy zones only; portfolio zone read-only | Drag timeline |
| 2 Editor | Editor + params_schema | Collaborative editing |
| 3 Sweep | Dynamic params + table + select | Full heatmap, permutations |
| 4 Robustness | Validation run + pass/fail | Full charts |
| 5 Addition | Gate ΔSR | Panel B correlation |
| 6 Commit | Commit CTA | Rich summary |

## UI vs backend

If a field is not in the deployed API response, do not show it. Deferred fields (dataset_version, scheduler, health status) render only when the endpoint returns them.

## Removed from MVP navigation

The following are **not** top-level nav items in MVP — they are tabs inside the portfolio hub:

| Removed nav item | Now lives at |
|-----------------|-------------|
| Strategy Library | `/portfolios/{id}/strategies` |
| Portfolio Builder | `/portfolios/{id}/compose` |
| Deployment | `/portfolios/{id}/deploy` |

Old routes (`/strategy-library`, `/portfolio-builder`, `/deployment`) redirect to the appropriate portfolio tab.
