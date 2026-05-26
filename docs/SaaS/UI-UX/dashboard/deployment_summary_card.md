# Component — Deployment Summary Card

**Component ID:** `DeploymentSummaryCard`  
**Used on:** [Dashboard page layout](page_layout.md) §B

> **Implement first:** [mvp_ui.md](mvp_ui.md) (minimal card). This document is the **full** component spec as features land.

## Purpose

One card per **Deployment** (not per portfolio draft). Primary glance surface for live signal health.

## MVP variant (ship first)

See [mvp_ui.md — Minimal deployment card](mvp_ui.md#minimal-deployment-card).

## Full variant — collapsed (default)

```
┌─────────────────────────────────────────────────────────────┐
│ [Portfolio name]  ·  v{n}              [Status pill]        │
│ Deployed {relative date} · {output_mode label}              │
├─────────────────────────────────────────────────────────────┤
│ Signals · {as_of_date} · {dataset_version} · {freshness}    │
│ ┌─────────┬──────────┬──────────┬──────────┐                │
│ │ Ticker  │ Forecast │ Pos %    │ Contracts│  (max 4 rows) │
│ └─────────┴──────────┴──────────┘  +N more if >4 tickers   │
├─────────────────────────────────────────────────────────────┤
│ Scheduler · {last_job_status} · {last_job_relative_time}    │
├─────────────────────────────────────────────────────────────┤
│ [View deployment]   [Signal API ↗]          [⌄ Expand]      │
└─────────────────────────────────────────────────────────────┘
```

Fields marked **deferred** in MVP: `dataset_version`, `freshness`, deployed date, output mode line, scheduler block, Signal API button on card, expand.

## Full variant — expanded

Toggle via **Expand** (⌄ / ⌃) — **not MVP**:

- Full signal table (all tickers)
- Last **5** scheduler events (timestamp, status, short message)

## Field definitions

| Field | Source | MVP |
|-------|--------|-----|
| Portfolio name | `Portfolio.name` | ✅ |
| Version | `PortfolioVersion.version_number` | ✅ |
| Status pill | `Deployment.status` only | ✅ `Running` / `Stopped` |
| Status pill (derived) | scheduler / data health | ❌ deferred |
| Deployed date | `Deployment.started_at` | ❌ deferred |
| Output mode | `Deployment.output_mode` | ❌ deferred on card |
| as_of_date | `latest_signal.as_of` | ✅ when signals exist |
| dataset_version | `latest_signal.dataset_version` | ❌ deferred |
| Freshness | computed from `generated_at` | ❌ deferred |
| Signal columns | per `output_mode` | ✅ when signals exist |
| last_job_* | scheduler API | ❌ deferred |

## Status pill — MVP

| Pill | Condition | Token |
|------|-----------|-------|
| `Running` | `status === 'running'` | `status-running` (muted Ember) |
| `Stopped` | `status === 'stopped'` | `status-stopped` |

## Status pill — full (deferred)

Requires scheduler and/or data-publish health APIs:

| Pill label | Condition |
|------------|-----------|
| `Waiting for data` | Data not published for expected session |
| `Stale signals` | Running but freshness > threshold |
| `Job failed` | Last scheduler job failed |
| `No signals yet` | Running, no `latest_signal` |

**Priority** (if multiple): Job failed > Waiting for data > Stale > Stopped > Running.

## Signal table columns

| `output_mode` | Columns |
|---------------|---------|
| `forecast_score` | Ticker, Forecast |
| `position_fraction` | Ticker, Position % |
| `contracts` | Ticker, Contracts |

**Formatting:**

- Forecast: `[-2.00, 2.00]`, 2 decimals
- Position %: 0 decimals if >10%, else 1 decimal
- Contracts: integer

**Sort:** descending `abs(position_fraction)` or `abs(forecast_score)`; tie-break by ticker.

MVP: cap at **5** rows; `+N more` text only (no expand).

## Click targets

| Target | Action | MVP |
|--------|--------|-----|
| Card body | `/deployment/{deployment_id}` | ✅ |
| View deployment | Same | ✅ |
| Signal API | `/deployment/{id}#signal-api` | ❌ defer to detail page |
| Expand | Toggle inline | ❌ deferred |

## Stopped deployment

- Pill: `Stopped`
- If last signal exists, show as-of + table with caption `Last signals (deployment stopped)`
- Else: `No signals yet`

## Phase 2 overlays

See [phase_2.md](phase_2.md).
