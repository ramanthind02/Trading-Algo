# Deployment (Deploy tab)

**Parent:** [UI/UX README](README.md)  
**Route:** `/portfolios/{id}/deploy` (old `/deployment` redirects here)

> **Navigation:** This is the **Deploy tab** inside the portfolio hub — not a top-level nav item. See [portfolios.md](portfolios.md) for the full hub spec.

## Purpose

Manage live deployments and **Signal API** access for this portfolio. Dashboard shows a cross-portfolio summary via [dashboard/mvp_ui.md](dashboard/mvp_ui.md).

## List page (MVP)

| Column | MVP |
|--------|-----|
| Portfolio name + version | ✅ |
| Status | ✅ `running` / `stopped` only |
| As of | ✅ if latest signal exists |
| Actions | Open detail |

**Deferred on list:** `waiting_for_data`, dataset version column, scheduler status.

## Detail page (MVP)

### Status header

| Field | MVP |
|-------|-----|
| Status | ✅ running / stopped + **Stop** when running |
| Output mode | ✅ display only (set at deploy) |
| Portfolio version | ✅ link or label |
| Dataset version | ❌ defer until Signal API returns it |

### Latest signals (MVP)

Table: ticker + columns for deployment `output_mode` (same as [dashboard/mvp_ui.md](dashboard/mvp_ui.md)).

If no signal: `No signals yet`.

### Signal API (`#signal-api`) — MVP

| Feature | MVP |
|---------|-----|
| Create key (secret shown once) | ✅ when API ships |
| List keys (prefix only) | ✅ |
| Revoke key | ✅ |
| Docs + example snippet | ✅ |

Multiple keys per deployment: defer cap UI until needed; single key is fine for alpha.

### Deferred on detail (MVP)

| Feature | When |
|---------|------|
| Scheduler / last run / recent events | Scheduler API |
| `waiting_for_data` banner | Health API |
| Live performance charts | Phase 2 |
| Monitoring suite | Phase 2 — [`../robustness_tests/monitoring.md`](../robustness_tests/monitoring.md) |

### Stop

- **Stop** — `POST /api/deployments/{id}/stop` on detail page only (not dashboard)

## Future

See [dashboard/phase_2.md](dashboard/phase_2.md) and original monitoring/chart list in prior spec.

## MVP vs future

| Feature | MVP | Future |
|---------|-----|--------|
| Status, keys, latest signals | ✅ | |
| Dataset version, scheduler UI | | ✅ |
| Live charts, monitoring | | ✅ |

See [mvp_scope.md](mvp_scope.md).
