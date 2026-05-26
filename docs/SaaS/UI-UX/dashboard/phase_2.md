# Dashboard — Phase 2+

Features **intentionally omitted from MVP** — add in this order (see [mvp_ui.md](mvp_ui.md#when-to-extend-the-card)).

## Card extensions

| Feature | Prerequisite |
|---------|----------------|
| `generated_at` / freshness (“2h ago”, Stale) | Signal payload + SLA |
| `dataset_version` on card | Field in Signal API |
| Output mode + deployed date captions | Product polish |
| Signal API button on card | Optional; detail page sufficient for MVP |
| Expand: full ticker table | UX choice |
| Scheduler line + event history | Scheduler run API |
| Derived status pills (`Waiting for data`, `Job failed`, `Stale`) | Health / `display_status` API |

## Page sections

| Feature | Doc |
|---------|-----|
| Account / quota strip | [account_strip.md](account_strip.md) |
| Library snapshot footer | [library_snapshot.md](library_snapshot.md) |
| `GET /api/dashboard` aggregate | [api_contract.md](api_contract.md) |

## Research rows

| Feature | Prerequisite |
|---------|----------------|
| Job badges | Job status on project list |
| Last strategy name | Project summary API |
| Mini zone bar | `zones_summary` on project list |

## Visualizations

| Feature | Notes |
|---------|-------|
| Live return sparkline | LivePerformanceRecord + charts policy |
| Monitoring badges | [monitoring.md](../../robustness_tests/monitoring.md) |
| Multi-portfolio aggregate header | 3+ deployments |

## Non-goals

- Editing monitoring thresholds from dashboard
- Portfolio weight changes
- Running backtests from dashboard
