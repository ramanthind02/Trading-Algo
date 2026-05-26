# Zone Manager — Field Reference

**UI surface:** [Project setup (step 1)](project_setup.md)  
**Flow:** [user_flow.md](user_flow.md)

Low-level zone behavior and API. User-facing labels:

| API / model | UI label | Where set |
|-------------|----------|-----------|
| Project test zone | **Portfolio zone** | [Portfolios create](../portfolios.md) — locked at create |
| Train / Validation zones | **Strategy zones** | [Research setup](project_setup.md) step 1 |

## Per-zone controls

| Control | Behavior |
|---------|----------|
| Date range | Non-overlapping; UTC inclusive per [`../../zone_manager.md`](../../zone_manager.md) |
| Zone name | Display label |
| Zone type | Train \| Validation (strategy); portfolio zone separate |

## Visual timeline

- Proportions, UTC range on hover, bar counts
- Gap/coverage warnings from API

## Universe (on setup page)

| Control | MVP |
|---------|-----|
| Tickers | Multi-select |
| Timeframe | Daily only |

## API

```http
POST /api/projects/{project_id}/zones
GET  /api/projects/{project_id}/zones
PUT  /api/projects/{project_id}/zones/{zone_id}
DELETE /api/projects/{project_id}/zones/{zone_id}
```

## Lock rule

Portfolio zone + zone boundaries: locked after first training/sweep job — UI read-only with explanation.
