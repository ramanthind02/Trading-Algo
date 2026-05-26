# Dashboard — API Contract

## MVP (compose existing endpoints)

No dedicated dashboard endpoint required for first frontend.

```http
GET /api/deployments
GET /api/v1/signals/latest?deployment_id={id}   # per deployment, when Signal API ships
GET /api/projects                               # sort by updated_at client-side, take 5
```

### MVP deployment + signal shape (minimal)

Use only fields the UI will render:

```json
{
  "deployment_id": "uuid",
  "portfolio_name": "Mean Reversion ES",
  "version_number": 2,
  "status": "running",
  "latest_signal": {
    "as_of": "2026-05-16",
    "signals": [
      { "ticker": "ES", "forecast_score": 1.25, "position_fraction": 0.42, "contracts": 1 }
    ]
  }
}
```

`latest_signal` may be `null` — card shows `No signals yet`.

**Do not require in MVP UI:** `dataset_version`, `generated_at`, `scheduler`, `display_status`, `quota`, `library_counts`.

### MVP projects shape

```json
{
  "project_id": "uuid",
  "name": "ES momentum v1",
  "updated_at": "2026-05-18T09:00:00Z"
}
```

## Optional aggregate (when N deployments grows)

```http
GET /api/dashboard
```

Add when polling `signals/latest` per deployment becomes costly. Response may include deferred fields as backends land — UI should ignore unknown keys.

### Full response shape (future)

```json
{
  "quota": { },
  "deployments": [
    {
      "deployment_id": "uuid",
      "portfolio_name": "Mean Reversion ES",
      "version_number": 2,
      "status": "running",
      "output_mode": "position_fraction",
      "display_status": "running",
      "latest_signal": {
        "as_of": "2026-05-16",
        "dataset_version": "2026-W20",
        "generated_at": "2026-05-16T22:15:00Z",
        "signals": []
      },
      "scheduler": {
        "last_run_at": "2026-05-16T22:10:00Z",
        "last_run_status": "completed",
        "recent_events": []
      }
    }
  ],
  "research_projects": [
    {
      "project_id": "uuid",
      "name": "ES momentum v1",
      "updated_at": "2026-05-18T09:00:00Z",
      "last_strategy_name": "EWMAC ES",
      "last_workspace_path": "/research/{id}/parameter-sweep",
      "zones_summary": { "train_pct": 60, "validation_pct": 20, "test_pct": 20 },
      "active_job": { "type": "parameter_sweep", "status": "running" }
    }
  ],
  "library_counts": {
    "committed_strategies": 8,
    "draft_portfolios": 2
  }
}
```

## Polling (MVP)

| Data | Interval |
|------|----------|
| Deployments + signals | 60s on dashboard (or manual refresh only in alpha) |
| Projects list | On mount + on return navigation |

Defer job polling until job badges ship.

## Related endpoints

| Action | API |
|--------|-----|
| Deployment list | `GET /api/deployments` |
| Latest signals | `GET /api/v1/signals/latest?deployment_id=...` |
| Stop deployment | `POST /api/deployments/{id}/stop` — **Deployment detail only** |

See [`../../technical_design.md`](../../technical_design.md) §10.6.
