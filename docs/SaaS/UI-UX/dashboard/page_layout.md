# Dashboard — Page Layout

> **MVP:** Two sections only (B + C). See [mvp_ui.md](mvp_ui.md).

## Global chrome

Inherited from [Navigation](../navigation.md):

- **Left sidebar:** Dashboard (active), Research Workspace, Strategy Library, Portfolio Builder, Deployment
- **Top bar:** Title `Dashboard`; optional user avatar (plan badge deferred)
- **No** global primary action on MVP dashboard

## Main content (MVP)

Scrollable column, **max-width ~1200px** centered.

```
┌──────────────────────────────────────────────────────────────┐
│ B. Deployed portfolios                                        │
├──────────────────────────────────────────────────────────────┤
│ C. Continue research                                          │
└──────────────────────────────────────────────────────────────┘
```

### Deferred sections (not in MVP layout)

| Section | Doc | When |
|---------|-----|------|
| A. Account strip | [account_strip.md](account_strip.md) | Quota API + product decision |
| D. Library snapshot | [library_snapshot.md](library_snapshot.md) | Counts API + product decision |

## Section B — header

```
Deployed portfolios                    [View all →]
```

- **View all** → `/deployment`
- Zero deployments → [empty state](empty_states.md)

## Responsive grid (section B)

| Viewport | Layout |
|----------|--------|
| ≥1280px | 2 columns |
| 768–1279px | 1 column |
| <768px | 1 column; signal values may wrap (no horizontal table required for MVP) |

## Card ordering (section B)

1. `running` before `stopped`
2. Within status: `started_at` descending (if available; else arbitrary)

Do **not** sort by derived health (stale/failed) in MVP — those states are not shown.

## List limits

| Section | Max visible | Overflow |
|---------|-------------|----------|
| Deployments | 6 cards | `View all N deployments →` |
| Research projects | 5 rows | Omit overflow link in MVP |

## Error handling

| Condition | UI |
|-----------|-----|
| 401 | Redirect to login |
| Deployments fetch fails | Message in section B + retry |
| Signals fetch fails per card | Card shows status + “Signals unavailable” |
| Projects fetch fails | Hide section C or inline retry |
