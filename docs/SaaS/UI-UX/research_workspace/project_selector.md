# Step 0 — Select Portfolio (research entry)

**Flow:** [user_flow.md](user_flow.md) §0  
**Route:** `/research`

## Purpose

Choose **which portfolio** to develop strategies in. Does **not** create portfolios or set the portfolio zone — that is [Portfolios](../portfolios.md).

## Screen layout

```text
Research Workspace

Develop strategies inside a portfolio. The portfolio zone is fixed at creation.

┌────────────────────────────────────────────────────────────┐
│ Select portfolio              [+ New portfolio → Portfolios]│
├────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Prop futures core                                     │   │
│ │ Portfolio zone 🔒 2018-01-01 → 2023-12-31            │   │
│ │ 2 drafts in progress · last edited 2d ago             │   │
│ │                                    [Continue →]       │   │
│ └──────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

## Portfolio card fields

| Field | Notes |
|-------|-------|
| Name | Portfolio name |
| Portfolio zone | Short locked date range (read-only cue) |
| In progress | Draft strategies count (optional) |
| Last edited | Last research activity |
| Continue | → last step or `/research/{id}/setup` |

## New portfolio

**[+ New portfolio]** navigates to `/portfolios/new` — **not** an inline modal on this page.

After create wizard completes, user may land here or go straight into research setup.

## Empty state

```text
No portfolios yet
Create a portfolio to define the holdout zone and universe, then research strategies.
[Create portfolio]
```

## Exit criteria

Portfolio selected → in-project routes with `portfolio_id` in path.

## No stepper

Zone timeline + pipeline stepper appear on steps 1–6. Portfolio zone on timeline is **locked segment** (non-interactive).
