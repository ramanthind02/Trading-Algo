# Component — Account Strip

**Component ID:** `DashboardAccountStrip`  
**Used on:** [Dashboard page layout](page_layout.md) §A

## MVP status: **not shipped**

Defer until quota/billing APIs are exposed and product wants usage visible on login. MVP dashboard has **no account strip** — see [mvp_ui.md](mvp_ui.md).

## Purpose (when enabled)

Surface quota and plan usage without blocking workflow. Low visual weight — single horizontal bar above deployment cards.

## Anatomy

```
┌──────────────────────────────────────────────────────────────┐
│ Backtests 12/50  ·  Jobs 1/3  ·  Dataset 2026-W20 active      │
└──────────────────────────────────────────────────────────────┘
```

**Note:** `Dataset … active` requires dataset publish pointer API — do not add until backend exists (same rule as signal `dataset_version` on cards).

## Fields (future)

| Element | Requires |
|---------|----------|
| Backtests used/limit | `UserPlan` + `UsageLedger` |
| Concurrent jobs | Job admission API |
| Active dataset | Dataset version / publish API |

## Prerequisite checklist

- [ ] `GET` quota or usage endpoint implemented
- [ ] Limits enforced on backtest submit (so strip is truthful)
- [ ] Product decision to show on dashboard vs settings only
