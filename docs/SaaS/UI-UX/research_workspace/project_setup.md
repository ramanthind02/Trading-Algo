# Step 1 — Research Setup (strategy zones)

**Flow:** [user_flow.md](user_flow.md) §1  
**Route:** `/research/{portfolio_id}/setup`  
**Field reference:** [zone_manager.md](zone_manager.md)

## Purpose

Configure **strategy-level** research data splits inside the selected portfolio. The **portfolio zone** is already fixed at portfolio creation — shown here **read-only**.

**Portfolio creation:** [../portfolios.md](../portfolios.md) — not on this page.

## Prerequisites

| Requirement | Source |
|-------------|--------|
| Portfolio exists | [Portfolios](../portfolios.md) create wizard |
| Portfolio zone locked | Set at create |

If user lands without `portfolio_id`, redirect to `/research` selector.

## Layout

```text
[Stepper: 1 Setup ● | 2 Strategy | 3 Sweep | …]

┌─ Portfolio zone (locked) ──────────────────────────────────┐
│ 🔒 Portfolio zone · 2018-01-01 → 2023-12-31              │
│ Reserved for final portfolio evaluation — not used below.  │
│ [View in Portfolios]                                       │
└──────────────────────────────────────────────────────────┘

Strategy zones (editable)
  [████ train ████][██ validation ██]   within pre-test window only
  add zone · edit dates · bar counts

Universe (from portfolio)
  ES, NQ, CL · Daily — inherited 🔒  (MVP: not editable here)

[Save]                              [Continue → Strategy editor]
```

## Section A — Portfolio zone (read-only)

| Element | Behavior |
|---------|----------|
| Dates | From `project_test_start` / `project_test_end` |
| Timeline segment | Amber / portfolio color on zone bar |
| Actions | None — no edit, no drag |
| Link | Portfolio detail for full metadata |

Copy: *“Fixed when this portfolio was created. Affects all strategies.”*

## Section B — Strategy zones (editable)

| Control | Spec |
|---------|------|
| Types | Train \| Validation only |
| Constraint | All `end_at_utc < project_test_start` |
| Default | Created at portfolio create (60/20 split of pre-test); user adjusts here |
| Add zone | Train or Validation; no overlapping ranges |
| Warnings | Gaps / coverage from API |

Train = `ember`; Validation = `ember-deep` — [design system](../design_system.md).

**Lock after first training/sweep job:** strategy zone boundaries become read-only (backend rule). Portfolio zone stays locked from create.

## Section C — Universe

| MVP | Behavior |
|-----|----------|
| Display only | Tickers + timeframe from portfolio record |
| Edit | Redirect to Portfolios — only if product allows pre-research edit (defer MVP) |

## Exit criteria

| Criterion | Required |
|-----------|----------|
| ≥1 Train + ≥1 Validation, valid vs portfolio zone | ✅ |
| Saved | ✅ |

**Continue** → `/research/{portfolio_id}/editor`

## What moved out of this step

| Formerly here | Now |
|---------------|-----|
| Set portfolio zone dates | [Portfolios create wizard](../portfolios.md) |
| Create portfolio + tickers | [Portfolios create wizard](../portfolios.md) |
| Pick portfolio | [project_selector.md](project_selector.md) |
