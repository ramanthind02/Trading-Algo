# Step 2 — Strategy Editor

**Flow:** [user_flow.md](user_flow.md) §2  
**Route:** `/research/{project_id}/editor`

## Purpose

Write the strategy `compute()` function, validate against the contract, and designate the **active strategy** used in steps 3–6.

Spec: [`../../strategy_spec.md`](../../strategy_spec.md)

## Layout

```text
[Stepper: 1 Setup ✓ | 2 Strategy ● | 3 Sweep | …]

Active strategy: [ EWMAC draft ▼ ]  [+ New strategy]

┌─ Code ─────────────────────┐  ┌─ Metadata ─────────────┐
│  def compute(candles, …)   │  │ name, max_lookback      │
│                            │  │ params_schema (drives   │
│                            │  │   step 3 controls)      │
└────────────────────────────┘  │ validation issues       │
                                │ version drafts list     │
                                └─────────────────────────┘

[Save draft]  [Quick backtest ↗]          [Continue → Parameter sweep]
```

## Active strategy selector

| Behavior | Spec |
|----------|------|
| Dropdown | All drafts in this project |
| New strategy | Creates empty draft; inherits project tickers/timeframe |
| Switch | Does not delete sweep results for other drafts |

Steps 3–6 read **active strategy id** from session or URL query `?strategy=`.

## Metadata panel

| Field | Notes |
|-------|-------|
| Strategy name | Required |
| `max_lookback` | Required for contract |
| `params_schema` | **Required for parameter sweep** — defines dynamic UI in step 3 |
| Tickers | Default from project setup; override if needed |

Live validation on save (errors block Continue).

## Quick backtest (auxiliary)

Link opens [backtest_runner.md](backtest_runner.md) in modal or `/backtest` with same active strategy — does not replace step 3.

## Exit criteria

| Criterion | Required |
|-----------|----------|
| Valid `compute()` saved | ✅ |
| `params_schema` present | ✅ for sweep |
| Active strategy selected | ✅ |

**Continue** → `/research/{project_id}/parameter-sweep`

## MVP vs full

| Feature | MVP | Later |
|---------|-----|-------|
| Editor + save + active selector | ✅ | |
| params_schema builder UI | Manual JSON OK | Visual schema builder |
| Collaborative editing | | ✅ |
