# Research Workspace — User Flow

**Parent:** [Research Workspace README](README.md)

Develop strategies inside a **portfolio** whose **portfolio zone was locked at creation** ([Portfolios](../portfolios.md)). Backend: [`../../research_flow.md`](../../research_flow.md), zones: [`../../data_flow.md`](../../data_flow.md) §2.

## Terminology

| UI label | Meaning |
|----------|---------|
| **Portfolio** | Container with locked holdout + universe. Created under **Portfolios**. |
| **Portfolio zone** | Project test holdout — **immutable after portfolio create** |
| **Strategy zones** | Train + Validation before portfolio zone — editable in Research step 1 until first training job |
| **Active strategy** | Draft selected in editor; drives steps 3–6 |

## Two-area model

```text
  PORTFOLIOS (once)                    RESEARCH (repeat per strategy)
  ─────────────────                    ──────────────────────────────
  Create portfolio                     0. Select portfolio
  Lock portfolio zone                  1. Strategy zones only
  Set tickers + timeframe              2. Editor → 3. Sweep → 4. Robustness
                                       5. Addition → 6. Commit
```

## Pipeline overview

```text
Portfolios:  [ Create portfolio + lock zone ]  (separate nav area)

Research:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 0. Select   │ →  │ 1. Strategy │ →  │ 2. Strategy │ →  │ 3. Parameter│ →  │ 4. Strategy │ →  │ 5. Portfolio│
│  portfolio  │    │  zone setup │    │    editor   │    │    sweep    │    │  robustness │    │   addition  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                                                        │
                                                                                                        ▼
                                                                                               ┌─────────────┐
                                                                                               │ 6. Commit   │
                                                                                               └─────────────┘
```

## Step index

| Step | Page | User goal |
|------|------|-----------|
| — | [Portfolios](../portfolios.md) | Create portfolio; **lock portfolio zone** |
| 0 | [Select portfolio](project_selector.md) | Enter research for one portfolio |
| 1 | [Research setup](project_setup.md) | Strategy zones only (portfolio zone read-only) |
| 2 | [Strategy editor](strategy_editor.md) | Code + active strategy |
| 3 | [Parameter sweep](parameter_sweep.md) | Grid, IS tests, select params |
| 4 | [Strategy robustness](strategy_robustness.md) | Validation on locked params |
| 5 | [Portfolio addition](portfolio_addition.md) | Correlation + gate |
| 6 | [Commit strategy](commit_strategy.md) | Immutable version |

**Auxiliary:** [Backtest runner](backtest_runner.md)

## Zone timeline (all in-project steps)

```text
[ train ][ validation ]  |  🔒 portfolio zone
   editable                 fixed at portfolio create
```

Clicking portfolio zone segment → tooltip + link to Portfolios detail (no edit).

## Progress stepper

Steps **1–6** only (portfolio create is outside Research). Locked steps show prerequisite tooltips.

## Multi-strategy

Repeat steps **2–6** per strategy. Portfolio zone never changes. Strategy zones lock after first training job on that portfolio.

## New portfolio for new holdout

User must **create a new portfolio** in Portfolios — cannot retarget holdout on an existing one.

## MVP

[mvp_ui.md](mvp_ui.md) — ship Portfolios create + lock before Research step 1.
