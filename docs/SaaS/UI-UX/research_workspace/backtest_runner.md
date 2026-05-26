# Backtest Runner (Auxiliary)

**Not a pipeline step** — [user_flow.md](user_flow.md)

## Purpose

Run **one** parameter set on a chosen zone for a quick check. Primary research path is [parameter sweep](parameter_sweep.md) → [strategy robustness](strategy_robustness.md).

## Entry points

| From | Context |
|------|---------|
| Strategy editor | “Quick backtest” with active strategy |
| Sweep page | Compare one-off vs grid (optional) |

**Route:** `/research/{project_id}/backtest` (modal or full page)

## Controls

| Control | Spec |
|---------|------|
| Zone | Any strategy zone; confirmation if portfolio zone (discourage) |
| Parameters | From schema — same controls as sweep but single point |
| Run | `POST /api/backtests` |

## Results

Equity, drawdown, Sharpe, Calmar, per-ticker table — same as prior spec.

## Does not

- Replace sweep or validation steps  
- Unlock step 4 without sweep selection flow  
- Gate commit

## MVP

Optional; ship after steps 1–3 if time-constrained — see [mvp_ui.md](mvp_ui.md).
