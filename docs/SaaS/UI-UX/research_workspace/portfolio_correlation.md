# Portfolio Correlation — Panel Spec

**Embedded in:** [Step 5 — Portfolio addition](portfolio_addition.md) (Panel 1)  
**Flow:** [user_flow.md](user_flow.md) §5

Advisory only — does not gate commit. Formal gate is addition panel on the same page.

## Panel A — Unconditional correlation

| Element | Spec |
|---------|------|
| Matrix | New strategy IS returns vs each committed strategy (shared IS range) |
| Heatmap | New strategy row/column highlighted |
| Summary | Mean pairwise ρ |
| Warning | Banner if mean ρ > **0.70** |

## Panel B — Drawdown conditional correlation

| Element | Spec |
|---------|------|
| Matrix | ρ_DD when either strategy DD < −5% |
| Layout | Second heatmap, same colour scale as Panel A |
| Overlap ratio | Per pair |
| Warning | Any ρ_DD > **0.60** or overlap > **0.60** |

## Interpretation (collapsible)

1. Low unconditional + high drawdown ρ → weak stress diversification  
2. Moderate unconditional + low drawdown ρ → may still help  
3. Neither auto-rejects — researcher decides with standalone merit

## Standalone route (optional)

`/research/{project_id}/portfolio-correlation` may redirect to step 5 or open as modal from Strategy Library — **not** a separate pipeline step.

## MVP

Panel A only acceptable for v1 — see [mvp_ui.md](mvp_ui.md).
