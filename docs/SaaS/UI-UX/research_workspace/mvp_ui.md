# Research Workspace — MVP UI

**Authority:** Ship order for the pipeline in [user_flow.md](user_flow.md). Hide stepper steps until backend exists.

## Prerequisites

| Area | Ship before Research |
|------|---------------------|
| [Portfolios](../portfolios.md) | Create wizard + lock portfolio zone + universe |

## MVP pipeline (recommended order)

| Step | Ship when | Minimum UI |
|------|-----------|------------|
| 0 Select portfolio | Portfolios API | List portfolios; link to create; no inline create |
| 1 Research setup | Zones API | Train/val only; portfolio zone read-only from portfolio |
| 2 Strategy editor | Strategy validate/save | Editor + metadata + active strategy selector |
| 3 Parameter sweep | Sweep job API | Dynamic params from schema; run; table + pick winner |
| 4 Strategy robustness | Validation job API | Run validation; show pass/fail summary |
| 5 Portfolio addition | Gate API | Run gate; block if ΔSR ≤ 0 |
| 6 Commit | Commit API | Commit button + success |

## Stepper (MVP)

- Show only shipped steps; later steps hidden or locked with tooltip
- **Continue** gated on minimal exit criteria per step doc
- No click-ahead on locked steps

## Defer in MVP UI

| Feature | Step |
|---------|------|
| Full heatmap / 1D sensitivity charts | 3 — table + best-by-metric OK first |
| On-demand permutation buttons | 3 — add when workers ready |
| Full correlation Panel B | 5 — Panel A only first |
| Monitoring pre-registration | Portfolio Builder / pre-holdout, not research pipeline |
| Backtest runner as tab | Optional link from editor only |

## Dynamic parameters (step 3 — core MVP)

When active strategy has `params_schema_json`:

- Auto-render controls (min/max/step or enum per param)
- Grid size estimate before submit
- Do not ship manual JSON grid editor unless schema missing

## Overfitting visibility (step 3 — core MVP)

Show automatically after sweep (when backend provides):

| Metric | Block Continue? |
|--------|-----------------|
| DSR | Soft warning if < 0.50 |
| NW t-stat rank | Used for “Best by metric” |
| Perturbation median | Soft warning if below floor |

Full permutation / CUSUM panels: add incrementally.

## Terminology in UI

Use **Portfolio zone** and **Strategy zones** in copy — map to project test + train/validation in API.
