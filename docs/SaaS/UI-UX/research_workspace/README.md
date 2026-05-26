# Research Workspace

**Parent:** [UI/UX README](../README.md)

## Purpose

Strategy development pipeline **inside a portfolio** selected from [Portfolios](../portfolios.md). The **portfolio zone is not configured here** — it was locked at portfolio creation.

**Start:** [user_flow.md](user_flow.md)

## Before Research

Create a portfolio → lock portfolio zone → set universe: **[../portfolios.md](../portfolios.md)**

## Pipeline

| Step | Page |
|------|------|
| 0 | [Select portfolio](project_selector.md) |
| 1 | [Research setup](project_setup.md) — strategy zones only |
| 2–6 | Editor → Sweep → Robustness → Addition → Commit |

**MVP:** [mvp_ui.md](mvp_ui.md)

## Zone timeline

```text
[ train ][ validation ]  |  🔒 portfolio zone
     step 1                  fixed (Portfolios)
```

## Display system

All research screens follow the patterns in [display_patterns.md](display_patterns.md):
- Verdict Card — the atomic unit for every test section
- Progressive disclosure — verdict → numbers → charts → raw data
- Chart system — color ramps, axis style, chart types
- Number formatting — mono, tabular-nums, always show sign on deltas

## Related

| Doc | Role |
|-----|------|
| [display_patterns.md](display_patterns.md) | **Design system for all research data display** |
| [zone_manager.md](zone_manager.md) | API fields |
| [portfolio_correlation.md](portfolio_correlation.md) | Step 5 panels |
