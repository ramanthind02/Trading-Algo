# T006 — In-Sample Cum-Sum Plots For Continuous Bin Counts

## Goal
Add a simple in-sample cumulative-sum plot artifact per parameter combo and `bin_count` so researchers can visually inspect `prediction * target` paths in the saved results folders.

## Context / References
- `feature_research/in_sample/pipeline.py`
- `feature_research/in_sample/run_is.py`
- `feature_selection/eda/eda_reporter.py`

## Scope
In scope:
- Continuous in-sample EDA artifact generation
- PNG plot output under each saved EDA report directory

Out of scope:
- Walkforward/OOS plotting changes
- New summary tables or ranking logic

## Interfaces (must match)
- Modify: `feature_research/in_sample/pipeline.py` — keep `run_eda_pipeline(...)` signature unchanged and append artifact generation after EDA report save

## Data Contracts
- Input series use aligned in-sample `feature` and `target` from the EDA loop
- Returns definition is `prediction * target` (target already volatility-scaled)
- Output files are PNGs named by `bin_count` under the report `plots/` directory

## Dependencies
- `matplotlib`
- `pandas`

## Invariants / Constraints
- No change to existing EDA report JSON/plot schema
- Deterministic outputs for fixed inputs/config
- Continuous path only (uses configured `bin_counts`)

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. python -m py_compile feature_research/in_sample/pipeline.py`

## Definition of done
- [x] Continuous in-sample EDA writes cum-sum plots per `bin_count`
- [x] Plot filenames are stable and discoverable in `plots/`
- [x] Targeted syntax check passes

## Notes
- Plot is intentionally simple (cum-sum of `prediction * target`) for quick visual QA.
