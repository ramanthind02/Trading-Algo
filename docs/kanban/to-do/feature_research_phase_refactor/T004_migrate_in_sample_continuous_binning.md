# T004 — Migrate In-Sample Continuous-Binning Research

## Goal
Move continuous-binning in-sample research modules (EDA, binning analysis, parameter sensitivity helpers, IS permutation entrypoints) from `feature_research/continuous_binning/` into `feature_research/in_sample/continuous_binning/` and update intra-package imports.

## Context / References
- Existing (to be migrated):
  - `feature_research/continuous_binning/config.py`
  - `feature_research/continuous_binning/data_loader.py`
  - `feature_research/continuous_binning/pipeline.py`
  - `feature_research/continuous_binning/binning_analysis.py`
  - `feature_research/continuous_binning/run_eda.py`
  - `feature_research/continuous_binning/run_binning_analysis.py`
- `docs/library/Feature_selection/feature_validator.md` (continuous vs rule-based split)

## Scope
In scope:
- Relocate continuous-binning in-sample code into `feature_research/in_sample/continuous_binning/`.
- Keep `run_binning_analysis.py` aligned with Phase 1 expectations.
- Move `param_sensitivity.ipynb` into phase folder (or replace with a pointer doc) to keep research assets co-located.

Out of scope:
- Walkforward scripts/entrypoints (handled by T006).

## Interfaces
New canonical imports (examples):
- `from feature_research.in_sample.continuous_binning.config import load_config`

## Invariants / Constraints
- Do not alter statistical computations (bin thresholds, metrics); refactor only.

## Acceptance Tests
- `python -c "from feature_research.in_sample.continuous_binning.config import load_config; load_config()"`
- `python -c "import feature_research.in_sample.continuous_binning.run_binning_analysis"`
- `python -m compileall feature_research/in_sample/continuous_binning`

## Definition of Done
- Continuous-binning in-sample modules live under `feature_research/in_sample/continuous_binning/`.
- Notebook/analysis assets are placed under the in-sample phase folder (or a clear link is added).
