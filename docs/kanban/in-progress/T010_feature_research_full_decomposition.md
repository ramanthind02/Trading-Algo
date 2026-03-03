# T010 — Feature Research Full Decomposition

## Goal
Decompose `feature_research/*` into clear, phase-focused modules while preserving behavior and public entrypoints, reducing duplication, and improving maintainability.

## Context / References
- `feature_research/pipeline.py`
- `feature_research/walkforward/run_walkforward_permutation.py`
- `feature_research/oos/run_oos_permutation.py`
- `feature_research/in_sample/data_loader.py`
- `docs/kanban/templates/feature.md`

## Scope
In scope:
- Shared bootstrap and helper extraction
- Pipeline split into `feature_research/pipelines/*` modules
- Shared research-data preparation for walkforward/OOS/permutation entrypoints
- Shared permutation-runtime helper extraction
- Internal import cleanup and targeted tests/docs updates

Out of scope:
- Algorithm/metric changes
- Artifact schema or config schema changes
- `feature_selection/*` refactors

## Interfaces (must match)
- Preserve signatures/behavior for:
  - `feature_research.pipeline.run_eda_pipeline`
  - `feature_research.pipeline.run_walkforward_pipeline`
  - `feature_research.pipeline.run_oos_pipeline`
  - `feature_research.pipeline.run_permutation_pipeline`
  - `feature_research.pipeline.write_permutation_summary`

## Data Contracts
- Existing walkforward/OOS/permutation artifact keys and filenames remain unchanged.
- Existing selected-bin 3D expansion behavior remains unchanged.
- Existing index normalization and duplicate-index handling semantics remain unchanged.

## Dependencies
- `pandas`
- `numpy`
- Existing `feature_research.walkforward.*` engine modules

## Invariants / Constraints
- Deterministic behavior for fixed inputs/seeds.
- No lookahead semantics unchanged.
- No new circular imports between phase modules and walkforward engine.

## Acceptance tests
1. `source venv/bin/activate && pytest -q tests/feature_research/test_import_smoke.py`
2. `source venv/bin/activate && pytest -q tests/feature_research/test_pipeline_3d_selected_bin.py`
3. `source venv/bin/activate && pytest -q tests/feature_research/test_continuous_pipeline_walkforward.py tests/feature_research/test_rule_based_pipeline_walkforward.py`
4. `source venv/bin/activate && pytest -q tests/feature_research/walkforward/test_run_walkforward_permutation.py tests/feature_research/test_run_oos_permutation.py`
5. `source venv/bin/activate && pytest -q tests/feature_research/walkforward/test_runner.py tests/feature_research/walkforward/test_portfolio_evaluator.py`
6. `source venv/bin/activate && pytest -q tests/feature_research`

## Definition of done
- [ ] New modular layout implemented under `feature_research/pipelines/` and shared helpers.
- [ ] Public pipeline entrypoints remain stable and behavior-compatible.
- [ ] Targeted tests pass; full `tests/feature_research` pass.
- [ ] API docs updated for moved module boundaries.

## Notes
- Non-goals are strict: no formula/statistical behavior updates, no artifact/config schema changes.
