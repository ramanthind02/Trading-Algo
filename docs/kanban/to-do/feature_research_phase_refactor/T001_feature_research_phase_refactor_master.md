# T001 — Feature Research Phase Refactor (Hard Cutover)

## Goal
Reorganize `feature_research/` into three phase folders (`in_sample/`, `walkforward/`, `oos/`) with `rule_based/` and `continuous_binning/` subfolders under each, and fully migrate imports/configs/scripts to the new structure (no legacy folders retained).

## Context / References
- `docs/library/Feature_selection/pipeline_overview.md`
- `docs/library/Feature_selection/feature_validator.md`
- Existing configs (current): `feature_research/rule_based/config.py`, `feature_research/continuous_binning/config.py`, `feature_research/walkforward/config.py`
- Existing engines (current): shared walkforward modules under `feature_research/walkforward/` (runner/io/selection/viz)

## Scope
In scope:
- Create new phase-first directory structure under `feature_research/`.
- Move/rename existing scripts/modules into the new locations.
- Update *all* imports (code + tests) to the new module paths.
- Keep the shared walkforward engine in `feature_research/walkforward/*.py`, but add phase/type entrypoints under `feature_research/walkforward/{rule_based,continuous_binning}/`.
- Add minimal smoke tests focused on importability / packaging (not correctness).
- Update docs references to new layout.

Out of scope:
- Making the research pipeline fully correct/end-to-end on real data.
- Changing statistical logic/thresholds/semantics of the pipeline.
- Re-designing result schemas or plots.

## Target Structure (End State)
```
feature_research/
  in_sample/
    rule_based/
    continuous_binning/
  walkforward/
    (shared engine modules remain here)
    rule_based/
    continuous_binning/
  oos/
    rule_based/
    continuous_binning/
```

## Migration Map (Old -> New)
Rule-based:
- `feature_research/rule_based/config.py` -> `feature_research/in_sample/rule_based/config.py`
- `feature_research/rule_based/data_loader.py` -> `feature_research/in_sample/rule_based/data_loader.py`
- `feature_research/rule_based/pipeline.py` -> `feature_research/in_sample/rule_based/pipeline.py` (keep Phase 1 logic; if it contains walkforward orchestration, move that portion to `feature_research/walkforward/rule_based/pipeline.py`)
- `feature_research/rule_based/run_eda.py` -> `feature_research/in_sample/rule_based/run_eda.py`
- `feature_research/rule_based/run_walkforward.py` -> `feature_research/walkforward/rule_based/run_walkforward.py`
- `feature_research/rule_based/param_sensitivity.ipynb` -> `feature_research/in_sample/rule_based/param_sensitivity.ipynb`

Continuous-binning:
- `feature_research/continuous_binning/config.py` -> `feature_research/in_sample/continuous_binning/config.py`
- `feature_research/continuous_binning/data_loader.py` -> `feature_research/in_sample/continuous_binning/data_loader.py`
- `feature_research/continuous_binning/binning_analysis.py` -> `feature_research/in_sample/continuous_binning/binning_analysis.py`
- `feature_research/continuous_binning/pipeline.py` -> `feature_research/in_sample/continuous_binning/pipeline.py` (keep Phase 1 logic; if it contains walkforward orchestration, move that portion to `feature_research/walkforward/continuous_binning/pipeline.py`)
- `feature_research/continuous_binning/run_eda.py` -> `feature_research/in_sample/continuous_binning/run_eda.py`
- `feature_research/continuous_binning/run_binning_analysis.py` -> `feature_research/in_sample/continuous_binning/run_binning_analysis.py`
- `feature_research/continuous_binning/run_walkforward.py` -> `feature_research/walkforward/continuous_binning/run_walkforward.py`
- `feature_research/continuous_binning/param_sensitivity.ipynb` -> `feature_research/in_sample/continuous_binning/param_sensitivity.ipynb`

Walkforward engine (shared):
- Keep as-is: `feature_research/walkforward/{config,runner,io,metrics,visualization,portfolio_evaluator,top_k_selection,stable_region_selection}.py`

New (no direct old equivalent):
- `feature_research/oos/rule_based/run_oos.py` (+ optional `run_oos_permutation.py`)
- `feature_research/oos/continuous_binning/run_oos.py` (+ optional `run_oos_permutation.py`)

Notes:
- Hard cutover: `feature_research/rule_based/` and `feature_research/continuous_binning/` must not exist at end of refactor.
- Preserve engine module paths where possible: e.g. `feature_research.walkforward.runner` remains importable.

## Interfaces (Module Paths)
New canonical import roots:
- In-sample rule-based: `feature_research.in_sample.rule_based.*`
- In-sample continuous: `feature_research.in_sample.continuous_binning.*`
- Walkforward engine (shared): `feature_research.walkforward.*` (engine modules)
- Walkforward entrypoints by type: `feature_research.walkforward.rule_based.*`, `feature_research.walkforward.continuous_binning.*`
- OOS entrypoints by type: `feature_research.oos.rule_based.*`, `feature_research.oos.continuous_binning.*`

Hard requirement:
- No re-export “compat” modules from the old paths.

## Dependencies
- Execute tasks in this order:
  - Create skeleton packages first.
  - Migrate/move code next (in-sample + walkforward + oos).
  - Update imports/tests.
  - Delete legacy folders only after new paths exist and are referenced everywhere.
  - Docs update near the end, once paths are final.

## Invariants / Constraints
- Pure refactor/organization: do not intentionally change pipeline behavior.
- Keep `feature_research/walkforward/*.py` engine modules as the shared implementation base.
- Avoid circular imports: type-specific entrypoints may import engine; engine must not import phase/type entrypoints.
- Prefer explicit packages (`__init__.py`) for new directories to avoid namespace-package ambiguity.

## Acceptance Tests
- `python -m compileall feature_research`
- `python -c "import feature_research; import feature_research.walkforward.runner"`
- `pytest -q tests/feature_research -k import` (or equivalent new smoke tests from T011)
- `python -c "import pkgutil, feature_research; assert any(m.name.endswith('in_sample') for m in pkgutil.iter_modules(feature_research.__path__))"`
- `python -c "import subprocess; import sys"` (sanity import, ensures no hidden sys.path hacks)

## Definition of Done
- New phase-first folders exist with both feature types under each.
- All imports updated; no references to `feature_research.rule_based` or `feature_research.continuous_binning` remain.
- Legacy directories removed.
- Import/packaging smoke tests pass.
- Docs updated to reference the new structure.

## Notes (Execution Plan)
- This refactor is intentionally staged for delegation:
  - Rule-based move/update tasks can run in parallel with continuous-binning tasks.
  - Import update + legacy deletion is a separate task to reduce merge conflicts.
