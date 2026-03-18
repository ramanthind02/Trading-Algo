# T005 - Tests, Docs, And Hard-Cut Cleanup

## Goal
Complete the hard cutover by migrating tests/docs to validation terminology and removing obsolete `feature_research/walkforward` package surface.

## Context / References
- `tests/feature_research/`
- `docs/api/data_pipeline.md`
- `docs/api/feature_selection.md`
- `feature_research/pipeline.py`
- `feature_research/walkforward/`

## Scope
In scope:
- Tests migration:
  - Move/rename `tests/feature_research/walkforward/*` to validation equivalents (or equivalent new structure).
  - Rename pipeline tests:
    - `test_continuous_pipeline_walkforward.py` -> validation naming.
    - `test_rule_based_pipeline_walkforward.py` -> validation naming.
  - Update import smoke tests to validate new public API names only.
- Docs updates:
  - Update walkforward references in:
    - `docs/api/data_pipeline.md`
    - `docs/api/feature_selection.md`
  - Add references to this new kanban task group directory.
- Cleanup:
  - Remove obsolete `feature_research/walkforward/` package after imports are migrated.
  - Ensure no feature-research runtime imports still target old walkforward paths.

Out of scope:
- New feature work unrelated to cutover.
- Non-feature-research documentation outside touched API pages.

## Interfaces (must match)
- Public facade expectation:
  - `run_validation_pipeline` exists.
  - `run_walkforward_pipeline` does not exist.
- Validation test module import paths are stable and discoverable by `pytest`.

## Data Contracts
- Validation artifact contracts reflected in docs:
  - subdir: `validation/`
  - validation-named outputs and summaries.
- OOS artifact docs remain explicit and consistent with new train/val/test flow.

## Dependencies
- `tests/feature_research/*`
- `docs/api/*`
- `feature_research/pipeline.py`

## Invariants / Constraints
- Hard cutover means no walkforward compatibility aliases in feature_research public API.
- Keep tests deterministic and aligned with current cache/test practices.

## Acceptance tests
1. `source venv/bin/activate && pytest -q tests/feature_research`
2. `source venv/bin/activate && python -c "import feature_research.pipeline as p; print(hasattr(p,'run_validation_pipeline'), hasattr(p,'run_walkforward_pipeline'))"` with expected output `True False`

## Definition of done
- [ ] Walkforward-named feature-research tests/docs migrated to validation naming.
- [ ] Obsolete `feature_research/walkforward/` package removed.
- [ ] Acceptance commands pass.

## Notes
- Run last after T001-T004 are merged to avoid rename churn.
