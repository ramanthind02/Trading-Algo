# T008 — Update Imports Across Repo + Tests

## Goal
Update all code and tests to use the new phase-first module paths, and remove all references to the legacy research package locations.

## Context / References
- Legacy import roots to eliminate:
  - `feature_research.rule_based.*`
  - `feature_research.continuous_binning.*`
- New canonical roots:
  - `feature_research.in_sample.rule_based.*`
  - `feature_research.in_sample.continuous_binning.*`
  - `feature_research.walkforward.*` (engine)
  - `feature_research.walkforward.rule_based.*`
  - `feature_research.walkforward.continuous_binning.*`
  - `feature_research.oos.*`

## Scope
In scope:
- Update imports in:
  - `feature_research/` code
  - `tests/feature_research/`
  - any `tests/integration/*` referencing research modules
  - any other modules (nodes/validators/etc) importing research configs/loaders

Out of scope:
- Changing test expectations beyond what is required by path changes (unless tests are path-coupled).

## Invariants / Constraints
- Prefer mechanical import updates; avoid logic changes.
- Keep the shared walkforward engine modules’ import paths stable where possible.

## Acceptance Tests
- `python -m compileall .`
- `pytest -q tests/feature_research`
- `pytest -q tests/integration -k feature_validator` (best-effort; may be flaky due to data)

## Definition of Done
- No remaining imports from legacy research paths.
- Unit tests compile/import with new paths.
