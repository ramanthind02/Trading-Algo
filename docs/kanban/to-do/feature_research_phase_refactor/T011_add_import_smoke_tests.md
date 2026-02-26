# T011 — Add Import/Packaging Smoke Tests for Refactor

## Goal
Add lightweight tests that verify the new `feature_research` package layout is importable and the canonical entrypoints/modules exist, without requiring data.

## Context / References
- The refactor intentionally does not guarantee the research pipeline runs end-to-end.
- Tests should focus on: importability, config instantiation, and basic module structure.

## Scope
In scope:
- Add a new test module under `tests/feature_research/` (or another appropriate unit-test location) that imports:
  - `feature_research.in_sample.rule_based`
  - `feature_research.in_sample.continuous_binning`
  - `feature_research.walkforward.runner` (engine)
  - `feature_research.walkforward.rule_based.run_walkforward`
  - `feature_research.walkforward.continuous_binning.run_walkforward`
  - `feature_research.oos.rule_based`
  - `feature_research.oos.continuous_binning`

Out of scope:
- Assertions about metrics/plots/output correctness.

## Acceptance Tests
- `pytest -q tests/feature_research -k import`

## Definition of Done
- New smoke tests exist and pass locally.
