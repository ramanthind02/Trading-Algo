# T004 — Permutation Config Phase Scoping

## Goal
Fix permutation configuration coupling so in-sample Stage 1/2 toggles are configurable without accidentally enabling/disabling walkforward Stage 3 or OOS permutation in other pipelines.

## Context / References
- `feature_research/config.py` (`PermutationSuiteConfig` shared research settings)
- `feature_research/in_sample/pipeline.py` (in-sample permutation config wiring)
- `feature_selection/validation/config.py` (`PermutationTestConfig`, `InSamplePermutationConfig`)
- `feature_selection/validation/orchestration.py` (stage execution gating)
- `docs/api/feature_selection/validation.md`
- Unit tests: `tests/unit-tests/validators/permutation/test_orchestration_unit.py`

## Observed behavior
- `PermutationSuiteConfig` mixes in-sample suite knobs with `run_stage3_walkforward` and `run_oos_permutation` booleans.
- In-sample pipeline forwards those shared booleans into `PermutationTestConfig`, so changing walkforward/OOS behavior in shared config changes in-sample execution.
- Setting shared booleans to protect in-sample can disable walkforward/OOS permutation when those phases build from the same shared config.

## Expected behavior
- `PermutationSuiteConfig` only controls in-sample permutation suite behavior (Stage 1/2 settings and toggles).
- In-sample pipeline always forces Stage 3 walkforward and OOS permutation off.
- `PermutationTestConfig` can explicitly skip Stage 1 and/or Stage 2 while preserving current Stage 3/OOS controls.

## Reproduction
1. Configure shared `PermutationSuiteConfig` with walkforward/OOS booleans for a non-in-sample workflow.
2. Run `feature_research/in_sample/run_is.py` and observe Stage 3/OOS behavior leaking into the in-sample pipeline.

## Regression window
- Unknown (design issue in current shared permutation suite config shape).

## Scope
In scope:
- Phase scoping of `PermutationSuiteConfig` in `feature_research/config.py`
- In-sample pipeline permutation config wiring in `feature_research/in_sample/pipeline.py`
- Stage 1/2 run toggles in validation permutation config/orchestration
- Unit tests and API docs for updated config interface

Out of scope:
- New walkforward/OOS permutation pipeline entrypoints
- Changes to permutation math/statistics

## Interfaces (must match)
- Modify: `feature_research/config.py` — `PermutationSuiteConfig` removes shared Stage 3/OOS flags and adds in-sample `run_stage1` / `run_stage2`
- Modify: `feature_selection/validation/config.py` — `InSamplePermutationConfig` / `PermutationTestConfig` support `run_stage1` and `run_stage2`
- Modify: `feature_selection/validation/orchestration.py` — Stage 1/2 execution gated by new flags
- Modify: `feature_research/in_sample/pipeline.py` — Stage 3 and OOS forced off for in-sample pipeline

## Constraints / Risk
- Preserve deterministic behavior for a fixed `random_seed` when a stage is enabled.
- When a stage is disabled, downstream stage inputs must be well-defined (Stage 1 skip => all params pass to Stage 2; Stage 2 skip => Stage 1 passers carry forward).
- Keep existing defaults backward-compatible (`run_stage1=True`, `run_stage2=True`).

## Acceptance tests
1. `pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_permutation_test_config_defaults -q`
2. `pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_stage2_skips_when_disabled_and_preserves_stage1_passers -q`
3. `pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_stage1_skip_runs_stage2_for_all_params -q`

## Definition of done
- [x] Tests updated/added under `tests/unit-tests/validators/permutation/`
- [x] Docs (interface changes) updated in `docs/api/feature_selection/validation.md`
- [x] Targeted pytest commands pass

## Notes
- Keep this change narrowly scoped to configuration boundaries and orchestration gating.
