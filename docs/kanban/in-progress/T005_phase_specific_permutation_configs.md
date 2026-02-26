# T005 — Phase-Specific Permutation Configs In feature_research

## Goal
Replace the single shared feature-research permutation config with separate phase-specific configs (in-sample, walkforward, OOS) so phase toggles/settings do not couple unrelated pipelines.

## Context / References
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `feature_research/in_sample/pipeline.py`
- `feature_research/in_sample/run_is.py`
- `feature_research/in_sample/profile_stage2.py`

## Observed behavior
- Shared `PermutationSuiteConfig` lives in `feature_research/config.py` and is reused across phases.
- Even after stage toggle fixes, the config ownership remains conceptually shared and can regress toward cross-phase coupling.

## Expected behavior
- `feature_research/config.py` exposes separate config objects for:
  - in-sample permutation settings
  - walkforward permutation settings
  - OOS permutation settings
- In-sample config loader consumes only the in-sample permutation config.
- In-sample pipeline/scripts continue to function using the in-sample phase config only.

## Scope
In scope:
- Shared feature_research config dataclasses and loader defaults
- In-sample config loader field mapping
- In-sample references/docs strings where needed

Out of scope:
- Implementing walkforward/OOS permutation runners
- Changes to permutation engine behavior

## Interfaces (must match)
- Modify: `feature_research/config.py` — add separate phase-specific permutation config dataclasses/fields
- Modify: `feature_research/in_sample/config.py` — map `base.in_sample_permutation`
- Modify: `feature_research/in_sample/pipeline.py` and scripts if field names change

## Constraints / Risk
- Keep current in-sample behavior unchanged.
- Avoid circular imports in `feature_research` config modules.
- Preserve strict typing and frozen dataclass usage.

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. python -m py_compile feature_research/config.py feature_research/in_sample/config.py feature_research/in_sample/pipeline.py`
2. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_permutation_test_config_defaults -q`

## Definition of done
- [x] feature_research config split implemented
- [x] In-sample loader/pipeline wired to in-sample phase config
- [x] Targeted checks pass

## Notes
- This is a config-ownership refactor; keep runtime semantics stable.
