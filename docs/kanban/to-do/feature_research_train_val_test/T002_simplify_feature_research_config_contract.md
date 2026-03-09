# T002 - Simplify Feature Research Config Contract

## Goal
Remove rolling walkforward orchestration config complexity from `feature_research` and switch runtime config to explicit train/validation/test controls.

## Context / References
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `utils/evaluation/walkforward/config.py`
- `tests/feature_research/test_config.py`

## Scope
In scope:
- In `feature_research/config.py`:
  - Remove `GlobalResearchDefaults`.
  - Remove `WalkforwardDefaultsConfig`.
  - Remove `BaseResearchConfig.build_walkforward()`.
  - Remove walkforward config imports from the feature-research config layer.
  - Keep `validation_window` and `oos_window` as canonical windows.
- In `feature_research/in_sample/config.py`:
  - Remove `walkforward: WalkforwardResearchConfig`.
  - Remove top-level `walkforward_selection_method`.
  - Remove top-level `weight_layer_algorithm`.
  - Remove consistency checks tied to those fields in `__post_init__`.
  - Add flat evaluation fields:
    - `top_k: int`
    - `objective_metric_name: str`
    - `smoothing_self_weight: float`
    - `n_jobs: int`
    - `output_root: Path`
- Ensure downstream code can build runtime engine config from flat fields.

Out of scope:
- Pipeline orchestration changes (handled by T003).
- Permutation script rewrite (handled by T004).

## Interfaces (must match)
- `load_config()` remains the public config entrypoint.
- `validation_window` and `oos_window` remain available and typed.
- New flat evaluation fields are present on `ResearchConfig`.

## Data Contracts
- Config no longer exposes nested walkforward runtime object in `ResearchConfig`.
- `validation_window` and `oos_window` define split boundaries consumed by pipelines.

## Dependencies
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `tests/feature_research/test_config.py`

## Invariants / Constraints
- No silent date-window reinterpretation.
- Multi-ticker target validation remains intact.
- Deterministic defaults remain explicit in one place.

## Acceptance tests
1. `source venv/bin/activate && python -c "from feature_research.config import load_config; load_config(); print('base ok')"`
2. `source venv/bin/activate && python -c "from feature_research.in_sample.config import load_config; c=load_config(); print(c.top_k, c.objective_metric_name)"`
3. `source venv/bin/activate && pytest -q tests/feature_research/test_config.py`

## Definition of done
- [ ] Removed rolling walkforward config classes/methods from feature-research config layer.
- [ ] `ResearchConfig` exposes flat eval fields.
- [ ] Acceptance commands pass.

## Notes
- Keep this task strictly contract-focused; no script or pipeline renaming here.
