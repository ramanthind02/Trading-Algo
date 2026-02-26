# T007 — Walkforward Config Single Source Of Truth

## Goal
Remove duplicated walkforward setting storage across feature-research config layers so walkforward defaults are defined once and phase configs consume the nested walkforward object.

## Context / References
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `feature_research/walkforward/config.py`
- `feature_research/walkforward/run_walkforward.py`

## Scope
In scope:
- Base feature-research walkforward defaults storage/refactor
- In-sample config duplication removal for walkforward selection/weight knobs
- Docstring references to the new edit location

Out of scope:
- Walkforward algorithm behavior changes
- Permutation config behavior changes

## Interfaces (must match)
- Modify: `feature_research/config.py` — preserve `load_config()` and `BaseResearchConfig.build_walkforward(...)`
- Modify: `feature_research/in_sample/config.py` — preserve `load_config()` return type and `ResearchConfig.walkforward`

## Data Contracts
- `WalkforwardResearchConfig` remains the runtime walkforward config object consumed by pipeline code
- Project-level defaults are stored once in `BaseResearchConfig.walkforward_defaults`

## Dependencies
- `dataclasses`
- `feature_research.walkforward.config`

## Invariants / Constraints
- Keep existing walkforward runtime semantics unchanged for current defaults
- Keep backward compatibility for common top-level BaseResearchConfig walkforward attribute reads (via aliases)

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. python -m py_compile feature_research/config.py feature_research/in_sample/config.py feature_research/walkforward/config.py feature_research/walkforward/run_walkforward.py`
2. `source venv/bin/activate && PYTHONPATH=. python - <<'PY' ... load_config/build_walkforward smoke ... PY`

## Definition of done
- [x] Single nested walkforward defaults object added to base config
- [x] In-sample config no longer stores duplicate walkforward selection/weight fields
- [x] Targeted compile and load/build smoke checks pass

## Notes
- `feature_research/walkforward/config.py` remains the schema/validation source; this task removes duplicated value storage, not the type definition.
- Extended follow-up: in-sample phase presets (bias spec / target / reports dir / walkforward overrides) are now centralized in `feature_research/config.py` and consumed by `feature_research/in_sample/config.py`.
