# T008 — Objective Metric Presets In Research Config

## Goal
Add reusable objective-metric presets to feature research config so researchers can quickly switch permutation objectives (e.g., sharpe/sortino/calmar/t_stat) without editing resolver internals.

## Context / References
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `feature_selection/validation/objective_metrics.py`

## Scope
In scope:
- Add config-level metric preset catalog for feature research
- Add `t_stat` builtin support in objective metric resolver so preset is executable

Out of scope:
- Walkforward metric resolver expansion (separate path/module)
- Composite multi-metric objective scoring

## Interfaces (must match)
- Modify: `feature_selection/validation/objective_metrics.py` — preserve `ObjectiveMetricSpec` API, extend builtin set
- Modify: `feature_research/config.py` — add reusable metric preset definitions and usage examples
- Modify: `feature_research/in_sample/config.py` — expose/read metric preset options in in-sample config comments/docs

## Data Contracts
- `ObjectiveMetricSpec` remains immutable and serializable via builtin + kwargs fields

## Dependencies
- `feature_selection.validation.objective_metrics`
- `feature_research.config`

## Invariants / Constraints
- Deterministic metric outputs for degenerate denominators remain unchanged
- Backward compatibility for existing `sortino` defaults

## Acceptance tests
1. `pytest tests/unit-tests/validators/permutation/test_objective_metrics_unit.py -q` — builtin resolver supports `t_stat`.

## Definition of done
- [ ] Presets available in config
- [ ] `t_stat` builtin resolves successfully
- [ ] Existing defaults remain `sortino`

## Notes
- Keep config additions lightweight and researcher-editable.
