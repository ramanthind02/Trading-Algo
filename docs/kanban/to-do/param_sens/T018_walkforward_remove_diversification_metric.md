# T018 — Walkforward Remove Diversification Metric

## Goal
Remove diversification-aware selection from walkforward enhanced mode and select top-k strictly by smoothed objective (after trade-frequency filter), with docs and outputs updated.

## Context / References
- `feature_research/walkforward/top_k_selection.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/io.py`
- `feature_research/walkforward/config.py`
- `tests/feature_research/walkforward/*`
- `docs/api/feature_selection.md`
- `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`

## Scope
In scope:
- Remove greedy diversity/correlation penalty from selection.
- Remove diversification config knobs.
- Replace `selected_by_diversity` with neutral top-k selection flag.
- Update docs and tests.

Out of scope:
- Changing fold/rank computation semantics.

## Acceptance tests
1. `pytest tests/feature_research/walkforward/test_top_k_selection.py -q`
2. `pytest tests/feature_research/walkforward/test_runner.py -q`
3. `pytest tests/feature_research/walkforward/test_io.py -q`
4. `pytest tests/feature_research/walkforward/test_config.py -q`
