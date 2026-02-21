# T028 — Walkforward top-k selected row flags bugfix

## Goal
Fix the enhanced/stable-region regression where row-level `selected_feature` is always forced to `False`, which breaks IO/visualization selected-row masks.

## Context / References
- `feature_research/walkforward/runner.py`
- `tests/feature_research/walkforward/test_runner.py`
- `tests/feature_research/walkforward/test_io.py`
- `tests/feature_research/walkforward/test_visualization.py`

## Observed behavior
- In enhanced/stable_region selection mode, `fold_scores_df.selected_feature` is hard-coded to `False` for all rows.
- Downstream consumers expecting selected-row masks receive empty selections even when `selected_in_top_k` contains true members.

## Expected behavior
- Keep selection-only semantics (no single winner summary fields).
- For enhanced/stable_region, set row-level `selected_feature` equal to per-member `selected_in_top_k`.
- Keep summary single-winner fields (`selected_feature`, selected objective columns) as `NaN`.

## Reproduction
1. `PYTHONPATH=. pytest tests/feature_research/walkforward/test_runner.py::test_enhanced_selection_outputs_selected_members_only -q`
2. `PYTHONPATH=. pytest tests/feature_research/walkforward/test_runner.py::test_stable_region_selection_outputs_selected_members_only -q`

## Regression window
- Unknown (introduced prior to this branch).

## Scope
In scope:
- `feature_research/walkforward/runner.py` selected-row assignment for enhanced/stable_region.
- Unit tests pinning row flags and summary NaN behavior.

Out of scope:
- Any changes to top-k/stable-region algorithm selection logic.
- Artifact schema changes beyond existing columns.

## Interfaces (must match)
- Modify: `feature_research/walkforward/runner.py::_build_fold_scores(...)`
- Modify: `tests/feature_research/walkforward/test_runner.py` (regression assertions)

## Constraints / Risk
- Deterministic fold ranking and top-k labels must remain unchanged.
- No-lookahead and fold boundary semantics unchanged.
- Risk if incorrect: visualization and IO selected-member exports remain empty.

## Acceptance tests
1. `PYTHONPATH=. pytest tests/feature_research/walkforward/test_runner.py::test_enhanced_selection_outputs_selected_members_only -q`
2. `PYTHONPATH=. pytest tests/feature_research/walkforward/test_runner.py::test_stable_region_selection_outputs_selected_members_only -q`
3. `PYTHONPATH=. pytest tests/feature_research/walkforward/test_io.py tests/feature_research/walkforward/test_visualization.py -q`

## Definition of done
- [ ] Tests updated/added under `tests/feature_research/walkforward/`
- [ ] Docs api unchanged (no public interface change)
- [ ] `pytest` commands in Acceptance tests pass

## Notes
- Keep `top_k_features` JSON output unchanged.
