# T014 — Rule-Based OOS Long/Short Target Alignment

## Goal
Fix OOS/validation rule-based portfolio evaluation so it uses the real aligned target series instead of an all-zero reference target, preventing one-sided fallback behavior in long-short mode.

## Context / References
- User bug report: seasonal bonds month OOS tearsheet appears long-only despite long-short configuration.
- Related code:
  - `feature_research/pipelines/_shared.py`
  - `utils/evaluation/walkforward/research_data.py`
  - `tests/feature_research/test_oos_pipeline_tearsheet.py`

## Observed behavior
- Rule-based OOS pipeline called `load_rule_based_research_data(..., capture_target_as_reference=False)`.
- `build_reference_target(...)` then produced a zero target series.
- Portfolio fold fitting consumed that zero target, collapsing directional mapping in rule-based bins.

## Expected behavior
- Rule-based OOS/validation should forward nonzero aligned target values into walkforward/portfolio simulation so long-short direction decisions use real return information.

## Reproduction
1. `source venv/bin/activate`
2. `python feature_research/oos/run_oos.py`
3. Observe OOS tearsheet behavior for `seasonal_bonds_month` vs cached rule-based signal profile.

## Regression window
- Unknown (needs investigation).

## Scope
In scope:
- Rule-based target forwarding in shared evaluation pipeline.
- Add regression assertion in OOS pipeline test.

Out of scope:
- Any redesign of rule-based binning semantics.
- Historical artifact regeneration beyond local verification.

## Interfaces (must match)
- Modify: `feature_research/pipelines/_shared.py`
- Modify: `tests/feature_research/test_oos_pipeline_tearsheet.py`

## Constraints / Risk
- Must preserve existing OOS/validation interfaces and artifact paths.
- Risk: if target forwarding changes unexpectedly for non-rule-based phases, selection and tearsheets could drift.

## Acceptance tests
1. `pytest tests/feature_research/test_oos_pipeline_tearsheet.py::test_rule_based_oos_passes_portfolio_inputs_to_runner -q`
2. `pytest tests/feature_research/test_oos_pipeline_tearsheet.py -q`

## Definition of done
- [x] Tests updated/added under `tests/<path>`
- [ ] Docs (if interfaces changed) under `docs/api/<module>.md`
- [x] Targeted pytest passes

## Notes
- This fix aligns rule-based OOS behavior with existing rule-based score construction that already uses aligned feature/target returns.
