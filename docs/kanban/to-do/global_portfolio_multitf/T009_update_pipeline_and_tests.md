# T009 — Update Research Pipeline and Tests for GlobalPortfolio

## Goal
Update `portfolio_research/pipelines/portfolio_test.py` and related test files to use `GlobalPortfolio` (optionally with multiple TFs), and ensure the existing single-TF research pipeline continues to work as a degenerate case. Add a multi-TF integration test.

## Context / References
- `portfolio_research/pipelines/portfolio_test.py` — primary pipeline to update
- `portfolio_research/weight_layer_report.py` — may reference `Portfolio`; check and update
- `ensemble/portfolio_tester.py` — check if any `Portfolio`-specific assumptions break
- `tests/portfolio_research/` — existing test files (identified in T002)
- T005 spec (GlobalPortfolio interface) and T008 (implementation)

## Scope
In scope:
- Update `portfolio_test.py` to construct `GlobalPortfolio` instead of bare `Portfolio`, passing a list of `TFPortfolio` instances (one per timeframe in the research config).
- Verify single-TF research runs (daily only) still produce the same output as before (regression guard).
- Add a multi-TF test run (e.g., daily + weekly) in the research pipeline config, confirming `GlobalPortfolio` produces non-equal TF weights when TFs have different downside profiles.
- Update any import of `Portfolio` in research pipeline files to import `GlobalPortfolio` (or use the alias — either is fine).
- Update `portfolio_research/weight_layer_report.py` to include cross-TF weight diagnostics from `GlobalPortfolio.get_diagnostics()`.
- Check `ensemble/portfolio_tester.py` for any hardcoded assumptions about single-TF that break with `GlobalPortfolio`.

Out of scope:
- Changing statistical logic in any pipeline.
- Updating deployment pipeline (separate task if needed).
- Updating `PositionSizer` (no changes required — output schema preserved).

## Interfaces
- Modify: `portfolio_research/pipelines/portfolio_test.py`
- Modify: `portfolio_research/weight_layer_report.py` (add cross-TF section if `GlobalPortfolio` detected)
- Possibly modify: `ensemble/portfolio_tester.py` (only if it breaks)

## Dependencies
- T008 (`GlobalPortfolio` implementation) must be complete.
- T006 and T007 must be complete.

## Invariants / Constraints
- Single-TF pipeline output must be numerically identical before and after this change (except for floating point — add tolerance).
- No new statistical methods introduced here — pure pipeline wiring.

## Acceptance Tests
1. `pytest tests/portfolio_research/ -q` — all existing tests pass.
2. Manual: run `portfolio_test.py` with single-TF config → tearsheet output matches pre-change output (eyeball or diff).
3. Manual: run `portfolio_test.py` with two-TF config (daily + weekly) → confirm cross-TF weights are NOT equal (i.e., GlobalWeightLayer is doing real work, not defaulting to equal).
4. `pytest tests/portfolio_research/test_global_portfolio_pipeline.py` — new integration test with two TFs passes.

## Definition of Done
- [ ] `portfolio_test.py` updated to use `GlobalPortfolio`
- [ ] Single-TF regression confirmed
- [ ] Multi-TF pipeline run confirmed
- [ ] `weight_layer_report.py` includes cross-TF weight section
- [ ] All existing tests pass

## Notes
- This task is the integration validation step — if anything is wrong with T006–T008, it will surface here.
- The multi-TF config for the test should use a short date range (2020–2022) to keep runtime reasonable.
- If `PortfolioTester` has assumptions that break, fix minimally — do not redesign it here.
