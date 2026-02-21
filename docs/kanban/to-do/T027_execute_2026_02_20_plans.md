# T027 - Execute 2026-02-20 plans

## Goal
Implement the four plan documents in `docs/plans/` to close walkforward correctness gaps, add portfolio-based stage-2 evaluation, enforce multi-ticker normalization invariants, and fix EWSD early-window behavior.

## Context / References
- `docs/plans/2026-02-20-walkforward-correctness-fix.md`
- `docs/plans/2026-02-20-portfolio-walkforward-integration.md`
- `docs/plans/2026-02-20-multiticker-normalization-invariant.md`
- `docs/plans/2026-02-20-ewsd-expanding-window.md`
- `docs/api/` walkforward and continuous-binning pages (update if interfaces change)

## Scope
In scope:
- Implement fold-aware evaluator refit behavior for walkforward scoring.
- Integrate `Portfolio`-based stage-2 fold evaluation into walkforward reports and artifacts.
- Add config/data-loader safety checks for raw multi-ticker targets.
- Update EWSD node long-run window and early-estimation behavior with regression tests.

Out of scope:
- New research objectives beyond existing plan files.
- Strategy changes in `rule_based` pipeline.

## Interfaces (must match)
- Modify: `feature_research/continuous_binning/pipeline.py`
- Modify: `feature_research/walkforward/runner.py`
- Modify: `feature_research/walkforward/io.py`
- Add: `feature_research/walkforward/portfolio_evaluator.py`
- Modify: `feature_research/continuous_binning/config.py`
- Modify: `feature_research/continuous_binning/data_loader.py`
- Modify: `nodes/ewsd.py`
- Add/modify tests under `tests/feature_research/walkforward/`, `tests/feature_research/continuous_binning/`, and `tests/nodes/`

## Invariants / Constraints
- Deterministic fold ranking and reproducible outputs.
- No lookahead in walkforward model fitting.
- Multi-ticker targets must be volatility-normalized.
- EWSD output remains finite and positive for valid price sequences.

## Acceptance tests
1. `pytest tests/feature_research/walkforward/test_runner.py -q`
2. `pytest tests/feature_research/walkforward/test_io.py -q`
3. `pytest tests/feature_research/walkforward/test_portfolio_integration.py -q`
4. `pytest tests/feature_research/continuous_binning/test_config.py -q`
5. `pytest tests/nodes/test_ewsd.py -q`

## Definition of done
- [ ] Code changes match all four plan documents, with walkforward-fix before portfolio integration.
- [ ] Tests added/updated in correct unit vs integration locations.
- [ ] Relevant `docs/api/*` pages updated for changed public interfaces.
- [ ] Acceptance tests pass in shared project venv.

## Notes
- User noted the walkforward bug may already be fixed; verify first and avoid duplicate work.
- Prefer reuse of `Portfolio` + `DiversifiedEnsemble` in research flow to keep implementation DRY.
