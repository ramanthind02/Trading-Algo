# T002 — Research: Document Current Portfolio, WeightLayer, and Pipeline Interfaces

## Goal
Read all relevant files and produce a concise interface summary that the design tasks (T003–T005) can build on, without requiring those tasks to re-read large files. This task is pure research — no code changes.

## Context / References
- `ensemble/portfolio.py` — `Portfolio.__init__`, `fit`, `predict`, `_compute_idm`, `_vol_scale` (or equivalent methods); check what vol scaling lives here vs ensemble
- `ensemble/weight_layer.py` — `BaseWeightLayer.fit`, `combine`; `WeightLayerConfig`; `DownsideHRPGroupedWeightLayer._fit_ticker_weights`, `_calculate_fdm`
- `ensemble/portfolio_tester.py` — `PortfolioTester`, `resample_positions_to_daily`, `aggregate_intraday_returns_to_daily`
- `portfolio_research/pipelines/portfolio_test.py` — how `Portfolio` and `WeightLayer` are constructed and called; what the pipeline assembles per timeframe
- `ensemble/__init__.py` — public exports

## Scope
In scope:
- Read and summarise the `__init__`, `fit`, and `predict`/`combine` signatures of `Portfolio` and `BaseWeightLayer`.
- Identify exactly where vol scaling happens (is it in `Portfolio`, `DiversifiedEnsemble`, or both?).
- Identify what `Portfolio.fit` receives and returns; what `Portfolio.predict` receives and returns.
- Identify what the portfolio_test pipeline does per-timeframe and how it currently merges multi-TF results.
- Note any state attributes that would need to move to `GlobalPortfolio` (IDM, instrument weights).
- Identify which tests cover `Portfolio` and `WeightLayer`.

Out of scope:
- Any code changes.
- Reading `DiversifiedEnsemble` internals beyond the output schema it produces.

## Deliverable
A markdown summary written as a comment or appended section to this task file, covering:
1. `Portfolio.__init__` params and their types
2. `Portfolio.fit` inputs/outputs/side-effects
3. `Portfolio.predict` inputs/outputs schema
4. Where vol scaling lives in the pipeline
5. What per-TF state `Portfolio` currently holds that is global (IDM, instrument weights)
6. How portfolio_test.py merges multi-TF outputs today (look for equal-weight combine step)
7. List of test files that import `Portfolio` or `WeightLayer`

## Dependencies
None — first task, can start immediately.

## Acceptance Tests
- No code changes → no tests required.
- Deliverable: this task file updated with the findings summary before T003/T004 begin.

## Notes
- Delegate entirely to a research subagent (Explore or general-purpose). Do not implement anything.
- Pay attention to `_compute_idm` and any instrument-weight normalisation — these must move to `GlobalPortfolio`.
- Note whether `Portfolio` has a `trading_timeframe` attribute — this is already a hint that per-TF isolation exists.
