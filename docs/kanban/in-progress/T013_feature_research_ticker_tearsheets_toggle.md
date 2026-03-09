# T013 — Feature Research Ticker Tearsheet Toggle

## Goal
Add an opt-in research config toggle to generate per-ticker walkforward tearsheets (per-fold and aggregate OOS) so researchers can inspect ticker-level behavior without changing default runtime/artifact volume.

## Context / References
- `feature_research/config.py`
- `utils/evaluation/walkforward/runner.py`
- `utils/evaluation/walkforward/portfolio_evaluator.py`
- `tests/feature_research/validation/engine/test_runner.py`
- `docs/api/utils.md`

## Scope
In scope:
- Add `ResearchConfig.generate_ticker_tearsheets` (default `False`) and wire through runtime.
- Extend fold evaluation output with per-ticker OOS return series.
- Generate optional per-fold and aggregate OOS ticker tearsheets under walkforward tearsheet output dir.
- Add/update unit tests for enabled/disabled behavior and timeframe forwarding.
- Update API docs for the new config and runner behavior.

Out of scope:
- Any change to tearsheet output format (HTML remains the only format).
- Any changes to portfolio construction, signal generation, or objective metric logic.

## Interfaces (must match)
- Modify: `feature_research/config.py` — add `generate_ticker_tearsheets: bool = False` to `ResearchConfig`; set explicit value in `load_config()`.
- Modify: `utils/evaluation/walkforward/portfolio_evaluator.py` — extend `FoldPortfolioResult` with `per_ticker_oos_returns: dict[str, pd.Series] | None`.
- Modify: `utils/evaluation/walkforward/runner.py` — read `research_config.generate_ticker_tearsheets` and generate per-ticker tearsheets when enabled.

## Data Contracts
- Per-ticker return series must use ticker labels from position/candle data and be keyed as deterministic strings.
- Per-ticker strategy returns must be reindexed to dense ticker baseline calendar with `fill_value=0.0` for tearsheet generation.
- Timestamps stay timezone-naive and sorted, consistent with existing walkforward tearsheet flow.

## Dependencies
- `utils.evaluation.walkforward.runner`
- `utils.evaluation.walkforward.portfolio_evaluator`

## Invariants / Constraints
- Deterministic output names and locations for same inputs.
- No lookahead behavior changes.
- Existing tearsheet generation stays unchanged unless `generate_ticker_tearsheets=True`.

## Acceptance tests
1. `pytest tests/feature_research/validation/engine/test_runner.py::test_run_portfolio_simulation_generates_per_fold_ticker_tearsheets_when_enabled -q` — per-fold ticker tearsheets generated when enabled.
2. `pytest tests/feature_research/validation/engine/test_runner.py::test_run_portfolio_simulation_generates_aggregate_ticker_tearsheets_when_enabled -q` — aggregate ticker tearsheets generated when enabled.
3. `pytest tests/feature_research/validation/engine/test_runner.py::test_run_portfolio_simulation_skips_ticker_tearsheets_when_disabled -q` — no ticker tearsheet generation when disabled.
4. `pytest tests/feature_research/validation/engine/test_runner.py::test_run_portfolio_simulation_forwards_timeframe_to_tearsheets -q` — timeframe forwarding includes ticker tearsheets.
5. `pytest tests/feature_research/validation/engine/test_runner.py::test_evaluate_fold_portfolio_returns_per_ticker_oos_returns_shape -q` — fold result includes per-ticker return contract.

## Definition of done
- [ ] Tests added under `tests/feature_research/validation/engine/`
- [ ] Docs updated under `docs/api/utils.md`
- [ ] Targeted pytest for modified tests passes

## Notes
- Ticker-level scope is fixed to per-fold + aggregate when enabled.
- Default remains OFF to preserve existing runtime and artifact footprint.
