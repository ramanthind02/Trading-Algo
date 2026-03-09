# T011 — Multi-Timeframe Portfolio Support

## Goal
Enable portfolio research runs to combine ensembles with different base timeframes while producing QuantStats-compatible daily combined tearsheets and safe per-timeframe outputs.

## Context / References
- `portfolio_research/run_portfolio_test.py`
- `ensemble/portfolio_tester.py`
- `portfolio_research/config.py`
- `docs/api/ensemble.md`

## Scope
In scope:
- Group loaded ensembles by `ensemble.base_tf` and run one `Portfolio` + `PortfolioTester` per timeframe.
- Always compute combined portfolio tearsheet on daily-aligned returns.
- Forward-fill non-daily positions to daily for combined aggregation.
- Aggregate intraday return series to daily before tearsheet generation.
- Generate per-timeframe portfolio tearsheets only for mixed-timeframe runs.
- Prefix per-ensemble/per-base-model tearsheets with timeframe labels in mixed runs.

Out of scope:
- Changing vault schema or adding new config fields.
- Modifying core `Portfolio` fitting/prediction internals.

## Interfaces (must match)
- Modify: `ensemble/portfolio_tester.py`
  - Add `resample_positions_to_daily(positions_df, daily_dates_per_ticker)`
  - Add `aggregate_intraday_returns_to_daily(returns)`
- Modify: `portfolio_research/run_portfolio_test.py`
  - Change `_load_candles(config, timeframe, start, end)`
  - Add `_group_ensembles_by_timeframe(...)`
  - Add `_build_daily_dates_per_ticker(...)`
  - Refactor `run_portfolio_test(...)` orchestration for multi-timeframe flow
- Modify: `portfolio_research/config.py`
  - Docstring clarification for timeframe semantics

## Data Contracts
- Positions DataFrame for resampling: `ticker`, `datetime`, `position_fraction`.
- Combined positions contract: sum by (`ticker`, `datetime`) after daily alignment, then clip to `±max_position_pct`.
- Intraday aggregation contract: for log-return series, aggregate by `index.normalize()` with sum.

## Dependencies
- `ensemble.portfolio`, `ensemble.portfolio_tester`, `ensemble.vault_manager`, `ensemble.weight_layer`
- `metrics.plotting.graphing.quantstats_reports.generate_tearsheet`

## Invariants / Constraints
- Deterministic output for same inputs.
- No lookahead changes to existing return alignment.
- Combined tearsheet frequency is always daily.
- Weekly/monthly per-timeframe tearsheets remain native-frequency.

## Acceptance tests
1. `pytest tests/unit-tests/ensemble/test_portfolio_tester_resampling.py -q`
2. `pytest tests/portfolio_research/test_run_portfolio_test_helpers.py -q`
3. `pytest tests/portfolio_research/test_run_portfolio_test_multitimeframe.py -q`
4. `pytest tests/portfolio_research/test_config.py -q`

## Definition of done
- [ ] New helper functions added and covered by unit tests
- [ ] Multi-timeframe runner orchestration implemented
- [ ] Mixed-timeframe output naming collision-proofed
- [ ] Portfolio research config docstring clarified
- [ ] API docs updated in `docs/api/ensemble.md`
- [ ] Targeted pytest commands pass

## Notes
- Current `TimeFrame` enum may not include intraday values yet; intraday aggregation is detected by duplicate normalized dates in return indices.
