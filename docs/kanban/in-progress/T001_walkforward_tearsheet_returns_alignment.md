# T001 — Walkforward Tearsheet Returns Alignment

## Goal
Fix walkforward stage-2 OOS/tearsheet return generation so strategy returns are computed from ticker-level portfolio positions against actual candle returns (not research target values), preserving multi-ticker aggregation and correct return scaling.

## Context / References
- `feature_research/walkforward/portfolio_evaluator.py`
- `feature_research/walkforward/runner.py`
- `ensemble/portfolio_tester.py` (`calculate_strategy_returns_from_positions`)
- Sample artifact: `feature_research/shared_results/continuous/rsi/walkforward/tearsheets/walkforward_ensemble_tearsheet.html`

## Observed behavior
- Walkforward tearsheet strategy volatility is ~`1.03%` annualized while benchmark is ~`19.72%` in sample results.
- Stage-2 OOS returns are currently built via datetime-only aggregation (`groupby("datetime").mean()`) and multiplied by the research `target` series (`log_return_atr` in the sample), which is normalized and not comparable to buy-and-hold candle returns.

## Expected behavior
- OOS portfolio and per-signal returns use ticker-level `position_fraction` outputs and candle-derived instrument returns (same return basis as baseline / portfolio tester).
- Multi-ticker aggregation preserves all ticker contributions when building daily return series.

## Reproduction
1. `source venv/bin/activate && python feature_research/walkforward/run_walkforward.py`
2. Open `feature_research/shared_results/continuous/rsi/walkforward/tearsheets/walkforward_ensemble_tearsheet.html` and compare strategy vs benchmark annualized volatility.
3. `pytest tests/feature_research/walkforward/test_portfolio_evaluator.py -q`

## Regression window
- Unknown (needs investigation).

## Scope
In scope:
- Walkforward stage-2 return generation in `feature_research/walkforward/portfolio_evaluator.py`
- Portfolio WeightLayer fit path bugfix in `ensemble/portfolio.py` (undefined `target_data`)
- Unit regression test(s) for multi-ticker OOS return aggregation path

Out of scope:
- Broader walkforward selection-stage target normalization design
- Refactoring research target construction in `feature_research/in_sample/pipeline.py`

## Interfaces (must match)
- Modify: `feature_research/walkforward/portfolio_evaluator.py` — keep `evaluate_fold_portfolio(...) -> FoldPortfolioResult` signature unchanged
- Modify: `ensemble/portfolio.py` — `_fit_weight_layer(...)` internal helper only (no public API changes)
- Add/modify internal helper(s) only (no public API changes)

## Constraints / Risk
- Preserve no-lookahead behavior in realized return calculation (positions applied to next bar returns)
- Keep deterministic index ordering for tearsheet generation
- Risk: incorrect merge keys (`ticker`, `datetime`) would under/over-count returns across instruments

## Acceptance tests
1. `pytest tests/feature_research/walkforward/test_portfolio_evaluator.py -q` — regression guard for candle-based OOS returns path.
2. `pytest tests/feature_research/walkforward/test_runner.py -q` — ensure walkforward orchestration still passes with unchanged interfaces.
3. `source venv/bin/activate && python - <<'PY' ... PY` deterministic smoke (optional local) to inspect duplicate datetime handling in stitched OOS series.

## Definition of done
- [ ] Tests updated/added under `tests/feature_research/walkforward/`
- [ ] Docs API update not required (no interface changes)
- [ ] `pytest tests/feature_research/walkforward/test_portfolio_evaluator.py -q` passes
- [ ] `pytest tests/feature_research/walkforward/test_runner.py -q` passes

## Notes
- Sample artifact volatility mismatch strongly suggests scale mismatch between strategy and baseline returns.
- Follow-up task may be needed for multi-ticker target usage in stage-2 fast-path training if further drift remains after this fix.
