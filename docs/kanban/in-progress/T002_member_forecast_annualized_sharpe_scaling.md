# T002 — Member Forecast Annualized Sharpe Scaling

## Goal
Scale walkforward fast-path member forecasts using annualized Sharpe magnitude (clipped) instead of binary sign-only activation so stronger features receive proportionally larger forecast amplitude.

## Context / References
- `ensemble/diversified_ensemble.py`
- `feature_research/walkforward/portfolio_evaluator.py` (fast-path emits member-level signals via `emit_member_signals`)
- Sample artifacts under `feature_research/shared_results/continuous/rsi/walkforward/tearsheets/`

## Observed behavior
- Member outputs are Sharpe-like per-bar values from continuous binning, but the ensemble member path currently discards magnitude and uses sign-only activation for vol targeting.
- This removes feature-strength differentiation and can understate risk when weak/strong features are treated identically.

## Expected behavior
- Convert per-bar Sharpe-like member outputs to annualized Sharpe-like strength using timeframe-aware scaling (`sqrt(bars_per_year)`).
- Preserve sign and clip magnitude to a bounded range suitable for forecast sizing (default `[-2, 2]`).

## Scope
In scope:
- `ensemble/diversified_ensemble.py` member forecast scaling path
- Unit test(s) for annualized Sharpe strength mapping helper

Out of scope:
- Changing base-model fit statistics or stored bin metrics
- Reworking non-member (legacy binary signal) forecast paths

## Interfaces (must match)
- No public API changes
- Internal helper additions only

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/ensemble/test_diversified_ensemble_flattened_signals.py -q`

