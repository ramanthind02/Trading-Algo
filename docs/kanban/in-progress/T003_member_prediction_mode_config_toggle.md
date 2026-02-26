# T003 — Walkforward Member Prediction Mode Config Toggle

## Goal
Add a researcher-facing config toggle to choose how walkforward member-level predictions are converted into forecast strength: binary sign-only vs annualized Sharpe-weighted magnitude.

## Context / References
- `feature_research/config.py`
- `feature_research/walkforward/config.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/portfolio_evaluator.py`
- `ensemble/diversified_ensemble.py`

## Expected behavior
- `feature_research/config.py` exposes a setting with options:
  - `binary`
  - `sharpe_weighted`
- Walkforward stage-2 portfolio evaluation forwards the selected mode into the ensemble member forecast path.
- Default preserves current behavior (`sharpe_weighted`).

## Scope
In scope:
- Config enum/dataclass field additions
- Wiring through walkforward runner/evaluator
- Internal ensemble member forecast-strength mode dispatch
- Unit tests for mode behavior

Out of scope:
- Changing base-model fit statistics
- Non-walkforward production config surfaces

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. pytest tests/feature_research/walkforward/test_runner.py -q`
2. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/ensemble/test_diversified_ensemble_flattened_signals.py -q -k member_signal_strength`

