# T007 — Implement: GlobalWeightLayer

## Goal
Implement `GlobalWeightLayer` in `ensemble/weight_layer.py` (or a new `ensemble/global_weight_layer.py` if the file becomes unwieldy) per the T004 spec. This class resamples per-TF forecast streams to a common daily grid, applies downside HRP across timeframes, and outputs cross-TF weights and FDM.

## Context / References
- T004 spec (must match exactly — implement against that spec, not against this task's description)
- `ensemble/weight_layer.py` — reuse `_compute_downside_semi_covariance`, `_hrp_weights_from_semi_cov`, `_compute_fdm_from_corr_matrix` directly
- `ensemble/portfolio_tester.py` — check `resample_positions_to_daily` for resampling pattern to reuse or adapt
- `docs/library/Ensemble/weight_layer.md` — FDM formula and no-Sharpe-tilt constraint

## Scope
In scope:
- `GlobalWeightLayer` class per T004 spec.
- Forward-fill resampling of weekly/monthly forecast streams to daily (no lookahead).
- Downside semi-covariance on daily-resampled TF group return streams (reuse existing helpers).
- HRP weight vector over K timeframes (reuse `_hrp_weights_from_semi_cov`).
- Cross-TF FDM (reuse `_compute_fdm_from_corr_matrix`).
- `get_diagnostics()` for per-TF weights, FDM, and mean cross-TF downside correlation.
- Fallback: single TF → weight=1.0, FDM=1.0.
- Export `GlobalWeightLayer` from `ensemble/__init__.py`.

Out of scope:
- Changes to existing `BaseWeightLayer` subclasses.
- `GlobalPortfolio` (T008).
- Per-TF intra-ensemble weighting.

## Interfaces
- Add: `GlobalWeightLayer` class in `ensemble/weight_layer.py` (or new file per T004 decision)
- Modify: `ensemble/__init__.py` — add `GlobalWeightLayer` to exports

## Data Contracts
- `fit(tf_forecast_streams: Dict[TimeFrame, pd.DataFrame], instrument_returns: pd.DataFrame)`:
  - `tf_forecast_streams`: each value is `DataFrame[ticker, datetime, forecast_score]`
  - `instrument_returns`: `DataFrame[datetime index, ticker columns, daily returns]`
- `combine(tf_forecast_streams: Dict[TimeFrame, pd.DataFrame]) -> pd.DataFrame`:
  - Returns `DataFrame[ticker, datetime, forecast_score]` on a daily grid

## Dependencies
- T004 spec finalised.
- T006 (TFPortfolio rename) does not block this task — T006 and T007 can proceed in parallel.

## Invariants / Constraints
- Resampling: `forecast_score` at day `t` uses only signals available at close of day `t` or earlier.
- All downside HRP logic must reuse existing helpers — no duplicate implementations.
- `GlobalWeightLayer` must be independently testable (no `TFPortfolio` or `GlobalPortfolio` import dependency).

## Acceptance Tests
1. Unit test: `pytest tests/ensemble/test_global_weight_layer.py::test_single_tf_passthrough` — single TF gives weight=1.0, FDM=1.0, output matches input.
2. Unit test: `pytest tests/ensemble/test_global_weight_layer.py::test_two_tf_weights_sum_to_one` — two TFs, weights sum to 1.0.
3. Unit test: `pytest tests/ensemble/test_global_weight_layer.py::test_no_lookahead_resampling` — verify daily-resampled weekly signal only changes on Mondays (or the first trading day of each week).
4. Unit test: `pytest tests/ensemble/test_global_weight_layer.py::test_fallback_insufficient_data` — < 2 rows → equal weights, FDM=1.0.

## Definition of Done
- [ ] `GlobalWeightLayer` implemented per T004 spec
- [ ] Unit tests written and passing
- [ ] `from ensemble import GlobalWeightLayer` works
- [ ] `get_diagnostics()` returns cross-TF weights and FDM

## Notes
- If `weight_layer.py` exceeds ~1600 lines after this addition, consider splitting into `weight_layer.py` (per-TF) and `global_weight_layer.py` (cross-TF) with a shared `_weight_helpers.py` for the pure functions.
- Write tests with synthetic data — daily returns from a normal distribution, 3 years, 2–3 tickers, 2–3 timeframes. No need for real market data in unit tests.
