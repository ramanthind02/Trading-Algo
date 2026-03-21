# T008 — Implement: GlobalPortfolio

## Goal
Implement `GlobalPortfolio` in `ensemble/portfolio.py` per the T005 spec. This is the new top-level class that owns multiple `TFPortfolio` instances, a `GlobalWeightLayer`, and applies global IDM and instrument weights to produce final position fractions.

## Context / References
- T005 spec (must match exactly)
- T003 spec (TFPortfolio interface — know what `TFPortfolio.predict` returns)
- T004 spec (GlobalWeightLayer.combine input/output)
- `ensemble/portfolio.py` — IDM computation methods to migrate or reuse here
- `docs/library/Ensemble/portfolio.md` — IDM formula

## Scope
In scope:
- `GlobalPortfolio` class per T005 spec.
- `fit(candles_per_tf, instrument_returns)` — fits each `TFPortfolio`, then `GlobalWeightLayer`, then global IDM.
- `predict(candles_per_tf)` — orchestrates TFPortfolio predictions → GlobalWeightLayer.combine → instrument weights + IDM → position fractions.
- Output schema: `['ticker', 'forecast_score', 'position_fraction']` (drop-in for current `Portfolio.predict`).
- `get_diagnostics()` aggregating per-TF and global WL diagnostics.
- `sector_allocation_config_path` support (migrated from `TFPortfolio`/`Portfolio`).
- Export as the new public `Portfolio` alias: `Portfolio = GlobalPortfolio` in `ensemble/__init__.py` (superseding the `TFPortfolio` alias from T006).
- Single-TF degenerate case: `GlobalPortfolio([one_tf_portfolio])` must be numerically equivalent to calling that `TFPortfolio` directly.

Out of scope:
- `PortfolioTester` changes (T009).
- New statistical methods — only reuse existing IDM computation.
- Changes to `PositionSizer`.

## Interfaces
- Add: `GlobalPortfolio` class in `ensemble/portfolio.py`
- Modify: `ensemble/__init__.py` — replace `Portfolio = TFPortfolio` alias with `Portfolio = GlobalPortfolio`; keep `TFPortfolio` export

## Data Contracts
- `fit(candles_per_tf: Dict[TimeFrame, pd.DataFrame], instrument_returns: pd.DataFrame) -> GlobalPortfolio`
- `predict(candles_per_tf: Dict[TimeFrame, pd.DataFrame]) -> pd.DataFrame`
  - Returns: `DataFrame` with columns `['ticker', 'forecast_score', 'position_fraction']`
  - Index: daily datetime (finest common grid)

## Dependencies
- T005 spec finalised.
- T006 (`TFPortfolio` rename) must be complete — `GlobalPortfolio` depends on `TFPortfolio`.
- T007 (`GlobalWeightLayer`) must be complete — `GlobalPortfolio` owns one.

## Invariants / Constraints
- `GlobalPortfolio` with a single `TFPortfolio` must produce the same `position_fraction` values as calling `TFPortfolio.predict` directly (± floating point). Write a property test.
- No Sharpe weighting: IDM uses instrument return correlations (risk-based), not forecast Sharpe.
- `from ensemble import Portfolio` must import `GlobalPortfolio` after this task (update the alias).

## Acceptance Tests
1. Unit test: `pytest tests/ensemble/test_global_portfolio.py::test_single_tf_equivalence` — `GlobalPortfolio([tf_p])` matches `tf_p.predict()` position fractions.
2. Unit test: `pytest tests/ensemble/test_global_portfolio.py::test_two_tf_output_schema` — output has correct columns and no NaNs.
3. Unit test: `pytest tests/ensemble/test_global_portfolio.py::test_output_schema_matches_legacy` — output schema identical to legacy `Portfolio.predict`.
4. `pytest tests/ -q` — full suite green.

## Definition of Done
- [ ] `GlobalPortfolio` implemented per T005 spec
- [ ] `Portfolio = GlobalPortfolio` alias in `__init__.py`
- [ ] Unit tests written and passing
- [ ] Single-TF equivalence property test passes
- [ ] `get_diagnostics()` returns per-TF + global diagnostics

## Notes
- Be careful with `sector_allocation_config_path`: this was attached to `Portfolio` (now `TFPortfolio`). Move its resolution to `GlobalPortfolio.__init__` and pass resolved `instrument_weights` dict to each `TFPortfolio` or hold it only at the global level — per T005 spec.
- The IDM computation method can be copied/moved from `Portfolio._compute_idm` rather than re-implemented.
