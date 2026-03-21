# T006 — Implement: Rename Portfolio → TFPortfolio and Strip Global Concerns

## Goal
Rename the existing `Portfolio` class in `ensemble/portfolio.py` to `TFPortfolio`, remove the global IDM and global instrument weight responsibilities (which will move to `GlobalPortfolio` in T008), and ensure the class is self-contained for per-timeframe use. Add `Portfolio = TFPortfolio` alias to preserve all existing import sites.

## Context / References
- T003 spec (TFPortfolio interface — must match exactly)
- `ensemble/portfolio.py` — target file
- `ensemble/__init__.py` — update exports
- All test files that import `Portfolio` (identified in T002)

## Scope
In scope:
- Rename class `Portfolio` → `TFPortfolio` throughout `ensemble/portfolio.py`.
- Add `Portfolio = TFPortfolio` alias at module level so existing imports continue to work.
- Strip IDM and global instrument weight logic from `TFPortfolio` per T003 spec, IF T003 specifies they move entirely to `GlobalPortfolio`. If T003 specifies `TFPortfolio` retains standalone IDM capability, keep it.
- Update `ensemble/__init__.py` to export both `TFPortfolio` and `Portfolio` (alias).
- Do NOT change any logic — this is a rename + scoping refactor only.

Out of scope:
- Implementing `GlobalPortfolio` or `GlobalWeightLayer`.
- Changing `portfolio_test.py` (T009).
- Any statistical logic changes.

## Interfaces
- Modify: `ensemble/portfolio.py` — class renamed; alias added; stripped attributes per T003 spec
- Modify: `ensemble/__init__.py` — add `TFPortfolio` to exports

## Data Contracts
- `TFPortfolio.predict(...)` output schema: unchanged from current `Portfolio.predict` — `['ticker', 'forecast_score', 'position_fraction']` (for backward compat in standalone use)

## Dependencies
- T003 spec must be finalised before implementation begins.
- T002 provides the list of test files to check after rename.

## Invariants / Constraints
- All existing tests must pass after this task (no broken imports, no logic changes).
- `from ensemble import Portfolio` must still work.
- `from ensemble import TFPortfolio` must now also work.

## Acceptance Tests
1. `pytest tests/ -q` — full suite green, no import errors.
2. `python -c "from ensemble import Portfolio, TFPortfolio; assert Portfolio is TFPortfolio"` — alias works.
3. `python -c "from ensemble.portfolio import TFPortfolio"` — direct import works.

## Definition of Done
- [ ] Class renamed in `portfolio.py`
- [ ] Alias `Portfolio = TFPortfolio` present
- [ ] `__init__.py` exports updated
- [ ] All tests pass
- [ ] No references to old class name remain (except the alias line itself)

## Notes
- Do a grep for `class Portfolio` and all `Portfolio(` instantiations before starting to know the full blast radius.
- The alias ensures zero changes needed at existing call sites — do not update call sites to `TFPortfolio` yet (that is a later cleanup).
