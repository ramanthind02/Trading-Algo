# T004 - Remove Sector Allocation Wiring

## Goal
Remove `sector_allocation_config_path` and its JSON resolution path from the active portfolio and research orchestration surfaces.

## Context / References
- `ensemble/portfolio.py`
- `portfolio_research/config.py`
- `portfolio_research/pipelines/portfolio_test.py`
- `scripts/benchmark_portfolio_backtest.py`
- `docs/methodology/sector_allocation.md`

## Scope
In scope:
- Remove sector-allocation config parameters from portfolio constructors used by research/global flows.
- Remove sector-allocation helper loading/validation/resolution logic from active orchestration.
- Remove config pass-through in research scripts and helpers.

Out of scope:
- Manual `instrument_weights` behavior not tied to sector JSON
- Refactoring unrelated config types

## Interfaces (must match)
- Remove: `sector_allocation_config_path` from `GlobalPortfolio.__init__`
- Remove: `sector_allocation_config_path` from `PortfolioResearchConfig`
- Remove: pass-through wiring in pipeline/scripts that currently forward the field

## Invariants / Constraints
- Removed constructor arg must fail clearly if still passed.
- Research/default configs should no longer reference the sector JSON file.
- No hidden sector-weight precedence logic remains.

## Acceptance Tests
1. Unit: constructing affected configs/portfolios with `sector_allocation_config_path` raises a clear error or is impossible by signature.
2. Unit: research pipeline no longer forwards sector config.
3. Search check: repo has no active portfolio/research call site still wiring sector JSON in this flow.

## Definition of Done
- [ ] Sector JSON is removed from active portfolio/research APIs
- [ ] Helper methods for sector resolution are removed from this flow
- [ ] Config defaults no longer point at a sector JSON file
