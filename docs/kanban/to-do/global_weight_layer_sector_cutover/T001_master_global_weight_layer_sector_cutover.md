# T001 - Global Weight Layer Sector Cutover (Master)

## Goal
Remove sector-allocation-config-driven weighting from the portfolio flow and replace it with a global diversification orchestration layer that reuses the current `WeightLayer` implementation unchanged.

## Context / References
- `ensemble/weight_layer.py` - existing clustered `WeightLayer`; must remain intact
- `ensemble/portfolio.py` - `TFPortfolio`, `GlobalPortfolio`, sector-allocation helpers
- `portfolio_research/pipelines/portfolio_test.py` - primary research pipeline
- `portfolio_research/weight_layer_report.py` - diagnostics/report export
- `portfolio_research/config.py` - research config surface

## Scope
In scope:
- Remove `sector_allocation_config_path` from portfolio/research orchestration.
- Introduce a global adapter/orchestration path that feeds aggregated streams into the current `WeightLayer`.
- Update tests, reports, and docs to describe the new orchestration model.

Out of scope:
- Rewriting clustering, weighting, or FDM logic inside `ensemble/weight_layer.py`
- Changing `cluster_equal` or `cluster_corr_ulcer`
- Introducing new statistical weighting methods

## Non-Negotiable Decisions
- Keep the current `WeightLayer` implementation as-is.
- Achieve cross-ticker/timeframe/strategy diversification by adapting inputs around `WeightLayer`, not by replacing it.
- `GlobalPortfolio` keeps only post-weight-layer IDM scaling and position capping.
- Hard-remove sector-allocation JSON from active sizing flows.
- Sector metadata may remain only for diagnostics/reporting, never for sizing inputs.

## Dependency Order
`T002 -> T003 -> T004 -> T005 -> T006 -> T007`

## Invariants / Constraints
- Deterministic: same aggregated stream inputs yield the same `WeightLayer` output.
- No lookahead in any daily-grid alignment or forward-fill.
- Output schema from `GlobalPortfolio.predict(...)` stays `['ticker', 'datetime', 'forecast_score', 'position_fraction']`.
- Existing single-timeframe `Portfolio` usage remains available.

## Definition of Done
- [ ] Sector-allocation config is removed from portfolio/research orchestration
- [ ] Current `WeightLayer` file remains implementation-identical or behaviorally unchanged
- [ ] Global adapter path is fully specified
- [ ] Tests cover adapter behavior and removed sector-config surface
- [ ] Docs updated to explain the new orchestration model
