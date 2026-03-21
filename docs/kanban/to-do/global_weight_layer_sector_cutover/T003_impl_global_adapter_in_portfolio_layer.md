# T003 - Implement Global Adapter in Portfolio Layer

## Goal
Implement the global stream adapter in the portfolio layer so `GlobalPortfolio` can reuse the current `WeightLayer` unchanged for cross-ticker/timeframe/strategy diversification.

## Context / References
- `ensemble/portfolio.py`
- `ensemble/weight_layer.py`
- T002 adapter contract

## Scope
In scope:
- Encode forecast vectors into the synthetic global `WeightLayer` input.
- Call the existing `WeightLayer.fit(...)` and `combine(...)`.
- Decode and aggregate weighted output back to ticker-level forecasts.
- Surface adapter rollups in `GlobalPortfolio.get_diagnostics()`.

Out of scope:
- Editing `ensemble/weight_layer.py`
- Changing TFPortfolio forecast generation semantics

## Interfaces (must match)
- Modify: `ensemble/portfolio.py`
- `GlobalPortfolio.fit(...)`:
  prepare raw forecast vectors, align to daily grid, encode to synthetic global ticker, fit current `WeightLayer`
- `GlobalPortfolio.predict(...)`:
  encode prediction streams, combine through current `WeightLayer`, decode, aggregate to ticker-level `forecast_score`, then apply IDM and cap

## Data Contracts
- Encoded vectors supplied to `WeightLayer` must still match the existing contract:
  `ticker`, `datetime`, `model_name`, `forecast`, `signal`
- Decoded combined output must aggregate all weighted streams by:
  `groupby(['ticker', 'datetime']) -> sum(weighted contribution)`

## Invariants / Constraints
- Current `WeightLayer` behavior must be preserved.
- Global aggregation cannot silently drop streams.
- Single-timeframe runs must still work through the same path.

## Acceptance Tests
1. Unit: single-timeframe global adapter run matches current single-TF aggregation behavior.
2. Unit: two-ticker, two-timeframe encoded streams aggregate back to the correct real tickers.
3. Unit: `GlobalPortfolio.get_diagnostics()` includes raw `WeightLayer` diagnostics plus adapter rollups.

## Definition of Done
- [ ] `GlobalPortfolio` uses adapter-based global weighting
- [ ] No direct sector-allocation weighting remains in global path
- [ ] No `WeightLayer` implementation changes are required
