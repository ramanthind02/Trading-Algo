# T002 - Design Global Weight Layer Adapter

## Goal
Specify the adapter layer that presents all ticker, timeframe, and strategy streams to the existing `WeightLayer` as one global weighting problem without changing `WeightLayer` internals.

## Context / References
- `ensemble/weight_layer.py`
- `ensemble/portfolio.py`
- Existing global model-name helpers in `ensemble/portfolio.py`

## Scope
In scope:
- Define the input transformation into a single synthetic `WeightLayer` problem.
- Define how weighted outputs map back to real tickers and datetimes.
- Define diagnostics expected from the adapter layer.

Out of scope:
- Editing `WeightLayer` clustering logic
- Report rendering details beyond required data contract

## Interfaces (must match)
- Add adapter helper(s) near `GlobalPortfolio` or in a small adjacent helper module.
- Standardize a stable stream identifier:
  `{ticker}::{timeframe}::{model_name}`
- Use a synthetic ticker key such as `__GLOBAL__` when invoking `WeightLayer`.

## Data Contracts
- Adapter input rows:
  `ticker`, `datetime`, `model_name`, `forecast`, `signal`, `timeframe`
- Adapter-to-`WeightLayer` rows:
  `ticker=__GLOBAL__`, `datetime`, `model_name=<stream_id>`, `forecast`, `signal`
- Adapter decode map:
  `stream_id -> {ticker, timeframe, original_model_name}`
- Adapter output:
  `ticker`, `datetime`, `forecast_score`

## Diagnostics Contract
- Preserve raw `WeightLayer.get_diagnostics()` output for the synthetic global ticker.
- Add adapter rollups:
  `ticker_rollups`, `timeframe_rollups`, `stream_decode_map`
- Do not require a new `WeightLayer` diagnostics schema.

## Invariants / Constraints
- No mutation of `WeightLayer` public API is required.
- Adapter must be reversible: every encoded stream ID decodes exactly once.
- All aggregation back to ticker level occurs after `WeightLayer.combine(...)`.

## Acceptance Tests
1. Unit: encoded stream IDs are unique across tickers, timeframes, and models.
2. Unit: adapter decode restores original ticker/timeframe/model identity.
3. Unit: passing encoded streams through current `WeightLayer` produces decodable weighted output.

## Definition of Done
- [ ] Synthetic global ticker approach is fully specified
- [ ] Stream ID format is fixed
- [ ] Decode/rollup diagnostics are fixed
