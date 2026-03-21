# T006 - Replace Sector Tests With Adapter Coverage

## Goal
Remove sector-allocation-specific tests and replace them with tests that pin the adapter-based global diversification workflow built around the unchanged `WeightLayer`.

## Context / References
- `tests/unit-tests/ensemble/test_portfolio_sector_allocation.py`
- `tests/unit-tests/ensemble/test_weight_layer.py`
- `tests/unit-tests/ensemble/test_global_portfolio.py`
- `tests/portfolio_research/test_weight_layer_report.py`

## Scope
In scope:
- Delete sector-allocation-only unit coverage.
- Add unit tests for adapter encoding, decoding, aggregation, and removed constructor/config surface.
- Update global portfolio/report tests to reflect adapter-driven diagnostics.

Out of scope:
- Broad integration redesign outside this refactor

## Test Cases
1. Global stream encoding produces unique `stream_id` values.
2. Current `WeightLayer` can fit encoded global streams without API changes.
3. Decoded combined output aggregates correctly to `(ticker, datetime, forecast_score)`.
4. Single-stream or single-cluster case still yields `FDM = 1.0`.
5. Daily alignment uses forward-fill only and does not look ahead.
6. Removed `sector_allocation_config_path` surface is rejected.

## Invariants / Constraints
- Tests using handcrafted synthetic data remain unit tests.
- Any cross-layer pipeline checks with persisted repo data belong under `tests/integration/`.

## Acceptance Tests
1. `pytest tests/unit-tests/ensemble/test_weight_layer.py -q`
2. `pytest tests/unit-tests/ensemble/test_global_portfolio.py -q`
3. `pytest tests/portfolio_research/test_weight_layer_report.py -q`

## Definition of Done
- [ ] Sector-allocation unit test file is removed or replaced
- [ ] Adapter coverage exists for encoding/decoding/aggregation
- [ ] Removed constructor/config surface is pinned by tests
