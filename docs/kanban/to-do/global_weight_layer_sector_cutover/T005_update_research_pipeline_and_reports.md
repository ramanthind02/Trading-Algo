# T005 - Update Research Pipeline and Reports

## Goal
Update research orchestration and diagnostics export so the current `WeightLayer` can still be reported correctly after the new adapter-based global weighting path is introduced.

## Context / References
- `portfolio_research/pipelines/portfolio_test.py`
- `portfolio_research/config.py`
- `portfolio_research/weight_layer_report.py`
- `tests/portfolio_research/test_weight_layer_report.py`

## Scope
In scope:
- Update research pipeline construction to use the new global adapter path.
- Remove sector-config references from research config and pipeline wiring.
- Extend reports to present adapter rollups while preserving current `WeightLayer` diagnostics consumption.

Out of scope:
- Redesigning report visuals unrelated to the new diagnostics
- Changing `WeightLayer` HTML/CSV semantics outside what the adapter requires

## Interfaces (must match)
- Modify: `portfolio_research/pipelines/portfolio_test.py`
- Modify: `portfolio_research/config.py`
- Modify: `portfolio_research/weight_layer_report.py`
- Modify: `scripts/benchmark_portfolio_backtest.py` if it still forwards removed sector config

## Data Contracts
- Report input continues to accept raw `WeightLayer` diagnostics.
- Global report adds adapter-derived sections:
  `ticker_rollups`, optional `timeframe_rollups`, decoded stream summaries

## Invariants / Constraints
- Single-timeframe research runs continue to export global weight-layer diagnostics.
- Report code must not assume per-real-ticker `WeightLayer` fitting if the adapter uses a synthetic global ticker.

## Acceptance Tests
1. `pytest tests/portfolio_research/test_weight_layer_report.py -q`
2. `pytest tests/portfolio_research/test_run_portfolio_test_multitimeframe.py -q`
3. Manual smoke: research pipeline writes global weight-layer report without sector-config inputs.

## Definition of Done
- [ ] Research config/pipeline no longer mention sector allocation
- [ ] Report exporter supports adapter rollups
- [ ] Existing report coverage is updated for the new orchestration
