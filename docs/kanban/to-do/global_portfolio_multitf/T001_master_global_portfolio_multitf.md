# T001 — Global Multi-Timeframe Portfolio Architecture (Master)

## Goal
Introduce a two-level portfolio hierarchy: `TFPortfolio` (per-timeframe, renamed from `Portfolio`) handles vol scaling and intra-TF forecast combination; `GlobalPortfolio` (new wrapper) manages multiple `TFPortfolio` instances, applies a cross-timeframe `GlobalWeightLayer` (downside HRP on daily-resampled TF streams), and owns the global IDM. Eliminates the current lazy equal-weight assumption across timeframes.

## Context / References
- `ensemble/portfolio.py` — current `Portfolio` class (2009 lines); rename target
- `ensemble/weight_layer.py` — `BaseWeightLayer`, `DownsideHRPGroupedWeightLayer`, `WeightLayerConfig`
- `docs/library/Ensemble/weight_layer.md` — design principles (no Sharpe tilt, risk-based diversification)
- `docs/library/Ensemble/portfolio.md` — pipeline overview and multiplier table
- `portfolio_research/pipelines/portfolio_test.py` — research pipeline that drives Portfolio
- `ensemble/portfolio_tester.py` — `PortfolioTester`, resampling helpers

## Motivation
Current problem: multiple `TFPortfolio` instances produce `forecast_score` per timeframe; these are combined with equal weights into the final position. This ignores that different timeframes have genuinely different downside volatility and cross-TF correlation structure. The fix applies the same risk-based methodology (downside HRP on daily-resampled return streams) that is already used within a timeframe.

## Target Architecture

```
Candles → TFPortfolio (daily)  → forecast_score_D  ─┐
          TFPortfolio (weekly) → forecast_score_W  ──┤→ GlobalWeightLayer → combined_score → GlobalPortfolio → position_fraction
          TFPortfolio (monthly)→ forecast_score_M  ─┘                        (cross-TF HRP
                                                                               + cross-TF FDM)
```

### Responsibility Split

| Class | Owns | Strips |
|---|---|---|
| `TFPortfolio` (renamed `Portfolio`) | vol scaling, per-TF `WeightLayer` (FDM), per-TF instrument weights | global IDM, cross-TF combination |
| `GlobalWeightLayer` | Resample all TF forecast streams to daily; downside HRP across TFs; cross-TF FDM | per-TF logic |
| `GlobalPortfolio` | Multiple `TFPortfolio` instances; `GlobalWeightLayer`; global IDM | vol scaling |

## Task Dependency Order

```
T002 (research) → T003 (design TFPortfolio) ─┐
                → T004 (design GlobalWL)    ──┼→ T005 (design GlobalPortfolio)
                                              │
T003 → T006 (impl TFPortfolio rename)        │
T004 → T007 (impl GlobalWeightLayer)         ├→ T008 (impl GlobalPortfolio)
T005 → T008                                  │
T008 → T009 (update pipeline + tests)        │
T009 → T010 (update docs)                   ─┘
```

## Invariants / Constraints
- `TFPortfolio` must remain backward-compatible at the call site level for the single-TF research pipeline (portfolio_test.py must still work for single-TF runs).
- `GlobalWeightLayer` uses downside HRP on the same machinery as `DownsideHRPGroupedWeightLayer` — no new statistical methods.
- All resampling is forward-fill (no lookahead).
- `GlobalPortfolio` must produce the same output schema as current `Portfolio`: `['ticker', 'forecast_score', 'position_fraction']`.
- No Sharpe weighting anywhere — purely risk-based (downside covariance).

## Definition of Done
- [ ] T002–T010 all complete
- [ ] `GlobalPortfolio` produces identical output to single-TF `Portfolio` when only one TF is provided
- [ ] Multi-TF integration test passes with at least two timeframes
- [ ] `pytest tests/ -q` green
- [ ] Docs updated
