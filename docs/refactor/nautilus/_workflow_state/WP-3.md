# WP-3 — Opt-in Nautilus BacktestEngine P&L lane (`PnLEngine`) · Workflow Ledger

**Parity expectation:** ADDITIVE / opt-in. Lane 1 (vectorized) = default, FROZEN behaviour,
byte-identical. Lane 2 (nautilus) = additive, never overwrites the parity baseline.
**Baseline commit:** `e662cfe` (parity 2 passed). **Mode:** autonomous loop.

## Phase 1 — DISCOVERY (done; read-only)
- **Vectorized P&L (FROZEN):** `ensemble/portfolio_impl/portfolio_tester.py:154`
  `calculate_strategy_returns_from_positions(positions_df, candles_df, *, instrument_return_kind='log_intraday')`.
  `log_intraday` (lines 207-212) = enter `open[t+1]`, exit `close[t+1]`; no-lookahead shift `:231-250`.
- **position_fraction producer:** `ensemble/portfolio_impl/global_portfolio_impl.py:359` `predict`
  → `['ticker','datetime','forecast_score','position_fraction']`; `predict_from_cache` `:481`.
- **Editable boundary (NON-FROZEN):** `research/portfolio/pipelines/portfolio_test.py` — global call `:794`
  (+ ensemble `:506`, tearsheets `:526/:543`); clean positions hand-off at `:777`.
- **Doc drift:** WP-3 doc's suggested `ensemble/evaluation/pnl_engine.py` VIOLATES the freeze →
  new code lives under `research/portfolio/pnl/`.
- **#1 risk:** Nautilus has no native next-bar-open fill (bar ts_init=close; `ingest.py:99`), so the
  nautilus lane CANNOT reproduce `log_intraday` bit-for-bit → additive only; reconcile only
  `nautilus + CLOSE_TO_CLOSE + BestPriceFillModel` vs the `log` (close-to-close) kind, never `log_intraday`.

## Phase 2 — IMPLEMENT
- **Unit 1 (DONE, committed):** `research/portfolio/pnl/{__init__,pnl_engine}.py` — `PnLEngine` protocol,
  `VectorizedPnLEngine` (verbatim delegate to the frozen fn), `make_pnl_engine` ("nautilus"→NotImplementedError).
  Config field `pnl_engine="vectorized"` (`research/portfolio/config.py:355`). Routed `portfolio_test.py:794`.
  **Gate: pytest tests/parity = 2 passed (byte-identical). No FROZEN touched.**
- **Unit 2 (NautilusPnLEngine / TargetRebalanceStrategy / BacktestEngine):** BLOCKED on WP-2 Unit-2
  data-precision fork (the nautilus lane needs the catalog candle/bar layer). Needs:
  BacktestEngine, Strategy(TargetRebalanceStrategy) fed position_fraction as registered CustomData,
  FillModel(BestPriceFillModel), OrderFactory.bracket, TWAP ExecAlgorithm, OMS NETTING, ExecutionPolicy.
  Reference dataset: NDX 2026 MT5 intraday (ingest built in WP-2 Unit-1b).

## Phase log
| ts | phase | unit | status | parity | frozen-touched? | evidence | notes |
|----|-------|------|--------|--------|-----------------|----------|-------|
| p1 | 1 | discovery | done | — | no | manifest | drift resolved; freeze-violation in doc target found |
| p2 | 2 | unit1 scaffold | done | 2 passed/2 passed | no | committed | vectorized default = byte-identical |
| p3 | 2 | unit2 nautilus lane | DONE | 5/5 lane tests + parity 2/2 | no | tests/research/test_nautilus_pnl_lane.py | NautilusPnLEngine + MultiTickerNautilusPnLEngine; frictionless CLOSE_TO_CLOSE≈vectorized log (corr≥0.99) reconciliation GREEN |
| p4 | 2 | CFD realism default | DONE | parity green (vectorized pinned) | no | scripts/capture_baselines.py | CFD feed (data_feed default), realistic Nautilus on TEST phase (realistic_phases). Test: futures 1.01 → CFD-frictionless 1.78 → CFD-realistic 2.27 Sharpe. Rollover overlay T-15 exit / post-deadzone reopen, MARKET, synth M1-spread quotes, windowed ephemeral catalog |

## Outcome (CFD + Nautilus migration, this session)

The opt-in lane is now the **default for the portfolio test/holdout phase** and the feature-research
validation + portfolio-addition phases (exploration stays frictionless). Data feed default = Darwinex
**CFD** (faithful %-returns; no additive back-adjustment distortion) — see the lead plan
`~/.claude/plans/keen-drifting-hejlsberg.md` and `tests/parity/snapshots/feed_comparison.md`
(futures↔CFD portfolio return corr 0.894; CFD better on nearly every sleeve). Gates: `pytest tests/parity`
(byte-identical, pinned to futures+vectorized) + `scripts/compare_feeds.py` (CFD similarity).
