# Multi-Timeframe Portfolio Orchestration

> **Scope:** How `GlobalPortfolio` combines signals across the supported timeframes (Daily, Weekly, Monthly) for both backtesting and live trading. The global combine runs on a **daily grid**; sub-daily rebalancing is not implemented.

---

## Core Principle

Each timeframe is processed **independently and sequentially**, then merged on a common daily grid. There is no interleaving of timeframes in a single loop.

```
Monthly candles  → TFPortfolio(M) → monthly forecast series ──┐
Weekly candles   → TFPortfolio(W) → weekly forecast series  ──┤ forward-fill → daily grid → weighted sum (+FDM) → position
Daily candles    → TFPortfolio(D) → daily forecast series   ──┘
```

Higher-TF forecasts are **constant** between their candle closes. Forward-filling naturally carries the last known forecast until new data arrives. `TFPortfolio.__init__` auto-loads vault ensembles from `vault/{D,W,M}` when no explicit ensembles are passed.

---

## Fit vs Predict

### Fit (training)

During fit, all forecast streams are resampled to a **daily grid** to ensure equal scaling across timeframes. Higher-TF signals are forward-filled, which guarantees the `WeightLayer` sees comparable signal distributions.

```python
from ensemble import GlobalPortfolio, TFPortfolio, WeightLayer
from ensemble.portfolio import PortfolioCacheQuery
from lib.core.enums import TimeFrame

global_p = GlobalPortfolio(
    tf_portfolios=[
        TFPortfolio(trading_timeframe=TimeFrame.D),
        TFPortfolio(trading_timeframe=TimeFrame.W),
    ],
    weight_layer=WeightLayer(weight_method="equal_signal", fdm_max=2.0),
)

query = PortfolioCacheQuery(
    tickers=("ES", "NQ"),
    start=train_start,
    end=train_end,
    timeframes=(TimeFrame.D, TimeFrame.W),
)
global_p.fit_from_cache(query, instrument_returns=returns_df)
```

`fit_from_cache` is an instance method: it loads candles + EWSD volatility from the central
cache and delegates to `fit(candles_per_tf, instrument_returns, daily_volatility_df)`.

Internally `GlobalPortfolio.fit` (in [`ensemble/portfolio_impl/global_portfolio_impl.py`](../../../ensemble/portfolio_impl/global_portfolio_impl.py)):
1. Fit each `TFPortfolio` on its native-frequency candles (`fit_from_candles`).
2. Collect per-TF forecast streams (`collect_tf_forecast_streams`).
3. Build a shared daily grid (`build_daily_grid`) and align every stream onto it via
   forward-fill (`align_forecast_vectors_to_daily_grid`, defined in
   `ensemble/portfolio_impl/portfolio_global_streams.py`).
4. Normalize by downside-vol (`normalize_global_signals_by_downside_vol`), encode into the
   synthetic `__GLOBAL__` ticker (`encode_forecast_vectors_for_global_weight_layer`), and fit
   the `WeightLayer`.
5. Compute global IDM from instrument return correlations (`calculate_idm_from_returns`,
   `IDM = min(√(1/(mean_corr + 0.01)), idm_max)`).

### Predict (inference)

Predict follows the same pattern — process each TF independently, align to the rebalance grid, combine with fitted weights.

```python
positions = global_portfolio.predict_from_cache(query)
```

---

## Backtesting

Backtesting is straightforward because all candle history is available upfront.

```python
query = PortfolioCacheQuery(
    tickers=("ES", "NQ", "YM", "RTY"),
    start=backtest_start,
    end=backtest_end,
    timeframes=(TimeFrame.M, TimeFrame.W, TimeFrame.D),
)
positions = global_portfolio.predict_from_cache(query)
```

Each `TFPortfolio` receives its **full** candle history, processes it vectorized, and returns a complete forecast series. `GlobalPortfolio.predict` merges them.

---

## Rebalance Frequency

The global combine rebalances on the **daily grid** (the finest supported timeframe).

Between candle closes for a given TF, that TF's forecast is constant (carried forward). This means:

- Monthly TFPortfolio produces one new forecast per month.
- Weekly TFPortfolio produces one new forecast per week.
- Daily TFPortfolio produces one new forecast per day.
- On each daily tick, the latest forecast from each TF is combined.

---

## Alignment Grid

`align_forecast_vectors_to_daily_grid` (in `ensemble/portfolio_impl/portfolio_global_streams.py`)
projects every per-TF stream onto a shared **daily** grid built by `build_daily_grid`:

1. Build the daily grid from the union of stream dates (and an optional reference index — the
   global-returns index at fit time, or the daily-candle reference grid at predict time).
2. For each stream: reindex to the grid, forward-fill, fill remaining NaN with 0.
3. Combine via fitted `WeightLayer` weights after encoding into the synthetic `__GLOBAL__` ticker.

Higher-TF forecasts simply repeat (via forward-fill) across the daily grid until a new candle
closes. (Finer-than-daily rebalancing is not implemented today — the global combine operates on
a daily grid.)

---

## Lookback & Stateful Bias Nodes

Bias nodes are stateful, but their warmup contract is now machine-readable. Each node exposes `lookback_contributions()`, `max_lookback()`, and `cold_rebuild_candle_count()` through the base `BiasNode` API.

### Design choice: stateless prediction via lookback window

Rather than serializing/deserializing bias node state (fragile across restarts, hard to test), the caller provides enough historical candles to warm up all nodes from scratch.

```
max_lookback = max(node.max_lookback() for all bias nodes in all ensembles for this TF)
```

The prediction path processes a stateless cold-rebuild window through the bias nodes, then uses only the **tail** of the resulting forecast series. In practice:

```
cold_rebuild_candles = max(node.cold_rebuild_candle_count() for all bias nodes in all ensembles for this TF)
```

`front_bad` still matters, but only as one contribution inside `max_lookback()`. Wrapper nodes may legitimately require more warmup than `front_bad` alone because they add their own rolling windows on top of a wrapped node.

This approach is:

- **Pure/deterministic** — same candles always produce the same output.
- **Trivially testable** — no hidden state to manage.
- **Fast enough** — processing 252 daily candles through 50+ bias nodes takes milliseconds.

| Timeframe | Typical max lookback | Typical cold rebuild window |
|-----------|---------------------|-----------------------------|
| Monthly   | ~24 bars (2 years)  | ~30 bars                    |
| Weekly    | ~52 bars (1 year)   | ~63 bars                    |
| Daily     | ~252 bars (1 year)  | ~303 bars                   |

These are illustrative magnitudes; the actual values come from each node's
`max_lookback()` / `cold_rebuild_candle_count()`. A common warmup heuristic is
`max_lookback + ceil(max_lookback * 0.2)`.

---

## Key Invariants

1. **No timeframe interleaving** — each TF is processed as a complete, independent series.
2. **Forward-fill bridges TFs** — higher-TF forecasts are carried forward on the daily grid.
3. **Fit and predict on a daily grid** — weights are learned on a daily grid, and the global combine runs on a daily grid at predict time.
4. **Stateless predict** — size the warmup window from `max_lookback()` / `cold_rebuild_candle_count()`; no serialized bias node state.
5. **Forecast scores are additive** — the weighted sum across TFs (with FDM) produces the combined forecast.

---

**See also:** [Portfolio pipeline](portfolio.md), [Weight layer](weight_layer.md), [Live multi-timeframe](../Deployment/live_multi_timeframe.md), [Cache architecture](../Cache/architecture.md)

> _Verified against current code via CodeGraph on 2026-06-07._
