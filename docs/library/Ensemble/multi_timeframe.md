# Multi-Timeframe Portfolio Orchestration

> **Scope:** How `GlobalPortfolio` combines signals from multiple timeframes (D, W, M, and intraday) for both backtesting and live trading.

---

## Core Principle

Each timeframe is processed **independently and sequentially**, then merged on a common rebalance grid. There is no interleaving of timeframes in a single loop.

```
Monthly candles  → TFPortfolio(M).predict → monthly forecast series ──┐
Weekly candles   → TFPortfolio(W).predict → weekly forecast series  ──┤ forward-fill → rebalance grid → weighted sum → position
Daily candles    → TFPortfolio(D).predict → daily forecast series   ──┤
Hourly candles   → TFPortfolio(H1).predict → hourly forecast series ──┘
```

Higher-TF forecasts are **constant** between their candle closes. Forward-filling naturally carries the last known forecast until new data arrives.

---

## Fit vs Predict

### Fit (training)

During fit, all forecast streams are resampled to a **daily grid** to ensure equal scaling across timeframes. Intraday signals lose some granularity, but this guarantees the `WeightLayer` sees comparable signal distributions.

```python
from ensemble.portfolio import PortfolioCacheQuery

query = PortfolioCacheQuery(
    tickers=("ES", "NQ"),
    start=train_start,
    end=train_end,
    timeframes=(TimeFrame.D, TimeFrame.W),
)
GlobalPortfolio.fit_from_cache(
    query=query,
    instrument_returns=returns_df,
)
```

Internally:
1. Fit each `TFPortfolio` on its native-frequency candles.
2. Collect per-TF base-model forecast vectors.
3. Align all vectors to a shared daily grid via forward-fill (`_align_forecast_vectors_to_daily_grid`).
4. Normalize by downside-vol, encode into a synthetic `__GLOBAL__` ticker, fit `WeightLayer`.
5. Compute global IDM from instrument return correlations.

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

The portfolio rebalances at the **lowest available timeframe**. If hourly signals exist, rebalance hourly. If the lowest TF is daily, rebalance daily.

Between candle closes for a given TF, that TF's forecast is constant (carried forward). This means:

- Monthly TFPortfolio produces one new forecast per month.
- Weekly TFPortfolio produces one new forecast per week.
- Daily TFPortfolio produces one new forecast per day.
- On each rebalance tick, the latest forecast from each TF is combined.

---

## Alignment Grid

`_align_forecast_vectors_to_daily_grid` currently projects everything onto a daily grid. To support intraday rebalancing, this generalizes to an arbitrary-frequency grid:

1. Build the rebalance grid at the target frequency (daily, hourly, etc.).
2. For each `(ticker, model, timeframe)` stream: reindex to the grid, forward-fill, fill remaining NaN with 0.
3. Combine via fitted `WeightLayer` weights.

Higher-TF forecasts simply repeat (via forward-fill) across the finer grid until a new candle closes.

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
| Hourly    | ~252 × 6.5 ≈ 1638  | ~1966 bars                  |

The current default is `max_lookback + ceil(max_lookback * 0.2)`.

---

## Key Invariants

1. **No timeframe interleaving** — each TF is processed as a complete, independent series.
2. **Forward-fill bridges TFs** — higher-TF forecasts are carried forward on the rebalance grid.
3. **Fit on daily, predict on any grid** — weights are learned on a daily grid; at predict time the grid can be finer.
4. **Stateless predict** — size the warmup window from `max_lookback()` / `cold_rebuild_candle_count()`; no serialized bias node state.
5. **Forecast scores are additive** — the weighted sum across TFs (with FDM) produces the combined forecast.

---

**See also:** [[portfolio]], [[weight_layer]], [[live_multi_timeframe]], [[Cache/architecture]]
