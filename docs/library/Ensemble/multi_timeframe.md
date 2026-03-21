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
GlobalPortfolio.fit(
    candles_per_tf={TimeFrame.D: daily_candles, TimeFrame.W: weekly_candles},
    instrument_returns=returns_df,
    daily_volatility_df=vol_df,
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
positions = global_portfolio.predict(
    candles_per_tf={TimeFrame.D: daily_candles, TimeFrame.W: weekly_candles},
    daily_volatility_df=vol_df,
)
```

---

## Backtesting

Backtesting is straightforward because all candle history is available upfront.

```python
forecasts_by_tf: Dict[TimeFrame, pd.DataFrame] = {}

# Process each timeframe independently — no interleaving
for tf in [TimeFrame.M, TimeFrame.W, TimeFrame.D]:
    tf_candles = all_candles[all_candles['timeframe'] == tf]
    forecasts_by_tf[tf] = tf_portfolio[tf].predict_base_model_vectors_from_candles(
        tf_candles, daily_volatility_df
    )

# GlobalPortfolio handles alignment and combination internally
positions = global_portfolio.predict(candles_per_tf, daily_volatility_df)
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

Bias nodes are stateful — they use rolling `deque` buffers that require warm-up candles before producing valid outputs. Each node has a `front_bad` period (typically equal to its `lookback` parameter) during which it outputs neutral values.

### Design choice: stateless prediction via lookback window

Rather than serializing/deserializing bias node state (fragile across restarts, hard to test), the caller provides enough historical candles to warm up all nodes from scratch.

```
max_lookback = max(node.front_bad for all bias nodes in all ensembles for this TF)
```

The prediction path processes all `max_lookback` candles through the bias nodes, but only the **tail** of the resulting forecast series is used. This is:

- **Pure/deterministic** — same candles always produce the same output.
- **Trivially testable** — no hidden state to manage.
- **Fast enough** — processing 252 daily candles through 50+ bias nodes takes milliseconds.

| Timeframe | Typical max lookback | Candles to fetch |
|-----------|---------------------|-----------------|
| Monthly   | ~24 bars (2 years)  | ~30 bars        |
| Weekly    | ~52 bars (1 year)   | ~60 bars        |
| Daily     | ~252 bars (1 year)  | ~300 bars       |
| Hourly    | ~252 × 6.5 ≈ 1638  | ~1700 bars      |

Add ~20% buffer above max lookback to be safe.

---

## Key Invariants

1. **No timeframe interleaving** — each TF is processed as a complete, independent series.
2. **Forward-fill bridges TFs** — higher-TF forecasts are carried forward on the rebalance grid.
3. **Fit on daily, predict on any grid** — weights are learned on a daily grid; at predict time the grid can be finer.
4. **Stateless predict** — pass enough lookback candles; no serialized bias node state.
5. **Forecast scores are additive** — the weighted sum across TFs (with FDM) produces the combined forecast.

---

**See also:** [[portfolio]], [[weight_layer]], [[live_multi_timeframe]]
