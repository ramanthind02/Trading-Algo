# Live Trading — Multi-Timeframe Operation

> **Scope:** Practical guide for running `GlobalPortfolio` in live/paper trading across multiple timeframes, including the candle fetch schedule, lookback strategy, and rebalance loop.

---

## Architecture

```
Scheduled candle fetch (per TF)
        │
        ▼
TFPortfolio.predict (with lookback candles)
        │
        ▼
Cache latest forecast per TF
        │
        ▼
Rebalance tick (lowest TF cadence)
        │
        ▼
Merge cached forecasts → GlobalPortfolio weights → position fractions
        │
        ▼
PositionSizer → broker execution
```

---

## Candle Fetch Schedule

Each timeframe has its own fetch trigger. Between triggers, its forecast is **constant** (carried forward from the last computation).

| Timeframe | Trigger | Example |
|-----------|---------|---------|
| Monthly   | First trading day after month close | 1st business day of month |
| Weekly    | First trading day after week close  | Sunday 18:00 EST / Monday pre-market |
| Daily     | After daily close | 00:00 EST (futures) or 16:00 EST (equities) |
| Hourly    | Every hour during RTH | On the hour |

On each trigger, fetch `max_lookback + buffer` candles for that TF, run `TFPortfolio.predict_base_model_vectors_from_candles`, and cache the result.

---

## Lookback Strategy

Bias nodes are stateful rolling-window calculators. Instead of persisting internal state across sessions (fragile), we re-warm from scratch on each prediction call by passing enough historical candles.

```python
# Determine max lookback across all bias nodes in a TFPortfolio
max_lookback = max(
    node.front_bad
    for ensemble in tf_portfolio.ensembles
    for base_model in ensemble.base_models.values()
    for node in base_model.bias_nodes  # however nodes are accessed
)

# Fetch with buffer
n_candles_to_fetch = int(max_lookback * 1.2)
```

Only the **last** prediction row matters for live trading. The lookback candles exist solely to warm up the bias nodes.

### Why not save bias node state?

| Approach | Pros | Cons |
|----------|------|------|
| Save state (serialize deques) | Skip lookback re-processing | Fragile across restarts/crashes; 50+ nodes × N tickers to serialize; hard to test; drift risk |
| Pass lookback candles | Pure, deterministic, trivially testable, no state management | Re-processes lookback candles each call (~ms cost) |

The lookback approach wins on simplicity. Daily-TF processing 300 candles through all bias nodes is sub-second.

---

## Rebalance Loop (Pseudocode)

```python
# On startup: fit or load fitted GlobalPortfolio
global_portfolio = load_fitted_global_portfolio()

# Cache for per-TF latest forecasts
cached_forecasts: Dict[TimeFrame, pd.DataFrame] = {}

# Per-TF fetch schedule
schedule = {
    TimeFrame.M: CronTrigger(day=1, hour=0),   # monthly
    TimeFrame.W: CronTrigger(day_of_week="mon", hour=0),  # weekly
    TimeFrame.D: CronTrigger(hour=0),           # daily
}

def on_tf_candle_close(tf: TimeFrame):
    """Triggered when new candles are available for a timeframe."""
    candles = fetch_candles(tf, n_bars=max_lookback_for[tf])
    vol_df = fetch_daily_volatility()

    vectors = global_portfolio.tf_portfolios[tf].predict_base_model_vectors_from_candles(
        candles, vol_df
    )
    # Keep only the latest prediction row per ticker
    cached_forecasts[tf] = (
        vectors.sort_values("datetime").groupby("ticker", as_index=False).last()
    )

def on_rebalance_tick():
    """Triggered at rebalance frequency (e.g., daily or hourly)."""
    if not cached_forecasts:
        return

    # Combine cached forecasts using fitted WeightLayer weights
    # This mirrors what GlobalPortfolio.predict does internally,
    # but avoids re-running all TFPortfolios from scratch.
    combined = combine_cached_forecasts(
        cached_forecasts,
        weight_layer=global_portfolio.weight_layer,
        instrument_weights=global_portfolio.instrument_weights,
        global_idm=global_portfolio.global_idm_,
    )
    execute_positions(combined)
```

---

## Startup Sequence

1. **Load fitted `GlobalPortfolio`** from persisted state (vault control files + fitted weights).
2. **Warm all TFs** — run `on_tf_candle_close` for every timeframe to populate `cached_forecasts`.
3. **Start the scheduler** — register per-TF triggers and the rebalance tick.
4. **First rebalance** — combine cached forecasts and send orders.

On restart after a crash, the same startup sequence runs. Because prediction is stateless (lookback candles re-warm the nodes), there is no lost state to recover.

---

## Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| Candle fetch fails for one TF | That TF's cached forecast goes stale | Carry forward last known forecast; alert operator; retry on next trigger |
| Candle fetch fails for all TFs | No rebalance possible | Hold existing positions; alert operator |
| Bias node error for one model | One base model missing from ensemble | Ensemble falls back to remaining models; `WeightLayer` handles missing model gracefully |
| Volatility data unavailable | Cannot scale forecasts | Block rebalance; alert operator (volatility is required) |

---

## Comparison: Backtest vs Live

| Aspect | Backtest | Live |
|--------|----------|------|
| Candle availability | Full history upfront | Fetched incrementally on schedule |
| Processing | Vectorized over full series | Lookback window → latest row |
| TF orchestration | Sequential per-TF, then merge | Scheduled per-TF, cached, merged on tick |
| State | None (pure function of candles) | None (re-warmed from lookback each call) |
| Rebalance grid | Built from full date range | Clock-driven (cron/scheduler) |

The prediction logic is **identical** in both paths — the only difference is how candles are sourced and when predict is called.

---

**See also:** [[multi_timeframe]], [[portfolio]], [[production]], [[bias_nodes/central_cache_architecture]]
