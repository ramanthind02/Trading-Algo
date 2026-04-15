# Bias nodes — library index

> [!summary]
> All bias-node documentation lives under this folder: implementing indicators, composing nodes, and tying research-time ideas to production-safe signed signals.

## Start here

| Doc | Purpose |
|-----|---------|
| [[bias_nodes/creating_nodes]] | Required attributes, templates, Cython, multi-ticker side channels, lookback metadata, checklists |
| [[bias_nodes/composed_nodes]] | Confirmation gates (`DualSignalNode`) — two signals must agree |
| [[bias_nodes/bias_node_arch]] | Redirect stub → use [[bias_nodes/creating_nodes]] |

## Where this sits in the stack

```text
OHLCV → Bias node(s) → (optional composition) → Research → Native discrete signal feature → Ensemble / vault
```

- **Default:** Features are **bias nodes**. Outputs are either already discrete (`-1` / `0` / `+1`) or real-valued **continuous** indicators.
- **Continuous path:** Use the feature-research pipeline to study the raw series. If the idea graduates to production, implement a separate native signed-signal node rather than a bin-wrapper.
- **Legacy note:** Runtime-fitted **continuous binning models** (`QuantileBinningModel`, etc.) are **discontinued** for new work. Do not add new features that rely on production-time bin geometry fitting.

## Related (outside this folder)

- [[Feature_selection/pipeline]] — selection phases after the feature exists as a discrete bias-node column
- [[Cache/user_guide]] — central cache for candles and bias artifacts
- [[Vault/user_guide]] — persisting frozen feature definitions
- [[Data/norgate]] — OHLCV sourcing (data layer, not a bias node doc)
