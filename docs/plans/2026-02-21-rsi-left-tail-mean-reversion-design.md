# RSI Left-Tail Mean-Reversion Design

## Goal
Create dedicated RSI-derived bias nodes tuned for short-horizon (2-10 period) left-tail behavior in stock-index mean reversion.

## Context
- User research indicates strongest edge in RSI left tail for short RSI periods.
- Existing library already contains standard RSI, cumulative RSI, percentile/zscore variants, and generic `ts_feature` wrappers.
- We want explicit, discoverable, standalone nodes rather than ad-hoc transform wrappers.

## Proposed Nodes
1. **LaggedRSI**
   - Output: RSI value lagged by `lag` bars (continuous 0-100).
   - Purpose: delayed oversold information often aligns with next-bar/next-few-bar reversion.

2. **RSILeftTailPressure**
   - Output: average positive shortfall below a threshold over trailing window.
   - Formula (window size `tail_window`, threshold `tail_level`):
     - `pressure_t = mean(max(0, tail_level - rsi_{t-i}) for i in 0..tail_window-1)`
   - Purpose: capture intensity/depth of persistent oversold conditions.

3. **RSILeftTailStreak**
   - Output: normalized consecutive count of bars with RSI <= threshold.
   - Formula: `streak / max_streak` capped to [0, 1].
   - Purpose: encode persistence duration of oversold regime.

4. **RSIReboundVelocity**
   - Output: short-horizon rebound in RSI from recent local tail minimum.
   - Formula:
     - `velocity_t = rsi_t - min(rsi_{t-rebound_window+1..t})`
   - Purpose: measure post-oversold snapback speed.

## Architecture
- Each new node is a standalone `BiasNode` in `nodes/mean_reversion/rsi/`.
- Reuse RSI incremental state flow and Cython-backed helpers from `utils.compute.fast_nodes` for base RSI updates.
- Add lightweight rolling state (deques or fixed arrays) per node for each engineered transformation.
- Keep outputs continuous and deterministic; no lookahead.

## Naming And Contracts
- `module_name` values remain snake_case to preserve parser consistency.
- One output feature per node: `signal`.
- Parameter names in `params` use existing camelCase convention.
- Warmup outputs:
  - RSI-scale features: `50.0`
  - Pressure/velocity features: `0.0`
  - Streak ratio: `0.0`

## Testing Strategy
- Unit tests under `tests/unit-tests/nodes/`:
  - Warmup neutral output checks.
  - Deterministic shape/range checks for each node.
  - Behavior checks for synthetic down-then-up paths (tail pressure rises in selloff; rebound velocity rises after bounce).

## Out Of Scope
- No new rule-based entry/exit policy nodes.
- No integration-pipeline cache tests in this change.
- No refactor of existing RSI variants.
