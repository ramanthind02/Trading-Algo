# Base Feature (Spec)

A **base feature** is the combination of a **bias node** and a **set of parameters**. The bias node turns raw market data (e.g. candles) into a **feature vector**; the parameters (e.g. lookback, period, threshold) fully define one variant of that feature. Both **continuous** and **rule-based** feature types are base features — they differ only in the **output format** of the bias node and thus which binning pipeline is applied. This doc defines the base feature concept and the **shared** Sharpe-based normalization and position-scaling methods used by both pipelines.

---

## Definition

| Component | Description |
|-----------|-------------|
| **Bias node** | A stateful component that processes candles and produces one or more output series. See [Bias Node Specification Guide](../../../bias_nodes/base_bias_node_specs.md) for implementation. |
| **Params** | A fixed set of parameters (e.g. `lookback=14`, `period=20`) that define a single variant. Column naming includes params so that e.g. `rsi_signal_D_lookback_14` and `rsi_signal_D_lookback_21` are distinct features. |

**Base feature** = one bias node + one param set → one **feature column** (e.g. continuous values or -1/0/1).

- **Continuous base feature:** Bias node outputs a continuous series (e.g. RSI 0–100). Downstream: [Continuous_binning](Continuous_binning.md) (quantile bins, regions, threshold filter, then position scaling).
- **Rule-based base feature:** Bias node outputs discrete -1, 0, 1. Downstream: [rule_based](rule_based.md) (three “bins,” no quantile binning, only position scaling).

---

## Shared: Target Variable

For **all** base features (continuous and rule-based), the **target** used for fit must be **vol-scaled returns**:

- **Definition:** e.g. `return / EWSD` or `return / ATR` per observation (same index as the feature).
- **Purpose:** Unitless, comparable across different markets and assets when training on multiple symbols in aggregate. Ensures Sharpe and adjusted Sharpe are on a common scale.

If the target is not vol-scaled, cross-asset and cross-feature comparison of edge strength is invalid.

---

## Shared: Sharpe and Adjusted Sharpe

Both continuous and rule-based pipelines use the same **per-bin** (or per-level) statistics and the same **position multiplier** formula.

### Per-bin / per-level statistics

For each bin (continuous) or level -1/0/1 (rule-based), compute on the **vol-scaled target** (returns where the feature falls in that bin/level):

- `mean_return` = mean of target in that bin/level  
- `volatility` = std of target in that bin/level  
- **Sharpe** = `mean_return / volatility` (optionally annualized). Unitless and comparable across features and markets when target is vol-scaled.  
- **Adjusted Sharpe** (shrinkage by sample size):  
  **`adjusted_Sharpe = Sharpe * sqrt(N) / (sqrt(N) + k)`**  
  - `N` = number of observations in the bin/level.  
  - `k` = shrinkage constant (e.g. 20). Bins/levels with more samples are trusted more; small-N are shrunk toward zero.

### Position multiplier (shared formula)

- **Long:** `position_multiplier = clip(1 + adjusted_Sharpe, 0.5, 2.0)` → output in [0.5, 2.0].  
- **Short:** Invert: `position_multiplier = -clip(1 + adjusted_Sharpe, 0.5, 2.0)` → output in [-2.0, -0.5].  
- **Long_short:** Both long and short bins/levels can be active; **output range is -2 to 2**.

No per-feature rescaling. Clipping is risk control; baseline exposure = 1, max tilt = 2.

### Why this is shared

- **Unitless:** Sharpe is already normalized by volatility.  
- **Comparable across features:** Same scale for RSI, momentum, rule-based breakouts, etc.  
- **Sample-size aware:** Adjusted Sharpe down-weights thin bins/levels.  
- **Bounded:** Prevents unbounded position size.

---

## Fit / predict (shared contract)

- **Fit:** Inputs are (feature_data, target_data, optional normalization_data). Target must be vol-scaled. Model fits bins/levels, computes Sharpe and adjusted Sharpe, and stores thresholds and multipliers.  
- **Predict:** Input is feature_data (and optional strategy, normalization_data, scaled). Output is a **continuous position multiplier** (or 0 for flat) in the range [-2, 2] when both long and short are used.

Details of **how** bins are formed (quantile vs three levels) and filtered (regions, t-stat, metric threshold) are in [Continuous_binning](Continuous_binning.md) and [rule_based](rule_based.md).

---

## Summary

- **Base feature** = bias node + param set → one feature column. See [Bias Node Specification Guide](../../../bias_nodes/base_bias_node_specs.md) for node design (rule-based vs continuous output, normalization, etc.).  
- **Shared across continuous and rule-based:**  
  - **Target:** Vol-scaled returns.  
  - **Per-bin/level:** Sharpe and **adjusted_Sharpe** = Sharpe × sqrt(N)/(sqrt(N)+k).  
  - **Position multiplier:** Long = clip(1 + adjusted_Sharpe, 0.5, 2.0); short = negative of same; long_short range **-2 to 2**.  
- **Continuous path:** [Continuous_binning](Continuous_binning.md). **Rule-based path:** [rule_based](rule_based.md).
