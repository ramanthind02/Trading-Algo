# Base Feature

> [!summary] Definition
> A **base feature** = one [[creating_nodes|bias node]] + one param set → one feature column.
> Column naming encodes params: e.g. `rsi_signal_D_lookback_14` vs `rsi_signal_D_lookback_21` are distinct features.

## Two Feature Types

| Type | Bias node output | Downstream pipeline |
|------|-----------------|---------------------|
| **Continuous** | Continuous series (e.g. RSI 0–100) | [[continuous_binning]] |
| **Rule-based** | Discrete -1 / 0 / +1 | [[rule_based]] |

---

## Shared: Target Variable

- Must be **vol-scaled returns**: e.g. `return / EWSD` or `return / ATR` per observation
- **Unitless** — enables valid Sharpe comparison across assets and features
- Without vol-scaling, cross-asset edge comparison is invalid

---

## Shared: Per-Bin / Per-Level Statistics

Computed on vol-scaled target within each bin (continuous) or level (rule-based):

- `mean_return` — mean of target in bin/level
- `volatility` — std of target in bin/level
- `sharpe` = `mean_return / volatility`
- **Adjusted Sharpe** (shrinkage by sample size):

$$\text{adjusted\_Sharpe} = \text{Sharpe} \times \frac{\sqrt{N}}{\sqrt{N} + k}$$

> [!note] Shrinkage constant `k` (e.g. 20)
> Small-N bins are shrunk toward zero. Large-N bins are trusted more.

---

## Shared: Position Multiplier Formula

| Direction | Formula | Output range |
|-----------|---------|-------------|
| **Long** | `clip(1 + adjusted_Sharpe, 0.5, 2.0)` | [0.5, 2.0] |
| **Short** | `-clip(1 + adjusted_Sharpe, 0.5, 2.0)` | [-2.0, -0.5] |
| **Long_short** | Both long and short bins active | **[-2, 2]** |

- No per-feature rescaling
- Baseline exposure = 1.0; max tilt = 2.0; clipping is risk control

---

## Fit / Predict Contract (Shared)

**Fit:**
- Inputs: `feature_data`, `target_data` (vol-scaled), optional `normalization_data`
- Fits bins/levels, computes Sharpe + adjusted Sharpe, stores thresholds and multipliers

**Predict:**
- Input: `feature_data` (+ optional `strategy`, `normalization_data`, `scaled`)
- Output: continuous position multiplier in **[-2, 2]** for `long_short`; 0 = flat

---

> [!info] See also
> - [[continuous_binning]] — quantile grid search, tail/hump detection, coverage bonus
> - [[rule_based]] — three fixed levels (-1, 0, +1), no binning step
> - [[creating_nodes]] — bias node design and output format specs
