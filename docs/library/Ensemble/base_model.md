# Base Model

> **What it is:** An ensemble of binning models — NOT a single model. One "base model" owns multiple binning models (one per param variant) and averages their outputs.

---

## Purpose

- Stability via **parameter smoothing**: averaging over a param neighborhood (e.g. RSI lookbacks 3, 4, 5) is more robust than betting on a single "optimal" value
- No performance weighting — simple mean across members
- Output: one combined signal per timestamp

> [!note] High correlation within ensemble is intentional
> Members like RSI_4 and RSI_5 may share 0.95+ correlation. This is **parameter smoothing**, not redundancy. We are not seeking independent signals.

---

## Member Selection

Members are chosen manually by the researcher after a multi-stage validation process:

1. **Permutation testing** ([[permutation_testing]]): all param combos tested individually; only those passing vector shuffle + pipeline permutation tests become candidates
2. **Walkforward stability** ([[walkforward]]): smoothed neighbor metric computed per fold; identifies which param regions are consistently strong across time
3. **Researcher review**: intersection of permutation pass + stable neighborhood forms the ensemble; pre-committed criteria prevent post-hoc rationalization

Typical ensemble size: **2–10 members**.

---

## Continuous Example: RSI Lookbacks

See [[continuous_binning]] for the binning pipeline.

- Members: RSI with lookbacks 2, 3, 4, 5 → four feature columns (`rsi_2_D`, `rsi_3_D`, …)
- Fit: each binning model fitted independently on `(feature_data_i, shared_vol_scaled_target)`
- Predict: each member outputs a position multiplier → `final_signal = mean(m1, m2, m3, m4)`

---

## Rule-Based Example: Param Variants

See [[rule_based]] for rule binning details.

- Members: breakout rule at periods 20, 40, 60 → three rule outputs (-1, 0, 1)
- Fit: per-member Sharpe and adjusted Sharpe computed on `(rule_output_i, target)`
- Predict: each member outputs a position multiplier → `final_signal = mean(m1, m2, m3)`

---

## Fit / Predict Workflow

**Fit**
1. For each member, obtain its feature series (from bias node / rule at that param)
2. Fit that member's binning model on `(feature_data_i, target)` — independently, no cross-member state

**Predict**
1. For each member, run `member.predict(feature_data_i, strategy=..., scaled=...)`
2. Aggregate: `final_signal = mean(member_outputs)`
3. Optional: clip averaged signal to binning bounds (e.g. `[-2, 2]`)

---

## Summary

| Aspect | Detail |
|---|---|
| Structure | N binning models (continuous or rule-based), one per param variant |
| Fit | Each model fitted independently on shared vol-scaled target |
| Predict | Mean of all member outputs |
| Correlation | High within-ensemble correlation is expected and correct |
| Size | 2–10 members, researcher-selected |

**Pipeline:** param variants → one binning model per variant → fit each → predict each → average

**See also:** [[continuous_binning]], [[rule_based]], [[weight_layer]]
