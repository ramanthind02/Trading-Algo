# Base Model (Spec) — Ensemble of Binning Models

The **base model** is always an **ensemble of binning models**. It does not own a single binning model; it owns **multiple** binning models (e.g. one per parametrization), fits each on its feature and the shared target, and **averages their outputs** at predict time. This applies to both **continuous** and **rule-based** feature types.

---

## Purpose

- **Ensemble:** One “base model” = many binning models (e.g. RSI with lookbacks 2, 3, 4, 5; or a rule with several param variants).
- **Aggregation:** Final signal = **mean** of the per-model outputs. No weighting by performance — simple average.
- **Stability:** Averaging across parametrizations smooths noise and reduces dependence on a single lookback or rule param.

---

## Composition

| Component | Description |
|-----------|-------------|
| **Member models** | Each member is a binning model (continuous or rule-based) with a **fixed parametrization** (e.g. RSI lookback 3, or rule threshold 0.6). |
| **Member selection** | Members are selected **manually by the researcher** after reviewing permutation test results and walkforward stability analysis (see [in-sample permutation testing §4](../permutation_testing/in-sample_pt.md#4-researcher-ensemble-formation)). |
| **Ensemble size** | Typically **2–10 members**. Researcher chooses based on stability report. |
| **Feature per member** | Each member has its own **feature series** (e.g. RSI_2, RSI_3, RSI_4, RSI_5). Same underlying idea, different params. |
| **Target** | **Single vol-scaled target** shared by all members (e.g. same return series for all). |
| **Output** | For each timestamp, **average** the position multipliers (or raw predictions) from all members. No weighting by performance — simple average. |

---

## Ensemble Member Selection

**How members are chosen:**

Ensemble members are selected **manually by the researcher** after a multi-stage validation process:

1. **Permutation testing (§1–§2):** Test all param combos individually. Only params that pass vector shuffle and pipeline permutation tests are candidates.
2. **Walkforward stability analysis (§3):** Evaluate all params on each walkforward fold. Compute smoothed neighbor metric per fold. Identify which param regions are consistently "best" across time periods.
3. **Researcher review:** The researcher examines which params passed permutation tests AND appear in a stable neighborhood across folds. The ensemble is formed from this intersection.

**Rationale:**
- **Regularization:** Averaging over a parameter neighborhood (e.g. RSI 3-5) is more robust than betting on a single "optimal" param (e.g. RSI 4 exactly).
- **Stability-informed:** The walkforward stability analysis reveals whether a good neighborhood exists. If the best param region is consistent across folds, the ensemble is well-supported. If it jumps around, the feature may not be worth deploying.
- **Researcher judgment:** The researcher brings domain knowledge and hypothesis context to the ensemble decision. Pre-committed stability criteria (defined before seeing results) prevent post-hoc rationalization.

**High correlation within ensemble is intentional:** Members like RSI_4 and RSI_5 may have 0.95+ correlation. This is not redundancy — it's **parameter smoothing**. We're not seeking independent signals; we're averaging over a stable region to avoid overfitting to a single param value.

For details, see:
- [Researcher Ensemble Formation](../permutation_testing/in-sample_pt.md#4-researcher-ensemble-formation)
- [Walkforward Stability Analysis](../permutation_testing/in-sample_pt.md#3-walkforward-stability-analysis)
- [Grid-Aware Neighbor Averaging](../../../to-do/grid_neighbor_smoothing_specs.md) (used as diagnostic tool in walkforward stability)

---

## Continuous Example: RSI Lookbacks

- **Members:** 4 binning models, each with [Continuous_binning](../feature_types/Continuous_binning.md) pipeline.
- **Features:** RSI with lookback 2, 3, 4, 5 → four feature columns (e.g. `rsi_2_D`, `rsi_3_D`, …).
- **Fit:** For each member, fit its binning model on `(feature_data_i, target)`.
- **Predict:** For each member, get position multiplier (range e.g. -2 to 2 in long_short). **Output = mean(member_1, member_2, member_3, member_4).**
- Result: One combined signal that averages over lookbacks instead of picking one.

---

## Rule-Based Example: Rule Param Variants

- **Members:** Several rule-based binning models, each with a different param (e.g. breakout period 20, 40, 60). See [rule_based](../feature_types/rule_based.md).
- **Features:** Rule output -1, 0, 1 for each param variant → one “feature” per member (or one column per variant).
- **Target:** Same vol-scaled returns for all.
- **Fit:** For each member, fit on `(rule_output_i, target)`; compute per-level Sharpe and adjusted Sharpe.
- **Predict:** Each member outputs a position multiplier; **output = mean over members.**

Same idea as continuous: parametrized variants, then average.

---

## Fit

1. For each **member** (param variant):
   - Obtain that member’s **feature** series (from bias node / rule with that param).
   - Fit that member’s **binning model** on `(feature_data, target)`. Target is vol-scaled and shared.
2. Store fitted binning models (and any thresholds, regions, scaling params) per member. No cross-member state — each model is fitted independently.

---

## Predict

1. For each **member**, run `member.predict(feature_data, strategy=..., scaled=...)` with that member’s feature series.
2. **Aggregate:** `final_signal = mean(member_1_output, member_2_output, ...)`.
3. Optional: apply a final clip to the averaged signal (e.g. keep in [-2, 2] if using the same bounds as the underlying binning specs).

---

## Summary

- **Base model = ensemble of binning models** (continuous or rule-based), each with its own parametrization and feature.
- **Fit:** Fit each binning model on (feature_i, shared vol-scaled target).
- **Predict:** Average the outputs of all members.
- **Continuous:** e.g. RSI lookbacks 2,3,4,5 → four binning models → average. See [Continuous_binning](../feature_types/Continuous_binning.md).
- **Rule-based:** e.g. rule with params 20, 40, 60 → three binning models → average. See [rule_based](../feature_types/rule_based.md).

Pipeline: **param variants → one binning model per variant → fit each → predict each → average.**
