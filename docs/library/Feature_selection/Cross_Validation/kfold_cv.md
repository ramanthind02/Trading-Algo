# K-Fold Cross-Validation for Time-Series Features

> **Status:** Library specification
> **Role in pipeline:** Phase 2 — IS Permutation Screening diagnostic (Stages 1+2 fold consistency) and IS Stability Analysis (Stage 3 formalization). See [pipeline_overview.md](../pipeline_overview.md) for the full pipeline.
> **Last updated:** 2026-02-19

## What This Document Covers

K-fold cross-validation applied to time-series feature data, with mandatory purging and embargoing at all fold boundaries. This document covers:

1. How k-fold is applied in **Phase 2, Stages 1+2** to produce fold-by-fold permutation p-values (consistency diagnostic)
2. How k-fold formalizes **Phase 2, Stage 3** (IS walkforward stability) with correct boundary handling
3. Purging and embargoing requirements
4. How to choose k

This is **not** a replacement for sequential walk-forward validation (Phase 3). K-fold within the IS period is an IS diagnostic tool. The definitive robustness test is Phase 3. See [pipeline_overview.md](../pipeline_overview.md).

---

## Time-Series K-Fold Structure

Standard k-fold randomly shuffles data into folds, which is invalid for time-series data: a model trained on observations from the future will leak information into the past. Instead, folds are contiguous temporal blocks:

```
IS period: 2000 ─────────────────────────────── 2023

K=5 folds (equal-width temporal blocks):
  Fold 1: 2000–2004 (4.6 years)
  Fold 2: 2005–2009 (4.6 years)
  Fold 3: 2010–2014 (4.6 years)
  Fold 4: 2015–2019 (4.6 years)
  Fold 5: 2020–2023 (3.8 years)
```

Each fold's boundaries are fixed before any data is examined. The assignment of which fold is "OOS" in a given cross-validation pass is determined by the CV scheme, not by feature performance.

---

## Purging and Embargoing

**Why purging and embargoing are mandatory:**

Financial features are autocorrelated. Raw returns over adjacent windows overlap. Model predictions trained on data adjacent to the test set will exhibit information leakage even without explicit look-ahead, because the training labels (forward returns) for observations near the boundary overlap temporally with the test observations.

**Purging:** Remove training observations whose label (forward return) extends into the test period. If a label is computed over a 5-day window, then the last 5 training observations before the fold boundary are purged from the training set.

**Embargo:** Remove a buffer of observations from the start of the test fold (and optionally the end of the training fold). The embargo length should be at least equal to the maximum forward-return horizon used in label construction.

```
Without purging/embargo:
  Training: ────────────────────│ Fold boundary
  Test:                         │────────────────
  Overlap:                   ───┤─── (leakage zone)

With purging + embargo:
  Training (used): ─────────────│
  Purged (excluded):            ▓▓▓
  Embargo (excluded):               ▓▓▓
  Test (used):                          ────────────
```

**Embargo length rule of thumb:** Use `max(label_horizon, feature_lookback_max) + 1` bars. For daily data with a 20-day forward return label and RSI max lookback of 20, embargo = 21 trading days (~1 month).

---

## Application 1: Phase 2 Stages 1+2 — Fold-by-Fold Permutation Diagnostics

### Purpose

Currently Phase 2 Stages 1+2 run the permutation test once on the full IS period (2000–2023) and return a single p-value per param combo. This gives a scalar measure of significance. K-fold applied here gives a vector of k p-values, one per fold, answering a richer question:

> "Is this feature's signal consistent across sub-periods, or is it driven by a single regime?"

A feature with p ≤ 0.10 in 4 of 5 folds is more convincing than a feature with p = 0.02 on the full block but p > 0.30 in 3 of 5 individual folds.

### Procedure

For each param combo, for each fold (as OOS fold), run the Stage 1 vector shuffle permutation test:

```
For k in 1..K:
  - Training set: all folds EXCEPT fold k (with purging + embargo at each boundary)
  - Test set: fold k
  - Compute observed metric on training set feature vector
  - Shuffle feature vector N times, compute permuted metrics
  - p_k = fraction of permuted metrics ≥ observed

Aggregate p-values using Fisher's combined probability test:
  χ² = -2 × Σ ln(p_k)      (sum over k folds)
  χ² ~ chi-squared with 2k degrees of freedom under the null

OR use the harmonic mean p-value (HMP):
  HMP = k / Σ (1/p_k)
  (more conservative; controls FWER under arbitrary dependence)
```

**Output:** Combined p-value per param combo + the vector of per-fold p-values as a diagnostic.

### Interpretation

| Pattern | Interpretation |
|---|---|
| p_k ≤ 0.10 in all k folds | Strong, consistent signal across sub-periods |
| p_k ≤ 0.10 in majority of folds | Signal present but with some regime variation |
| p_k ≤ 0.10 in only 1 fold | Signal concentrated in one regime — treat with caution |
| p_k > 0.10 in all folds | No signal in any sub-period; feature should fail Stage 1 |

The fold-level p-values are **diagnostic only**. The feature-level gate (Stage 1+2 combined) remains the same: does ANY param pass at the aggregate level? The k-fold analysis provides the researcher with evidence quality context.

### Cost

K-fold permutation testing multiplies Stage 1 compute by k. With K=5 and N=200 permutations per fold, total permutations per param combo = 1000. This is acceptable for Stage 1 (vector shuffle only, no refitting). For Stage 2 (pipeline/candle shuffle), k-fold is expensive — apply to Stage 2 only for features of particular interest, not by default.

---

## Application 2: Phase 2 Stage 3 — IS Stability Analysis Formalization

### Current State

Stage 3 evaluates ALL param combos on non-overlapping IS sub-folds, where each fold is fit and evaluated on its own data — there is no held-out test set within each fold. The current implementation uses contiguous non-overlapping blocks without trimming at boundaries.

### Formalized K-Fold IS Stability

Replace raw non-overlapping block evaluation with proper purged k-fold:

```
IS period split into K non-overlapping folds (K=5 or K=8)
For each fold k:
  - Trim edge observations at each fold boundary by the label horizon
    (forward-return labels at the boundary overlap with adjacent folds)
  - Compute raw metric and neighbor-smoothed metric for all params on fold k's data
  - Record which params are in the top-K for this fold

Stability assessment:
  - How many folds does param p appear in the selected set?
  - If count ≥ min_folds_stable (default 3), param p is "IS-stable"
```

**Edge trimming at Stage 3 boundaries matters** because forward-return labels computed at fold boundaries overlap temporally with the adjacent fold's data. An observation at the end of fold k with a 20-day forward return label "sees" data that belongs to fold k+1. Including these boundary observations in fold k's metric computation introduces a mild cross-fold dependency. Trimming the last `label_horizon` observations from each fold's tail (and the first `label_horizon` from each fold's head if concerned about the leading edge) removes this dependency. This is distinct from the classical train/test purging described in Application 1 — here each fold is evaluated on its own data, not split into training and test sets.

### Fold Count Recommendations

| IS data length | Recommended K | Min fold size |
|---|---|---|
| 10–15 years daily | K = 4–5 | 2 years |
| 15–24 years daily | K = 5–8 | 2–3 years |
| > 24 years daily | K = 8–12 | 2 years |

Stage 3 requires each fold to have enough data for reliable neighbor-smoothed metric estimation. With RSI max lookback = 20 days, fold sizes of 2 years (~500 bars) are adequate. With longer-lookback features (EWMAC-256, lookback = 512 days), fold sizes should be ≥ 3 years.

---

## Choosing K

**Too small K (K=2–3):** Very few IS folds; stability assessment has little resolution. With K=2, you can only observe "consistent" or "inconsistent" — not a spectrum.

**Too large K (K=15+):** Each fold is small → unreliable metric estimates per fold → noisy stability signal. Permutation tests per fold have low power with few observations.

**Recommended starting points:**
- Daily data, 24-year IS period: K = 5 (≈ 4.6-year folds) or K = 8 (≈ 3-year folds)
- Daily data, 15-year IS period: K = 4 or K = 5 (≈ 3-year folds)

---

## Limitations

1. **Reduced power per fold.** Each fold has 1/k of the IS data. Permutation tests have less data to distinguish signal from noise. The fold-level p-values will be noisier than a single full-IS p-value.

2. **Correlation between folds.** Adjacent folds share market conditions (similar macro regime, volatility regime). Fold-level p-values are not independent. Fisher's combined test and HMP are more robust than naive combination under this correlation.

3. **K-fold is NOT live-trading simulation.** When fold k is held out, training uses data BEFORE and AFTER fold k (in non-expanding configurations). This does not mirror production. K-fold is an IS diagnostic tool. Sequential expanding WF (Phase 3) is the live-trading simulation.

---

## Related Documentation

| Document | Relationship |
|---|---|
| [pipeline_overview.md](../pipeline_overview.md) | Master pipeline spec; see Phases 2 and 3 |
| [in-sample_pt.md](../Permutation Testing/in-sample_pt.md) | Stage 1+2+3 specifications; k-fold integrates here |
| [cpcv.md](cpcv.md) | Combinatorial Purged CV — alternative for Stage 3 with more paths |
| [param_selection_rule.md](../Parameter%20Sensitivity/param_selection_rule.md) | Pre-committed selection rule used inside CV folds |
| [grid_search_parameter_stability.md](../Parameter%20Sensitivity/grid_search_parameter_stability.md) | Neighbor smoothing theory |
