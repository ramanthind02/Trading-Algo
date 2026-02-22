# In-Sample Permutation Testing

> [!note] Role: **Phase 2** of the full validation pipeline — a coarse IS computational filter (full IS period, e.g. 2000–2023). Feeds into [[walkforward]] (Phase 3), which is the definitive robustness test.

Relates to: [[pipeline]] | [[candle_permutation]] | [[kfold]]

---

## Purpose and Design Constraints

> *"Does this feature class have any stable, exploitable signal? If yes, graduate it to the definitive walkforward. If no, discard it cheaply."*

**Critical design decisions:**

1. **Feature-level graduation, not param-level.** The per-param pass/fail verdict is *diagnostic only*. The binary gate is: "did ANY param pass?" If yes, the feature advances. **All param combos enter the walkforward regardless of IS permutation result** — pre-filtering corrupts the neighbor smoothing in walkforward training folds.

2. **Full IS period maximises statistical power.** Permutation tests compare the feature against shuffled versions of itself, not a hold-out set — so using all IS data is valid.

3. **Coarse filter only.** Features that fail here would almost certainly fail OOS walkforward. The purpose is to avoid paying the compute cost of walkforward on obvious noise.

4. **No per-fold permutation inside walkforward.** Within each walkforward training fold, use neighbor smoothing + metric threshold (cheap). Full IS permutation runs once here.

---

## Stage 1 — Vector Shuffle Test

**Input:** The fitted feature vector (binned continuous or rule-based output). No pipeline refit.

**Method:**
- Randomly permute the feature vector — same values, new temporal order. Breaks feature–target alignment while preserving marginal distribution.
- Compute objective metric (e.g. Sharpe) on the shuffled series.
- Run 500–1000 replicates to build null distribution.
- **Pass condition:** original metric beats the `(1 − α₁)` quantile of the null. Default `α₁ = 0.10` (90th percentile; lax to reduce false negatives).

**Scope:** Applied per parameter combination independently. Params that fail Stage 1 are **excluded from Stage 2** (early stopping).

---

## Stage 2 — Pipeline Permutation Test

Only Stage 1 passers proceed here. Runs the full pipeline under the null.

### Continuous Features

Two modes (candle shuffle is recommended):

| Mode | What is shuffled | Null destroyed |
|------|-----------------|----------------|
| Shuffle raw feature | The raw bias-node series before binning | Feature–target link |
| **Shuffle candles** (recommended) | The bars in time; feature recomputed from shuffled stream | Feature–target link AND temporal/serial structure |

Full [[pipeline|binning pipeline]] runs on the shuffled input with the **pre-specified metric threshold** (set before seeing any data). If no valid bins satisfy the threshold, that replicate scores 0.

### Rule-Based Features

**Shuffle bars (candles)** — not the feature vector. Rule-based outputs are serially correlated; shuffling the vector would produce unnatural sequences. Feed the shuffled candle stream into all param-combo rule models.

See [[candle_permutation]] for the candle shuffling algorithm.

### Pass Condition

Same logic as Stage 1: original metric beats `(1 − α₂)` quantile of null. Default `α₂ = 0.05` (stricter than Stage 1). Applied per param combo. Params that fail Stage 2 are excluded from Stage 3.

---

## Stage 3 — IS Walkforward Stability

> [!important] Stage 3 is NOT a permutation test. It is a temporal stability analysis. See [[walkforward]] for the full OOS walkforward (Phase 3), which is the separate definitive robustness test.

**Purpose:** Assess whether the optimal parameter region is consistent across different IS time periods.

**All params enter Stage 3** — including those that failed Stages 1 and 2. Neighbor smoothing requires the full parameter grid; a sparse filtered grid produces meaningless averages.

**Procedure per fold (non-overlapping IS folds, e.g. 2-year chunks):**
1. Compute objective metric for ALL params on that fold's data.
2. Compute smoothed neighbor metric: `smoothed(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])`.
3. Select top-K params by smoothed objective (K = 3 recommended).

**Stable feature:** Same parameter neighborhood (e.g. `{4, 5, 3}`) appears in top-K across most folds.

**Unstable feature:** Top-K jumps across parameter space fold-to-fold → likely noise or regime-dependent → reject.

k-fold with boundary trimming or CPCV can be applied within Stage 3 for richer diagnostic coverage. See [[kfold]] and `cpcv.md`.

---

## Feature-Level Graduation

After Stages 1–3, the researcher reviews:
- Which params passed Stages 1 + 2 (statistical significance).
- The IS stability report from Stage 3 (temporal consistency).

**Feature verdict:** If any param shows signal AND the stable region makes structural sense → graduate to OOS walkforward. **All param combos enter the walkforward** (not just IS passers).

Typical ensemble formed from params that are both statistically significant (Stages 1–2) AND temporally stable (Stage 3): 2–10 members from a consistent parameter neighborhood.

---

## Summary Flow

```
Stage 1 (all params)        — vector shuffle, p ≤ α₁=0.10, fast, no refit
    ↓ passers only
Stage 2 (S1 passers)        — pipeline/candle permutation, p ≤ α₂=0.05, expensive
    ↓ passers only (but all params used for smoothing below)
Stage 3 (ALL params)        — IS walkforward stability, neighbor smoothing, no permutation
    ↓
Researcher review           — feature-level decision, all params enter OOS walkforward
    ↓
Phase 3: OOS Walkforward    — definitive test (see [[walkforward]])
```
