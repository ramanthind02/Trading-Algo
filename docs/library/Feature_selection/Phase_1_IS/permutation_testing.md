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

**Null hypothesis:** The temporal ordering of the feature vector carries no predictive information — the observed metric is consistent with random reassignment of the same signal values across time. Equivalently: a strategy with identical time-in-market and position-size distribution but random entry/exit timing performs as well as the original.

**Input:** The fitted feature vector (binned continuous or rule-based output). No pipeline refit.

**Method:**
- Randomly permute the feature vector — same values, new temporal order. Breaks feature–target alignment while preserving marginal distribution.
- Compute objective metric (e.g. Sharpe) on the shuffled series.
- Run 500–1000 replicates to build null distribution.
- **Pass condition:** original metric beats the `(1 − α₁)` quantile of the null. Default `α₁ = 0.10` (90th percentile; lax to reduce false negatives).

**Scope:** Applied per parameter combination independently. Params that fail Stage 1 are **excluded from Stage 2** (early stopping).

> [!note] **Why early stopping is correct:** Stage 2 (candle shuffle) is a *stricter* null than Stage 1 — it destroys everything the vector shuffle destroys, plus serial autocorrelation and price process structure. A feature that cannot beat random timing of its own signals cannot pass a harder test. Early stopping therefore reduces false negatives, not false positives, and saves the compute cost of Stage 2 on clear noise.

---

## Stage 2 — Pipeline Permutation Test

**Null hypothesis:** The strategy's observed metric is consistent with what a random temporal ordering of the market's bar-structure would produce — the entire data-generating process (price dynamics → feature computation → signal → returns) could have arisen by chance. More precisely: any arrangement of bars with the same marginal bar-shape and gap distributions (but random temporal sequence) would produce this result.

This is a strictly stronger null than Stage 1. Stage 1 asks "is the signal assigned at the right times?"; Stage 2 asks "does the price process itself contain genuine exploitable structure, or would any randomly ordered market with the same statistical fingerprint produce this result?"

Only Stage 1 passers proceed here. Runs the full pipeline under the null.

> [!important] **Revised pipeline order:** Stage 3 (IS stability) now runs **before** Stage 2. Stage 2 only runs on the stable region identified in Stage 3 (typically 3–10 params), making compute viable regardless of original grid size. See the Summary Flow below.

### Continuous Features

Two options (candle shuffle is default):

| Option | What is shuffled | When to use |
|--------|-----------------|-------------|
| **Candle shuffle** (default) | Bars reconstructed; feature recomputed from shuffled stream | Default — tests price process structure |
| Raw feature shuffle + refit | Continuous feature series before binning; binning pipeline refit on each replicate | Cheaper alternative for features with a wide Stage 1 margin |

For candle shuffle, see [[candle_permutation]] for the algorithm. The full [[pipeline|binning pipeline]] runs on the shuffled input with the **pre-specified metric threshold**. If no valid bins satisfy the threshold, that replicate scores 0.

### Rule-Based Features

Rule-based outputs are serially correlated — shuffling the feature vector produces unnatural sequences, and candle shuffle (while valid) is expensive. The preferred approach is a **target return shuffle**.

**Target return shuffle procedure:**

1. Reserve the last ~20% of the IS period as `IS_val` (a within-IS holdout; real, unshuffled returns).
2. For each replicate:
   a. Randomly shuffle the `IS_train` return column. The feature is kept **unchanged**.
   b. Run the stability algorithm on `IS_train` with shuffled returns → identify stable region (if any).
   c. Apply an **absolute metric threshold** gate: if no region passes the minimum raw metric, score = 0.
   d. Select best-region params → evaluate on `IS_val` (original returns) → record metric.
3. Score = 0 if no stable region found.
4. Build null distribution of `IS_val` metrics across replicates. Compare original `IS_val` metric against the `(1 − α₂)` quantile.

**Why absolute metric threshold is critical:** with shuffled returns, the best metric across params will be near zero. Floor-based selection (`best × (1 − δ)`) sets a near-zero floor and will admit spurious "stable" regions. An absolute floor (e.g. minimum Sharpe ≥ 0.1) prevents this inflation of the null.

**H₀ (rule-based):** The feature's predictive relationship with returns is consistent with a completely random target — the IS stability algorithm cannot identify a region that genuinely generalises to `IS_val` returns.

### Pass Condition

Same logic as Stage 1: original metric beats `(1 − α₂)` quantile of null. Default `α₂ = 0.05` (stricter than Stage 1). Applied per param combo. Params that fail Stage 2 are excluded from the ensemble, but all params still enter the OOS walkforward for neighbor smoothing.

---

## Stage 3 — IS Walkforward Stability

> [!important] Stage 3 is NOT a permutation test. It is a temporal stability analysis. See [[walkforward]] for the full OOS walkforward (Phase 3), which is the separate definitive robustness test.

> [!note] **Revised pipeline order:** Stage 3 now runs **before** Stage 2. Its output — the stable parameter region — constrains which params enter the expensive Stage 2 test. This is why Stage 3 still requires the full param grid (see below): the smoothing must happen over all params, not just Stage 1 passers.

**Purpose:** Assess whether the optimal parameter region is consistent across different IS time periods, and identify the stable neighborhood to pass to Stage 2.

**All params enter Stage 3** — including those that failed Stage 1. Neighbor smoothing requires the full parameter grid; a sparse filtered grid produces meaningless averages.

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

## What to Shuffle: IS vs. OOS Contexts

The answer depends on what question you are asking and whether parameters are fixed or free.

| Context | What to shuffle | Rationale |
|---------|----------------|-----------|
| Phase 1 IS — Stage 1 | Feature vector (full IS period) | Params already fitted; tests signal ordering only |
| Phase 1 IS — Stage 2 (continuous) | IS candles (or raw feature series); refit pipeline on each replicate | Params free; must account for fitting degrees of freedom |
| Phase 1 IS — Stage 2 (rule-based) | IS_train return column; evaluate on IS_val holdout (last ~20% of IS) | Preserves serial correlation; tests IS→IS_val generalisation under random target |
| Phase 3 OOS — fixed system | OOS candles only | Params locked; shuffle only the evaluation period |
| WF Phase 5 — sub-stage 1 | OOS fold returns only; signals and params fixed | Tests whether OOS performance is genuine given fixed strategy (cheapest) |
| WF Phase 5 — sub-stage 2 | IS fold returns; full param grid; OOS returns real | Tests whether IS fitting genuinely identifies predictive params; full grid required |
| WF Phase 5 — sub-stage 3 | Full WF candle series (IS + OOS); feature recomputed | Tests whether a random market could produce this result end-to-end (strongest null) |

**Key principle:** if parameters are being fitted as part of the replicate, shuffle all the data that feeds into the fitting. If parameters are fixed and you are evaluating a deployed system, shuffle only the evaluation period. Shuffling IS data when parameters are already locked is a no-op that blurs the test.

---

## Summary Flow

```
Stage 1 (all params)           — vector shuffle, p ≤ α₁=0.10, fast, no refit
    ↓ passers only
Stage 3 (ALL params)           — IS stability, neighbor smoothing; identifies stable region
    ↓
Researcher review of EDA       — manually restrict to stable neighborhood (5–10 params)
                                  using param sensitivity heatmap + Stage 3 output
    ↓
Stage 2 (stable region only)   — continuous: candle shuffle (or raw feature shuffle + refit)
                                  rule-based: target return shuffle + IS_val evaluation
                                  p ≤ α₂=0.05; score=0 if no stable region found
    ↓
Researcher review              — feature-level decision; all params enter OOS walkforward
    ↓
Phase 3: OOS Walkforward       — definitive test (see [[walkforward]])
```

> [!note] The researcher manual restriction step (between Stage 3 and Stage 2) is intentionally not automated. It requires inspecting the param sensitivity landscape (EDA Phase 1) and Stage 3 stability output to identify a coherent stable neighbourhood. This judgment prevents mechanical over-restriction and ensures the chosen region makes structural sense.
