# Grid Search Parameter Stability Analysis

**Purpose:** Conceptual foundation and decision framework for identifying stable parameter regions in grid search results. This document explains *why* parameter stability matters, *how* to assess it, and *when* to trust it.

**Scope:** Knowledge reference and source of truth for parameter stability concepts. For implementation details, see [Grid-Aware Neighbor Averaging (Task Spec)](../../../to-do/grid_neighbor_smoothing_specs.md).

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Conceptual Framework](#2-conceptual-framework)
3. [Algorithm Comparison](#3-algorithm-comparison)
4. [Integration with Validation Pipeline](#4-integration-with-validation-pipeline)
5. [Interpretation Guidelines](#5-interpretation-guidelines)
6. [Decision Rules for Ensemble Formation](#6-decision-rules-for-ensemble-formation)
7. [Trading-Specific Examples](#7-trading-specific-examples)
8. [Limitations and Edge Cases](#8-limitations-and-edge-cases)
9. [References and Related Concepts](#9-references-and-related-concepts)

---

## 1. Overview and Motivation

### Why Parameter Stability Matters

When testing a parametrized trading feature (e.g., RSI with lookback = 2, 3, 4, ..., 20), researchers often face a critical question:

> "This parameter combination has the best in-sample performance. Should I trust it?"

**The problem:** Parameter overfitting is ubiquitous in trading system development. A parameter may perform well for two reasons:

1. **Genuine signal**: The parameter captures a real market inefficiency (stable across time and nearby parameter values)
2. **Lucky noise**: The parameter happened to align with random fluctuations in the sample period (isolated peak, breaks down out-of-sample)

Traditional approaches focus on **statistical significance** (permutation tests, p-values). This answers: "Is this parameter better than chance?" But it doesn't answer: "Will this parameter generalize?"

**Parameter stability analysis** addresses the generalization question by asking: **"Is this parameter in a stable region, or is it an isolated spike?"**

### The Isolated Peak Problem

Consider an RSI mean reversion feature tested on lookback values [2, 3, 4, 5, 10, 14, 20]:

**Scenario A: Stable region (good)**
```
Lookback:    2     3     4     5    10    14    20
Sharpe:    0.6   0.9   1.1   1.0   0.5   0.3   0.1
            └─── stable region ───┘
```
→ Lookback 4 has the best Sharpe (1.1), but its neighbors (3, 5) also perform well. This suggests a **genuine signal**: short-term mean reversion is robust across a range of lookbacks.

**Scenario B: Isolated peak (bad)**
```
Lookback:    2     3     4     5    10    14    20
Sharpe:    0.1   0.2   1.1   0.3   0.2   0.1   0.0
                        ↑
                   isolated spike
```
→ Lookback 4 still has the best Sharpe (1.1), but its neighbors perform poorly. This suggests **overfitting**: lookback 4 happened to align with noise, but the signal isn't robust.

**Key insight:** In Scenario B, even if lookback 4 passes a permutation test (statistically significant), it's likely to fail out-of-sample because the edge isn't stable.

### How This Fits Into the Validation Framework

Parameter stability analysis is **Stage 3** of the feature validation pipeline (see [In-Sample Permutation Testing](../permutation_testing/in-sample_pt.md)):

1. **Stage 1–2: Permutation tests** → Validate individual param combos (statistical significance)
2. **Stage 3: Walkforward stability analysis** → Assess temporal consistency of optimal param regions
3. **Stage 4: Researcher ensemble formation** → Manually select params that are both significant AND stable

Permutation tests filter out statistical noise. Stability analysis filters out parameter overfitting. Together, they identify features that are **both real and robust**.

---

## 2. Conceptual Framework

### What is a "Stable Parameter Region"?

A **stable parameter region** is a set of nearby parameter values that all exhibit good performance. Formally:

> A parameter P is in a stable region if its **1-step neighbors** (parameters differing by one grid step in one dimension) also have objective metrics close to P's metric.

**Visual intuition (1D):**
```
     Stable region               Isolated peak

  Obj │    ╱‾‾‾╲                Obj │     ╱╲
      │   ╱     ╲                   │    ╱  ╲
      │  ╱       ╲                  │   ╱    ╲
      │ ╱         ╲                 │__╱______╲___
      └──────────────> Param       └──────────────> Param

  Wide plateau with               Narrow spike that
  gradual changes                 drops off sharply
```

**Mathematical definition:**
- Let `obj(P)` = objective metric at parameter P
- Let `neighbors(P)` = all 1-step axis-aligned neighbors of P in the grid
- Define **smoothed objective**: `smoothed(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])`
- **Stability ratio**: `stability(P) = smoothed(P) / obj(P)`

A parameter is **stable** if `stability(P) ≈ 1` (smoothed ≈ raw). It's **unstable** if `stability(P) << 1` (smoothed much lower than raw).

### Relationship to Robustness and Generalization

**Robustness** = performance doesn't degrade significantly when assumptions are violated

**Generalization** = in-sample performance carries over to out-of-sample data

**Stability → Robustness → Generalization:**

1. **Stable parameters are robust to parameter misspecification**
   - If the true optimal lookback is 4.5, but you can only test integers, a stable region (3, 4, 5) means you're close to optimal regardless
   - Isolated peaks are fragile: if the true optimal is 4.1, testing lookback 4 vs 5 makes a huge difference

2. **Stable parameters are robust to regime changes**
   - If the optimal parameter shifts slightly over time (e.g., lookback 4 in 2015, lookback 5 in 2020), a stable region hedges against this
   - Isolated peaks break immediately when the regime shifts

3. **Stable parameters generalize better**
   - Empirically, parameters in stable regions tend to maintain performance out-of-sample
   - Isolated peaks are often curve-fitted to sample-specific noise

**Bayesian interpretation:** A stable region provides **more evidence** for a genuine effect than an isolated peak. If many nearby parameters work well, it's less likely they all got lucky independently.

### Why Stability Predicts Out-of-Sample Performance

**Occam's Razor argument:**
- A stable region implies a **simple** underlying relationship (e.g., "mean reversion works for short lookbacks 3–7")
- An isolated peak implies a **complex** relationship (e.g., "mean reversion works ONLY for lookback 4, and fails for 3 or 5")
- Simpler relationships are more likely to be real and to persist OOS

**Noise structure argument:**
- Random noise in performance metrics is typically **high-frequency** (varies rapidly across parameter values)
- Genuine signal is typically **low-frequency** (smooth, changes gradually across parameter space)
- Smoothing filters out high-frequency noise while preserving low-frequency signal
- If a parameter survives smoothing (stability ratio ≈ 1), its performance is likely signal, not noise

**Empirical evidence:**
- Walkforward stability analysis (Stage 3) tests this directly: if the same param region is optimal across multiple time folds, it's stable over time
- Features that pass walkforward stability tests have higher OOS success rates than those that don't (anecdotal but consistent)

---

## 3. Algorithm Comparison

This section compares **four approaches** to assessing parameter stability. The comparison assumes the context of **hypothesis-driven feature testing** with **small parameter grids** (5–20 param combos per dimension).

### Grid-Aware Neighbor Averaging (Recommended)

**How it works:**

For each parameter combination P in a grid, compute the **smoothed objective** as the mean of:
- The objective at P itself
- The objective at all **1-step axis-aligned neighbors** of P

**1-step axis-aligned neighbor** = differs in exactly one parameter dimension by one grid step (previous or next value in the ordered parameter list).

Example (2D grid):
```
      Threshold
        30   50   70
L  7   A    B    C
o 14   D    E    F
o 21   G    H    I
k

For cell E (lookback=14, threshold=50):
  - Neighbors: B (7,50), H (21,50), D (14,30), F (14,70)
  - Smoothed = mean(E, B, H, D, F)
```

**Strengths:**

| Strength | Explanation |
|----------|-------------|
| **Respects grid structure** | Uses only actual tested parameter combinations (no interpolation or extrapolation) |
| **Interpretable** | Smoothed metric has clear meaning: "How good is this param if I average with its neighbors?" |
| **Computationally cheap** | O(N × D) where N = # grid points, D = # dimensions. Scales well to large grids. |
| **Boundary-aware** | Naturally handles boundary params (fewer neighbors → less smoothing, which is correct) |
| **Dimensionality-agnostic** | Works for 1D, 2D, 3D+

 grids without modification |

**Weaknesses:**

| Weakness | Explanation |
|----------|-------------|
| **Requires regular/semi-regular grid** | Assumes parameter values are ordered and adjacency is meaningful. Breaks down for highly irregular grids (e.g., [1, 2, 100, 1000]). |
| **Partial grids reduce smoothing** | If many neighbors are missing, smoothing degenerates (becomes closer to raw objective). In extreme cases, isolated cells have no neighbors → no smoothing. |
| **No distance weighting** | All neighbors contribute equally, even if grid spacing is non-uniform. A neighbor at +1 and a neighbor at +10 are treated the same. |

**When to use:**
- Hypothesis-driven testing with structured parameter grids (most common use case)
- Parameter values are ordered and somewhat evenly spaced (e.g., lookback [5, 10, 15, 20, 30])
- Grid is reasonably complete (not too many missing cells)

---


## 4. Integration with Validation Pipeline

Parameter stability analysis integrates into the **three-stage validation pipeline** as follows:

### Stage 1–2: Permutation Tests (Statistical Significance)

**Purpose:** Filter out param combos that are statistically indistinguishable from chance

**Method:**
- **Stage 1 (Vector Shuffle):** Shuffle feature vector, compare objective to null. Fast filter.
- **Stage 2 (Pipeline Permutation):** For continuous: shuffle raw feature or candles, run full binning pipeline. For rule-based: shuffle candles, feed to rule models. Rigorous test.

**Output:** List of param combos that **pass** permutation tests (objective beats 90th percentile of null, α = 0.1)

**What this tells you:** These params have statistically significant edge (not just luck)

**What this doesn't tell you:** Whether the edge is robust (stable across parameter space and time)

### Stage 3: Walkforward Stability Analysis (Temporal Consistency)

**Purpose:** Assess whether the optimal parameter region is stable across time

**Method:**
1. Divide in-sample data into **non-overlapping walkforward folds** (e.g., 2-year chunks: 2015-2016, 2017-2018, ...)
2. For **each fold independently**:
   - Evaluate ALL param combos on that fold's data (not just those that passed Stage 1–2)
   - Compute **smoothed neighbor metric** for each param (grid-aware neighbor averaging)
   - Select **top K params** by smoothed objective (e.g., K = 3)
3. **Compare across folds:** Are the same param combos consistently in the top-K?

**Output:** Stability report containing:
- Per-fold top-K param selections (with smoothed objectives)
- Overlay: which params also passed permutation tests
- Assessment: stable (consistent top-K) vs unstable (jumping around)

**What this tells you:** Whether the optimal param region persists across different time periods

**Why use ALL params, not just permutation test passers:**
- Neighbor smoothing needs the full grid to compute meaningful averages
- A param that narrowly failed permutation tests might still be in a stable region (useful context for the researcher)
- The researcher will only select from params that passed BOTH tests (permutation + stability)

### Stage 4: Researcher Ensemble Formation (Manual Selection)

**Purpose:** Combine individually-validated, temporally-stable params into an ensemble

**Method:**
1. Researcher reviews:
   - Which params passed permutation tests (Stage 1–2)
   - Walkforward stability report (Stage 3)
2. **Selection criteria** (pre-committed before seeing results):
   - Param must have passed permutation tests (statistical significance)
   - Param must appear in or near the top-K across 3+ of 5 folds (temporal stability)
   - Params should form a tight neighborhood (not scattered across parameter space)
3. Form ensemble: 2–10 members from the stable region
4. Final signal = mean(member outputs)

**Example decision process:**
```
RSI lookback grid: [2, 3, 4, 5, 10, 14, 20]

Stage 1–2 output (passed permutation tests):
  ✅ lookback 3 (Sharpe = 0.9, p < 0.1)
  ✅ lookback 4 (Sharpe = 1.1, p < 0.05)
  ✅ lookback 5 (Sharpe = 1.0, p < 0.1)
  ✅ lookback 14 (Sharpe = 0.8, p < 0.1)
  ❌ lookback 2, 10, 20 (failed permutation tests)

Stage 3 output (walkforward stability, top-3 per fold):
  Fold 1 (2015-2016): {4, 5, 3}  → center ≈ 4
  Fold 2 (2017-2018): {5, 4, 10} → center ≈ 5
  Fold 3 (2019-2020): {4, 5, 3}  → center ≈ 4
  Fold 4 (2021-2022): {5, 4, 14} → center ≈ 5

Researcher decision:
  - lookback 3, 4, 5: ✅ Passed permutation AND consistently in top-3 across folds → INCLUDE
  - lookback 14: ❌ Passed permutation BUT only in top-3 in 1 fold (2021-2022) → EXCLUDE
  - lookback 10: ❌ In top-3 in fold 2 BUT failed permutation test → EXCLUDE

Final ensemble: {3, 4, 5} → mean(RSI_3, RSI_4, RSI_5)
```

**Key principle:** The ensemble is formed from params that are **both statistically significant (Stage 1–2) AND temporally stable (Stage 3)**.

### How Smoothed Metric is Used

The **smoothed neighbor metric** serves two purposes in Stage 3:

1. **Ranking params within each fold:** Instead of ranking by raw objective (which rewards isolated peaks), rank by smoothed objective (which rewards stable regions). This naturally filters out overfitted params.

2. **Consistency check across folds:** If the same smoothed param is top-ranked across multiple folds, it's robust. If different params are top-ranked in each fold, the feature is unstable.

**Why smoothing matters:**
- Without smoothing, an isolated peak in fold 1 might be top-ranked, then disappear in fold 2 (unstable)
- With smoothing, isolated peaks are penalized (low smoothed objective), so only stable regions consistently rank high

---

## 5. Interpretation Guidelines

How to interpret the **raw objective** vs **smoothed objective** for a parameter combination:

### The Four Quadrants

|                     | **High Smoothed Objective** | **Low Smoothed Objective** |
|---------------------|----------------------------|---------------------------|
| **High Raw Objective** | ✅ **Excellent candidate**<br>Strong param in stable region | ⚠️ **Suspicious**<br>Isolated peak, likely overfit |
| **Low Raw Objective**  | ⚠️ **Mediocre but stable**<br>Acceptable ensemble member | ❌ **Reject**<br>Poor param in poor region |

### Quadrant 1: High Raw, High Smoothed (✅ Excellent)

**Example:**
```
Lookback: 4
Raw objective (Sharpe): 1.1
Neighbors: {3: 0.9, 5: 1.0}
Smoothed objective: (1.1 + 0.9 + 1.0) / 3 = 1.0
Stability ratio: 1.0 / 1.1 = 0.91
```

**Interpretation:**
- This param performs well (Sharpe = 1.1)
- Its neighbors also perform well (Sharpe 0.9–1.0)
- High stability ratio (0.91) indicates a stable region

**Action:** ✅ **Strong candidate for ensemble**. This is the ideal scenario — genuine signal, robust.

---

### Quadrant 2: High Raw, Low Smoothed (⚠️ Suspicious)

**Example:**
```
Lookback: 14
Raw objective (Sharpe): 1.1
Neighbors: {10: 0.2, 20: 0.3}
Smoothed objective: (1.1 + 0.2 + 0.3) / 3 = 0.53
Stability ratio: 0.53 / 1.1 = 0.48
```

**Interpretation:**
- This param performs well (Sharpe = 1.1)
- But its neighbors perform poorly (Sharpe 0.2–0.3)
- Low stability ratio (0.48) indicates an isolated peak

**Action:** ⚠️ **Suspicious, likely overfit**. Even if it passed a permutation test, it's probably not robust. Consider:
- Checking if there's a theoretical reason why *only* lookback 14 works (unlikely)
- Rejecting this param unless you have strong prior evidence
- If deployed, monitor closely for OOS degradation

---

### Quadrant 3: Low Raw, High Smoothed (⚠️ Mediocre but Stable)

**Example:**
```
Lookback: 5
Raw objective (Sharpe): 0.7
Neighbors: {4: 1.1, 3: 0.9}
Smoothed objective: (0.7 + 1.1 + 0.9) / 3 = 0.9
Stability ratio: 0.9 / 0.7 = 1.29 (smoothed > raw!)
```

**Interpretation:**
- This param has mediocre performance (Sharpe = 0.7)
- But its neighbors perform well (Sharpe 0.9–1.1)
- High stability ratio (1.29, even > 1) indicates it's in a good neighborhood

**Action:** ⚠️ **Acceptable ensemble member**. This param is "riding the coattails" of its neighbors. It's not the best param, but it's in a stable region. Consider:
- Including it in the ensemble if neighbors (4, 3) are also included (it provides regularization)
- Skipping it if you want a smaller ensemble (focus on the best params in the region)

**Why smoothed > raw is possible:** If a param is surrounded by better neighbors, averaging pulls the smoothed objective *up*. This is actually a good sign — the param is in a "valley" within a stable plateau.

---

### Quadrant 4: Low Raw, Low Smoothed (❌ Reject)

**Example:**
```
Lookback: 20
Raw objective (Sharpe): 0.1
Neighbors: {14: 0.3, (no neighbor at 30 in grid)}
Smoothed objective: (0.1 + 0.3) / 2 = 0.2
Stability ratio: 0.2 / 0.1 = 2.0 (but both are low!)
```

**Interpretation:**
- This param performs poorly (Sharpe = 0.1)
- Its neighbors also perform poorly (Sharpe 0.3)
- Even though stability ratio is high, *absolute* smoothed objective is low

**Action:** ❌ **Reject**. This param is in a poor region. Stability doesn't help if the entire region is bad.

**Important caveat:** A high stability ratio with low absolute smoothed objective means "consistently poor", not "good". Always check the absolute value, not just the ratio.

---

### Quantitative Thresholds

**Stability ratio** = `smoothed_objective / raw_objective`

| Stability Ratio | Interpretation | Action |
|----------------|----------------|--------|
| **> 0.9** | Very stable (smoothed ≈ raw) | ✅ Strong candidate if raw is good |
| **0.7 – 0.9** | Moderately stable | ✅ Acceptable if raw is good |
| **0.5 – 0.7** | Weakly stable | ⚠️ Caution, check neighbors |
| **< 0.5** | Isolated peak | ❌ Likely overfit, reject |

**Absolute smoothed objective:**
- Always check that `smoothed_objective > threshold` (e.g., Sharpe > 0.5)
- A high stability ratio with low absolute smoothed objective is still a reject

**Combined decision rule:**
```
IF raw_objective > threshold AND stability_ratio > 0.8:
    → Excellent candidate
ELIF raw_objective > threshold AND stability_ratio > 0.6:
    → Acceptable candidate (monitor)
ELSE:
    → Reject or scrutinize further
```

---

### Visual Interpretation (1D Example)

```
Raw objective:
Sharpe │         B
       │        /|\
       │       / | \        D
       │   A  /  |  \      /|\
       │   |\/   |   \    / | \
       │___/\____|____\__/____|____
           2  3  4  5 10 14 20    Lookback

Smoothed objective:
Sharpe │      ___
       │     /   \___
       │    /        \___
       │   /             \___
       │__/                  \____
           2  3  4  5 10 14 20    Lookback

Interpretation:
  - A (lookback 3): moderate raw, high smoothed → in stable region around 4
  - B (lookback 4): high raw, high smoothed → peak of stable region ✅
  - C (lookback 5): moderate raw, high smoothed → in stable region around 4
  - D (lookback 14): high raw, low smoothed → isolated peak ⚠️
```

**Key visual cue:** Smoothing "flattens" the landscape. Broad plateaus remain high; narrow spikes get reduced. Look for params where smoothed ≈ raw.

---

## 6. Decision Rules for Ensemble Formation

After completing Stage 3 (Walkforward Stability Analysis), the researcher uses these decision rules to form the ensemble.

### Pre-Committed Stability Criteria

**Principle:** Define what "stable" means *before* seeing the walkforward results. This prevents post-hoc rationalization.

**Example criteria** (choose one or combine):

1. **Top-K consistency:**
   - "At least 3 of 5 folds must select params within a 3-step neighborhood"
   - Example: If folds select {4,5,3}, {5,4,10}, {4,5,3}, {5,4,14}, the consistent neighborhood is {3,4,5} (appears in 3+ folds)

2. **Smoothed objective threshold:**
   - "Selected params must have smoothed objective within 10% of raw objective"
   - Example: If raw Sharpe = 1.0, smoothed Sharpe must be ≥ 0.9

3. **Neighborhood tightness:**
   - "All ensemble members must be within 2 grid steps of each other"
   - Example: If best param is lookback 4, only include lookback {2,3,4,5,6}

4. **Stability ratio minimum:**
   - "Selected params must have stability ratio ≥ 0.8"
   - Example: smoothed / raw ≥ 0.8 for all ensemble members

**Recommendation:** Use criterion 1 (top-K consistency) + criterion 4 (stability ratio ≥ 0.8). This ensures params are both temporally stable AND locally stable.

---

### How to Use the Walkforward Stability Report

The walkforward stability report (output of Stage 3) shows:
- Per-fold top-K param selections
- Per-fold smoothed objective landscape
- Overlay: which params passed permutation tests

**Three possible outcomes:**

#### Outcome 1: Consistent Top-K (✅ Stable Feature, Deploy)

**Example:**
```
RSI lookback grid: [2, 3, 4, 5, 10, 14, 20]

Fold 1: top-3 = {4, 5, 3}   smoothed = {1.0, 0.95, 0.90}
Fold 2: top-3 = {5, 4, 3}   smoothed = {1.05, 1.00, 0.88}
Fold 3: top-3 = {4, 3, 5}   smoothed = {1.02, 0.92, 0.98}
Fold 4: top-3 = {4, 5, 10}  smoothed = {1.01, 0.96, 0.70}
Fold 5: top-3 = {5, 4, 3}   smoothed = {1.03, 0.99, 0.91}

Permutation test passers: {3, 4, 5, 14}
```

**Interpretation:**
- Lookback {3, 4, 5} consistently appear in top-3 across all folds
- Lookback 10 appears once (fold 4), but with lower smoothed objective (0.70 vs ~1.0 for others)
- Lookback 14 passed permutation test but never appears in top-3

**Decision:** ✅ **Deploy ensemble = {3, 4, 5}**
- These params are both statistically significant (passed permutation) AND temporally stable (top-3 in all folds)
- Exclude 10 (not consistently top) and 14 (never top, likely isolated peak in full in-sample data)

**Deployment param selection:** Since the top-K are consistent, use **full in-sample data** to compute final smoothed objectives and select the stable region {3, 4, 5}.

---

#### Outcome 2: Gradual Drift (⚠️ Use Recency Weighting)

**Example:**
```
EWMAC (spanFast, spanSlow) grid: (8,32), (16,64), (20,80), (32,128)

Fold 1 (2015-2016): top-3 = {(16,64), (20,80), (8,32)}   → slower MAs
Fold 2 (2017-2018): top-3 = {(16,64), (20,80), (32,128)} → slower MAs
Fold 3 (2019-2020): top-3 = {(20,80), (16,64), (32,128)} → mixed
Fold 4 (2021-2022): top-3 = {(20,80), (32,128), (16,64)} → faster MAs
Fold 5 (2023-2024): top-3 = {(32,128), (20,80), (16,64)} → faster MAs

Permutation test passers: all params
```

**Interpretation:**
- The optimal MA speeds are gradually increasing over time (slower MAs in early folds, faster MAs in recent folds)
- No single param region is consistently best
- This suggests a **regime shift** (markets becoming faster/more efficient?)

**Decision:** ⚠️ **Deploy with recency bias**
- Option A: Use only the most recent fold's top-K: {(32,128), (20,80)}
- Option B: Weight folds by recency, select params that are top-ranked in recent folds
- **Do NOT** use full in-sample data (would average over the regime shift and pick a mediocre middle ground)

**Risk:** If the drift reverses (markets slow down again), the ensemble will underperform. Monitor closely.

---

#### Outcome 3: Unstable (❌ Feature is Noise, Reject)

**Example:**
```
Breakout rule (lookback, threshold) grid

Fold 1: top-3 = {(10, 2.0), (20, 1.5), (5, 2.5)}
Fold 2: top-3 = {(40, 1.0), (60, 0.8), (30, 1.2)}
Fold 3: top-3 = {(15, 1.8), (10, 2.0), (20, 1.5)}
Fold 4: top-3 = {(50, 1.0), (40, 1.1), (60, 0.9)}
Fold 5: top-3 = {(20, 1.5), (25, 1.4), (15, 1.6)}

Permutation test passers: many params (overfitting?)
```

**Interpretation:**
- Top params are jumping all over the parameter space (short/long lookbacks, high/low thresholds)
- No consistent pattern or neighborhood
- This suggests the feature is **fitting to noise** in each fold

**Decision:** ❌ **Reject the feature entirely**
- Even if some params passed permutation tests, the lack of temporal stability indicates overfitting
- Revisit the hypothesis: maybe the breakout rule itself is flawed, or the thresholds/lookbacks are poorly chosen

**Alternative:** If you have strong theoretical reasons to believe in the feature, try:
- Narrowing the parameter grid (maybe you're testing too wide a range)
- Testing on a different asset or timeframe
- Checking for data quality issues (maybe folds 2 and 4 had bad data?)

---

### Ensemble Size Heuristics

**How many members should the ensemble have?**

| Ensemble Size | When to Use |
|---------------|-------------|
| **2–3 members** | Very stable feature, clear top param (e.g., lookback 4 dominates, include 3 and 5 for regularization) |
| **4–7 members** | Moderately stable, broader good region (e.g., lookback 3–7 all work well) |
| **8–10 members** | Large stable region, or high-dimensional grid (e.g., 2D grid with multiple good neighborhoods) |
| **> 10 members** | Rarely justified; if you have this many, either (1) your grid is too coarse or (2) you're including too many mediocre params |

**Principle:** The ensemble should be **as small as possible while capturing the stable region**. Don't include params just to increase ensemble size.

**Example:**
```
Stable region: lookback {3, 4, 5, 6, 7}

Option A: Ensemble = {4, 5} (2 members)
  → Focuses on the best params, but might be sensitive to param choice

Option B: Ensemble = {3, 4, 5, 6, 7} (5 members)
  → Captures the full stable region, more robust to param misspecification

Option C: Ensemble = {2, 3, 4, 5, 6, 7, 10} (7 members)
  → Includes boundary params (2, 10) that might not be in the stable region → unnecessary

Recommendation: Option B (5 members). Captures the stable region without diluting with boundary params.
```

---

## 7. Trading-Specific Examples

### Example 1: RSI Mean Reversion (1D Parameter: Lookback)

**Hypothesis:** Oversold RSI levels predict short-term rebounds (mean reversion).

**Parameter grid:** RSI lookback = [2, 3, 4, 5, 7, 10, 14, 20, 30]

**Stage 1–2 results (Permutation tests on full in-sample data):**

| Lookback | Raw Sharpe | Permutation p-value | Pass? |
|----------|------------|---------------------|-------|
| 2 | 0.50 | 0.15 | ❌ (p > 0.10) |
| 3 | 0.85 | 0.05 | ✅ |
| 4 | 1.10 | 0.02 | ✅ |
| 5 | 0.95 | 0.04 | ✅ |
| 7 | 0.60 | 0.12 | ❌ |
| 10 | 0.40 | 0.25 | ❌ |
| 14 | 0.85 | 0.06 | ✅ |
| 20 | 0.20 | 0.40 | ❌ |
| 30 | 0.10 | 0.50 | ❌ |

**Stage 3 results (Walkforward stability analysis):**

Compute smoothed objectives per fold:

**Fold 1 (2015-2016):**

| Lookback | Raw Sharpe | Neighbors | Smoothed Sharpe | Stability Ratio |
|----------|------------|-----------|-----------------|-----------------|
| 2 | 0.55 | {3} | (0.55+0.90)/2 = **0.73** | 1.33 |
| 3 | 0.90 | {2,4} | (0.90+0.55+1.15)/3 = **0.87** | 0.97 |
| 4 | 1.15 | {3,5} | (1.15+0.90+1.00)/3 = **1.02** | 0.89 |
| 5 | 1.00 | {4,7} | (1.00+1.15+0.65)/3 = **0.93** | 0.93 |
| 7 | 0.65 | {5,10} | (0.65+1.00+0.45)/3 = **0.70** | 1.08 |
| 10 | 0.45 | {7,14} | (0.45+0.65+0.80)/3 = **0.63** | 1.40 |
| 14 | 0.80 | {10,20} | (0.80+0.45+0.25)/3 = **0.50** | 0.63 |
| 20 | 0.25 | {14,30} | (0.25+0.80+0.15)/3 = **0.40** | 1.60 |
| 30 | 0.15 | {20} | (0.15+0.25)/2 = **0.20** | 1.33 |

**Top-3 by smoothed Sharpe:** {4, 5, 3} (smoothed = 1.02, 0.93, 0.87)

**Fold 2 (2017-2018):**

| Lookback | Smoothed Sharpe | Top-3? |
|----------|-----------------|--------|
| 3 | 0.92 | ✅ (3rd) |
| 4 | 0.98 | ✅ (2nd) |
| 5 | 1.05 | ✅ (1st) |
| 7 | 0.75 | |
| 10 | 0.55 | |
| 14 | 0.48 | |

**Fold 3 (2019-2020):**

| Lookback | Smoothed Sharpe | Top-3? |
|----------|-----------------|--------|
| 3 | 0.88 | ✅ (2nd) |
| 4 | 1.00 | ✅ (1st) |
| 5 | 0.90 | ✅ (3rd) |
| 7 | 0.68 | |
| 10 | 0.50 | |
| 14 | 1.10 | ❌ (isolated spike in this fold) |

**Fold 4 (2021-2022):**

| Lookback | Smoothed Sharpe | Top-3? |
|----------|-----------------|--------|
| 3 | 0.85 | ✅ (3rd) |
| 4 | 1.05 | ✅ (1st) |
| 5 | 0.92 | ✅ (2nd) |
| 14 | 0.52 | |

**Summary across folds:**

| Lookback | Appears in top-3 | Passed permutation? | Decision |
|----------|------------------|---------------------|----------|
| 3 | 4 of 4 folds | ✅ | ✅ **INCLUDE** |
| 4 | 4 of 4 folds | ✅ | ✅ **INCLUDE** |
| 5 | 4 of 4 folds | ✅ | ✅ **INCLUDE** |
| 7 | 0 of 4 folds | ❌ | ❌ Exclude |
| 14 | 1 of 4 folds | ✅ | ❌ **Exclude (isolated peak in full data, unstable in walkforward)** |

**Final ensemble:** RSI lookback {3, 4, 5}

**Interpretation:**
- **Stable region identified:** Lookback 3–5 form a stable neighborhood with consistent performance across time
- **Isolated peak rejected:** Lookback 14 passed permutation tests on full in-sample data (Sharpe = 0.85), but walkforward shows it's unstable (only good in fold 3)
- **Deployment strategy:** Use full in-sample data to fit the ensemble (stable region is consistent across folds)

---

### Example 2: EWMAC Crossover (2D Parameters: spanFast, spanSlow)

**Hypothesis:** Exponential moving average crossovers capture trend-following signals.

**Parameter grid:**
- spanFast: [8, 16, 20, 32]
- spanSlow: [32, 64, 80, 128]
- Total: 16 combinations (4×4 grid)

**Stage 1–2 results (Permutation tests):**

Passed permutation tests: (8,32), (16,64), (20,80), (32,128), (16,80), (20,64)

**Stage 3 results (Walkforward, abbreviated):**

**Fold 1 (2015-2016) — Top-3 by smoothed Sharpe:**

1. (16, 64): smoothed = 1.20 (neighbors: (8,64)=0.9, (20,64)=1.1, (16,32)=0.8, (16,80)=1.3)
2. (20, 80): smoothed = 1.15
3. (16, 80): smoothed = 1.10

**Fold 2 (2017-2018):**

1. (16, 64): smoothed = 1.18
2. (20, 80): smoothed = 1.12
3. (32, 128): smoothed = 0.95

**Fold 3 (2019-2020):**

1. (20, 80): smoothed = 1.22
2. (16, 64): smoothed = 1.15
3. (20, 64): smoothed = 1.00

**Fold 4 (2021-2022):**

1. (16, 64): smoothed = 1.25
2. (20, 80): smoothed = 1.20
3. (16, 80): smoothed = 1.05

**Summary:**

| Param Combo | Appears in top-3 | Passed permutation? | Decision |
|-------------|------------------|---------------------|----------|
| (16, 64) | 4 of 4 folds | ✅ | ✅ **INCLUDE** |
| (20, 80) | 4 of 4 folds | ✅ | ✅ **INCLUDE** |
| (16, 80) | 2 of 4 folds | ✅ | ✅ **INCLUDE (borderline)** |
| (20, 64) | 1 of 4 folds | ✅ | ⚠️ Exclude (less consistent) |
| (32, 128) | 1 of 4 folds | ✅ | ⚠️ Exclude |

**Visualization of stable region (2D grid):**

```
       spanSlow
         32   64   80  128
   8    [0.6] 0.9  0.7  0.4
  16    [0.8] 1.2  1.1  0.6    ← (16,64) and (16,80) are neighbors
s 20    [0.7] 1.0  1.2  0.8    ← (20,64) and (20,80) are neighbors
p 32    [0.5] 0.6  0.7  0.9
a
n
F
a
s
t

[brackets] = failed permutation tests
Bold = top-3 in most folds

Stable region: {(16,64), (16,80), (20,64), (20,80)}
  → All are 1-step neighbors of each other
  → Form a 2×2 sub-grid in the center
```

**Final ensemble:** (16,64), (20,80), (16,80) — **3 members from the stable 2×2 region**
- Include (16,64) and (20,80): top-3 in all folds
- Include (16,80): top-3 in 2 folds, bridges the other two (regularization)
- Exclude (20,64): only top-3 in 1 fold (less consistent than the others)

**Interpretation:**
- **Stable region in 2D:** The classic (16,64) crossover and its neighbors form a stable region
- **Boundary params excluded:** (8,32) and (32,128) are at the edges, less stable
- **Deployment:** 3-member ensemble averages over the stable region for robustness

---

### Example 3: Breakout Rule (2D Parameters: Lookback, Threshold)

**Hypothesis:** Breakouts above a rolling high predict continuation.

**Parameter grid:**
- lookback: [10, 20, 30, 40, 60]
- threshold (% above rolling high): [0.5%, 1.0%, 1.5%, 2.0%, 2.5%]
- Total: 25 combinations (5×5 grid)

**Stage 3 results (Walkforward, hypothetical unstable case):**

**Fold 1:** Top-3 = {(10, 2.0%), (20, 1.5%), (30, 1.0%)}
**Fold 2:** Top-3 = {(40, 0.5%), (60, 1.0%), (40, 1.0%)}
**Fold 3:** Top-3 = {(20, 2.5%), (10, 2.0%), (20, 2.0%)}
**Fold 4:** Top-3 = {(60, 0.5%), (40, 1.0%), (30, 0.5%)}

**Interpretation:**
- **No consistent pattern:** Short lookbacks with high thresholds in fold 1, long lookbacks with low thresholds in fold 2, mixed in folds 3-4
- **Jumping across parameter space:** No stable neighborhood emerges
- **Likely noise:** The feature is fitting to fold-specific noise, not a genuine signal

**Decision:** ❌ **Reject the feature entirely**
- Even if some params passed permutation tests, the lack of walkforward stability indicates overfitting
- Recommendation: Revisit the breakout rule hypothesis or try a different asset/timeframe

**Contrast with stable feature:**

If the walkforward had shown:
```
Fold 1: {(20, 1.0%), (20, 1.5%), (30, 1.0%)}
Fold 2: {(20, 1.5%), (30, 1.0%), (20, 1.0%)}
Fold 3: {(30, 1.0%), (20, 1.5%), (30, 1.5%)}
Fold 4: {(20, 1.0%), (30, 1.0%), (20, 1.5%)}
```
→ **Stable region: (20-30, 1.0-1.5%)** would emerge, and the feature would be deployable.

---

## 8. Limitations and Edge Cases

### When Grid-Aware Neighbor Averaging Works Well

✅ **Regular or semi-regular grids**
- Parameter values are ordered and somewhat evenly spaced
- Example: lookback = [5, 10, 15, 20, 30] (gaps increase but still ordered)
- Adjacency is meaningful (moving from 10 to 15 is a "small step")

✅ **Hypothesis-driven testing with small grids**
- 5–20 parameter combinations per dimension
- Grid density is chosen deliberately (not random)
- Researcher has priors about which regions to test

✅ **Reasonably smooth objective surfaces**
- Performance changes gradually as parameters change
- Typical of mean reversion (RSI, Bollinger), trend-following (EWMAC, donchian)
- Not pure noise (if objective is random, no algorithm helps)

✅ **Complete or mostly-complete grids**
- Most cells in the grid have been tested
- Some missing cells are OK (partial grids), but not too many

---

### When Grid-Aware Neighbor Averaging Struggles

❌ **Very sparse grids (most neighbors missing)**

Example:
```
2D grid: (lookback, threshold)
Tested: {(5, 1.0), (10, 2.0), (40, 0.5)}
Missing: all other combinations

For (10, 2.0):
  - Neighbor (5, 2.0)? Missing
  - Neighbor (15, 2.0)? Missing (no lookback 15 in grid)
  - Neighbor (10, 1.5)? Missing
  - Neighbor (10, 2.5)? Missing
  → No neighbors exist!
  → Smoothed = raw (no smoothing happens)
```

**Solution:** Use KNN smoothing or increase grid density.

---

❌ **Highly irregular grids (adjacency is misleading)**

Example:
```
Lookback = [1, 2, 100, 1000]

For lookback 2:
  - Previous neighbor: 1 (gap = 1)
  - Next neighbor: 100 (gap = 98)

Averaging performance at {1, 2, 100} is meaningless — the "neighbors" are in completely different regimes.
```

**Solution:** Either:
- Restructure the grid (test [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] to have more uniform log-spacing)
- Use KNN with a distance metric that accounts for the irregular spacing

---

❌ **Random search results (no grid structure)**

Example:
```
Bayesian optimization sampled:
  {(lookback=7.3, threshold=1.42), (lookback=23.8, threshold=0.61), ...}

No "grid" → no "1-step neighbors" → grid-aware averaging doesn't apply
```

**Solution:** Use KNN smoothing (distance-based) instead of grid-based.

---

❌ **Objective is pure noise (no signal)**

Example:
```
All parameters have Sharpe ~ 0 ± 0.3 (random fluctuations)
No stable regions exist because there's no genuine signal
```

**Result:** Grid-aware neighbor averaging will still compute smoothed metrics, but they're all ~0. Permutation tests should catch this (no params pass), but if you're not using permutation tests, you'll see random patterns.

**Solution:** Always combine stability analysis with permutation tests (Stages 1–2 filter out noise, Stage 3 filters out overfitting).

---

### Edge Cases

#### Boundary Parameters (Fewer Neighbors)

**Scenario:** Parameters at the edge of the grid have fewer neighbors.

Example (1D):
```
Lookback = [2, 3, 4, 5, 10]

For lookback 2:
  - Previous neighbor: none (boundary)
  - Next neighbor: 3
  → Smoothed = (obj(2) + obj(3)) / 2  (only 2 values)

For lookback 4:
  - Previous: 3
  - Next: 5
  → Smoothed = (obj(3) + obj(4) + obj(5)) / 3  (3 values)
```

**Effect:** Boundary params are smoothed less (fewer neighbors), so their smoothed objective is closer to their raw objective.

**Is this a problem?** **No, this is correct behavior:**
- Boundary params *should* have less regularization because we have less information about their neighborhood
- If a boundary param has high raw objective but few neighbors, it's inherently more risky (less evidence of stability)

**Interpretation:** A boundary param with `smoothed ≈ raw` doesn't mean "stable" — it just means "under-smoothed". Check if interior params in the same performance range have better stability ratios.

---

#### Partial Grids (Missing Cells)

**Scenario:** Some parameter combinations weren't tested (holes in the grid).

Example (2D):
```
      Threshold
        1.0  1.5  2.0
L 10   ✓    ✓    ✗     (2.0 missing)
o 20   ✓    ✗    ✓     (1.5 missing)
o 30   ✗    ✓    ✓     (1.0 missing)
k

For (20, 1.0):
  - Neighbors: {(10, 1.0), (30, 1.0)?missing, (20, 1.5)?missing, (20, 2.0)}
  - Existing neighbors: {(10, 1.0), (20, 2.0)}
  → Smoothed = (obj(20,1.0) + obj(10,1.0) + obj(20,2.0)) / 3
```

**Effect:** Missing neighbors are simply excluded from the average. The param is smoothed over fewer values.

**Is this a problem?** **Depends on how many neighbors are missing:**
- 1-2 missing neighbors: Fine, still have enough for smoothing
- Most neighbors missing: Degenerates to under-smoothing (similar to boundary case)

**Solution:** If the grid is very sparse, consider filling in the holes or using KNN.

---

#### Multi-Dimensional Grids (3D, 4D, ...)

**Scenario:** Testing 3+ parameters simultaneously.

Example (3D):
```
Parameters: lookback, threshold, smoothing
Grid: 5 × 5 × 3 = 75 combinations

For a param in the interior:
  - 1D: 2 neighbors (prev/next in 1 dimension)
  - 2D: 4 neighbors (up/down/left/right)
  - 3D: 6 neighbors (add front/back)
  - 4D: 8 neighbors
  - ND: 2×D neighbors
```

**Effect:** Smoothing uses more neighbors in higher dimensions → more regularization.

**Is this a problem?** **Generally no:**
- More neighbors = more information about the neighborhood = better stability assessment
- But: harder to visualize (can't plot 3D+ grids easily)

**Solution for interpretation:**
- Compute and trust the smoothed metric (the math still works)
- Visualize 2D slices (fix one param, plot the other two)
- Use the stability ratio (smoothed / raw) as a single number summary

---

#### High Stability Ratio but Low Absolute Smoothed Objective

**Scenario:** A param has `smoothed / raw > 1.0` but both are low.

Example:
```
Lookback 20:
  Raw Sharpe = 0.1
  Smoothed Sharpe = 0.2
  Stability ratio = 0.2 / 0.1 = 2.0 (very high!)

Is this stable? Yes, but it's stable at a LOW level (consistently poor).
```

**Interpretation:** **High stability ratio doesn't mean "good" — it means "consistent".**

**Action:** Always check the **absolute value** of smoothed objective:
```
IF smoothed_objective > threshold AND stability_ratio > 0.8:
    → Good and stable ✅
ELIF smoothed_objective < threshold AND stability_ratio > 0.8:
    → Consistently poor ❌ (reject)
```

---

## 9. References and Related Concepts

### Related Documentation

- **Task specification (implementation):** [Grid-Aware Neighbor Averaging](../../../to-do/grid_neighbor_smoothing_specs.md) — Algorithm details, API signature, edge case handling, implementation scope
- **Validation framework:** [In-Sample Permutation Testing](../permutation_testing/in-sample_pt.md) — Full testing pipeline (§3: Walkforward Stability Analysis)
- **Ensemble structure:** [Base Model (Ensemble)](../base_models/base_model.md) — How ensembles are formed and aggregated
- **Continuous features:** [Continuous Binning](../feature_types/Continuous_binning.md) — Quantile binning, region selection, thresholds
- **Rule-based features:** [Rule-Based Features](../feature_types/rule_based.md) — Discrete signal handling

### Related Code

- **Parameter analysis (existing):** `eda/parameter_analysis.py` (`ParameterAnalyzer` class)
  - `analyze_parameter()`, `analyze_2d_parameters()`, `analyze_nd_parameters()` — Grid search evaluation
  - `compute_robustness_metrics()` — Variance-based robustness (Alternative 1 in this spec)
- **Parameter plotting:** `metrics/plotting/parameter_plots.py`
  - `plot_parameter_sensitivity()`, `plot_2d_parameter_surface()` — Visualization of grid search results

### Theoretical Background

**Concepts:**
- **Parameter overfitting:** Selecting parameter values that are optimal for the sample but fail out-of-sample
- **Regularization:** Penalizing complexity to improve generalization (ensemble averaging is a form of regularization)
- **Occam's Razor:** Simpler explanations (stable regions) are more likely to be true than complex ones (isolated peaks)
- **Cross-validation:** Stability analysis is conceptually similar to cross-validation (test on multiple folds to assess robustness)


### Key Takeaways

1. **Parameter stability ≠ statistical significance:** A param can pass a permutation test (significant) but still be an isolated peak (unstable). Both tests are needed.

2. **Neighbor smoothing is regularization:** Averaging over a neighborhood penalizes isolated peaks and rewards stable regions — this is a form of regularization that improves generalization.

3. **Stability predicts OOS performance:** Empirically, params in stable regions generalize better than isolated peaks. This is why walkforward stability analysis (Stage 3) is critical.

4. **Grid-aware neighbor averaging is the right default:** For hypothesis-driven testing with structured grids, it's simple, interpretable, and effective. Use alternatives (KNN, regularization) only for irregular grids or random search.

5. **Always combine with permutation tests:** Stability analysis filters overfitting, but permutation tests filter noise. Use both (Stages 1–2 + Stage 3) for robust feature validation.

6. **Researcher judgment is essential:** The framework provides data (permutation results, stability reports), but the researcher makes the final ensemble decision using pre-committed criteria and domain knowledge.

---

**End of specification.**
