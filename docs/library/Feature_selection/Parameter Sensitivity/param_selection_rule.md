# Pre-Committed Parameter Selection Rule

> **Status:** Library specification

**Date:** 2026-02-19

**Purpose:** Formal specification of the algorithm that selects which parameter combinations to trade from a grid. This rule runs identically during walkforward validation and in production — that identity is the core property that makes walkforward evaluation realistic.

**Scope:** Algorithm definition, design rationale, threshold reference, and worked examples. For the neighbor smoothing algorithm that this rule depends on, see [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md).

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [The Algorithm](#2-the-algorithm)
3. [Pre-Committed Global Thresholds](#3-pre-committed-global-thresholds)
4. [Design Rationale](#4-design-rationale)
5. [Neighbor Definition](#5-neighbor-definition)
6. [Worked Examples](#6-worked-examples)
7. [Integration with the Pipeline](#7-integration-with-the-pipeline)
8. [Related Docs](#8-related-docs)

---

## 1. Overview and Motivation

### The Problem Being Solved

Trading features such as RSI, EWMAC, and Donchian channels have many candidate parameter combinations. A naive approach would:

1. Test all combinations on a training window
2. Pick the single best-performing combination ("best param wins")
3. Trade that combination going forward

This approach fails for two reasons:

- **Overfitting:** The best parameter in-sample is usually an isolated spike caused by noise, not genuine signal. It breaks down out-of-sample.
- **Non-stationarity:** Even if the best parameter is real, markets shift over time. A single fixed parameter has no robustness to drift.

The pre-committed parameter selection rule replaces this with a principled, automated algorithm that:

- Selects a small **ensemble** of stable parameter combinations, not a single "best" one
- Requires **neighborhood support** — qualifying parameters must have similar neighbors
- Has **no free choices at decision time** — all thresholds are committed in advance
- Runs **identically** in walkforward backtests and live production

### What "Pre-Committed" Means

The rule's thresholds (`stability_threshold`, `metric_threshold`, `K_min`, `K_max`) are set **once, before any data is examined**, and are **never adjusted after seeing results**. This prevents the common research failure mode of tuning selection criteria to make the backtest look good — which invalidates the walkforward.

The rule is an algorithm, not a judgment call. Given a training fold and a set of candidate parameters, it deterministically produces either a selected ensemble or an empty result (no position).

---

## 2. The Algorithm

```
Pre-Committed Parameter Selection Rule
Input:  candidate_params    — all parameter combinations to consider
        training_data       — data for this fold's training window
        thresholds          — pre-committed config (stability_threshold,
                              metric_threshold, K_min, K_max,
                              objective_metric)
Output: selected_ensemble   — ordered list of qualifying params
                              (empty = no position this period)

Step 1: Fit all candidate params on training data
   For each param P in candidate_params:
       Fit the base model (binning or rule-based) using P on training_data
       Compute raw_objective(P) = Sharpe or Sortino of fitted model
       Store (P, raw_objective(P))

Step 2: Apply neighbor smoothing (grid-aware)
   For each param P:
       neighbors(P) = all 1-step axis-aligned neighbors of P in the grid
       smoothed(P)  = mean([raw_objective(P)] + [raw_objective(N)
                            for N in neighbors(P) if N exists])
       stability_ratio(P) = smoothed(P) / raw_objective(P)
                            (if raw_objective(P) = 0, set ratio = 0)

Step 3: Apply selection gates
   qualifying = [P for P in candidate_params
                 if stability_ratio(P) >= stability_threshold   (gate A)
                 and raw_objective(P)  >  metric_threshold]     (gate B)

Step 4: Rank and cap
   Sort qualifying by smoothed(P) descending
   If len(qualifying) > K_max:
       qualifying = qualifying[:K_max]

Step 5: Minimum membership check
   If len(qualifying) < K_min:
       return []  # no position this period

Step 6: Return ensemble
   return qualifying

Ensemble signal = mean(signal(P) for P in qualifying)
```

### Step-by-Step Notes

**Step 1 — Fit all params:** Every candidate parameter combination is fitted on the training window, not just a preselected subset. The full grid is needed to compute meaningful neighbor averages in Step 2.

**Step 2 — Neighbor smoothing:** Each parameter's smoothed objective is computed by averaging over itself and its 1-step axis-aligned neighbors. The stability ratio measures how close the smoothed objective is to the raw objective. A ratio near 1.0 means neighbors perform similarly (stable region). A ratio well below 1.0 means the parameter's performance collapses in its immediate neighborhood (isolated peak). See [Section 5](#5-neighbor-definition) for the formal neighbor definition.

**Step 3 — Two selection gates:**

- **Gate A (stability gate):** `stability_ratio >= stability_threshold`. Rejects isolated peaks.
- **Gate B (quality gate):** `raw_objective > metric_threshold`. Rejects parameters that are stable but genuinely poor.

Both gates must be passed. A parameter that is stable but weak (e.g., Sharpe = 0.05 with stability ratio 0.95) is rejected by Gate B. A parameter that is strong but isolated (e.g., Sharpe = 1.5 with stability ratio 0.3) is rejected by Gate A.

**Step 4 — Rank by smoothed objective:** Ranking by smoothed (not raw) objective means the selection favors stable high performers over isolated high performers. `K_max` prevents the ensemble from growing large when many mediocre parameters marginally pass the gates.

**Step 5 — K_min check:** If too few parameters qualify, there is insufficient neighborhood support. Rather than trading a single isolated qualifier, the rule returns an empty ensemble. An empty ensemble means **no position** for this period — not a flat signal, but a genuine absence of a tradeable hypothesis.

---

## 3. Pre-Committed Global Thresholds

These thresholds **must be set before running any validation** and must remain fixed throughout all walkforward folds and production runs. Changing them after seeing results is data snooping and invalidates the walkforward.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stability_threshold` | 0.8 | Minimum `stability_ratio` to pass Gate A. A ratio of 0.8 means the smoothed objective is at least 80% of the raw objective. |
| `metric_threshold` | Feature-dependent | Minimum `raw_objective` to pass Gate B. Must be set per-feature before data is examined (e.g., Sharpe > 0.3 for a trend-following feature). |
| `K_min` | 2 | Minimum qualifying params. Fewer than this → empty ensemble (no position). |
| `K_max` | 7 | Maximum ensemble members. Limits ensemble size if many params qualify. |
| `objective_metric` | Sharpe or Sortino | The metric computed in Step 1 and used in all subsequent steps. Must match what the walkforward measures. |

### Setting `metric_threshold`

The `metric_threshold` is the only parameter that is feature-dependent. It should be set based on theoretical priors about what constitutes a meaningful edge for that feature class, not by examining the data.

Example priors:
- Mean reversion features (RSI, Bollinger Bands): Sharpe > 0.3 is a modest hurdle; reasonable prior.
- Trend-following features (EWMAC, Donchian): Sharpe > 0.2 may be more appropriate given lower average metrics in choppy markets.
- Rule-based features with binary signals: Sortino > 0.3 or Information Coefficient > 0.05.

**Critical constraint:** Do not look at the distribution of training fold results and then set `metric_threshold` to be "just below the good ones." That is data snooping. Set it before the first fold is run.

---

## 4. Design Rationale

### Why Neighbor Smoothing?

Isolated peaks in parameter space are likely noise. When a parameter performs well but its immediate neighbors do not, there are two possible explanations:

1. **There is a genuine, narrow edge at exactly that parameter value** — but this is implausible for most trading features, where underlying market dynamics vary smoothly.
2. **The in-sample data happened to have noise that aligned with that parameter** — this is the common case.

Genuine signal produces **stable regions** — broad plateaus where nearby parameters also perform well. Noise produces **narrow spikes** — sharp peaks that collapse as soon as the parameter shifts by one grid step.

Neighbor smoothing filters this by computing a weighted average over the neighborhood. It acts as a low-pass filter on the parameter space:

- **High-frequency variation** (spike-to-trough within 1–2 grid steps) = noise → smoothed out
- **Low-frequency variation** (broad plateaus) = signal → preserved

The stability ratio then makes this explicit: a high ratio means the raw performance survives smoothing (stable), a low ratio means it collapses under smoothing (spike).

### Why Select Multiple Params?

Selecting the single best parameter overfits the training window. The "best" parameter shifts with each new training fold as the optimal market behavior changes. This produces unstable, noisy trading.

Selecting 3–7 parameters from a stable neighborhood provides **implicit diversification over parameter uncertainty**:

- The ensemble averages over a small range of parameter values
- Because the parameters are from a stable neighborhood, they are highly correlated (they all capture the same underlying signal)
- But the ensemble smooths out noise specific to any one parameter setting
- The ensemble signal has lower variance than a single-parameter signal without sacrificing expected return

This is not diversification in the sense of combining uncorrelated signals. It is diversification in the sense of reducing parameter-specific noise while preserving the shared edge.

### Why K_min = 2?

If only one parameter qualifies, there is no neighborhood corroboration. The single qualifier might be a borderline isolated peak that barely passed Gate A, or a genuine stable parameter that happens to be on the grid boundary with only one tested neighbor.

Requiring at least two qualifiers means the ensemble has at least minimal internal support. It is analogous to requiring at least two independent witnesses before acting on testimony — one data point is insufficient evidence.

The minimum is intentionally low (2, not 3 or 5) because the stability gate and quality gate already do the heavy filtering. If two parameters both pass both gates, that is meaningful evidence.

### Why stability_ratio >= 0.8?

The threshold of 0.8 was derived from empirical analysis of what constitutes a "plateau" versus a "spike" in typical trading feature grids:

- At 0.8, the smoothed objective is at most 20% lower than the raw objective. This allows for moderate variation in the neighborhood while still requiring meaningful support.
- At values below 0.8 (e.g., 0.5–0.7), parameters are allowing substantial performance degradation in immediate neighbors — this is the hallmark of an isolated spike.
- Setting the threshold too high (e.g., 0.95) would be too conservative: it would require near-perfect uniformity in the neighborhood and would reject legitimate stable regions with natural variation.

The 0.8 threshold is pre-committed and does not vary by feature. Using a uniform threshold is a feature of the design, not a limitation — it prevents per-feature threshold tuning, which would be another form of overfitting.

### Why Identical in Walkforward and Production?

The walkforward test measures the expected future performance of a trading rule. For this measurement to be valid, **the walkforward must emulate exactly what would happen in production**.

If the walkforward uses a different selection criterion than production — even a slightly more generous one — then the walkforward results are not representative of live performance. They measure a strategy that was never deployed.

By using the identical algorithm, with the identical pre-committed thresholds:

- The walkforward fold results are genuine simulations of what production will do
- No "testing special cases" or "relaxed thresholds for research" in the backtest
- The walkforward OOS metrics are directly interpretable as expected live metrics

This is the core of live trading emulation. It is why the rule must be specified precisely before any data is seen.

---

## 5. Neighbor Definition

### Formal Definition

For a D-dimensional parameter grid, the **1-step axis-aligned neighbors** of a parameter combination P are all combinations that:

- Differ from P in **exactly one parameter dimension**
- Differ by **exactly one grid step** (i.e., the immediately adjacent value in the ordered list of tested values for that dimension)

Let `grid[d]` denote the ordered list of tested values for dimension `d`, and let `P[d]` denote parameter P's value in dimension `d`. Then:

```
neighbors(P) = {
    P' : exists exactly one d such that
         (P'[d] is the predecessor of P[d] in grid[d]
          OR P'[d] is the successor of P[d] in grid[d])
         AND P'[d'] = P[d'] for all d' != d
         AND P' was actually tested (exists in candidate_params)
}
```

### 1D Example (RSI lookback)

```
Grid: [2, 3, 4, 5, 10, 14, 20]

neighbors(2)  = {3}           (no predecessor at boundary)
neighbors(3)  = {2, 4}
neighbors(4)  = {3, 5}
neighbors(5)  = {4, 10}       (next step in grid is 10, not 6)
neighbors(10) = {5, 14}
neighbors(14) = {10, 20}
neighbors(20) = {14}          (no successor at boundary)
```

Note: the "1 grid step" is not a fixed numerical distance — it is one step in the ordered list of tested values. The step from 5 to 10 is one grid step even though it is a numerical gap of 5.

### 2D Example (EWMAC: spanFast x spanSlow)

```
Grid dimensions:
  spanFast: [8, 16, 20, 32]
  spanSlow: [32, 64, 80, 128]

For P = (spanFast=16, spanSlow=64):
  Vary spanFast by ±1 step, hold spanSlow fixed:
    (8,  64) — predecessor of 16 in spanFast
    (20, 64) — successor   of 16 in spanFast
  Vary spanSlow by ±1 step, hold spanFast fixed:
    (16, 32) — predecessor of 64 in spanSlow
    (16, 80) — successor   of 64 in spanSlow

neighbors((16,64)) = {(8,64), (20,64), (16,32), (16,80)}
```

For a D-dimensional interior parameter, there are at most 2D neighbors (2 per dimension). Boundary parameters and sparse grids have fewer.

### Boundary Behavior

Boundary parameters have fewer neighbors. This is correct behavior, not a bug:

- A boundary parameter has inherently less neighborhood evidence
- Less smoothing (fewer neighbors in the average) means the smoothed objective is closer to the raw objective
- This makes the stability ratio less informative at the boundary
- As a consequence, boundary parameters are held to a slightly less stringent effective standard

Researchers should be aware that boundary parameters that pass the stability gate may have passed because of limited smoothing rather than genuine stability. This is a known limitation, documented in [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md) under edge cases.

### Sparse and Incomplete Grids

If a neighbor was not tested (the grid has a hole), it is excluded from the neighborhood average. The denominator of the mean is the number of existing neighbors (including P itself), not the theoretical maximum.

For very sparse grids where most neighbors are missing, smoothing degenerates toward the raw objective, and the stability ratio loses discriminatory power. In these cases, the algorithm still produces a result, but the result should be interpreted with more caution. See [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md) for a full treatment of grid sparsity edge cases.

---

## 6. Worked Examples

### Example A: RSI Lookback (1D Grid)

**Feature:** RSI mean reversion signal
**Grid:** RSI lookback = [2, 3, 4, 5, 10, 14, 20]
**Thresholds:** `stability_threshold = 0.8`, `metric_threshold = 0.5` (Sharpe), `K_min = 2`, `K_max = 7`

**Step 1 — Raw objectives (hypothetical training fold):**

| Lookback | Raw Sharpe |
|----------|------------|
| 2        | 0.45       |
| 3        | 0.72       |
| 4        | 0.88       |
| 5        | 0.79       |
| 10       | 0.51       |
| 14       | 0.30       |
| 20       | 0.05       |

**Step 2 — Neighbor smoothing:**

| Lookback | Raw Sharpe | Neighbors (existing) | Smoothed | Stability Ratio |
|----------|------------|----------------------|----------|-----------------|
| 2        | 0.45       | {3: 0.72}            | (0.45 + 0.72) / 2 = **0.585** | 0.585 / 0.45 = **1.30** |
| 3        | 0.72       | {2: 0.45, 4: 0.88}   | (0.72 + 0.45 + 0.88) / 3 = **0.683** | 0.683 / 0.72 = **0.95** |
| 4        | 0.88       | {3: 0.72, 5: 0.79}   | (0.88 + 0.72 + 0.79) / 3 = **0.797** | 0.797 / 0.88 = **0.91** |
| 5        | 0.79       | {4: 0.88, 10: 0.51}  | (0.79 + 0.88 + 0.51) / 3 = **0.727** | 0.727 / 0.79 = **0.92** |
| 10       | 0.51       | {5: 0.79, 14: 0.30}  | (0.51 + 0.79 + 0.30) / 3 = **0.533** | 0.533 / 0.51 = **1.05** |
| 14       | 0.30       | {10: 0.51, 20: 0.05} | (0.30 + 0.51 + 0.05) / 3 = **0.287** | 0.287 / 0.30 = **0.96** |
| 20       | 0.05       | {14: 0.30}            | (0.05 + 0.30) / 2 = **0.175** | 0.175 / 0.05 = **3.50** |

**Step 3 — Apply gates:**

| Lookback | Raw Sharpe | Stability Ratio | Gate B (raw > 0.5)? | Gate A (ratio >= 0.8)? | Qualifies? |
|----------|------------|-----------------|---------------------|------------------------|------------|
| 2        | 0.45       | 1.30            | No                  | Yes                    | **No** (fails Gate B) |
| 3        | 0.72       | 0.95            | Yes                 | Yes                    | **YES** |
| 4        | 0.88       | 0.91            | Yes                 | Yes                    | **YES** |
| 5        | 0.79       | 0.92            | Yes                 | Yes                    | **YES** |
| 10       | 0.51       | 1.05            | Yes                 | Yes                    | **YES** |
| 14       | 0.30       | 0.96            | No                  | Yes                    | **No** (fails Gate B) |
| 20       | 0.05       | 3.50            | No                  | Yes                    | **No** (fails Gate B) |

Note: lookback 2 fails Gate B (raw Sharpe 0.45 < 0.5). Lookback 20 has an extreme stability ratio (3.50) because its smoothed objective (0.175) is much higher than its own raw objective (0.05) — this is the "mediocre but in a better neighborhood" case. It still fails Gate B because its raw Sharpe is negligible.

**Step 4 — Rank by smoothed objective, cap at K_max = 7:**

Qualifying: {3 (0.683), 4 (0.797), 5 (0.727), 10 (0.533)}

Sorted by smoothed descending: [4 (0.797), 5 (0.727), 3 (0.683), 10 (0.533)]

4 members < K_max = 7, so no capping needed.

**Step 5 — K_min check:**

4 >= K_min (2) → proceed.

**Result:** Selected ensemble = {lookback=4, lookback=5, lookback=3, lookback=10}

**Ensemble signal** = mean(RSI_4_signal, RSI_5_signal, RSI_3_signal, RSI_10_signal)

**Interpretation:** The rule selected a cluster of short-to-medium lookbacks. Lookback 4 dominates by smoothed objective, but lookbacks 3, 5, and 10 all have sufficient raw quality and neighborhood support. The ensemble averages over this stable short-term mean reversion region. Lookback 14 was rejected not because of instability (ratio = 0.96) but because its raw Sharpe (0.30) fell below the pre-committed quality threshold.

---

### Example B: EWMAC on a 2D Grid (spanFast x spanSlow)

**Feature:** EWMAC trend-following crossover signal
**Grid dimensions:** spanFast = [8, 16, 20, 32], spanSlow = [32, 64, 80, 128]
**Thresholds:** `stability_threshold = 0.8`, `metric_threshold = 0.25` (Sharpe), `K_min = 2`, `K_max = 5`

**Step 1 — Raw objectives (hypothetical training fold):**

```
              spanSlow
              32     64     80    128
spanFast  8  [0.15]  0.60   0.52  0.30
         16   0.40   1.10   0.95  0.55
         20   0.35   0.90   1.05  0.70
         32   0.20   0.50   0.60  0.65
```

**Step 2 — Neighbor smoothing for selected cells:**

For **(16, 64)** — interior cell:
- Neighbors: (8,64)=0.60, (20,64)=0.90, (16,32)=0.40, (16,80)=0.95
- Smoothed = (1.10 + 0.60 + 0.90 + 0.40 + 0.95) / 5 = **0.79**
- Stability ratio = 0.79 / 1.10 = **0.718**

For **(20, 80)** — interior cell:
- Neighbors: (16,80)=0.95, (32,80)=0.60, (20,64)=0.90, (20,128)=0.70
- Smoothed = (1.05 + 0.95 + 0.60 + 0.90 + 0.70) / 5 = **0.84**
- Stability ratio = 0.84 / 1.05 = **0.80**

For **(20, 64)** — interior cell:
- Neighbors: (16,64)=1.10, (32,64)=0.50, (20,32)=0.35, (20,80)=1.05
- Smoothed = (0.90 + 1.10 + 0.50 + 0.35 + 1.05) / 5 = **0.78**
- Stability ratio = 0.78 / 0.90 = **0.867**

For **(16, 80)** — interior cell:
- Neighbors: (8,80)=0.52, (20,80)=1.05, (16,64)=1.10, (16,128)=0.55
- Smoothed = (0.95 + 0.52 + 1.05 + 1.10 + 0.55) / 5 = **0.834**
- Stability ratio = 0.834 / 0.95 = **0.878**

**Stable 2x2 sub-region:**

```
              spanSlow
              64      80
spanFast 16  [1.10]  [0.95]    <- both have strong raw + neighbor support
         20  [0.90]  [1.05]    <- the 2x2 stable block
```

All four cells in this sub-region have raw Sharpe > 0.25 (pass Gate B). Now checking Gate A:

| Combo      | Smoothed | Stability Ratio | Gate A (>= 0.8)? |
|------------|----------|-----------------|------------------|
| (16, 64)   | 0.790    | 0.718           | **No** — isolated relative to surroundings |
| (20, 80)   | 0.840    | 0.800           | **Yes** |
| (20, 64)   | 0.780    | 0.867           | **Yes** |
| (16, 80)   | 0.834    | 0.878           | **Yes** |

Note: (16, 64) has the highest raw Sharpe (1.10) but fails the stability gate because its neighborhood includes (16, 32)=0.40 and (8, 64)=0.60 — the wider neighborhood drags down the smoothed value. This is a mild isolated peak: still a strong parameter, but less stable than its neighbors.

**Contrast with a true isolated peak vs. a stable block:**

- If (16, 64) were surrounded by values of 0.1–0.2, stability ratio would be ~0.2 — clearly isolated spike, rejected.
- In this case, stability ratio = 0.718 — a mild spike. It fails the 0.8 gate but is borderline. This is expected behavior: the gate is conservative by design.

**Qualifying (after both gates):** {(20,80), (20,64), (16,80)}
**Ranked by smoothed:** [(20,80): 0.840, (16,80): 0.834, (20,64): 0.780]
**K_min check:** 3 >= 2 — proceed.

**Result:** Selected ensemble = {(20,80), (16,80), (20,64)}

**Ensemble signal** = mean(EWMAC_20_80_signal, EWMAC_16_80_signal, EWMAC_20_64_signal)

**Interpretation:** The rule selected the stable medium-speed crossover region. The highest raw performer (16,64) was excluded by the stability gate — its raw Sharpe of 1.10 was too much better than its neighbors, a signature of mild overfit. The ensemble of three covers a coherent neighborhood: medium-fast spans (16–20) with medium-slow spans (64–80). These all share the underlying signal of medium-horizon trend following.

---

### Example C: No Position (K_min not Met)

**Feature:** Rule-based breakout signal
**Grid:** lookback = [10, 20, 40, 60], threshold_pct = [1.0, 1.5, 2.0, 2.5]
**Thresholds:** `stability_threshold = 0.8`, `metric_threshold = 0.3`, `K_min = 2`

Suppose the training fold produces scattered results with no coherent region:

| Combo       | Raw Sharpe | Smoothed | Stability Ratio | Passes both gates? |
|-------------|------------|----------|-----------------|-------------------|
| (20, 2.0)   | 0.70       | 0.42     | 0.60            | No (ratio < 0.8)  |
| (10, 1.0)   | 0.55       | 0.31     | 0.56            | No (ratio < 0.8)  |
| (40, 1.5)   | 0.48       | 0.35     | 0.73            | No (ratio < 0.8)  |
| All others  | < 0.30     | —        | —               | No (raw < 0.3)    |

Zero parameters pass both gates. `len(qualifying) = 0 < K_min = 2`.

**Result:** Empty ensemble — no position this period.

**Interpretation:** No stable region exists in this training window. Trading this feature during this period would mean acting on noise with no neighborhood support. The pre-committed rule returns an empty ensemble, and the position sizing layer receives zero allocation for this feature. This is the correct conservative response.

---

## 7. Integration with the Pipeline

### Where the Rule Runs

The pre-committed selection rule is called in exactly two contexts:

**Context 1: Walkforward Phase 3 (one call per fold)**

The walkforward (see [`../Walkforward/walkforward.md`](../Walkforward/walkforward.md)) splits the in-sample data into multiple training/test folds. For each fold:

1. The training window is defined
2. The selection rule is called on that training window
3. The selected ensemble is applied to the hold-out test window
4. OOS metrics are recorded

This produces a sequence of OOS results, one per fold, that simulate what production would have done.

**Context 2: Production (one call per refit period)**

In production, the rule is called on an expanding training window at each refit date:

1. The training window extends from the feature's inception date to the current refit date
2. The selection rule is called on this full training window
3. The selected ensemble is the live trading ensemble until the next refit
4. No test window — production immediately trades the selected ensemble

**The walkforward and production calls are identical.** The same function, same thresholds, same algorithm. The only difference is the training window's end date.

### Where the Rule Does NOT Run

| Pipeline Phase | What Happens Instead |
|----------------|---------------------|
| **IS EDA (Phase 1)** | Diagnostic exploration only — researchers examine grid search results manually, no automated selection. Purpose is hypothesis formation, not parameter commitment. |
| **IS Permutation Screening (Phase 2)** | Uses a simpler fitted-vector approach — runs permutation tests on individual parameter combinations in isolation. No neighborhood smoothing. Purpose is statistical filtering, not ensemble selection. |
| **OOS Evaluation (after walkforward)** | The rule has already selected the ensemble during each training fold. OOS evaluation only applies and measures the selected ensemble. No re-selection on OOS data. |

### Data Flow

```
Training fold data
       |
       v
Step 1: Fit ALL candidate params on training data
       | (one base model fit per param)
       v
Step 2: Compute smoothed objective for each param
       | (ParameterAnalyzer.compute_neighbor_smoothed_grid)
       v
Step 3: Apply stability gate (ratio >= threshold)
        Apply quality gate (raw > metric_threshold)
       |
       v
Step 4: Sort by smoothed, cap at K_max
       |
       v
Step 5: K_min check
       |
       +---> Empty []    -> no position allocated
       |
       +---> [P1, P2, ...] -> base models fitted with these params
                             -> signals averaged
                             -> ensemble signal used in rest of pipeline
```

### Connection to the Broader Ensemble

The selection rule produces a list of parameter combinations. Each combination corresponds to a fitted `BinningModelBase` instance (continuous) or rule-based model instance. These individual model signals are averaged to produce the ensemble signal for this feature.

The ensemble signal is then passed up the pipeline:

```
Param selection rule (this document)
    → ensemble signal for one feature
        → DiversifiedEnsemble (across features)
            → WeightLayer (FDM application)
                → Portfolio (IDM application)
                    → PositionSizer (contracts)
```

The selection rule is thus responsible for the first level of aggregation: from individual parameter combinations to a single feature-level signal.

---

## 8. Related Docs

- [`../pipeline_overview.md`](../pipeline_overview.md) — Master pipeline specification from candles to positions
- [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md) — Neighbor smoothing theory, algorithm comparison, interpretation guidelines, edge cases
- [`../Walkforward/walkforward.md`](../Walkforward/walkforward.md) — Walkforward implementation and fold structure
- [`../feature_validator.md`](../feature_validator.md) — Phase 1 (IS EDA) and Phase 2 (IS Permutation Screening) details

---

**End of specification.**
