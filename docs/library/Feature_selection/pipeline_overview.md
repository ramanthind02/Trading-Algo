# Feature Validation Pipeline — Master Specification

> **Status:** Library specification (single source of truth)
> **Date:** 2026-02-19
> **Scope:** All feature validation in the systematic trading research codebase

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Data Setup and Time Partitions](#2-data-setup-and-time-partitions)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Design Goals](#4-design-goals)
5. [Pre-Committed Param Selection Rule](#5-pre-committed-param-selection-rule)
6. [Phase 1: IS EDA](#6-phase-1-is-eda)
7. [Phase 2: IS Permutation Screening](#7-phase-2-is-permutation-screening)
8. [Phase 3: Walkforward Validation](#8-phase-3-walkforward-validation)
9. [Phase 4: Walkforward Permutation Test](#9-phase-4-walkforward-permutation-test)
10. [Phase 5: Graduation and Production](#10-phase-5-graduation-and-production)
11. [Cross-Validation Methods](#11-cross-validation-methods)
12. [Gates vs Informational Outputs](#12-gates-vs-informational-outputs)
13. [Two-Tier Structure Rationale](#13-two-tier-structure-rationale)
14. [Related Documentation](#14-related-documentation)

---

## 1. Purpose

This document is the **single source of truth** for the Feature Validation Pipeline. It describes the complete sequence of steps used to determine whether a trading feature (e.g., RSI, EWMAC, a breakout rule) has genuine predictive power and is safe to deploy in live trading.

All other docs in this library describe individual components of this pipeline. This document describes how those components fit together, why they are ordered as they are, and which outputs are hard gates versus informational diagnostics.

**What problem does this pipeline solve?**

A researcher discovers that RSI-14 has a Sharpe ratio of 0.45 over the last 20 years. Does that mean it will work going forward? Possibly. Or the researcher may have unconsciously chosen the lookback that happened to perform best, evaluated on data that informed the choice, and found a relationship that is noise. The pipeline exists to distinguish these cases mechanically, with no researcher discretion at execution time.

---

## 2. Data Setup and Time Partitions

The pipeline assumes daily OHLCV data for the instruments under study. The full data history is partitioned into two regions that must never be crossed:

```
2000                    2023        2025
 |------------------------|-----------|
 |   In-Sample (IS)       |   OOS     |
 |   2000 – 2023          | 2024–2025 |
 |   All validation and   | Strict    |
 |   training happens     | hold-out; |
 |   here                 | untouched |
 |------------------------|-----------|
```

- **In-Sample (IS) period: 2000–2023** — All validation, training, screening, and walkforward folds operate within this window. No result from this window provides "proof of OOS validity"; it provides evidence.
- **Strict OOS period: 2024–2025** — This window is never examined during research or validation. It is touched only once: at final production deployment, as a sanity check after the feature has graduated through all pipeline phases.

This hard partition prevents look-ahead and ensures the strict OOS period retains its statistical validity as an independent check.

---

## 3. Pipeline Overview

The pipeline has five sequential phases organized into two tiers.

**Tier 1 — IS Screening** (coarse, cheap, runs on full IS period):
- Phase 1: IS EDA
- Phase 2: IS Permutation Screening

**Tier 2 — Walkforward Validation** (definitive, expensive, runs on OOS folds within IS):
- Phase 3: Walkforward Validation
- Phase 4: Walkforward Permutation Test
- Phase 5: Graduation and Production

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE VALIDATION PIPELINE                          │
│                                                                         │
│  Input: Feature class (e.g., RSI) + full param grid                    │
│         (e.g., lookback ∈ [2, 3, 4, 5, 10, 14, 20])                   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  TIER 1: IS SCREENING (2000–2023, full IS period)             │      │
│  │                                                               │      │
│  │  Phase 1: IS EDA                                              │      │
│  │  ┌────────────────────────────────────────────────────────┐   │      │
│  │  │ Distributions, decile plots, temporal stability,       │   │      │
│  │  │ rolling objective metric, param grid heatmaps,         │   │      │
│  │  │ neighbor-smoothed stability landscape                  │   │      │
│  │  │                          (informational, no gate)      │   │      │
│  │  └────────────────────────────────────────────────────────┘   │      │
│  │                   │                                            │      │
│  │                   ▼                                            │      │
│  │  Phase 2: IS Permutation Screening                            │      │
│  │  ┌────────────────────────────────────────────────────────┐   │      │
│  │  │ Stage 1: Vector shuffle test (all param combos)        │   │      │
│  │  │ Stage 2: Pipeline/candle shuffle (Stage 1 passers)     │   │      │
│  │  │ → Feature-level verdict: did ANY param pass S1+S2?     │   │      │
│  │  │                              (GATE: yes/no for Tier 2) │   │      │
│  │  └────────────────────────────────────────────────────────┘   │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                          │                                              │
│               FAIL ──────┤──────── PASS                                │
│               (reject)   │         │                                    │
│                          ▼         ▼                                    │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  TIER 2: WALKFORWARD VALIDATION (OOS folds within IS)        │      │
│  │                                                               │      │
│  │  Phase 3: Walkforward Validation                              │      │
│  │  ┌────────────────────────────────────────────────────────┐   │      │
│  │  │ 8 expanding folds, initial train 2000–2015,            │   │      │
│  │  │ test 2015–2016, step 1 year                            │   │      │
│  │  │ Per fold: fit ALL params → selection rule → ensemble   │   │      │
│  │  │ → evaluate OOS; track neighborhood consistency         │   │      │
│  │  │              (GATE: neighborhood stable ≥ min_folds)   │   │      │
│  │  └────────────────────────────────────────────────────────┘   │      │
│  │                   │                                            │      │
│  │                   ▼                                            │      │
│  │  Phase 4: Walkforward Permutation Test                        │      │
│  │  ┌────────────────────────────────────────────────────────┐   │      │
│  │  │ Permute feature values (two-region shuffle)            │   │      │
│  │  │ Run full walkforward on permuted data × N trials       │   │      │
│  │  │ Build null distribution of aggregate walkforward Sharpe│   │      │
│  │  │ → original Sharpe > (1-α) quantile?                   │   │      │
│  │  │                          (GATE: statistical threshold) │   │      │
│  │  └────────────────────────────────────────────────────────┘   │      │
│  │                   │                                            │      │
│  │                   ▼                                            │      │
│  │  Phase 5: Graduation                                          │      │
│  │  ┌────────────────────────────────────────────────────────┐   │      │
│  │  │ All three gates passed → feature graduates             │   │      │
│  │  │ Researcher reviews diagnostics, confirms deployment    │   │      │
│  │  │ Production: same pre-committed selection rule,         │   │      │
│  │  │ expanding training window, automatic refit/reselect    │   │      │
│  │  └────────────────────────────────────────────────────────┘   │      │
│  └───────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Design Goals

The pipeline was designed around four non-negotiable properties:

### Goal 1: Automatic Execution

No researcher discretion is permitted at execution time. All rules — thresholds, selection criteria, fold boundaries, permutation counts — are pre-committed before any data is examined. The pipeline produces outputs; the researcher reads them. The researcher does not choose which outputs to act on mid-pipeline.

This matters because: if a researcher can adjust rules after seeing preliminary results, the rules no longer provide statistical protection. The p-values and confidence intervals assume the test was designed before the data was seen.

### Goal 2: Live Trading Emulation

The pipeline is structured so that every decision made in walkforward validation is made using only information that would be available in production at that moment in time. When the walkforward evaluates a test fold beginning in January 2018, the selection rule uses only data up to December 2017. This mirrors exactly what production would do.

The pre-committed param selection rule (Section 5) is identical in the walkforward and in production. The walkforward therefore tests real production behavior, not an idealized version of it.

### Goal 3: Reduce Researcher Selection Bias

Parameter selection is the primary source of selection bias in trading research. A researcher who computes performance for RSI lookbacks [2, 3, 4, 5, 10, 14, 20] and then deploys the best-performing one has implicitly optimized on the test set.

The pipeline eliminates this by:
- Pre-committing the selection rule before any evaluation
- Using neighbor smoothing to prefer stable parameter regions rather than point optimums
- Requiring walkforward consistency across multiple folds (not just aggregate performance)
- Applying a permutation test to the entire walkforward, including the selection rule

### Goal 4: Sequential Gates with Clear Roles

Each phase has a single defined role. Phases do not overlap in purpose. This makes the pipeline auditable: for any feature decision, a researcher can trace exactly which gate caused graduation or rejection and why.

---

## 5. Pre-Committed Param Selection Rule

This section describes the rule that selects which parameter combinations contribute to the trading signal in any given period. It is called "pre-committed" because it must be specified, in full, before the walkforward begins. It is never adjusted based on walkforward results.

**The rule is used identically in two places:**
1. Inside each walkforward training fold (to select params for the subsequent test fold)
2. In production (to select params for the upcoming trading period)

This identity is essential. Because the walkforward uses exactly the rule that production will use, the walkforward OOS metrics are a valid estimate of production performance.

### The Rule

```
Within each training fold (or production training window):

  1. Fit ALL candidate param combos on the training data.

  2. For each param combo p, compute:
         raw_metric(p) = Sharpe (or Sortino) on training data

  3. Apply neighbor smoothing:
         smoothed_metric(p) = mean(raw_metric(neighbors(p)))
         stability_ratio(p) = smoothed_metric(p) / raw_metric(p)

  4. Select p if:
         stability_ratio(p) ≥ 0.80
         AND raw_metric(p) > pre_committed_threshold

  5. If |selected| < K_min (default: K_min = 2):
         → no position this period

  6. Ensemble signal = mean(signal(p) for p in selected)
```

### Key Properties

**Thresholds are global and pre-committed.** The values 0.80, `pre_committed_threshold`, and `K_min = 2` are set once for the entire pipeline. They are not tuned per feature and not adjusted based on observed results.

**Neighbor smoothing requires the full param grid.** Neighbor smoothing computes, for each parameter combination, the average performance of its nearby neighbors in parameter space. This is only meaningful if the full grid is present. Pre-filtering the grid (e.g., "only include lookbacks > 5") distorts the neighborhood structure and makes stability_ratio unreliable. All param combos must be fit in every fold.

**Example — RSI lookback grid [2, 3, 4, 5, 10, 14, 20]:**

Say in training fold ending 2017:
- raw Sharpe: [0.2, 0.3, 0.35, 0.4, 0.5, 0.45, 0.1]
- smoothed Sharpe: [0.25, 0.28, 0.32, 0.38, 0.45, 0.38, 0.3]
- stability_ratio: [1.25, 0.93, 0.91, 0.95, 0.90, 0.84, 3.0]

Lookbacks 14 and 20 have stability_ratio < 0.80 (or in lookback 20's case, the smoothed is higher than raw, which may indicate it is in a good neighborhood but its own raw metric is low). Lookbacks 2, 3, 4, 5, 10 qualify on stability_ratio. After applying the raw_metric threshold, lookbacks 5, 10, 14 may survive. The ensemble for test fold 2017–2018 uses those that qualify.

**Multiple params = implicit diversification.** When several param combos are selected, the ensemble signal is their mean. This reduces single-param risk: if lookback-10 happens to perform poorly in the test fold but lookback-5 and lookback-14 perform well, the ensemble is partially protected. This also means the pipeline naturally prefers features with stable regions in param space over features with single, narrow peaks.

**The K_min=2 rule prevents degenerate ensembles.** If only one param passes, the selection rule has effectively made a point estimate. The pipeline requires at least two to ensure some diversification. If no params pass, the pipeline takes no position rather than forcing a trade with insufficient evidence.

---

## 6. Phase 1: IS EDA

**Period:** Full IS period, 2000–2023
**Gate:** None. This phase is purely informational.
**Role:** Equip the researcher to interpret later results correctly.

### What Happens

EDA is run across all param combos in the grid:

- **Distribution plots** — histogram and QQ-plot of feature values per param combo; checks for fat tails, bimodality, or degenerate distributions (all zeros, all ones)
- **Decile plots** — sort feature values into deciles; compute forward return for each decile; a monotone pattern suggests predictive structure
- **Temporal stability** — rolling mean of the feature value over 1-year windows; large shifts indicate regime sensitivity
- **Rolling objective metric** — compute Sharpe or Sortino in rolling 2-year windows; persistent positive values are encouraging; consistent sign-flipping is a warning
- **Parameter grid search** — compute raw Sharpe across the full param grid; visualize as heatmap or line plot
- **Neighbor-smoothed stability landscape** — apply neighbor smoothing to the raw grid; visualize smoothed metric and stability_ratio across param space; identify stable regions

### What the Researcher Learns

- Are there obviously degenerate param combos (flat signal, all-zero output)?
- Is there a stable region in param space where performance is consistent across neighbors?
- Is the feature's signal persistent over time or concentrated in specific regimes?
- What is a reasonable expectation for IS performance before the permutation tests?

### What IS EDA Does Not Do

EDA does not select which params enter subsequent phases. All params enter Phase 2 and Phase 3 regardless of EDA output. EDA is a diagnostic instrument, not a filter.

---

## 7. Phase 2: IS Permutation Screening

**Period:** Full IS period, 2000–2023
**Gate:** YES — feature-level binary go/no-go for Tier 2
**Role:** Coarse filter to rule out pure noise features before running expensive walkforward

### Why Permutation Screening on the Full IS Period

The full IS period (24 years of daily data) provides maximum statistical power to discriminate signal from noise. Running permutation tests here, before any walkforward, means that features which would never survive walkforward are rejected early, saving computation.

Permutation screening tests the null hypothesis: "This feature's relationship with forward returns is no better than chance." If the null cannot be rejected on 24 years of data, it will certainly not be rejected on any subset.

### Stage 1: Vector Shuffle Permutation Test

**Input:** All param combos in the grid (e.g., RSI lookbacks [2, 3, 4, 5, 10, 14, 20])
**Procedure:**

For each param combo independently:
1. Compute the observed objective metric (Sharpe or Sortino) on the full IS period
2. Repeat N times: shuffle the feature vector randomly; recompute the objective metric
3. p-value = fraction of permuted metrics ≥ observed metric
4. Param combo passes Stage 1 if p-value ≤ α_1 (e.g., 0.10)

**Stage 1 outcome per param combo:** pass or fail (diagnostic label)

**Stage 1 is fast** because it only shuffles the feature vector, not the full pipeline. It does not refit any model. This allows testing all param combos quickly.

### Stage 2: Pipeline / Candle Shuffle Permutation Test

**Input:** Only param combos that passed Stage 1
**Procedure:**

For each Stage-1-passing param combo:
1. Compute the observed objective metric using the full pipeline (including signal generation, binning, and return computation)
2. Repeat M times: shuffle candles within a block structure that preserves local autocorrelation; recompute the full-pipeline metric
3. p-value = fraction of permuted metrics ≥ observed metric
4. Param combo passes Stage 2 if p-value ≤ α_2 (e.g., 0.05)

**Stage 2 is expensive** because it reruns the full pipeline per permutation. It is therefore applied only to Stage 1 passers to limit computation.

**Stage 2 outcome per param combo:** pass or fail (diagnostic label)

### Feature-Level Verdict (the Gate)

After Stages 1 and 2, a feature-level verdict is computed:

```
Feature PASSES IS screening if:
    ANY param combo passes both Stage 1 AND Stage 2.

Feature FAILS IS screening if:
    No param combo passes both Stage 1 AND Stage 2.
```

**This is the only gate in Phase 2.** The feature-level verdict determines whether the feature advances to Tier 2. It is binary: yes or no.

### Critical Design Choice: Per-Param Results Are Diagnostic Only

The individual Stage 1 and Stage 2 results per param combo are **not used to filter which params enter the walkforward**. All param combos in the grid enter the walkforward regardless of their IS permutation pass/fail status.

**Why?** The pre-committed param selection rule uses neighbor smoothing. Neighbor smoothing requires the full param grid to correctly compute stable regions. If params that failed IS permutation were excluded from the walkforward grid, the neighbor structure would be distorted, the stability_ratio values would be incorrect, and the selection rule would make different (and less reliable) decisions than production would make.

There is also a subtler reason: a param combo that appears weak in isolation on the IS period may contribute genuine diversification within an ensemble. The walkforward selection rule evaluates params in the context of their neighborhood; the IS permutation test evaluates them in isolation. These are different questions.

### What IS Screening Rules Out

IS screening catches:

- Features with no relationship to forward returns across any lookback (e.g., a random indicator, a feature computed from the wrong data)
- Features that appear to work in EDA but fail permutation tests, suggesting the IS EDA result was data-mined
- Features with degenerate output (constant signal, all zeros) that would trivially fail but should be caught early

IS screening does not catch:

- Subtle overfitting to the IS period (this is what the walkforward catches)
- Parameter-specific overfitting (this is what neighbor smoothing and walkforward consistency catch)
- Regime-specific features that will fail in future regimes (this is partially addressed by walkforward fold diversity)

---

## 8. Phase 3: Walkforward Validation

**Period:** OOS folds within IS (typically 2015–2023 for test folds)
**Gate:** YES — neighborhood consistency must be stable in ≥ min_folds_stable folds
**Role:** Definitive robustness test; emulates what production would have done historically

### Fold Structure

The walkforward uses an expanding training window:

```
Fold 1:  Train: 2000–2015  |  Test: 2015–2016
Fold 2:  Train: 2000–2016  |  Test: 2016–2017
Fold 3:  Train: 2000–2017  |  Test: 2017–2018
Fold 4:  Train: 2000–2018  |  Test: 2018–2019
Fold 5:  Train: 2000–2019  |  Test: 2019–2020
Fold 6:  Train: 2000–2020  |  Test: 2020–2021
Fold 7:  Train: 2000–2021  |  Test: 2021–2022
Fold 8:  Train: 2000–2022  |  Test: 2022–2023

Total OOS coverage: 2015–2023 (8 years, each year evaluated exactly once)
```

**Training window requirement:** The initial training window must cover multiple distinct market regimes. For daily data, a minimum of 15 years (2000–2015) is required. This ensures that the selection rule has seen enough variation in market conditions to be meaningfully calibrated.

**Why expanding windows?** Expanding windows use all available historical data for each fold. This gives the selection rule more information in later folds, which mirrors what production does (it also accumulates data over time). Rolling windows would discard early data, wasting information and creating artificial regime breaks.

### Per-Fold Workflow

In each fold, the following sequence is executed:

```
Step 1: Fit ALL param combos on training data.
        (e.g., for RSI, fit lookbacks [2,3,4,5,10,14,20] on 2000–2015)

Step 2: Compute raw_metric and smoothed_metric for each param.
        Identify the stable neighborhood (params with stability_ratio ≥ 0.80).

Step 3: Apply pre-committed selection rule.
        → Set of selected params for this fold's test period.

Step 4: Compute ensemble signal = mean(signal(p) for p in selected).

Step 5: Evaluate ensemble signal on test fold (OOS).
        Record: test Sharpe, selected param set, neighborhood centroid.

Step 6: Record diagnostic outputs:
        - Which params were selected?
        - How many params qualified?
        - What was the neighborhood's location in param space?
```

### Stability Assessment

After all 8 folds are evaluated, the selected param neighborhoods are compared across folds. Three stability outcomes are possible:

**Stable (good):** The selected param neighborhood is consistent across ≥ 3 out of 8 folds. The same region of param space (e.g., lookbacks 5–10 for RSI) is selected repeatedly. This means the feature's optimal region is not shifting arbitrarily; the selection rule is latching onto a genuine structural property.

**Drifting (acceptable):** The selected neighborhood moves gradually in param space as the training window expands. For example, in early folds the selection clusters around lookback-5; in later folds it shifts to lookback-10. Gradual drift can indicate that the feature's optimal timescale is evolving with market structure. This is acceptable if the drift is monotone and slow. Recency weighting can be used to account for drift in production.

**Unstable (bad):** The selected neighborhood jumps randomly across folds with no consistent pattern. In some folds lookback-2 is selected; in others lookback-20; in others nothing qualifies. This indicates the feature has no stable signal; the selection rule is picking up noise that happens to be significant in each training window by chance.

**The gate:** The feature must exhibit stable or drifting behavior in at least `min_folds_stable` folds (default: 3 out of 8). Features with unstable neighborhoods are rejected here, before the permutation test.

### No Per-Fold IS Permutation Tests

Phase 3 does not run permutation tests within each training fold. The reason is twofold:

1. **Redundant with OOS metrics.** If the selection rule consistently selects a coherent neighborhood and the test fold Sharpes are positive, this is more informative than a per-fold IS permutation test. The OOS metric is the ground truth.

2. **Computationally prohibitive.** Running N=500 permutation trials inside each of 8 folds, where each trial involves refitting all param combos, would multiply computation by 500×. Phase 4 provides the permutation-based statistical test at the walkforward level, which is the correct level of analysis.

---

## 9. Phase 4: Walkforward Permutation Test

**Period:** Same fold structure as Phase 3
**Gate:** YES — original walkforward Sharpe must exceed (1-α) quantile of null distribution
**Role:** Statistical test that the selection rule + feature class produces genuine OOS signal

### What Is Being Tested

Phase 3 produces an aggregate walkforward Sharpe: the combined performance across all 8 test folds. Phase 4 tests whether this aggregate Sharpe is significantly better than what would be expected by chance.

The null hypothesis is: "The feature's signal, when processed by the pre-committed selection rule through the full walkforward procedure, produces no more aggregate OOS Sharpe than a randomly permuted version of the same feature."

This is more powerful than testing individual params because it tests the entire procedure: the selection rule, the ensemble formation, and the OOS evaluation across all folds.

### Two-Region Shuffle

The permutation is a two-region shuffle, not a simple global shuffle:

```
Region 1: First training window (2000–2015)
Region 2: Remaining data (2015–2023)

Permutation:
  - Shuffle feature values within Region 1 (preserving index within region)
  - Shuffle feature values within Region 2 (preserving index within region)
  - Do NOT shuffle across the region boundary
```

Why two-region? The first training window is used to initialize the walkforward; its composition affects all subsequent folds. The boundary between regions corresponds to the start of the OOS evaluation period. Shuffling across this boundary would destroy the temporal structure that defines "training" vs "test", making the null distribution unrealistically permissive.

### Procedure

```
1. Run Phase 3 walkforward with original feature values.
   Record: aggregate_sharpe_observed

2. Repeat N times (e.g., N = 500):
   a. Generate permuted feature by two-region shuffle.
   b. Run full Phase 3 walkforward on permuted feature.
      (This includes fitting all params, applying selection rule,
       forming ensemble, evaluating OOS — the complete procedure.)
   c. Record: aggregate_sharpe_permuted[i]

3. Build null distribution from {aggregate_sharpe_permuted[1..N]}.

4. p-value = fraction of permuted Sharpes ≥ aggregate_sharpe_observed.

5. Feature passes Phase 4 if: p-value ≤ α (e.g., α = 0.05)
   Equivalently: aggregate_sharpe_observed > (1-α) quantile of null.
```

### Why This Is the Correct Test

The walkforward permutation test is more stringent than simple IS permutation because:

- It tests OOS performance (not IS performance)
- It includes the selection rule, so it accounts for any luck in parameter selection
- It builds a null distribution that reflects the actual procedure's variance
- N=500 permutations gives a reliable estimate of the (1-α=0.95) quantile

A feature can pass IS permutation but fail walkforward permutation if the selection rule happened to select lucky params in the IS period that did not generalize. The walkforward permutation test catches this.

---

## 10. Phase 5: Graduation and Production

**Gate:** All three gates must pass (IS screening, walkforward consistency, walkforward permutation)
**Role:** Confirm feature graduation and specify production behavior

### Graduation Criteria

A feature graduates to production if and only if ALL of the following hold:

```
Criterion 1 (IS Screening Gate):
    At least one param combo passed both Stage 1 and Stage 2 IS permutation.

Criterion 2 (Walkforward Consistency Gate):
    The selected param neighborhood is stable or drifting in ≥ min_folds_stable folds.
    (Default: min_folds_stable = 3 out of 8)

Criterion 3 (Walkforward Permutation Gate):
    Aggregate walkforward Sharpe > (1 - α) quantile of null distribution.
    (Default: α = 0.05)
```

If any criterion fails, the feature is rejected. The researcher receives the full diagnostic output (EDA plots, per-fold selections, permutation histograms) to understand why.

### Researcher Review

After all criteria are evaluated, the researcher reviews the diagnostic output before confirming deployment. This review is the one point of researcher judgment in the pipeline. The researcher is looking for:

- Obvious artifacts in the data (missing data periods, corporate actions causing spurious signals)
- Whether the walkforward consistent neighborhood is economically sensible (e.g., RSI lookback-5 is a plausible signal; lookback-2 may be noise)
- Whether drift in the neighborhood is sensible given market structure changes

The researcher **confirms** graduation; they do not **decide** it. The pipeline decides; the researcher validates that no procedural error occurred.

### Production Behavior

In production, the exact same pre-committed selection rule is applied:

```
On a scheduled basis (e.g., daily or weekly):
  1. Take expanding training window (all available historical data).
  2. Fit ALL param combos on training data.
  3. Apply pre-committed selection rule → select qualifying params.
  4. Compute ensemble signal = mean(signal(p) for p in selected).
  5. Pass ensemble signal to position sizing.
  6. Log which params were selected and their metrics.
```

No researcher input is required per period. The system automatically refits, reselects, and generates signals. The logged selection history can be used to monitor for neighborhood drift over time, triggering a review if drift exceeds expected bounds.

### Strict OOS Check

After graduation, and only after graduation, the researcher may evaluate the feature on the strict OOS period (2024–2025). This is a one-time sanity check, not a gate. If the strict OOS performance is wildly inconsistent with walkforward performance, it warrants investigation but does not automatically reverse the graduation decision (the strict OOS period is too short for reliable statistical inference).

---

## 11. Cross-Validation Methods

Two cross-validation schemes are used as diagnostics within this pipeline. Neither replaces the sequential expanding walk-forward (Phase 3), which remains the primary live-trading simulation. Both are applied within the IS period (2000–2023).

CV applications in Stages 1+2 use **purging and embargoing** at every fold boundary (Purge length = max label horizon + max feature lookback; Embargo length = max label horizon + 1 bar) to prevent information leakage where a training set and test set are separated by a boundary. Stage 3 IS stability uses **boundary trimming** instead — each fold is evaluated on its own data with no internal train/test split, so trimming the edge observations by the label horizon is sufficient to remove cross-fold autocorrelation contamination.

### K-Fold CV (Non-Shuffled)

K-fold divides the IS period into k contiguous temporal blocks. For each fold held out as OOS, the remaining k-1 folds form the IS set. Temporal ordering is preserved (no shuffling). Purging and embargoing are applied at every boundary.

**Where it is used in this pipeline:**

| Phase | Application |
|---|---|
| Phase 2, Stages 1+2 | Fold-by-fold permutation p-values — runs the vector shuffle permutation test independently on each of k IS folds, then aggregates p-values using Fisher's combined probability test. Provides a consistency diagnostic: does the feature beat chance in each sub-period, or only in aggregate? |
| Phase 2, Stage 3 | Formalizes IS stability analysis with correct boundary trimming at fold edges. Replaces raw non-overlapping block splits with properly trimmed k-fold. |

See [`Cross_Validation/kfold_cv.md`](Cross_Validation/kfold_cv.md) for the full specification.

### Combinatorial Purged CV (CPCV)

CPCV generates all C(k, n_test) combinations of which n_test groups are held out as OOS, while the remaining k-n_test groups form the IS set. With k=8 and n_test=2, this yields 28 distinct backtests — far more than the 8 paths from sequential k-fold. Each path applies the full pre-committed selection rule and ensemble formation.

**Where it is used in this pipeline:**

| Phase | Application | Role |
|---|---|---|
| Phase 2, Stage 3 | Param selection frequency over C(k, n_test) paths | Supplementary / stronger alternative to k-fold IS stability |
| Phase 3 | Sharpe distribution + PBO (Probability of Backtest Overfitting) | Supplementary diagnostic only |

**CPCV does NOT replace Phase 3 sequential WF.** Sequential WF always trains on past data to predict future data, exactly as production does. In CPCV paths, IS and OOS groups can be interleaved in time — group 5 (2018–2020) may be in the IS set while group 3 (2012–2014) is OOS. This is statistically valid with purging but does not simulate production. Phase 3 primary metric remains the sequential WF aggregate Sharpe. CPCV provides the distribution and PBO (ω_n = logit of OOS rank of the IS-selected strategy; ω_n < 0 indicates overfit).

See [`Cross_Validation/cpcv.md`](Cross_Validation/cpcv.md) for the full specification including PBO computation.

### What CV Does Not Replace

| Sequential Walk-Forward | K-Fold / CPCV |
|---|---|
| Live-trading simulation: always train on past, test on future | IS diagnostic: IS and OOS can be interleaved in time |
| Primary Phase 3 gate | Supplementary diagnostics |
| Proves production-like generalization | Estimates distribution of outcomes; catches concentrated regime performance |

---

## 12. Gates vs Informational Outputs

The following table classifies every output of the pipeline as either a hard gate (a binary pass/fail that determines whether the feature advances) or an informational diagnostic (available to the researcher but not used as a decision criterion).

| Output | Phase | Role | Gate? |
|---|---|---|---|
| IS EDA plots (distributions, decile plots, temporal stability) | Phase 1 | Researcher understanding of feature | No |
| IS EDA rolling objective metric | Phase 1 | Researcher understanding of persistence | No |
| IS param grid raw metric heatmap | Phase 1 | Landscape intuition before smoothing | No |
| IS neighbor-smoothed stability landscape | Phase 1 | Intuition about stable regions | No |
| IS Stage 1 per-param pass/fail | Phase 2 | Diagnostic label per param combo | No — does NOT gate walkforward param entry |
| IS Stage 2 per-param pass/fail | Phase 2 | Diagnostic label per param combo | No — does NOT gate walkforward param entry |
| IS feature-level pass/fail | Phase 2 | Coarse filter verdict | YES — gates Tier 2 entry |
| Walkforward per-fold OOS metrics | Phase 3 | Evidence of fold-by-fold performance | No — input to next gate |
| Walkforward per-fold selected param set | Phase 3 | Diagnostic for consistency analysis | No — input to next gate |
| Walkforward neighborhood consistency verdict | Phase 3 | Stability assessment | YES — must be ≥ min_folds_stable |
| Walkforward permutation null distribution | Phase 4 | Statistical context | No — used to compute next gate |
| Walkforward permutation p-value | Phase 4 | Statistical gate | YES — must pass α threshold |
| Strict OOS performance | Phase 5 | Sanity check post-graduation | No — informational only |

**Key insight on per-param pass/fail:** The IS Stage 1 and Stage 2 results for individual param combos appear in the researcher's report as diagnostic labels. A researcher who sees that lookback-14 passed Stage 1 and Stage 2 while lookback-20 failed Stage 1 gains information about the feature landscape. But this information does not filter which params enter the walkforward. The pipeline never uses per-param IS results to gate walkforward participation.

---

## 13. Two-Tier Structure Rationale

### Why Two Tiers?

The pipeline could, in principle, skip IS screening and run the walkforward on every feature a researcher proposes. The two-tier structure exists for a practical reason: the walkforward (especially Phase 4 with N=500 permutation trials) is expensive. Running it on a feature that any permutation test on the full IS period would reject wastes computation.

IS screening catches the clear rejections cheaply. Walkforward validation provides the definitive answer for features that survive screening.

### Why IS Screening Graduates Features, Not Param Combos

The IS screening verdict is at the feature level (RSI: pass/fail), not the param level (RSI-14: pass/fail, RSI-20: pass/fail). This is intentional and critical.

If IS screening could gate which params enter the walkforward, a researcher might reason: "RSI-14 passed IS permutation, so I'll only run RSI-14 in the walkforward." But the walkforward's selection rule requires the full param grid for neighbor smoothing. Selectively entering params based on IS results would:

1. Distort the neighbor structure used by the selection rule
2. Create an inconsistency between the walkforward's decision procedure and production's decision procedure (production sees all params, walkforward would not)
3. Introduce implicit selection bias: the params entering the walkforward would be pre-selected to look good on IS data

By graduating the feature (not the params), the pipeline ensures the walkforward is run with the full grid, exactly as production will run it.

### Why Walkforward Folds Are OOS but Within IS

The walkforward folds use data from 2015–2023 as test folds. This data is within the IS period (2000–2023), which means it has been "seen" in the sense that EDA was run on it. However, the walkforward test folds are OOS relative to each training fold: when evaluating the test fold 2015–2016, the selection rule only saw data up to 2015.

The distinction matters: the IS period is not "training data." It is the data available for all research. The strict OOS period (2024–2025) is the data that has never been touched, not even by EDA. The walkforward folds are genuine OOS tests of the selection rule, but they are not as strong as a test against the strict OOS period. This is why the walkforward permutation test (Phase 4) is needed: it provides statistical rigor even when the test folds have been seen at a high level.

---

## 14. Related Documentation

The following documents describe specific components of this pipeline in detail. This master specification describes how those components fit together; the linked docs describe how to implement or use each component.

| Document | Phases Covered | Contents |
|---|---|---|
| `feature_validator.md` | Phases 1 & 2 | IS EDA implementation, IS permutation screening setup, output formats, parameter configuration |
| `Permutation Testing/in-sample_pt.md` | Phase 2 | Stage 1 (vector shuffle) and Stage 2 (pipeline/candle shuffle) permutation test specifications, null distribution construction, significance thresholds |
| `Walkforward/walkforward.md` | Phases 3 & 4 | Walkforward fold structure, expanding window logic, per-fold workflow implementation, walkforward permutation test implementation |
| `Parameter Sensitivity/grid_search_parameter_stability.md` | All phases | Neighbor smoothing theory, stability_ratio computation, param grid heatmap generation |
| `Parameter Sensitivity/param_selection_rule.md` | Phases 3, 4, 5 | Pre-committed selection rule full specification, K_min behavior, ensemble formation, production refit schedule |
| `Cross_Validation/kfold_cv.md` | Phase 2 | K-fold CV specification: fold-by-fold permutation diagnostics, IS stability formalization, purging and boundary trimming requirements |
| `Cross_Validation/cpcv.md` | Phases 2 & 3 | CPCV specification: param selection frequency, PBO computation (logit-rank formulation), Sharpe distribution, limitations |

### Reading Order for New Researchers

1. Start here (this document) to understand the full pipeline and the role of each phase.
2. Read `Parameter Sensitivity/param_selection_rule.md` to understand the selection rule in depth, since it is central to Phases 3, 4, and 5.
3. Read `Parameter Sensitivity/grid_search_parameter_stability.md` to understand neighbor smoothing, which underpins the selection rule.
4. Read `feature_validator.md` for Phase 1 and Phase 2 implementation details.
5. Read `Walkforward/walkforward.md` for Phase 3 and Phase 4 implementation details.
6. Read `Permutation Testing/in-sample_pt.md` for the statistical tests in Phase 2.

---

*End of Feature Validation Pipeline Master Specification.*
