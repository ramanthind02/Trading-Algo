# Combinatorial Purged Cross-Validation (CPCV)

> **Status:** Library specification
> **Role in pipeline:** Phase 2 Stage 3 (IS stability — param selection frequency over many paths) and Phase 3 supplementary (Sharpe distribution + PBO). See [pipeline_overview.md](../pipeline_overview.md) for the full pipeline.
> **Last updated:** 2026-02-19
> **Reference:** López de Prado (2018), *Advances in Financial Machine Learning*, Chapter 12.

## What This Document Covers

Combinatorial Purged Cross-Validation (CPCV) applied to feature validation. This document covers:

1. What CPCV is and how it differs from sequential walk-forward and k-fold
2. Purging and embargoing requirements with interleaved IS/OOS groups (more complex than k-fold)
3. Application in **Phase 2 Stage 3** (IS stability): param selection frequency over C(k, n_test) paths
4. Application in **Phase 3** (supplementary): Sharpe distribution and Probability of Backtest Overfitting (PBO)
5. What CPCV does NOT replace

**Critical:** CPCV does not replace the sequential expanding walk-forward (Phase 3). Sequential WF is the live-trading simulation — IS data always comes before OOS data. CPCV interleaves IS and OOS groups in time, which cannot simulate production behavior. CPCV is used as a supplementary diagnostic to estimate the distribution of outcomes and compute PBO.

---

## What Is CPCV?

Standard k-fold generates k backtests (one per fold held out as OOS). CPCV generates C(k, n_test) backtests by considering all combinations of which n_test groups are held out as OOS at once, while the remaining k-n_test groups form the IS set.

```
K = 6 groups, n_test = 2:
  C(6, 2) = 15 distinct backtests

Example backtests:
  Backtest 1:  IS = groups {1,2,3,4}  OOS = groups {5,6}
  Backtest 2:  IS = groups {1,2,3,5}  OOS = groups {4,6}
  Backtest 3:  IS = groups {1,2,3,6}  OOS = groups {4,5}
  ...
  Backtest 15: IS = groups {3,4,5,6}  OOS = groups {1,2}

Note: In some backtests, OOS groups are not contiguous in time
(e.g., groups {4,6} as OOS means group 6 is OOS but group 5 is IS).
```

Each of the 15 backtests produces an OOS metric. The distribution of these 15 metrics is far richer than the single path from sequential WF.

### Key Properties

**More backtests → better distribution estimate.** With k=6, n_test=2, you get 15 paths. With k=10, n_test=2, you get 45 paths. This allows computing mean, variance, and tail quantiles of OOS performance.

**IS/OOS ratio is (k - n_test) / k.** With k=6, n_test=2: IS uses 4 groups, OOS uses 2 → IS = 66%, OOS = 33% of total data per path.

**IS and OOS groups can be non-contiguous in time.** In backtest 2 above, IS groups are {1,2,3,5} — groups 1,2,3 come before group 4, but group 5 comes after. This means the model is trained on data that is partially after the test data. This is valid statistically (with correct purging), but does NOT mirror production. Production always trains on data from the past to predict the future.

---

## Purging and Embargoing in CPCV

CPCV purging is more complex than k-fold purging because IS and OOS groups can be interleaved in time. Every IS-OOS boundary requires purging and embargoing, regardless of whether the IS group precedes or follows the OOS group in time.

```
Example: IS = {1,2,3,5}, OOS = {4,6}

Time ordering: [Group 1][Group 2][Group 3][Group 4][Group 5][Group 6]
                 IS        IS       IS      OOS       IS      OOS

Boundaries requiring purge+embargo:
  → Between Group 3 (IS) and Group 4 (OOS): purge end of Group 3, embargo start of Group 4
  → Between Group 4 (OOS) and Group 5 (IS): purge start of Group 5 (contaminated by Group 4 labels), embargo
  → Between Group 5 (IS) and Group 6 (OOS): purge end of Group 5, embargo start of Group 6
```

**Purging rule for CPCV:** For each IS-OOS boundary (in either direction), purge observations whose label horizon extends across the boundary. Embargo length = same as k-fold (max label horizon + 1 bar).

The purging must be applied bilaterally — not just at the leading edge of the IS-OOS transition but at every boundary where IS and OOS groups touch.

---

## Application 1: Phase 2 Stage 3 — Param Selection Frequency

### Purpose

Stage 3 IS stability analysis currently uses K non-overlapping folds and counts, for each param combo, how many folds it appears in the selected set. With K=8 folds, you get at most 8 stability data points.

CPCV dramatically increases the number of paths:

```
K=8, n_test=2: C(8,2) = 28 paths
K=10, n_test=2: C(10,2) = 45 paths
K=10, n_test=3: C(10,3) = 120 paths
```

For each of the C(k, n_test) paths, run the IS stability procedure:
1. Fit ALL param combos on the IS groups
2. Compute raw metric and neighbor-smoothed metric per param
3. Apply pre-committed selection rule → set of selected params for this path
4. Record which params were selected

**Aggregate:** For each param combo p, compute:
```
selection_frequency(p) = (number of paths in which p is selected) / C(k, n_test)
```

A param with `selection_frequency ≥ 0.70` (selected in ≥ 70% of all CPCV paths) is far more convincingly stable than one that merely appears in 3 of 8 sequential folds.

### Recommended Configuration for IS Stability

| IS data length | K | n_test | C(k, n_test) paths |
|---|---|---|---|
| 15–24 years daily | 8 | 2 | 28 |
| 15–24 years daily | 10 | 2 | 45 |
| > 24 years daily | 10 | 3 | 120 |

Start with K=8, n_test=2 for 24 years of IS data. Each IS fold has approximately 3 years of data (24 / 8 = 3 years per group), and each path uses 6 groups as IS (18 years). This is enough for reliable param fitting.

---

## Application 2: Phase 3 Supplementary — Sharpe Distribution and PBO

### Purpose

Sequential walk-forward (Phase 3) produces **one** aggregate Sharpe across all test folds. This is a point estimate. CPCV produces a **distribution** of aggregate Sharpes (one per combinatorial path), enabling:

1. **Sharpe distribution:** Mean, standard deviation, 5th/95th percentile of OOS Sharpe
2. **PBO (Probability of Backtest Overfitting):** What fraction of CPCV paths produce a worse Sharpe than the median across all paths?

**This supplements sequential WF; it does not replace it.** Use sequential WF as the primary Phase 3 metric (it simulates production). Use CPCV to assess how robust the result is.

### PBO Computation

PBO is computed as follows (López de Prado, Chapter 14):

```
1. Run CPCV with C(k, n_test) paths.
   Each path i produces: IS_sharpe_i, OOS_sharpe_i

2. For each path i, rank the IS Sharpe relative to all other paths:
   rank_IS_i = rank of IS_sharpe_i among all {IS_sharpe_j}

3. OOS Sharpe for the "best IS" path (path with highest IS Sharpe):
   Let i* = argmax IS_sharpe_i
   OOS_i* = OOS Sharpe of path i*

4. If OOS_i* < median(all OOS Sharpes):
   This path counts as "overfit" — the path that looked best IS performs below median OOS.

5. PBO = fraction of all CPCV configurations (k, n_test choices) where this happens.
   High PBO (> 0.5) → the selection process is likely selecting noise.
   Low PBO (< 0.1) → the best-IS result generalizes well.
```

In practice, a simpler interpretation: run CPCV, plot IS Sharpe vs OOS Sharpe across all paths. If the correlation is positive and strong, the IS screening process generalizes. If the correlation is weak or negative, IS performance is not predictive of OOS performance — selection bias is likely.

### Recommended Configuration for Phase 3 Supplement

Use the same fold boundaries as Phase 3 walk-forward when possible, to ensure comparability:

```
Phase 3 walk-forward: 8 sequential folds (2015–2023 test period)
CPCV supplement: K=8, n_test=2 → 28 paths, using same 8-group boundaries
```

Each CPCV path uses 6 groups as IS (not necessarily temporally contiguous) and 2 groups as OOS. The pre-committed selection rule and ensemble formation are run identically in each path.

---

## What CPCV Does NOT Replace

### Sequential Walk-Forward (Phase 3) — Primary Test

Sequential WF (train 2000–2015, test 2015–2016; train 2000–2016, test 2016–2017; etc.) **always** uses data from the past to predict data from the future. This is the only configuration that accurately emulates what production would do.

CPCV paths where OOS group 3 (2010–2012) is evaluated with IS groups that include group 5 (2014–2017) are statistically valid but do not mirror production — a production system in 2011 would not have access to 2014 data.

**For this reason, Phase 3 primary metric remains the sequential WF aggregate Sharpe.** CPCV provides a richer distributional view, not a replacement for the production-emulation test.

### IS Permutation Tests (Phase 2 Stages 1+2)

CPCV is not a permutation test and should not be used as one. CPCV evaluates how a model's param selection generalizes across different IS/OOS splits. Permutation tests (Stages 1+2) evaluate whether observed performance is better than chance under the null. These are different questions.

---

## Limitations

1. **Temporal realism breaks down.** In CPCV paths where OOS groups precede IS groups in time, the model is trained on "future" data relative to the test period. This is not a statistical problem (purging handles leakage), but it means these paths cannot be interpreted as simulating a historically plausible trading strategy.

2. **Compute cost.** C(k, n_test) × cost_per_path. With K=10, n_test=2, 45 paths: if each path takes 30 seconds (fit all params + apply selection rule), total = 22.5 minutes. With n_test=3 (120 paths), total = 60 minutes. For Phase 3 supplement, this may be acceptable; for Phase 2 Stage 3, evaluate the tradeoff against simpler k-fold.

3. **Dependency between paths.** CPCV paths are not independent — they share groups. Statistical corrections for path-level inference (e.g., hypothesis tests on PBO) are complex. Use PBO and selection frequency as diagnostics rather than formal statistical tests.

4. **Small IS per path with large n_test.** If n_test is too large relative to k, each path has a small IS set (few groups) → unreliable param fitting → noisy selection frequency estimates. Keep IS fraction ≥ 0.60 (i.e., n_test ≤ 0.40 × k).

---

## Related Documentation

| Document | Relationship |
|---|---|
| [pipeline_overview.md](../pipeline_overview.md) | Master pipeline spec; see Phases 2 and 3 |
| [in-sample_pt.md](../Permutation Testing/in-sample_pt.md) | Stage 3 IS stability spec; CPCV integrates here |
| [kfold_cv.md](kfold_cv.md) | K-fold CV — simpler alternative with fewer paths but correct temporal structure |
| [param_selection_rule.md](../Parameter Sensitivity/param_selection_rule.md) | Pre-committed selection rule run inside each CPCV path |
| [walkforward.md](../Walkforward/walkforward.md) | Phase 3 sequential WF — primary test that CPCV supplements |
