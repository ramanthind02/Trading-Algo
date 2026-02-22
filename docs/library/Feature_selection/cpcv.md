# Combinatorial Purged Cross-Validation (CPCV)

> **Role:** Phase 2 Stage 3 (IS stability — param selection frequency) and Phase 3 supplementary (Sharpe distribution + PBO)
> Reference: López de Prado (2018), *Advances in Financial Machine Learning*, Ch. 12 & 14.
> See [[pipeline]] for the full feature selection pipeline.

---

## What It Is

Standard k-fold produces **k** backtests (one fold held out at a time). CPCV generates **C(k, n_test)** backtests by exhausting all combinations of which `n_test` groups are held out as OOS simultaneously.

```
K=8, n_test=2:  C(8,2) = 28 paths
K=10, n_test=2: C(10,2) = 45 paths
K=10, n_test=3: C(10,3) = 120 paths

Example (K=6, n_test=2):
  Path 1:  IS = {1,2,3,4}   OOS = {5,6}
  Path 2:  IS = {1,2,3,5}   OOS = {4,6}   ← OOS groups non-contiguous in time
  Path 3:  IS = {1,2,3,6}   OOS = {4,5}
  ...
  Path 15: IS = {3,4,5,6}   OOS = {1,2}
```

- **IS/OOS ratio:** `(k - n_test) / k` — keep IS fraction ≥ 0.60 (`n_test ≤ 0.40 × k`)
- IS and OOS groups **can be non-contiguous in time** — valid statistically (with purging), but does NOT mirror production

---

## Purging and Embargoing

CPCV purging is more complex than k-fold because IS and OOS groups can interleave:

- Every IS-OOS boundary (in either temporal direction) requires purge + embargo
- **Bilateral purging:** purge both the leading and trailing edges at each IS-OOS transition
- Embargo length = same rule as k-fold: `max(label_horizon, feature_lookback_max) + 1` bars

```
IS={1,2,3,5}, OOS={4,6}  (time: Group 1 → 2 → 3 → 4 → 5 → 6)

Boundaries requiring purge+embargo:
  Group 3 (IS) → Group 4 (OOS): purge end of 3, embargo start of 4
  Group 4 (OOS) → Group 5 (IS): purge start of 5, embargo
  Group 5 (IS) → Group 6 (OOS): purge end of 5, embargo start of 6
```

---

## Phase 2 Stage 3 — Param Selection Frequency

**Purpose:** dramatically increase the number of IS stability paths beyond what k-fold provides.

**Procedure (per CPCV path):**
1. Fit ALL param combos on IS groups (including Stage 1+2 failures — required for neighbor smoothing neighborhood integrity)
2. Compute raw metric + neighbor-smoothed metric per param
3. Apply pre-committed selection rule → record selected param set for this path

**Aggregate:**
```
selection_frequency(p) = paths where p is selected / C(k, n_test)
```
- Threshold: `selection_frequency ≥ 0.70` → convincingly IS-stable

| IS data length | K | n_test | Paths |
|---|---|---|---|
| 15–24 years daily | 8 | 2 | 28 |
| 15–24 years daily | 10 | 2 | 45 |
| > 24 years daily | 10 | 3 | 120 |

> [!note] Why include Stage 1+2 failures?
> Neighbor smoothing requires the full param grid. Pre-filtering distorts neighborhood structure and makes `stability_ratio` unreliable.

---

## Phase 3 Supplementary — Sharpe Distribution and PBO

Sequential WF (Phase 3) produces **one** aggregate Sharpe. CPCV produces a **distribution** across C(k, n_test) paths.

**Uses:**
- Sharpe distribution: mean, std, 5th/95th percentile of OOS Sharpe
- **PBO (Probability of Backtest Overfitting)**

**PBO computation:**
```
1. Each path i → IS_sharpe_i, OOS_sharpe_i
2. i* = argmax IS_sharpe_i   (best in-sample path)
3. P_n = OOS rank of i* among all paths / (C(k, n_test) + 1)   ∈ (0,1)
4. ω_n = logit(P_n) = log(P_n / (1 - P_n))
```

| ω_n | Interpretation |
|---|---|
| ω_n < 0 (P_n < 0.5) | IS-best ranks below OOS median → selection bias / overfit |
| ω_n ≈ 0 | IS performance no better than chance at predicting OOS rank |
| ω_n > 0 (P_n > 0.5) | IS performance generalizes to OOS |

> [!tip] Simpler diagnostic
> Plot IS Sharpe vs OOS Sharpe across all paths. Strong positive correlation → IS performance is predictive. Weak or negative → selection bias likely.

**Recommended config for Phase 3 supplement:** use same fold boundaries as sequential WF
```
Phase 3 WF: 8 folds (2015–2023)
CPCV supplement: K=8, n_test=2 → 28 paths, same group boundaries
```

---

## What CPCV Does NOT Replace

> [!important] Sequential WF (Phase 3) remains the primary test
> CPCV paths where an OOS group precedes IS groups in time are statistically valid but cannot simulate production — a live system in 2011 cannot train on 2014 data. Phase 3 primary metric stays the sequential WF aggregate Sharpe.

- CPCV is not a permutation test — it does not test whether performance beats chance under the null
- IS permutation tests (Stages 1+2) answer a different question entirely

---

## Limitations

1. **Temporal realism breaks down** on paths where OOS precedes IS in time — valid statistically, not production-realistic
2. **Compute cost:** C(k, n_test) × cost-per-path; at 30s/path: K=10, n_test=3 → 120 paths = 60 min
3. **Path dependency:** paths share groups; p-values on PBO are complex — use as diagnostics, not formal tests
4. **Small IS with large n_test:** keep IS fraction ≥ 60% to ensure reliable param fitting per path

---

## Related

- [[pipeline]] — master pipeline; Phases 2 and 3
- [[walkforward]] — Phase 3 sequential WF; the primary test CPCV supplements
- [[kfold]] — simpler alternative with fewer paths but strict temporal ordering
- [[param_stability]] — neighbor smoothing theory used inside each CPCV path
