# Out-of-Sample (OOS) Validation

> [!note] Status: Library reference — OOS phase implementation (Phases 6–8)
> Last updated: 2026-02-23

---

## Overview

After a feature passes all in-sample (IS) and walkforward (WF) gates (Phases 1–5), it enters the OOS period (2024–2025) for final validation. This period was untouched during all prior analysis.

**Three OOS phases:**
- **Phase 6: OOS Walk-Forward Validation** — confirm param stability on fresh OOS folds (same algorithm as Phase 4 WF Validation)
- **Phase 7: OOS Permutation Test** — confirm feature significance on OOS data (same threshold as Phase 5 WF Permutation)
- **Phase 8: Graduation** — apply pre-committed param selection rule, persist to vault

---

## Phase 6: OOS Walk-Forward Validation

Replicate Phase 4 (WF Validation) on out-of-sample folds.

### Fold Structure

Same expanding or rolling window logic as Phase 4, but applied to OOS data (2024–2025):

- Initial training window: last IS fold (e.g., 2022–2024)
- Test folds: rolling or expanding on OOS period
- Rolling window step: typical 1 year or user-specified

### Algorithm

Per fold:
1. Fit ALL param combos on training candles (up to `train_end`)
2. Compute raw objective metric (e.g., Sharpe) for each param
3. Apply **axis-aligned 1-step neighbor smoothing** → `smoothed_objective(P) = mean(obj(P), obj(neighbors))`
4. Identify stable region: contiguous params where `stability_ratio > threshold` (default 0.8)
5. Form ensemble from stable region params
6. Evaluate portfolio on test fold candles

### Pass Criterion

- Stable region reproducible in ≥ `min_folds_stable` of OOS folds (typical 2–3 folds out of 2–3 total OOS folds)
- If < 2 params selected globally → no position (insufficient evidence)

> [!important] Param Stability
> Phase 6 uses the same neighbor smoothing + stable region algorithm as Phase 4. See [[param_stability]] for algorithm details, quadrant interpretation, and configuration.

### Why Phase 6?

- IS/WF gates confirm theoretical signal + temporal stability within training data
- OOS validation confirms generalization to fresh data not touched during any prior analysis
- If param region becomes unstable or disappears in OOS, feature is rejected (overfitting detected)

---

## Phase 7: OOS Permutation Test

Replicate Phase 5 (WF Permutation) on out-of-sample data.

### Test Procedure

For each feature/param combo that passed Phase 6:

1. **Fit on original OOS data** → compute objective metric (`Sharpe`, `Sortino`, etc.)
2. **Shuffle candles** N times (default N=500) → refit → recompute metric
3. **Null distribution** → metrics from all shuffled runs
4. **p-value** = fraction of shuffled metrics ≥ original
5. **Gate**: p ≤ α (default α=0.05)

### Why Candle Shuffle?

- Preserves first/last OHLC anchors + intra-bar structure
- Destroys temporal order of trend/volatility clusters
- OOS period is short (1–2 years), so permutation power is reduced compared to IS (20+ years)

### Pass Criterion

If p ≤ 0.05, feature passes Phase 7 and is eligible for graduation.

> [!warning] Reduced power on OOS
> OOS permutation test has ~1/10 the sample size of IS permutation test. A feature that passes IS + WF + OOS permutation is highly robust (three independent gates).

---

## Phase 8: Graduation

Once all gates pass, apply the pre-committed param selection rule to choose final parameters.

### Param Selection Rule

Fit ALL param combos on full IS data (2000–2023):

1. Compute objective metric (Sharpe) for each param
2. Apply neighbor smoothing → `smoothed_objective(P)`
3. Hard filters: `bin_count ≥ bin_count_min` (default 5), `trade_frequency ≥ min_freq` (default 0.05)
4. Relative floor: `floor = best - δ * σ` (default δ=0.20) or adaptive
5. Select stable region: contiguous params where `smoothed(P) > floor`
6. From valid region, select top-K params by smoothed objective (default K=3)

### Selection Criteria

- **0 params selected** → no position (feature exists but not deployable)
- **1 param selected** → questionable, marginal confidence
- **2–6 params selected** → solid ensemble (typical case)
- **>6 params selected** → broad region, high stability

If fewer than 2 params selected, the feature is kept in the vault but marked "no_position" — it passed all gates but has insufficient param-level evidence for ensemble formation.

### Vault Storage

Persist to [[vault]] with metadata:
- Feature name, ticker, timeframe, bias node + parameter
- Selected param list (or "no_position" if < 2 params)
- Permutation test results (p-values from Phases 3, 5, 7)
- IS + WF + OOS performance metrics

Feature is now approved for live deployment.

---

## Summary: OOS vs IS vs WF

| Aspect | IS (Phase 3) | WF (Phase 5) | OOS (Phase 7) |
|--------|------|-----|-----|
| Data | 20–23 years (2000–2023) | Full IS period, rolling folds | 1–2 years (2024–2025) |
| Sample size | Large (high power) | Medium (per-fold) | Small (low power) |
| Gate | p ≤ 0.10 (coarse filter) | p ≤ 0.05 (strict filter) | p ≤ 0.05 (out-of-sample validation) |
| Shuffle method | Vector + candle | Candle (full WF) | Candle (fresh data) |
| Purpose | Screen for basic signal | Confirm temporal stability | Confirm generalization |

**All three gates must pass** for graduation.

---

> [!info] See also
> - [[pipeline]] — full validation pipeline overview (Phases 1–8)
> - [[param_stability]] — neighbor smoothing, stable region selection, param selection rule
> - [[walkforward]] — Phase 4–5 walkforward (replicated in Phase 6)
> - [[vault]] — feature storage and deployment
