# Weight Layer — Specification

> **Status:** Active library specification
> **Date:** 2026-02-21
> **Scope:** Forecast combination and diversification multiplier

---

## Table of Contents

1. [Role in the Pipeline](#1-role-in-the-pipeline)
2. [Design Principle: No Sharpe Tilt](#2-design-principle-no-sharpe-tilt)
3. [Grouping Strategy](#3-grouping-strategy)
4. [Algorithm](#4-algorithm)
5. [Method Ladder](#5-method-ladder)
6. [Forecast Diversification Multiplier](#6-forecast-diversification-multiplier)
7. [Configuration](#7-configuration)
8. [Output Contract](#8-output-contract)
9. [References](#9-references)

---

## 1. Role in the Pipeline

```
Base Models → DiversifiedEnsemble → WeightLayer → Portfolio → PositionSizer
```

The WeightLayer receives per-model volatility-scaled forecasts from one or more DiversifiedEnsembles and combines them into a single `forecast_score` per ticker. It applies the Forecast Diversification Multiplier (FDM) to account for the diversification benefit of combining multiple signals.

The WeightLayer is the primary forecast combiner.

Upstream selection methods are selection-only and determine membership; this layer performs the first portfolio-impacting forecast combination.

**What the WeightLayer must do:**
- Combine signals to maximise diversification, especially in downside regimes
- Produce stable weights fold-to-fold (weight instability is a sign of overfitting)
- Apply FDM to preserve the volatility-targeting properties of the forecasts

**What the WeightLayer must not do:**
- Re-optimise for Sharpe or expected return
- Invert a large or near-singular covariance matrix
- Make decisions that depend on high-variance estimates from small training windows

---

## 2. Design Principle: No Sharpe Tilt

Each base model forecast is already volatility-targeted and exposure-adjusted:

```
F_i = (τ / (σ × √h_i)) × X_i
```

where `h_i = 1/n_bins` is the exposure fraction (how often the signal fires). The scaling ensures that when a signal is active, it contributes a consistent expected volatility regardless of its parameter or bin structure.

Additionally, the param selection step (see [top_k_ensemble_selection.md](../Feature_selection/Parameter%20Sensitivity/top_k_ensemble_selection.md)) already filters for stable, high-quality param combos before any signal reaches the WeightLayer. What enters the WeightLayer is a pre-screened set.

**Therefore: the WeightLayer's objective is purely diversification — not Sharpe ranking.**

Introducing Sharpe-based weights at this stage would:
1. Amplify training-window estimation noise (Sharpe from a single fold is a high-variance estimate)
2. Add a second layer of performance ranking on top of param selection, creating implicit look-ahead into which signals performed best on training data
3. Undermine the clean separation between signal quality (handled by param selection) and portfolio construction (handled by WeightLayer)

All weighting methods in this spec are purely risk-based. No expected return estimation is performed.

---

## 3. Grouping Strategy

Before any correlation-based weighting is applied, signals are assigned to groups. This is the most important structural decision in the WeightLayer.

### Why grouping matters

Within a feature family (e.g., RSI param combos [3, 4, 5]), signals are near-perfectly correlated by construction — they are the same indicator at nearby parameter values in the same stable region. Applying a correlation-based weighting scheme to these signals directly produces a near-singular matrix and numerically unstable weights.

Grouping separates two distinct sub-problems:
- **Within-group**: signals are near-identical → equal weights are mathematically near-optimal with zero estimation error
- **Across-group**: signals come from genuinely different feature families → correlation-based allocation adds real value and is reliably estimated

This reduces the effective dimension of the allocation problem from N (number of individual signals) to K (number of groups, typically 3–8), making the correlation matrix tractable and well-conditioned.

### How groups are defined

**Primary method — pre-specified by feature family (preferred):**

Groups are assigned based on the feature family of each base model. All RSI param combos form one group, all EWMAC param combos form another, etc. This requires zero estimation and cannot overfit.

```
Group assignment is determined by the DiversifiedEnsemble configuration,
not by training data. It does not change fold-to-fold.
```

**Fallback — data-driven clustering (when feature families are ambiguous):**

Apply hierarchical clustering to the full-period signal correlation matrix with a pre-committed cutoff `ρ_cut = 0.70`. Signals with full-period correlation > 0.70 are assigned to the same group. This is a binary membership decision, not a continuous weight — estimation noise has limited impact on a binary outcome.

Use full-period correlation for grouping (not downside). Grouping detects structural similarity between signals; full-period correlation is more stably estimated and appropriate for this structural purpose. Downside correlation is reserved for the allocation step (Step 3 below).

The cutoff `ρ_cut = 0.70` is pre-committed. It is not tuned per feature or per fold.

---

## 4. Algorithm

### Step 1: Within-Group Aggregation

For each group k, compute the group signal and group return stream using equal weights over all group members:

```
group_signal_k(t) = mean(F_i(t)  for i in group_k)
group_return_k(t) = mean(r_i(t)  for i in group_k)
```

Equal weights within groups are the default and the recommended production setting. The one alternative — weighting proportionally to each member's smoothed training objective (already computed by param selection, no additional estimation) — may be tested experimentally but adds sensitivity without clear theoretical justification.

### Step 2: Downside Semi-Covariance

For each group signal, compute the lower semi-return at every observation:

```
r_k^-(t) = min(group_return_k(t), 0)
```

This uses each signal's own negative returns rather than filtering to joint-negative periods. Advantages: no reference portfolio required (avoids circularity), every observation contributes, and the matrix captures how signals co-move on their own downside.

Compute the K×K downside semi-covariance matrix:

```
Σ^down_kl = (1/T) × Σ_t [ r_k^-(t) × r_l^-(t) ]
```

Apply Ledoit-Wolf shrinkage to Σ^down. With K ≤ 8 and ~3750+ daily observations, shrinkage has modest impact but is included as a robustness guard against near-singularity.

### Step 3: HRP on Group Matrix

Derive downside correlation and distance:

```
C^down_kl = Σ^down_kl / sqrt(Σ^down_kk × Σ^down_ll)

D_kl = sqrt(0.5 × (1 - C^down_kl))
```

Apply Ward linkage hierarchical clustering to D. Ward linkage is preferred over single or complete linkage for near-equal distances, which are common when group signals are moderately correlated.

Perform HRP recursive bisection on the resulting dendrogram:

```
At each binary split (left cluster L, right cluster R):
  - Compute downside variance of each cluster:
      Var_L = w_L^T × Σ^down_LL × w_L       (w_L = equal weights within L)
      Var_R = w_R^T × Σ^down_RR × w_R
  - Fraction to L:  v_L = Var_R / (Var_L + Var_R)
  - Fraction to R:  v_R = Var_L / (Var_L + Var_R)
  - Recurse within each cluster
```

Normalise final group weights to sum to 1:

```
w_1, ..., w_K   (Σ w_k = 1)
```

### Step 4: Combined Forecast

```
raw_forecast(t) = Σ_k  w_k × group_signal_k(t)
```

### Step 5: FDM

```
scaled_forecast(t) = raw_forecast(t) × FDM
```

FDM computation is described in Section 6.

---

## 5. Method Ladder

The WeightLayer supports multiple methods, ordered by complexity. In a walkforward experiment, all methods can be evaluated simultaneously on the same folds. The selection criterion is: **use the simplest method that consistently beats the next simpler method by a meaningful margin across multiple ensembles.**

Pre-committed threshold for "meaningful margin": aggregate Sharpe improvement ≥ 0.10 **and** maximum drawdown reduction ≥ 10%, sustained consistently across ensembles. Without a pre-committed threshold, there is a risk of rationalising whichever complex method looked better in-sample.

| Level | Method | Free parameters | Description |
|-------|--------|----------------|-------------|
| 0 | **Equal weights (flat)** | 0 | All N signals equal weight. The hard baseline. |
| 1 | **Equal within group, equal across group** | 0 | Grouping applied; no risk-based allocation. Tests whether grouping structure alone adds value. |
| 2 | **Equal within group, inverse downside vol across group** | K scalars | Group weights ∝ 1/σ^down_k. No matrix estimation. Directly targets drawdown without correlation. |
| 3 | **Equal within group, Downside-HRP across group** | K×K matrix | Full algorithm from Section 4. Default production method candidate. |
| 4 | **Downside-HRP on all N signals (no grouping)** | N×N matrix | Control experiment: does grouping improve stability over raw HRP on individuals? |

**Evaluation metrics applied to OOS walkforward folds:**
- Aggregate walkforward Sharpe
- Maximum drawdown across folds
- Average drawdown duration
- Fold-to-fold weight stability — std of group weight vector across folds

Weight stability is a primary diagnostic. A method producing dramatically different allocations each fold is overfitting to the training window even if aggregate Sharpe looks acceptable. Expect Level 4 (flat Downside-HRP) to show more weight instability than Level 3 (grouped) as N grows.

---

## 6. Forecast Diversification Multiplier

The FDM preserves the volatility-targeting calibration of the individual forecasts after combination. Without FDM, combining positively correlated signals produces a portfolio with lower realised volatility than the individual target, which systematically under-sizes positions.

FDM is computed from the mean pairwise downside correlation across groups (consistent with the weighting method):

```
mean_corr = mean of off-diagonal entries of C^down

FDM = min(sqrt(1 / (mean_corr + 0.01)), FDM_max)
```

`FDM_max = 2.0` (pre-committed cap).

When equal weights are used (Levels 0–1), FDM is computed from the full-period correlation matrix instead, which is more stably estimated and appropriate when there is no downside-weighted allocation to be consistent with.

**Interpretation:**
- `mean_corr → 1` (all signals identical): FDM → 1. No scaling — combination adds no diversification.
- `mean_corr → 0` (signals independent): FDM → 10 (capped at 2.0). Full diversification benefit captured.
- Typical range in practice: FDM 1.2–1.8 depending on cross-group correlation structure.

---

## 7. Configuration

All parameters are pre-committed before any walkforward fold begins. They are not adjusted based on observed results.

This layer is controlled by the `weighting_method` knob, while upstream membership is controlled by the feature-selection `selection_method` knob described in [top_k_ensemble_selection.md](../Feature_selection/Parameter%20Sensitivity/top_k_ensemble_selection.md#configuration).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weighting_method` | `inverse_correlation` | One of: `inverse_correlation`, `equal_flat`, `equal_grouped`, `inv_downside_vol_grouped`, `downside_hrp_grouped`, `downside_hrp_flat` |
| `group_method` | `feature_family` | `feature_family` (pre-specified) or `correlation_clustering` (data-driven fallback) |
| `rho_cut` | `0.70` | Correlation cutoff for data-driven grouping |
| `within_group_weights` | `equal` | `equal` or `smoothed_metric_proportional` |
| `linkage` | `ward` | Hierarchical clustering linkage: `ward`, `single`, `complete` |
| `shrinkage` | `ledoit_wolf` | Shrinkage estimator for Σ^down: `ledoit_wolf` or `none` |
| `fdm_max` | `2.0` | Cap on FDM |
| `fdm_correlation_source` | `downside` | `downside` or `full_period` for FDM computation |
| `weight_stability_threshold` | `0.20` | Max allowed fold-to-fold shift in any group weight before diagnostic warning |

---

## 8. Output Contract

**Per-ticker forecast output:**

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | str | Instrument identifier |
| `forecast_score` | float | FDM-scaled combined forecast |

**Fitted state logged per fold (for diagnostics and monitoring):**

| Field | Description |
|-------|-------------|
| `group_assignments` | Dict mapping model name → group ID |
| `group_weights` | w_1..w_K for this fold |
| `fdm` | FDM value for this fold |
| `mean_downside_corr` | Mean pairwise downside correlation across groups |
| `downside_corr_matrix` | Full K×K downside correlation matrix |
| `weight_method` | Which method was applied |
| `weight_stability_flag` | Whether any group weight shifted > threshold vs. prior fold |

---

## 9. References

- [forecast_pipeline_architecture.md](../Portfolio/forecast_pipeline_architecture.md) — Full pipeline context, FDM/IDM summary, existing implementation
- [top_k_ensemble_selection.md](../Feature_selection/Parameter%20Sensitivity/top_k_ensemble_selection.md) — Param selection: what enters the WeightLayer
- [pipeline_overview.md](../Feature_selection/pipeline_overview.md) — Pre-committed rule design principles

---

*End of specification.*
