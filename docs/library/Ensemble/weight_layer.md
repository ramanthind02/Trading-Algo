# Weight Layer

> **Role:** Receives per-model vol-scaled forecasts from one or more DiversifiedEnsembles → combines into a single `forecast_score` per ticker → applies FDM.

```
Base Models → DiversifiedEnsemble → WeightLayer → Portfolio → PositionSizer
```

---

## Design Principle: No Sharpe Tilt

Each forecast entering the WeightLayer is already vol-targeted:

```
F_i = (τ / (σ × √h_i)) × X_i
```

- `τ` = target volatility (e.g. 0.20)
- `σ` = instrument blended volatility
- `h_i` = exposure fraction (`1 / n_bins`)
- `X_i` = binary signal (0 or 1)

Upstream [[param_stability]] already filters for stable, high-quality params. **The WeightLayer's objective is purely diversification — not Sharpe ranking.**

> [!warning] Why no Sharpe weighting here
> - Sharpe from a single fold is a high-variance estimate — amplifies noise
> - Adds a second layer of performance ranking on top of param selection (implicit look-ahead)
> - Breaks the clean separation: signal quality → param selection; portfolio construction → WeightLayer

All weighting methods are purely **risk-based**. No expected return estimation.

---

## Grouping Strategy

Signals are assigned to groups before any correlation-based weighting. This is the most important structural decision.

**Why group?** Within a feature family (e.g. RSI params [3, 4, 5]), signals are near-perfectly correlated by construction → near-singular matrix → unstable weights. Grouping reduces the allocation problem from N signals to K groups (typically 3–8).

- **Within-group:** signals near-identical → equal weights are near-optimal with zero estimation error
- **Across-group:** genuinely different feature families → correlation-based allocation adds real value

**Primary method (preferred):** pre-specified by feature family — determined by DiversifiedEnsemble config, never changes fold-to-fold, cannot overfit.

**Fallback (data-driven):** hierarchical clustering with pre-committed cutoff `ρ_cut = 0.70` on full-period correlation. Binary membership decision — estimation noise has limited impact on a binary outcome.

---

## Algorithm (5 Steps)

### Step 1: Within-Group Aggregation
```
group_signal_k(t) = mean(F_i(t)  for i in group_k)
group_return_k(t) = mean(r_i(t)  for i in group_k)
```
Equal weights within groups are the default production setting.

### Step 2: Downside Semi-Covariance Matrix `Σ^down`
```
r_k^-(t) = min(group_return_k(t), 0)
Σ^down_kl = (1/T) × Σ_t [ r_k^-(t) × r_l^-(t) ]
```
Apply Ledoit-Wolf shrinkage. Uses each signal's own negative returns (no reference portfolio required).

### Step 3: HRP Recursive Bisection on `Σ^down`
```
C^down_kl = Σ^down_kl / sqrt(Σ^down_kk × Σ^down_ll)
D_kl      = sqrt(0.5 × (1 - C^down_kl))
```
Ward linkage hierarchical clustering on D. At each binary split:
```
v_L = Var_R / (Var_L + Var_R)
v_R = Var_L / (Var_L + Var_R)
```
Normalise final group weights to sum to 1.

### Step 4: Combined Forecast
```
raw_forecast(t) = Σ_k  w_k × group_signal_k(t)
```

### Step 5: Apply FDM
```
scaled_forecast(t) = raw_forecast(t) × FDM
```

---

## Method Ladder

| Level | Method | Free params | Notes |
|---|---|---|---|
| 0 | Equal weights (flat) | 0 | Hard baseline |
| 1 | Equal within group, equal across group | 0 | Tests grouping structure alone |
| 2 | Equal within group, inverse downside vol across group | K scalars | No matrix estimation |
| 3 | Equal within group, Downside-HRP across group | K×K matrix | Default production candidate |
| 4 | Downside-HRP on all N signals (no grouping) | N×N matrix | Control: does grouping improve stability? |

Use the **simplest method that consistently beats the next simpler method** by: Sharpe improvement ≥ 0.10 AND max drawdown reduction ≥ 10%, sustained across ensembles. Thresholds pre-committed before seeing results.

---

## Forecast Diversification Multiplier (FDM)

Without FDM, combining positively correlated signals produces lower realised vol than the individual target → systematically under-sizes positions.

```
mean_corr = mean of off-diagonal entries of C^down
FDM = min(sqrt(1 / (mean_corr + 0.01)), FDM_max)
```

- `FDM_max = 2.0` (pre-committed cap)
- `mean_corr → 1` (identical signals): FDM → 1 (no scaling)
- `mean_corr → 0` (independent signals): FDM → 10, capped at 2.0
- Typical range in practice: **FDM 1.2–1.8**

For Levels 0–1 (equal weights), FDM uses full-period correlation instead of downside.

---

## Key Config Params

| Parameter | Default | Description |
|---|---|---|
| `weighting_method` | `inverse_correlation` | `equal_flat`, `equal_grouped`, `inv_downside_vol_grouped`, `downside_hrp_grouped`, `downside_hrp_flat` |
| `group_method` | `feature_family` | `feature_family` or `correlation_clustering` |
| `rho_cut` | `0.70` | Cutoff for data-driven grouping |
| `fdm_max` | `2.0` | Cap on FDM |
| `shrinkage` | `ledoit_wolf` | Shrinkage estimator for `Σ^down` |
| `weight_stability_threshold` | `0.20` | Max fold-to-fold shift before diagnostic warning |

All parameters are pre-committed before any walkforward fold begins.

---

## Output Contract

| Field | Type | Description |
|---|---|---|
| `ticker` | str | Instrument identifier |
| `forecast_score` | float | FDM-scaled combined forecast |

Diagnostics logged per fold: `group_weights`, `fdm`, `mean_downside_corr`, `downside_corr_matrix`, `weight_stability_flag`.

---

**See also:** [[portfolio]], [[param_stability]], [[base_model]]
