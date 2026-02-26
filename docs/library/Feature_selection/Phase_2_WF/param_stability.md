# Parameter Stability

> [!important] Pre-committed rule: The stable region selection algorithm runs **identically** inside each walkforward training fold and in production refits — no recalibration between contexts.

Relates to: [[pipeline]] | [[Phase_1_EDA/eda]] | [[Phase_3_4_Walkforward/walkforward]]

---

## Why This Matters: Stability vs. Significance

Permutation tests answer: **"Is this parameter better than chance?"**
Parameter stability answers: **"Will this parameter generalize?"**

Both are required. A param can pass a permutation test (statistically significant) yet still be an isolated peak that collapses OOS.

### The Isolated Peak Problem

```
Scenario A — Stable region (deploy):
Lookback:   2    3    4    5    10   14   20
Sharpe:    0.6  0.9  1.1  1.0  0.5  0.3  0.1
                └── stable plateau ──┘

Scenario B — Isolated peak (reject):
Lookback:   2    3    4    5    10   14   20
Sharpe:    0.1  0.2  1.1  0.3  0.2  0.1  0.0
                      ↑
                 isolated spike
```

In Scenario B, lookback 4 may pass a permutation test — but its neighbors perform poorly. **The edge isn't robust.** OOS, it breaks down.

**Why stability predicts OOS performance:**
- Genuine signal is low-frequency across parameter space (smooth plateau)
- Noise is high-frequency (narrow spike). Smoothing filters noise, preserves signal.
- A stable region also hedges against slight regime shifts (optimal lookback 4 in 2015, 5 in 2020 → both in the plateau, ensemble stays valid).

---

## Neighbor Smoothing

For each parameter combo `P`, the **smoothed objective** averages `P` with its **1-step axis-aligned neighbors** (params differing by exactly one grid step in one dimension).

```
smoothed(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])
stability_ratio = smoothed(P) / obj(P)
```

**2D example** — cell `E` at `(lookback=14, threshold=50)`, neighbors are `B(7,50)`, `H(21,50)`, `D(14,30)`, `F(14,70)`:
```
      Threshold
        30   50   70
L  7   A    B    C
o 14   D  [ E ]  F
o 21   G    H    I
```
`smoothed(E) = mean(E, B, H, D, F)`

Boundary parameters have fewer neighbors and are smoothed less — this is correct (less neighborhood evidence = less certainty).

---

## Four Quadrants

|                      | High smoothed | Low smoothed             |
|----------------------|---------------|--------------------------|
| **High raw**         | ✅ Genuine signal in stable region | ⚠️ Isolated peak, likely overfit |
| **Low raw**          | ⚠️ Param in good neighborhood — acceptable ensemble filler | ❌ Consistently poor — reject |

> [!warning] A high stability ratio with low absolute smoothed objective means "consistently poor", not "good". Always check the absolute value.

**Numeric examples:**

```
Quadrant 1 (High raw, High smoothed — deploy):
  Lookback 4: raw=1.1, neighbors={3:0.9, 5:1.0} → smoothed=1.0, ratio=0.91 ✅

Quadrant 2 (High raw, Low smoothed — suspicious):
  Lookback 14: raw=1.1, neighbors={10:0.2, 20:0.3} → smoothed=0.53, ratio=0.48 ⚠️

Quadrant 4 (Low raw, High ratio — reject):
  Lookback 20: raw=0.1, neighbors={14:0.3} → smoothed=0.2, ratio=2.0 ❌
  (high ratio but both values are poor — "consistently bad")
```

---

## Stable Region Selection Algorithm

### Step 1 — Hard Filters

- `bin_count < bin_count_min`: blocks degenerate low-activity signals (continuous features only).
- `trade_freq_min`: enforced upstream by the walkforward runner before params reach this selector.

### Step 2 — Relative Floor

Computed from surviving params only:

```
best = max(smoothed_objective)

Option A — adaptive (recommended):
    floor = best - 1.0 × σ(smoothed_objectives)

Option B — fixed relative:
    floor = best × (1 - δ)      # δ = 0.20 default
```

Adaptive self-normalises to the landscape spread without requiring feature-specific tuning.

### Step 3 — Qualifying Set

```
qualifying = { p : smoothed_objective(p) > floor }
```

### Step 4 — Connected Components

Find connected components of `qualifying` using grid adjacency (1-step axis-aligned). Two params are in the same component if connected by a chain of qualifying adjacencies.

### Step 5 — Filter Degenerate Components

Discard a component if:
- `size < min_region_size` (default: 2) — no genuine plateau is a single point.
- All members have `smoothed_objective ≤ 0` — absolute positivity floor.

### Step 6 — Select from Valid Regions

Per valid component: rank by `smoothed_objective` descending, take up to `k_per_region`. If total across all regions exceeds `k_max`, trim to top `k_max` globally.

**If no valid component exists → no position taken this fold.**

---

## Example: Single vs Multiple Stable Regions

**Single region (common):**
```
Params:           [2,   3,   4,   5,   7,   14,  20]
Smoothed Sharpe:  [0.2, 0.6, 0.9, 1.0, 0.95, 0.4, 0.1]
best=1.0, δ=0.20 → floor=0.80

Qualifying: {4, 5, 7} → one connected component (size 3)
Selected (k_per_region=3): {5, 4, 7}
```

**Multiple regions (structural diversity emerges from data):**
```
Params:           [2,   3,    4,   5,   10,  14,   20]
Smoothed Sharpe:  [0.1, 0.85, 0.9, 0.3, 0.2, 0.88, 0.82]
best=0.9, adaptive floor≈0.72

Qualifying: {3, 4, 14, 20}
Region A: {3, 4}    Region B: {14, 20}
Selected: top-2 from A + top-2 from B → {4, 3, 14, 20}
```

Both regions contribute. This is data-driven structural diversity, not a forced correlation penalty.

---

## Walkforward Outcomes

After running walkforward stability analysis, three outcomes are possible:

### ✅ Outcome 1 — Consistent Top-K (deploy)

Same param neighborhood appears in top-3 across 3+ folds:
```
Fold 1: {4, 5, 3}
Fold 2: {5, 4, 3}
Fold 3: {4, 3, 5}
Fold 4: {4, 5, 3}
```
→ Deploy ensemble `{3, 4, 5}`. Use full in-sample data for final fit.

### ⚠️ Outcome 2 — Gradual Drift (recency bias)

Optimal region shifts across folds (e.g., slower MAs early → faster MAs late):
```
Fold 1: {(16,64), (20,80), ...}   ← slower
Fold 4: {(32,128), (20,80), ...}  ← faster
```
→ Don't average over the regime shift. Use recent fold's top-K only. Monitor for reversal.

### ❌ Outcome 3 — Unstable (reject)

Top params jump across parameter space with no consistent pattern → feature is fitting noise.
→ Reject the feature. Revisit the hypothesis or try a different asset/timeframe.

---

## Ensemble Size Guidance

| Size | When to Use |
|------|-------------|
| 2–3 | Clear top param, tight neighborhood |
| 4–7 | Broader stable plateau |
| 8–10 | Large stable region or 2D+ grid |
| > 10 | Rarely justified |

**Principle:** Smallest ensemble that captures the stable region. Don't pad with mediocre params.

---

## Key Configuration

| Parameter              | Default    | Description                                          |
|------------------------|------------|------------------------------------------------------|
| `selection_method`     | `stable_region` | Algorithm identifier |
| `floor_method`         | `adaptive` | `adaptive` (sigma-based) or `relative` (fixed δ)    |
| `delta`                | `0.20`     | Relative tolerance when `floor_method=relative`     |
| `adaptive_sigma_multiplier` | `1.0` | σ multiplier for adaptive floor                   |
| `k_per_region`         | `3`        | Max params selected per stable region                |
| `k_max`                | `6`        | Hard cap on total selected params across all regions |
| `min_region_size`      | `2`        | Minimum component size to be treated as valid        |
| `bin_count_min`        | `5`        | Min bin count for continuous features                |
| `trade_freq_min`       | feature-specific | Upstream walkforward prefilter — applied before selector input |

All parameters are pre-committed before any fold is evaluated — not tuned per feature.

---

## Output: Per-Param Fields

| Field               | Description                                          |
|---------------------|------------------------------------------------------|
| `smoothed_objective`| Neighbour-smoothed metric value                      |
| `trade_frequency`   | Signal activity rate                                 |
| `passed_hard_filters` | Whether param survived Step 1                     |
| `above_floor`       | Whether param is in the qualifying superlevel set    |
| `region_id`         | Connected component ID (`null` if not qualifying)    |
| `region_size`       | Size of the param's connected component              |
| `selected`          | Whether param was selected for the ensemble          |

---

## Key Takeaways

1. **Stability ≠ significance.** Permutation tests filter noise. Stability analysis filters overfitting. Both required.
2. **Smoothing is regularization.** Averaging over neighbors penalizes isolated peaks, rewards plateaus.
3. **Stable regions predict OOS.** Broad plateaus generalize better than narrow spikes — empirically and theoretically.
4. **Self-normalising thresholds.** Relative floor means no feature-specific absolute threshold tuning needed.
5. **Multiple regions = structural diversity.** If two disconnected plateaus exist, include both — genuine signal diversity, not forced correlation penalty.
6. **If |selected| < 2, no position taken.** The minimum ensemble rule is hard-coded, not a soft recommendation.

---

**See also:** [[Ensemble/weight_layer]] (forecast combination of selected params), [[pipeline]] (pre-committed selection rule context)
