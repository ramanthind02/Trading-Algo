# Parameter Stability

> [!important] The stable region selection rule is IDENTICAL inside each walkforward training fold and in production refits. No recalibration between contexts.

Relates to: [[pipeline]] | [[eda]] | [[walkforward]]

---

## Neighbor Smoothing

For each parameter combo `P` on the grid, the **smoothed objective** averages `P` with its **1-step axis-aligned neighbors** (parameters differing by one grid step in exactly one dimension).

```
smoothed(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])
stability_ratio = smoothed(P) / obj(P)
```

A **2D example**: for cell `E` at `(lookback=14, threshold=50)`, neighbors are `B(7,50)`, `H(21,50)`, `D(14,30)`, `F(14,70)`.

Boundary parameters have fewer neighbors and are smoothed less — this is correct behavior (less information about neighborhood = less certainty).

### Four Quadrants

|                      | High smoothed | Low smoothed             |
|----------------------|---------------|--------------------------|
| **High raw**         | Strong candidate — genuine signal in stable region | Suspicious — isolated peak, likely overfit |
| **Low raw**          | Acceptable — param "in a good neighborhood" | Reject — consistently poor region |

> [!warning] A high stability ratio with low absolute smoothed objective means "consistently poor", not "good". Always check the absolute value.

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

Adaptive is preferred: it self-normalises to the spread of the landscape without requiring feature-specific tuning.

### Step 3 — Qualifying Set

```
qualifying = { p : smoothed_objective(p) > floor }
```

### Step 4 — Connected Components

Find connected components of `qualifying` using grid adjacency (1-step axis-aligned). Two params are in the same component if connected by a chain of qualifying adjacencies.

### Step 5 — Filter Degenerate Components

Discard a component if:
- `size < min_region_size` (default: 2) — isolated single-param passers are suspect; no genuine plateau is a single point.
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

## Key Configuration

| Parameter              | Default    | Description                                          |
|------------------------|------------|------------------------------------------------------|
| `floor_method`         | `adaptive` | `adaptive` (sigma-based) or `relative` (fixed δ)    |
| `delta`                | `0.20`     | Relative tolerance when `floor_method=relative`     |
| `adaptive_sigma_multiplier` | `1.0` | σ multiplier for adaptive floor                   |
| `k_per_region`         | `3`        | Max params selected per stable region                |
| `k_max`                | `6`        | Hard cap on total selected params across all regions |
| `min_region_size`      | `2`        | Minimum component size to be treated as valid        |
| `bin_count_min`        | `5`        | Min bin count for continuous features                |

All parameters are pre-committed before any fold is evaluated — they are not tuned per feature.

---

## Output: Per-Param Fields

| Field               | Description                                          |
|---------------------|------------------------------------------------------|
| `smoothed_objective`| Neighbour-smoothed metric value                      |
| `above_floor`       | Whether param is in the qualifying superlevel set    |
| `region_id`         | Connected component ID (`null` if not qualifying)    |
| `region_size`       | Size of the param's connected component              |
| `selected`          | Whether param was selected for the ensemble          |

---

## Relationship to Walkforward

This algorithm runs inside each walkforward training fold and identically in production. The selection outputs `selected_params = {p_1, ..., p_k}` only. Forecast combination of selected members is performed downstream by the [[weight_layer|WeightLayer]].
