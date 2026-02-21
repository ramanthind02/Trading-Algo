# Stable Region Ensemble Selection

> **Status:** Active library specification
> **Date:** 2026-02-21
> **Supersedes:** Enhanced Top-K Ensemble Selection (Smoothed-Only)

---

## Overview

Param combo selection uses a **stable region detection** algorithm: the smoothed performance landscape is analysed to find connected plateaus of high performance, then the best representatives from each plateau are selected.

This replaces the previous fixed top-K ranking approach, which had two weaknesses:
- Required absolute thresholds (stability_ratio > 0.8, smoothed_metric > X) that needed manual tuning per feature.
- Selected only the K best points regardless of whether they came from the same isolated cluster or from genuinely distinct signal regions.

The new approach is self-normalising (thresholds are relative to the landscape), finds however many stable regions exist, and produces natural structural diversity when multiple valid regions are present.

Selection is selection-only: no forecast averaging is performed at the selection stage.

---

## Conceptual Foundation

The smoothed objective surface over the parameter grid is treated as a terrain. A **stable region** is a broad plateau — a connected set of parameter combos that all perform within a relative tolerance of the best point. Isolated spikes that drop sharply to their neighbours fail the relative threshold and are excluded automatically.

The algorithm is parameter-free in the sense that no absolute performance cutoffs are required. The one tuning parameter (relative tolerance δ) has a direct economic interpretation and transfers across features without tuning.

---

## Inputs

- `smoothed_objectives`: dict mapping `param_label → smoothed_metric` (from grid-aware neighbour averaging — see [grid_search_parameter_stability.md](grid_search_parameter_stability.md))
- `raw_objectives`: dict mapping `param_label → raw_metric`
- `trade_frequencies`: dict mapping `param_label → trade_freq`
- `grid_adjacency`: dict mapping `param_label → list[param_label]` (1-step axis-aligned neighbours)
- Config (see [Configuration](#configuration))

---

## Algorithm

### Step 1: Hard Filters

Apply hard constraints before any ranking.

At the walkforward runner boundary, `trade_freq_min` is enforced as an upstream prefilter/input contract. The stable-region selector receives only params that already pass that trade-frequency gate.

Inside the stable-region selector, the hard filter is:

```
bin_count < bin_count_min   (continuous features only)
```

`bin_count_min` prevents degenerate low-activity signals that produce near-zero outputs and achieve spurious decorrelation with real signals.

### Step 2: Relative Floor Computation

Compute the performance floor from the surviving params only:

```
best = max(smoothed_objective over surviving params)

Option A (fixed relative):
    floor = best × (1 - δ)            δ = 0.20 default

Option B (adaptive, recommended):
    μ, σ = mean and std of smoothed_objectives over surviving params
    floor = best - 1.0 × σ
```

Option B is preferred because it adapts to the spread of the performance landscape. For features where all params perform similarly (tight spread), it is more inclusive. For features with a clear winner and poor alternatives (wide spread), it is more selective. Either way, no feature-specific absolute threshold is needed.

### Step 3: Superlevel Set

```
qualifying = {p : smoothed_objective(p) > floor}
```

### Step 4: Connected Components

Find connected components of `qualifying` using the grid adjacency graph. Two params are in the same component if they are connected by a chain of 1-step adjacencies, all of which are in `qualifying`.

Each connected component is a candidate stable region.

### Step 5: Filter Degenerate Components

A component is valid if:
- Size ≥ 2 (isolated single-param passers are treated as suspect; no genuine plateau is a single point)
- At least one member has `smoothed_objective > 0` (absolute positivity floor — prevents selecting consistently negative regions)

Discard components that fail either criterion.

### Step 6: Select from Each Valid Region

For each valid component, rank members by `smoothed_objective` descending. Select up to `k_per_region` members.

If the total number of selected params across all regions exceeds `k_max`, trim to the top `k_max` by `smoothed_objective` globally.

If no valid component exists, take no position this fold (consistent with the `K_min = 2` rule from the pre-committed selection rule — see [pipeline_overview.md](../pipeline_overview.md#5-pre-committed-param-selection-rule)).

---

## What the Algorithm Produces

**Single stable region (common case):**
```
Smoothed Sharpe: [0.2, 0.6, 0.9, 1.0, 0.95, 0.4, 0.1]
Params:           [2,   3,   4,   5,   7,   14,  20]
floor = 1.0 × (1 - 0.20) = 0.80

Qualifying: {4, 5, 7}  →  one connected component (size 3)
Selected (k_per_region=3): {5, 4, 7}
```

**Multiple stable regions:**
```
Smoothed Sharpe: [0.1, 0.85, 0.9, 0.3, 0.2, 0.88, 0.82]
Params:           [2,   3,    4,   5,   10,  14,   20]
best = 0.9, floor = 0.72

Qualifying: {3, 4, 14, 20}

Connected components:
  Region A: {3, 4}    (3 and 4 are adjacent; 4 and 5 not both qualifying)
  Region B: {14, 20}  (14 and 20 are adjacent)

Selected: top-2 from Region A + top-2 from Region B
→ {4, 3, 14, 20}
```

Both regions contribute to the ensemble. This is structural diversity that emerges from the data, not forced diversity from a correlation penalty.

**No valid regions (degenerate case):**
```
All qualifying components have size 1 → no position taken this fold.
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection_method` | `top_k` | Selection algorithm identifier. `top_k` is default; `enhanced` and `stable_region` are available alternatives. |
| `floor_method` | `adaptive` | `adaptive` (sigma-based) or `relative` (fixed δ) |
| `delta` | `0.20` | Relative tolerance when `floor_method=relative` |
| `adaptive_sigma_multiplier` | `1.0` | σ multiplier when `floor_method=adaptive` |
| `k_per_region` | `3` | Max params selected per stable region |
| `k_max` | `6` | Hard cap on total selected params across all regions |
| `trade_freq_min` | feature-specific | Upstream walkforward prefilter (`WalkforwardResearchConfig`), applied before stable-region selection input |
| `bin_count_min` | `5` | Min bin count for continuous features (blocks degenerate signals) |
| `min_region_size` | `2` | Minimum connected component size to be treated as a valid region |

All parameters are pre-committed before any fold is evaluated. They are not tuned per feature.

---

## Output Fields

Per-param fields appended to the fold result:

| Field | Description |
|-------|-------------|
| `smoothed_objective` | Neighbour-smoothed metric value |
| `trade_frequency` | Signal activity rate |
| `passed_hard_filters` | Whether param survived Step 1 |
| `above_floor` | Whether param is in the qualifying superlevel set |
| `region_id` | Connected component ID (`null` if not qualifying) |
| `region_size` | Size of the param's connected component |
| `selected` | Whether param was selected for the ensemble this fold |

The `selected_params_detailed.csv` report includes all params with the fields above, allowing the researcher to inspect the landscape, the floor level, and the region structure per fold.

---

## Relationship to the Pre-Committed Selection Rule

This algorithm is the implementation of the pre-committed param selection rule described in [pipeline_overview.md §5](../pipeline_overview.md#5-pre-committed-param-selection-rule). It operates identically inside each walkforward training fold and in production. Because the selection is driven by relative thresholds, the rule behaves consistently across regimes without requiring threshold recalibration.

This stage outputs a selected parameter set only:

```
selected_params = {p_1, ..., p_k}
```

Any forecast combination of selected members is performed downstream by the WeightLayer.

If `|selected| < 2`, no position is taken.

---

## Comparison to Previous Approach

| Aspect | Previous (top-K smoothed) | Current (stable region) |
|--------|--------------------------|------------------------|
| Threshold type | Absolute (stability_ratio > 0.8) | Relative (adaptive floor) |
| Feature-agnostic | No — thresholds needed tuning | Yes — self-normalising |
| Multi-region detection | No | Yes |
| Structural diversity | Forced via correlation penalty (caused degenerate low-activity params) | Emerges from landscape structure |
| Degenerate signal protection | Trade-freq filter only | Upstream trade-freq prefilter + bin_count hard filter |
| Isolated peak handling | Implicit via stability_ratio | Explicit via min_region_size ≥ 2 |

---

## References

- [grid_search_parameter_stability.md](grid_search_parameter_stability.md) — Neighbour smoothing theory and stability ratio computation
- [pipeline_overview.md §5](../pipeline_overview.md#5-pre-committed-param-selection-rule) — Pre-committed selection rule context
- [feature_validator.md](../feature_validator.md) — Full IS validation pipeline
