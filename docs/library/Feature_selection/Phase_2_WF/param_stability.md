# Parameter Stability

> [!important] Pre-committed rule: Marginal Peak Selection (MPS) runs **identically** inside each walkforward training fold and in production refits — no recalibration between contexts.

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

For each parameter combo `P`, the **smoothed objective** is a weighted average of `P` and its **1-step axis-aligned neighbors** (params differing by exactly one grid step in one dimension).

```
smoothed(P) = (self_weight × obj(P) + Σ obj(N)) / (self_weight + n_neighbors)
stability_ratio = smoothed(P) / obj(P)
```

`self_weight` (configured via `smoothing_self_weight`, default `2.0`) controls how much the center param's own performance counts relative to each neighbor. `self_weight=1.0` gives equal weight to center and each neighbor (most aggressive); higher values reduce dilution. This matters most for **boundary params** — those at the grid edge have fewer neighbors, so without center-weighting their smoothed value is aggressively pulled toward the (possibly weaker) neighbors on one side.

**2D example** — cell `E` at `(lookback=14, threshold=50)`, neighbors are `B(7,50)`, `H(21,50)`, `D(14,30)`, `F(14,70)`:
```
      Threshold
        30   50   70
L  7   A    B    C
o 14   D  [ E ]  F
o 21   G    H    I
```
`smoothed(E) = (self_weight × E + B + H + D + F) / (self_weight + 4)`

Boundary parameters have fewer neighbors. With `self_weight > 1` their own performance still anchors the smoothed value — the center's evidence is not overwhelmed by a smaller set of weaker neighbors.

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

## Selection Workflow

Marginal Peak Selection (MPS) has been retired. Parameter selection now relies on manual review of the **top-k** combos by smoothed objective, ensuring stability and alignment with the three-period workflow:

1. **Training (2000–2017)** — full parameter sweeps and stability analysis produce the smoothed surface and candidate plateau.
2. **Validation (2018–2022)** — fixed candidates from training are tested on unseen data; failing strategies are discarded.
3. **Test (2023–2025)** — locked parameters are evaluated via forward simulation to ensure final robustness.

Neighbor smoothing continues to regularize the surface so that plateau regions dominate isolated spikes. Researchers hand-select a few parameter combinations from the smoothed ranking (TOP_K or ENHANCED in walkforward), rather than relying on any automated gap-based marginal analysis.

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
| `selection_method`     | `top_k` | Walkforward: `top_k` or `enhanced`. |
| `smoothing_self_weight` | `2.0`–`3.0` | Center param weight vs. each neighbor in smoothing. **Must match EDA config so researcher and WF see the same landscape.** |
| `trade_freq_min`       | feature-specific | Upstream walkforward prefilter — applied before selector input. |

All parameters are pre-committed before any fold is evaluated — not tuned per feature.

---

## Key Takeaways

1. **Stability ≠ significance.** Permutation tests filter noise. Parameter stability (e.g. MPS, smoothing) filters overfitting. Both required.
2. **Smoothing is regularization.** Averaging over neighbors penalizes isolated peaks, rewards plateaus; used for plots and for TOP_K/enhanced selection.
3. **Top-K smoothed selection identifies the plateau.** The smoothed objective surface ranks stable neighborhoods; researchers hand-select from the top-k rather than relying on gap-based marginal tables.
4. **Pre-committed config.** All selection parameters are fixed before any fold — no recalibration between walkforward and production.

---

**See also:** [[Ensemble/weight_layer]] (forecast combination of selected params), [[pipeline]] (pre-committed selection rule context)
