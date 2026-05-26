# Step 3 — Parameter Sweep

**Flow:** [user_flow.md](user_flow.md) §3  
**Route:** `/research/{id}/parameter-sweep`  
**Display patterns:** [display_patterns.md](display_patterns.md)

---

## Purpose

Explore the active strategy's parameter space on the **Train zone**, surface overfitting diagnostics, and lock one combination to carry into validation. Sweep (exploration) and validation (confirmation) are two separate pages — this is exploration only.

Specs: [in_sample.md](../../robustness_tests/in_sample.md) · [parameter_sensitivity.md](../../robustness_tests/parameter_sensitivity.md) · [parameter_selection.md](../../robustness_tests/parameter_selection.md)

---

## Page layout

```
[Stepper — sticky]

Active strategy: EWMAC ES · v3              Train zone: 2004-01 → 2019-12

──────────────────────────────────────────────────────────────────────────

PHASE A — Configure (before run)
PHASE B — Results (after run)
PHASE C — Selection (always at bottom)
```

The three phases are always present in the DOM — Configure and Selection are always visible; Results appears after the first run completes.

---

## Phase A — Configure

```
SEARCH SPACE                                         Est. 48 combinations

  fast_period    min [  8 ]  max [ 32 ]  step [ 4 ]
  slow_period    min [ 32 ]  max [128 ]  step [ 8 ]

                                                         [ Run sweep → ]
```

| Control | Behavior |
|---------|----------|
| Range inputs | Auto-generated from `params_schema` — min, max, step |
| Enum params | Dropdown multi-select |
| Bool params | Toggle |
| Combination count | Computed live as inputs change; warn in amber if > quota |
| Run sweep | Primary CTA; disables while running; transitions to running state inline |

Grid size warning at > 500 combinations: *"Large grid — estimated run time 3 min. Results quality improves with more combinations."*

---

## Phase B — Results

Appears after the sweep completes. Three sections rendered in reading priority order: **Robustness first**, then table, then heatmap.

---

### B1 — Robustness Summary (always first)

The most important output of the sweep. Rendered immediately above the results table so it is the first thing the researcher reads.

```
SEARCH BIAS                                               ● PASS
  DSR: 0.94   N_eff: 33 / 286   avg ρ: 0.76   NW λ: 1.43
  Strong evidence of real edge after correcting for search depth.

TEMPORAL STABILITY                                        ● PASS
  Rolling positive: 73%   CUSUM: 1.19 / 1.36   Break: none
  Edge is consistent across the training period.
  [View rolling Sharpe chart ▾]
```

**Search Bias card — inline metrics:**

| Metric | Label | Tooltip |
|--------|-------|---------|
| DSR probability | `DSR` | "Probability of real edge after correcting for N_eff trials" |
| Effective trials | `N_eff: 33 / 286` | "Effective independent trials out of total combinations" |
| Avg pairwise correlation | `avg ρ` | "Mean pairwise correlation of strategy returns across parameter grid" |
| NW inflation factor | `NW λ` | "t-stat inflation from autocorrelation; NW-adjusted t is divided by √λ" |

**Temporal Stability card — inline metrics:**

| Metric | Label |
|--------|-------|
| Rolling positive fraction | `Rolling positive: 73%` |
| CUSUM statistic / threshold | `CUSUM: 1.19 / 1.36` |
| Break detected | `Break: none` or `Break: detected` |

The rolling Sharpe chart is the only chart that may auto-expand — if CUSUM detects a break, the chart auto-expands and the break point is marked with a vertical ember dashed line and annotation.

**DSR verdict thresholds:**

| DSR | Verdict | Border |
|-----|---------|--------|
| ≥ 0.75 | PASS | Default |
| 0.50 – 0.75 | WARNING | Amber — "Marginal. Consider running the grid permutation test." |
| < 0.50 | FAIL | Red — continue disabled; acknowledge required |

---

### B2 — Results Table

Dense, sortable. Every combination in one row. Designed for quick scanning, not deep reading.

```
RESULTS   286 combinations                            Sort: NW t-stat ▾

  params                │ NW t-stat │  Sharpe  │   DSR   │  rows
  ──────────────────────┼───────────┼──────────┼─────────┼───────
  fast 16  slow 64      │  3.07 ★   │  1.42    │  0.94   │  3,780
  fast 16  slow 56      │  2.94     │  1.38    │  0.91   │  3,780
  fast 12  slow 64      │  2.81     │  1.31    │  0.88   │  3,780
  ...
```

| Column | Notes |
|--------|-------|
| Params | One cell per param, space-separated — no separate columns |
| NW t-stat | Primary sort column by default; ★ marks current selection |
| Sharpe | Raw annualised; always show alongside NW-adjusted |
| DSR | Color-coded: ≥ 0.75 ember · 0.50–0.75 amber · <0.50 muted red |
| Row count | Bar count used for this run |

Row click: sets as selected combination (updates Phase C). No separate "select" button.

Rows per page: 25. Pagination at bottom. No infinite scroll — the table is not the primary interface; the researcher needs to scan the first page and move on.

**No column for: skewness, kurtosis, individual permutation p-value, N_eff per-row** — these live in the detail drawer, accessible by clicking a row's expand arrow.

---

### B3 — Heatmap (collapsed by default)

```
PARAMETER HEATMAP                              [Show ▾]
```

When expanded:

```
  X-axis [ fast_period ▾ ]    Y-axis [ slow_period ▾ ]

  [2D heatmap — warm ember ramp, chosen combo marked]
```

- Dropdown selects which two params to map when strategy has > 2 params
- Color ramp: ink-2 (low) → ember (high) — no blue, no diverging scale
- Selected combination: `paper-2` 4px dot with tooltip showing exact metric
- Perturbation box: `paper-2` dashed border (±10% band around selected params)
- Side-by-side IS / Val heatmaps available after validation runs (step 4)

---

### B4 — On-Demand: Grid Permutation Test

Collapsed section below the heatmap. Never auto-runs — expensive.

```
GRID PERMUTATION TEST                           [Not run]

  Answer: "Could random search over this grid produce my result by chance?"
  Est. time: ~18 seconds (1000 iterations, 286 combinations)

  [ Run permutation test ]
```

After run:

```
GRID PERMUTATION TEST                            p = 0.028  ● PASS

  Best NW t-stat: 3.07 (real)    Null median: 2.31    p = 0.028
  [View null distribution histogram ▾]
  1000 iterations · seed 42
```

The histogram auto-expands after the permutation test completes. The researcher just ran something expensive — they want to see the result immediately.

---

## Phase C — Selection

Always visible at the bottom of the page. Updates live as the researcher clicks rows in the results table.

```
SELECTION

  Current: fast 16 · slow 64           NW t-stat 3.07   DSR 0.94

  [ Use best by NW t-stat ]            [ Clear selection ]

  ⚠ DSR 0.56 — Marginal edge evidence. Acknowledge to continue.
    [ ] I understand and want to proceed.

                    [ ← Back to editor ]   [ Lock & continue → Robustness ]
```

**Warnings in Selection:**

| Condition | Warning |
|-----------|---------|
| DSR < 0.50 | Hard block — must acknowledge: "IS evidence is weak. Running validation on an overfit result." |
| DSR 0.50–0.75 | Soft warning — acknowledge checkbox visible |
| CUSUM break detected | Soft warning — acknowledge checkbox visible |
| No combination selected | Continue disabled — "Select a combination to continue" |

**Lock & continue:** sets the selected combination as **immutable** in the backend before navigating. Step 4 will show a locked-params banner.

---

## IS recap (for steps 4–6)

After locking, a collapsed IS recap panel is created and appears at the top of steps 4, 5, and 6. See [display_patterns.md](display_patterns.md) — IS recap panel.

```
IS RESULTS (STEP 3)    EWMAC ES  ·  fast 16 · slow 64    [Expand ▾]
DSR 0.94  ·  NW λ 1.43  ·  Rolling PASS  ·  N_eff 33 / 286
```
