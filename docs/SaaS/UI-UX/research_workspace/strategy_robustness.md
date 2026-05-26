# Step 4 — Strategy Robustness (Validation)

**Flow:** [user_flow.md](user_flow.md) §4  
**Route:** `/research/{id}/robustness`  
**Display patterns:** [display_patterns.md](display_patterns.md)

---

## Purpose

Evaluate the locked parameter combination on the **Validation zone** — data the strategy has never touched. This is confirmation, not exploration. The researcher cannot change parameters here; they can only observe whether IS edge survived OOS.

Six tests across three concern areas. Each concern area gets one Verdict Card.

Spec: [validation.md](../../robustness_tests/validation.md)

---

## Page layout

```
[Stepper — sticky]
  IS recap panel (collapsed)                          [Expand ▾]

  LOCKED: EWMAC ES · fast 16 · slow 64
  Parameters are fixed. Validation tests generalization, not a new search.

  [ Run validation ]    Status: Not run

  ── (Results appear here after run) ──

  [ ← Back to sweep ]           [ Continue → Portfolio addition ]
```

---

## Locked-params banner

Persistent across the full page, directly below the stepper. Cannot be dismissed.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Locked: EWMAC ES  ·  fast 16  ·  slow 64  ·  selected in step 3   │
│  Validation tests generalization, not a new search.    [View sweep] │
└─────────────────────────────────────────────────────────────────────┘
```

`[View sweep]` opens the step 3 results in a read-only drawer — does not navigate away.

---

## Run state

```
[ Run validation ]   →   [ ◌ Running · 8s ]   →   [ ● Complete · Rerun ]
```

All six tests run as a single job. Progress shown inline on the button — no modal, no separate progress panel. The results section appears below once complete.

---

## Results — three Verdict Cards

Results are split into three cards by **concern area**. Each card's verdict is the most severe outcome of its constituent tests. A researcher scanning quickly sees three verdicts and expands only the cards they need.

---

### Card 1 — Performance Survived OOS

**Tests:** Sharpe degradation ratio · CI overlap · degradation z-score  
**Question:** Did the IS edge survive on data the strategy never saw?

```
PERFORMANCE SURVIVED OOS                                    ● PASS
  IS SR (NW): 1.19   Val SR (NW): 0.84   Degradation: 71%
  IS edge degraded moderately — within expected sampling variation.
  [View Sharpe comparison chart ▾]
```

**Inline metrics (always visible):**

| Label | Value | Source |
|-------|-------|--------|
| `IS SR (NW)` | NW-adjusted IS Sharpe | Step 3 |
| `Val SR (NW)` | NW-adjusted validation Sharpe | Validation run |
| `Degradation` | `sr_val / sr_is` as percentage | Computed |

**Verdict thresholds:**

| Degradation ratio | Verdict |
|-------------------|---------|
| ≥ 70% | PASS |
| 40–70% | WARNING — "Meaningful degradation; signal present but reduced" |
| 10–40% | FAIL (soft) — "Most IS edge did not survive" — acknowledge required |
| ≤ 10% | FAIL (hard) — "IS performance was primarily noise" — continue blocked |

Also FAIL (soft) if: `degradation_z ≥ 2.0` and CI does not overlap.

**Chart (collapsed by default):**
Horizontal bar comparison — IS bar (ember) and Val bar (ember-deep) side by side, each with 95% CI error bars. CI overlap region shaded with ember-muted hatching. Degradation ratio annotation below.

Auto-expands on hard fail.

---

### Card 2 — Strategy Alive

**Tests:** CUSUM break detected · rolling z-score visual  
**Question:** Is the strategy's return-generating process still intact, or has it structurally changed?

```
STRATEGY ALIVE                                              ● PASS
  CUSUM: 1.19 / 1.36   Break: none   Val period: 2020-01 → 2022-12
  Return distribution is consistent with IS baseline.
  [View CUSUM chart ▾]
```

**Inline metrics:**

| Label | Value |
|-------|-------|
| `CUSUM` | stat / threshold (1.36) |
| `Break` | `none` or `detected YYYY-MM` |
| `Val period` | Date range validated |

**Verdict thresholds:**

| CUSUM result | Verdict |
|-------------|---------|
| Not triggered | PASS |
| Triggered | FAIL (hard) — "Strategy's return process has structurally changed. Do not proceed." |

Hard CUSUM fail: continue blocked, no override. The interpretation card shows when the break occurred and the rolling z-score auto-expands.

**Chart (collapsed by default, auto-expands on fail):**

Two-panel time series covering the validation period:

- **Upper panel** — bar chart of daily z-scores (validation returns standardized by IS μ and σ); 60-bar rolling mean overlaid as ember line; horizontal dashed lines at z = ±1 in `paper-2` at 0.35 opacity
- **Lower panel** — CUSUM series in ember; horizontal `paper-2` dashed line at threshold 1.36; vertical ember-red dashed line at break point if detected

Shared x-axis. No separate legend — lines are annotated inline at their endpoints.

---

### Card 3 — Parameter Surface Real

**Tests:** Neighbourhood val_p10 · rank correlation ρ  
**Question:** Was the IS parameter landscape meaningful, or did the IS winner survive OOS by luck?

```
PARAMETER SURFACE                                           ✓ PASS
  Neighbourhood p10: 2.1   Rank ρ: 0.44   (moderate)
  IS parameter rankings are partially preserved on validation data.
  [View IS vs Val heatmaps ▾]    [View rank scatter ▾]
```

**Inline metrics:**

| Label | Value |
|-------|-------|
| `Neighbourhood p10` | 10th percentile NW t-stat across ±10% perturbation set on validation |
| `Rank ρ` | Spearman rank correlation of IS vs val metric ordering |

**Verdict thresholds:**

| Condition | Verdict |
|-----------|---------|
| val_p10 ≥ floor AND ρ ≥ 0.20 | PASS |
| val_p10 ≥ floor AND ρ < 0.20 | WARNING — "Strategy works but IS rankings were mostly noise" |
| val_p10 < floor | FAIL (soft) — "Parameter surface collapsed OOS" |

**Charts (both collapsed by default):**

**IS vs Val Heatmaps** — side by side with shared color ramp (ink-2 → ember). Chosen combination marked on both. Perturbation box overlaid on both. If the high-performance region shifts dramatically between IS and val, that shift is visible immediately without annotation.

**Rank Scatter** — IS NW t-stat (x) vs Val NW t-stat (y), one dot per combination. Chosen combination highlighted in ember. Regression line in `paper-2` at 0.4 opacity. Spearman ρ annotated top-left.

---

## Overall verdict banner

Appears after all three cards, above the Continue button. Summarizes across all concern areas.

**All pass:**
```
┌─────────────────────────────────────────────────────────── ● PASS ──┐
│  Validation complete. Strategy edge survived OOS across all checks.  │
│  Ready for portfolio addition.                                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Mixed (some warnings):**
```
┌──────────────────────────────────────────────────────── ⚠ WARNING ──┐
│  Validation complete with 1 warning. Review the flagged card above   │
│  before proceeding. Continuing is allowed but noted.                  │
│                                      [ ] I have reviewed the warning  │
└──────────────────────────────────────────────────────────────────────┘
```

**Hard fail:**
```
┌──────────────────────────────────────────────────────────── ✗ FAIL ──┐
│  CUSUM break detected — strategy's return distribution has changed.   │
│  Do not add this strategy to the portfolio.                            │
│  Start a new research cycle if you want to develop a replacement.     │
└───────────────────────────────────────────────────────────────────────┘
```

Hard fail: Continue button is hidden entirely. Only "← Back to sweep" is shown.

---

## IS recap panel (read-only)

Collapsed panel at the top of the page — see [display_patterns.md](display_patterns.md):

```
IS RESULTS (STEP 3)    EWMAC ES  ·  fast 16 · slow 64    [Expand ▾]
DSR 0.94  ·  NW λ 1.43  ·  Rolling PASS  ·  N_eff 33 / 286
```

Expanded view shows all IS Verdict Cards in read-only mode. No rerun control.

---

## Exit criteria

| Criterion | Required for Continue |
|-----------|----------------------|
| Validation job completed | ✅ |
| No hard fail (CUSUM, degradation < 10%) | ✅ |
| Soft fail warnings acknowledged | ✅ (checkbox) |

**Continue** → `/research/{id}/portfolio-addition`
