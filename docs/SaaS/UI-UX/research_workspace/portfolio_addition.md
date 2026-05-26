# Step 5 — Portfolio Addition Gate

**Flow:** [user_flow.md](user_flow.md) §5  
**Route:** `/research/{id}/portfolio-addition`  
**Display patterns:** [display_patterns.md](display_patterns.md)

---

## Purpose

A binary gate: does this strategy improve the portfolio on the data it has already seen? Pass → proceed to commit. Fail → discard the strategy entirely. There is no tuning, no middle path.

Spec: [portfolio_addition.md](../../robustness_tests/portfolio_addition.md)

---

## Data context banner

Always visible at the top. Tells the researcher exactly what data this gate evaluates — critical because contamination discipline depends on it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Evaluation data: IS + Validation   2018-01-01 → 2021-12-31         │
│  No project test zone data is used here.                             │
│  Result is binary: pass or fail. Do not adjust the strategy          │
│  in response to this result.                                [?]      │
└─────────────────────────────────────────────────────────────────────┘
```

The `[?]` opens a one-paragraph explainer on contamination discipline — inline drawer, not a new page.

**First strategy special case:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  This is the first strategy in this portfolio. Portfolio addition    │
│  gate is skipped — there is no existing portfolio to compare         │
│  against.                                                            │
│                                          [ Continue → Commit → ]    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Run state

```
[ Check portfolio fit ]   →   [ ◌ Running · 4s ]   →   [ ● Complete ]
```

Single button. All four tests run as one job.

---

## Results — two Verdict Cards + one Info card

Three cards. Two produce verdicts (pass/fail). One is always observational.

---

### Card 1 — Theoretical Case

**Tests:** Analytical hurdle (SR vs correlation hurdle) · drawdown correlation analysis  
**Question:** Can this strategy theoretically improve the portfolio given its Sharpe and correlation?

```
THEORETICAL CASE                                            ● PASS
  SR new: 0.61   Correlation hurdle: 0.29   Margin: +0.32
  Strategy's Sharpe clears the correlation-adjusted hurdle.
  [View drawdown correlation detail ▾]
```

**Inline metrics:**

| Label | Value | Meaning |
|-------|-------|---------|
| `SR new` | NW-adjusted Sharpe of new strategy on IS + val | What the strategy delivers |
| `Correlation hurdle` | `ρ_effective × SR_portfolio` | The minimum SR required |
| `Margin` | `SR_new − hurdle` | Positive = clears; Negative = fails |

Margin: ember if positive (`+0.32`), error red if negative (`−0.18`). Always show sign.

**Verdict thresholds:**

| Margin | Verdict |
|--------|---------|
| > 0 | PASS |
| ≤ 0 | WARNING (soft fail) — "Sharpe does not clear the correlation hurdle. Continue if empirical test passes." |

Analytical hurdle is a **soft gate** — the empirical ΔSR is the primary decision.

**Drawdown detail (collapsed by default):**

```
  Effective correlation:     0.31   (unconditional: 0.28 · drawdown: 0.31)
  Drawdown overlap ratio:    0.43
  Joint drawdown depth:      −7.2%  vs individual avg −5.1% / −4.3%
```

Color-coded against advisory thresholds (amber label if exceeded, no hard gates):

| Metric | Advisory threshold |
|--------|-------------------|
| Drawdown correlation uplift | > 0.25 above unconditional |
| Overlap ratio | > 0.60 |
| Joint drawdown depth | > 1.5× individual avg |

---

### Card 2 — Empirical Result (Primary Gate)

**Tests:** ΔSR (primary) · 95% bootstrap CI on ΔSR · weight assigned  
**Question:** Does the portfolio's Sharpe actually improve when this strategy is added?

```
EMPIRICAL RESULT                                            ● PASS
  Portfolio SR without: 0.94   Portfolio SR with: 1.08   ΔSR: +0.14
  CI: [+0.02, +0.26]   Weight assigned: 14%
  Portfolio Sharpe improves with strategy included.
  [View ΔSR bootstrap distribution ▾]
```

**Inline metrics:**

| Label | Value |
|-------|-------|
| `Portfolio SR without` | NW-adjusted portfolio Sharpe without this strategy |
| `Portfolio SR with` | NW-adjusted portfolio Sharpe with this strategy |
| `ΔSR` | Point estimate, always with sign (ember if positive, red if negative) |
| `CI` | 95% bootstrap CI on ΔSR |
| `Weight assigned` | Weight the weight layer gives this strategy |

**Verdict thresholds:**

| ΔSR | Verdict |
|-----|---------|
| > 0 | PASS |
| ≤ 0 | FAIL (hard) — continue blocked; only "Discard Strategy" available |

**Weight warning (inline, not a separate card):**

If `weight < 3%` and `ΔSR > 0`: amber annotation directly below the weight number:

```
  Weight assigned: 2%  ⚠ Below floor (3%) — strategy adds value in
                         principle but is effectively zero-weighted.
```

The researcher can still continue — advisory only.

**Chart (collapsed by default):**

Bootstrap histogram of ΔSR distribution. Real ΔSR as ember vertical line. CI bounds as `paper-2` dashed vertical lines. Positive ΔSR region shaded ember at 0.10 opacity. Annotation showing the 95% CI range.

---

### Card 3 — Portfolio Impact (Observational)

**Tests:** ΔIDM · IDM before/after  
Always `· INFO` verdict — never pass/fail.

```
PORTFOLIO IMPACT                                            · INFO
  IDM without: 1.42   IDM with: 1.51   ΔIDM: +0.09
  Modest diversification benefit — some regime coverage added.
```

ΔIDM interpretation annotation (text only, no color gating):

| ΔIDM | Annotation |
|------|-----------|
| > 0.10 | "Material diversification — low correlation with existing portfolio" |
| 0.02–0.10 | "Modest diversification benefit" |
| < 0.02 | "Negligible diversification — strategy highly correlated with existing set" |
| < 0 | "IDM decreases — strategy more correlated than average existing pair" |

---

## Overall verdict banner

**PASS:**
```
┌──────────────────────────────────────────────────────────── ● PASS ──┐
│  Portfolio Sharpe improves with this strategy.                         │
│  Strategy is approved for portfolio inclusion.                          │
│                                              [ Continue → Commit → ]  │
└───────────────────────────────────────────────────────────────────────┘
```

**FAIL:**
```
┌──────────────────────────────────────────────────────────── ✗ FAIL ──┐
│  Portfolio Sharpe does not improve with this strategy.                 │
│  This strategy cannot be added to this portfolio.                      │
│                                                                         │
│  Do not modify parameters in response to this result.                  │
│  To develop a replacement: start a new strategy from scratch.          │
│                                                                         │
│  [ Discard strategy ]                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

Fail state: Continue button is hidden. "Discard strategy" shows a confirmation dialog before removing the strategy from the pipeline.

---

## Contamination notice

Below the overall verdict, always visible after results complete:

```
·  ·  ·

This result is final. The following are not permitted:
  — Adjusting parameters in response to this result
  — Changing the weight layer to make a failing strategy pass
  — Re-running the gate after modifying strategy design
```

Muted text. Small. Not a banner — present so it cannot be missed.

---

## IS + Validation recap panel (read-only)

Collapsed at the top — see [display_patterns.md](display_patterns.md):

```
IS + VALIDATION SUMMARY                                    [Expand ▾]
DSR 0.94  ·  Val degradation 71%  ·  CUSUM PASS  ·  fast 16 · slow 64
```

---

## Exit criteria

| Criterion | Required for Continue |
|-----------|----------------------|
| Gate job completed | ✅ |
| ΔSR > 0 | ✅ — hard gate; no override |

**Continue** → `/research/{id}/commit`
