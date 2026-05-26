# Research Display Patterns

Design system for the Research Workspace. Governs how metrics, charts, and test results are rendered across all research steps. Apply these patterns consistently — they are the product's visual language for data.

**Parent:** [UI/UX README](../README.md) · [Research user flow](user_flow.md)

---

## Core principle: verdict first, detail on demand

The research area produces a lot of numbers. The user's first question is always binary: **did this pass or fail?** Show the verdict at the top of every test section, then let the user pull the details they want. Never bury the verdict below a wall of charts.

Three levels of disclosure — always in this order:

| Level | What shows | When |
|-------|-----------|------|
| 1. Verdict | Pass/fail pill + 3–4 key numbers | Always visible |
| 2. Chart | Time series, heatmap, scatter | Expand toggle — user chooses |
| 3. Raw data | Full table, computation inputs | Link to drawer/modal — rarely needed |

---

## The Verdict Card

Every test section is a **Verdict Card**. This is the atomic unit of the research UI.

```
┌── [Section label in mono caps] ─────────────── ● PASS ──┐
│  metric_a: 1.42   metric_b: 0.94   metric_c: 33 / 286   │
│  One-sentence interpretation in muted text               │
│  [View chart ▾]                                          │
└──────────────────────────────────────────────────────────┘
```

**Rules:**
- Section label: JetBrains Mono, 11px, uppercase, `muted`, letter-spacing 1.6px
- Verdict pill: right-aligned in the header row
- Key numbers: max **4** per card — choose the most decision-relevant
- All numbers: JetBrains Mono, `paper`, tabular-nums
- Interpretation: Space Grotesk, 13px, `text-secondary` — one sentence only
- Chart toggle: `[View chart ▾]` / `[Hide chart ▲]` — ember text, no button border
- Chart area: collapses/expands with a smooth 200ms ease-out; defaults collapsed

**Hard fail state:**
```
┌── [Section label] ──────────────────────────── ✗ FAIL ──┐   ← red border, 2px
│  metric_a: 0.07   metric_b: —                            │
│  Degradation ratio critically low — IS edge did not       │
│  survive OOS. Strong indication of IS overfitting.        │
│  [View chart ▾]                   [Acknowledge risk →]   │   ← only if override allowed
└──────────────────────────────────────────────────────────┘
```

Hard fail cards **auto-expand** (chart visible immediately). The card border turns `error` red. If an override is possible, the acknowledge CTA appears inline — not in a separate modal.

**Warning state (soft fail):**
```
┌── [Section label] ───────────────────────── ⚠ WARNING ──┐   ← warm amber border
│  ...                                                      │
└──────────────────────────────────────────────────────────┘
```

---

## Verdict pill system

| State | Visual | CSS |
|-------|--------|-----|
| Pass | `● PASS` — ember text, ember-muted fill | `color: var(--ember)` |
| Warning | `⚠ WARNING` — amber text, amber-muted fill | `oklch(0.78 0.16 75)` |
| Fail | `✗ FAIL` — error red text, error-muted fill | `oklch(0.58 0.20 25)` |
| Info | `· INFO` — muted text, ink-3 fill | `color: var(--muted)` |
| Pending | `· NOT RUN` — muted, ink-3 fill | `color: var(--muted)` |
| Running | `◌ RUNNING` — paper, pulsing opacity | spinner animation |

Always include the text label — never use color alone to convey state.

---

## Number formatting

| Type | Format | Font |
|------|--------|------|
| Sharpe ratio | 2 decimal places: `1.42` | JetBrains Mono |
| t-statistic | 2 decimal places: `3.07` | JetBrains Mono |
| Probability / DSR | 2 decimal places: `0.94` | JetBrains Mono |
| Percentage | 1 decimal place: `71.4%` | JetBrains Mono |
| Count | No decimals: `286` | JetBrains Mono |
| Ratio (N_eff / N) | `33 / 286` — slash separated | JetBrains Mono |
| Date | `2022-01-03` — ISO 8601 | JetBrains Mono |
| p-value | `p = 0.031` — always show leading zero | JetBrains Mono |
| CI | `[1.13, 1.71]` — brackets, 2dp | JetBrains Mono |
| Delta | `+0.14` or `−0.03` — always show sign | JetBrains Mono, ember or red |

Positive deltas: ember. Negative deltas: error red. Zero/negligible: muted.

---

## Chart system

All charts render on the **ink** background — no white or paper backgrounds inside chart areas. Charts are instruments, not dashboards.

### Common properties

| Property | Value |
|----------|-------|
| Background | `ink` — no border, no box |
| Grid lines | 1px `ink-3`, horizontal only, 4–5 lines max |
| Axis labels | JetBrains Mono 10px, `muted`, uppercase |
| Axis ticks | None (labels only) |
| Legend | Inline annotation on the line end, not a floating box |
| Padding | 16px inner, 0 outer — chart bleeds edge-to-edge within card |
| Interaction | Crosshair on hover; tooltip in JetBrains Mono; no zoom in MVP |

### Chart types

**Time series (line)**
- Primary line: `ember` (#D57044), 1.5px stroke
- Secondary line: `ember-deep`, 1.5px stroke, dashed
- Reference line (zero / threshold): `paper-2` at 0.35 opacity, 1px dashed
- Zone shading: `ember-muted` fill (10% opacity) for highlighted windows
- Break point marker: vertical `error` dashed line with annotation

**Rolling metric (e.g. rolling Sharpe)**
- Line: `ember`, 1.5px
- 0-line: `paper-2` at 0.35 opacity
- Positive area fill: `ember` at 0.08 opacity
- Negative area fill: error red at 0.08 opacity

**Histogram (null distribution)**
- Bars: `ink-3` fill, 1px `ink-2` gap between bars
- Researcher's real value: `ember` vertical line, 2px
- p-value region: ember fill from line to right at 0.15 opacity
- Annotation: `p = 0.031` in JetBrains Mono next to line

**Heatmap (parameter grid)**
- Color scale: single warm ramp — `ink-2` (low) → `ember-deep` → `ember` (high)
- Never use blue-red diverging scale — warm spectrum only per brand
- Cell text: hidden by default, show on hover
- Selected combination: `paper-2` dot marker, 3px
- Perturbation box: `paper-2` dashed border at 0.5 opacity

**Bar comparison (IS vs Val Sharpe)**
- IS bar: `ember` fill
- Val bar: `ember-deep` fill
- Error bars: `paper-2` line + CI fill at 0.15 opacity
- Overlap shading: `ember-muted` hatched fill over the overlap region
- Labels above bars in JetBrains Mono 10px

**Scatter (rank correlation)**
- Dots: `ink-3` fill, `ink-2` border, 4px radius
- Chosen combination: `ember` fill, `paper-2` border, 5px radius
- Regression line: `paper-2` at 0.4 opacity, 1px dashed
- ρ annotation: top-left of chart in JetBrains Mono 10px

---

## Grouping test results

Multiple tests that answer the same underlying question are grouped into a single Verdict Card. The group verdict is the most severe of its constituent tests.

### Research step 3 (IS robustness) — groups

| Group card | Tests inside | Verdict |
|-----------|-------------|---------|
| Search Bias | DSR · N_eff · NW inflation factor | Most severe of sub-tests |
| Temporal Stability | Rolling Sharpe positive fraction · CUSUM | Most severe |
| (On-demand) Permutation | Full Grid p-value | Separate card, not auto-run |

### Research step 4 (Validation) — groups

| Group card | Tests inside | Verdict |
|-----------|-------------|---------|
| Performance Survived | Sharpe degradation ratio · CI overlap · degradation z | Most severe |
| Strategy Alive | CUSUM break detected · rolling z-score | Most severe |
| Parameter Surface | Neighbourhood val_p10 · rank correlation ρ | Most severe |

### Research step 5 (Portfolio Addition) — groups

| Group card | Tests inside | Verdict |
|-----------|-------------|---------|
| Theoretical Case | Analytical hurdle margin · effective correlation · drawdown overlap | Most severe |
| Empirical Result | ΔSR · ΔSR CI · weight assigned | Most severe |
| Observational | ΔIDM · IDM before/after | Always INFO — no pass/fail |

---

## Layout principles for research screens

**Content width:** 860px max. Research is focused work — narrower than the portfolio hub. The stepper and content share the same column.

**Reading order:** Verdict → Numbers → Sentence interpretation → Chart. Top to bottom, no lateral scanning required.

**Whitespace:** 24px between cards. 16px between elements inside a card. No section dividers beyond card boundaries.

**Sticky stepper:** The 6-step stepper is sticky to the top of the viewport. When the user scrolls down through many cards, they always know where they are in the pipeline.

**No sidebars in research:** The research workspace is full-width content within the app shell. No secondary navigation panels, no floating sidebars. Focus entirely on the current step.

**Collapsed by default:** Every chart and every detail drawer starts collapsed. The page renders fast and uncluttered. The user expands what they care about.

**Run states:**
```
[ Run validation ]   →   [ ◌ Running · 14s ]   →   [ ● Complete · Rerun ]
```
Single button, state transitions inline, no separate progress modal.

---

## IS recap panel (cross-step persistence)

On steps 4, 5, and 6, a collapsed panel at the top of the page shows the IS summary:

```
IS Results (step 3)  [Expand ▾]
DSR 0.94 · NW λ 1.43 · Rolling PASS · {fast: 16, slow: 64}
```

This panel is always collapsed by default. It prevents the researcher from needing to navigate back to review IS numbers while working on validation or portfolio addition. Numbers are read-only — no re-run control.

---

## Empty and pending states

| State | Display |
|-------|---------|
| Step not yet reached | Card greyed out — `text-muted`, `ink-2` background — "Run [previous step] to unlock" |
| Run not triggered | `[ Run validation ]` CTA only — no metrics, no charts |
| Running | Spinner in card header, key metrics show `—` placeholders |
| Completed, no issues | All cards collapsed, overall verdict at top |
| Completed, failures | Failed cards auto-expanded, other cards collapsed |

---

## Accessibility

| Requirement | Implementation |
|-------------|----------------|
| Color not sole signal | Every verdict uses text label (PASS/FAIL) + icon |
| Focus ring | 2px `ember` outline, 2px offset — all interactive elements |
| Chart alt text | Each chart has a `title` + one-sentence `aria-description` |
| Reduced motion | Charts render without animation; no pulsing state transitions |
| Keyboard expand | Chart toggles operable with Enter/Space |
