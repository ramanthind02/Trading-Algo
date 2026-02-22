# Feature Selection Pipeline

> [!note] Status: Library reference — master pipeline overview
> Last updated: 2026-02-22

## Data Splits

- **In-Sample (IS):** 2000–2023 — all screening, permutation testing, and walkforward validation
- **Strict OOS:** 2024–2025 — untouched until graduation; used only for final confirmation

---

## Pipeline Overview

```
IS Data (2000-2023)
│
├── TIER 1: IS Screening
│   ├── Phase 1: EDA ──────────────────── [no gate — informational]
│   │   └── distributions, decile plots, param grid heatmap
│   │
│   └── Phase 2: IS Permutation Screen ── [GATE: any param passes S1+S2?]
│       └── vector shuffle → pipeline permutation (per param combo)
│           └── FAIL → feature class rejected (no walkforward)
│
└── TIER 2: Walkforward
    ├── Phase 3: WF Validation ─────────── [GATE: neighborhood stable ≥ min_folds]
    │   └── rolling folds, neighbor-smoothed metric, stable region ID
    │       └── FAIL → feature class rejected
    │
    ├── Phase 4: WF Permutation Test ───── [GATE: p ≤ α on WF metric]
    │   └── candle shuffle on full WF
    │       └── FAIL → feature class rejected
    │
    └── Phase 5: Graduation ────────────── param selection rule applied → [[vault]]
```

---

## Gates vs. Diagnostics

| Phase | Type | Passes On | Notes |
|-------|------|-----------|-------|
| 1 EDA | Diagnostic | — (always) | Researcher inspection only |
| 2 IS Permutation | **GATE** | Any param combo passes S1 + S2 | Feature-level decision |
| 3 WF Validation | **GATE** | Stable neighborhood ≥ min\_folds | Param-level; see [[param_stability]] |
| 4 WF Permutation | **GATE** | p ≤ α on WF metric | Destroys temporal structure |
| 5 Graduation | Decision | All gates passed | OOS lock broken |

> [!note] IS permutation results are DIAGNOSTIC per param combo — they do NOT gate individual params from entering the walkforward. If the feature class has any signal (any param passes), all params enter WF together.

---

## Pre-Committed Param Selection Rule (Summary)

Applied at Phase 5 graduation. Full spec: [[param_stability]].

1. Fit ALL param combos on IS data → compute objective metric
2. Apply grid-aware neighbor smoothing → `smoothed_objective(P) = mean(P + neighbors)`
3. Identify stable regions: contiguous params where `stability_ratio > 0.8`
4. Select representative params from each stable region (e.g., centroid or best-in-region)
5. **If < 2 params selected → no position taken** (insufficient stability evidence)

> [!warning] All param combos enter WF regardless of IS pass/fail. IS screening answers "does this feature class have signal?" — not "which params are good?"

---

## Feature Types

- **Continuous** (RSI, momentum, vol ratios): require binning → see [[continuous_binning]]
- **Rule-based** (breakouts, regime filters, discrete -1/0/+1): already discrete → see [[rule_based]]

Each type follows distinct validation paths in Phases 1–2. Phases 3–5 are shared.

---

## Related Docs

- [[eda]] — Phase 1 EDA detail
- [[permutation_testing]] — IS + WF permutation test specs
- [[walkforward]] — Phase 3–4 walkforward implementation
- [[param_stability]] — Neighbor smoothing, stability ratio, param selection rule
- [[kfold]] — k-fold cross-validation reference
- [[cpcv]] — Combinatorial purged CV reference
- [[base_feature]] — Base feature interface
- [[continuous_binning]] — Quantile binning pipeline
- [[rule_based]] — Rule-based feature handling
- [[vault]] — Feature persistence after graduation
