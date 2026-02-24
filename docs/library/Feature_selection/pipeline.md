# Feature Selection Pipeline

> [!note] Status: Library reference — master pipeline overview
> Last updated: 2026-02-22

## Data Splits

- **In-Sample (IS):** 2000–2023 — all screening, permutation testing, and walkforward validation
- **Walk-Forward (WF):** overlapping windows within IS; rolling or expanding; see [[walkforward]]
- **Strict OOS:** 2024–2025 — untouched until all WF gates pass; used only for final confirmation

---

## Pipeline Overview

```
IS Data (2000-2023)
│
├── TIER 1: In-Sample (IS)
│   ├── Phase 1: EDA ──────────────────────────── [no gate — informational]
│   │   └── distributions, decile plots, param sensitivity heatmap
│   │       └── rule-based: level stats + transition matrix
│   │
│   ├── Phase 2: Binning Analysis ───────────── [continuous only; rule-based skips]
│   │   └── grid search over bin_counts → single best bin per direction
│   │
│   └── Phase 3: IS Permutation Screen ────────── [GATE: any param passes S1+S2?]
│       └── vector shuffle → pipeline permutation (per param combo)
│           └── FAIL → feature class rejected (no walkforward)
│
├── TIER 2: Walk-Forward (WF)
│   ├── Phase 4: WF Validation ────────────────── [GATE: stable neighborhood ≥ min_folds]
│   │   └── expanding/rolling folds, neighbor-smoothed metric, stable region ID
│   │       └── FAIL → feature class rejected
│   │
│   └── Phase 5: WF Permutation Test ─────────── [GATE: p ≤ α on WF metric]
│       └── candle shuffle on full WF
│           └── FAIL → feature class rejected
│
└── TIER 3: Out-of-Sample (OOS)
    ├── Phase 6: OOS Validation ───────────────── [GATE: stable on OOS folds]
    │   └── expanding/rolling folds on 2024–2025
    │       └── FAIL → feature class rejected
    │
    ├── Phase 7: OOS Permutation Test ────────── [GATE: p ≤ α on OOS metric]
    │   └── candle shuffle on OOS period
    │       └── FAIL → feature class rejected
    │
    └── Phase 8: Graduation ──────────────────── param selection rule applied → [[vault]]
```

---

## Gates vs. Diagnostics

| Phase | Type | Passes On | Notes |
|-------|------|-----------|-------|
| 1 EDA | Diagnostic | — (always) | Researcher inspection only |
| 2 Binning (continuous only) | Diagnostic | — (always) | Inform param selection |
| 3 IS Permutation | **GATE** | Any param combo passes S1 + S2 | Feature-level decision |
| 4 WF Validation | **GATE** | Stable neighborhood ≥ min\_folds | Param-level; see [[param_stability]] |
| 5 WF Permutation | **GATE** | p ≤ α on WF metric | Destroys temporal structure |
| 6 OOS Validation | **GATE** | Stable neighborhood ≥ min\_folds | Same region algorithm as Phase 4 |
| 7 OOS Permutation | **GATE** | p ≤ α on OOS metric | Confirms Phases 4–5 robustness |
| 8 Graduation | Decision | All gates passed | OOS lock broken → vault |

> [!note] **Feature type**: rule-based features skip Phase 2 (Binning Analysis) — they are already discrete (-1/0/+1). IS permutation results (Phase 3) are DIAGNOSTIC per param combo — if the feature class has any signal (any param passes), all params enter WF together.

---

## Pre-Committed Param Selection Rule (Summary)

Applied at Phase 8 graduation. Full spec: [[param_stability]].

1. Fit ALL param combos on IS data → compute objective metric
2. Apply grid-aware neighbor smoothing → `smoothed_objective(P) = mean(P + neighbors)`
3. Identify stable regions: contiguous params where `stability_ratio > 0.8`
4. Select representative params from each stable region (e.g., centroid or best-in-region)
5. **If < 2 params selected → no position taken** (insufficient stability evidence)

> [!warning] All param combos enter WF regardless of IS pass/fail. IS screening answers "does this feature class have signal?" — not "which params are good?"

---

## Feature Types

- **Continuous** (RSI, momentum, vol ratios): require Phase 2 binning analysis → see [[feature_model]]
- **Rule-based** (breakouts, regime filters, discrete -1/0/+1): skip Phase 2, use fixed 3 levels → see [[feature_model]]

Both types follow the same Phases 1, 3–8. Only Phase 2 (Binning Analysis) is type-specific.

---

## Related Docs

**Phase 1 (In-Sample):**
- [[eda]] — Phase 1 EDA detail
- [[feature_model]] — Phase 2 binning (continuous) + shared feature interface
- [[permutation_testing]] — Phase 3 IS permutation test specs
- [[kfold]] — k-fold cross-validation reference
- [[cpcv]] — Combinatorial purged CV reference

**Phase 2 (Walk-Forward):**
- [[walkforward]] — Phases 4–5 walkforward implementation
- [[param_stability]] — Neighbor smoothing, stable region selection, param selection rule

**Phase 3 (Out-of-Sample):**
- [[oos_validation]] — Phases 6–8 OOS validation and graduation

**Shared:**
- [[base_feature]] — Shared target definition and Sharpe formula
- [[vault]] — Feature persistence after graduation
