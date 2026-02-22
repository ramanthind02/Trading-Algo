# K-Fold Cross-Validation

> **Role:** Phase 2 IS diagnostic — fold consistency (Stages 1+2) and IS stability formalization (Stage 3)
> See [[pipeline]] for the full feature selection pipeline.

---

## What It Is

- Time-series k-fold: data split into **K contiguous temporal blocks** — no shuffling
- Each fold is held out in turn as the "test" set; training uses remaining folds
- Folds are pre-assigned before any data is examined (no adaptive splitting)

```
IS period: 2000 ──────────────────────── 2023

K=5:
  Fold 1: 2000–2004
  Fold 2: 2005–2009
  Fold 3: 2010–2014
  Fold 4: 2015–2019
  Fold 5: 2020–2023
```

---

## Purging and Embargoing

Financial features are autocorrelated — labels near fold boundaries overlap temporally with adjacent folds, causing leakage even without explicit look-ahead.

- **Purging:** remove training observations whose forward-return label extends into the test fold
- **Embargo:** remove a buffer at the start (and optionally end) of each test fold
- **Rule of thumb:** embargo = `max(label_horizon, feature_lookback_max) + 1` bars
  - Example: 20-day return label + RSI max lookback 20 → 21 bar embargo (~1 month daily)

> [!warning] Stage 3 boundary trimming is distinct
> In Stage 3, each fold is evaluated on its own data (no train/test split per fold). Trim the last `label_horizon` bars from each fold's tail to prevent cross-fold label overlap. This is not the same as train-test purging.

---

## Phase 2 Stages 1+2 — Fold-by-Fold Permutation Diagnostics

**Purpose:** convert a single IS p-value into a vector of k p-values — one per sub-period.

> "Is this feature consistent across regimes, or driven by a single lucky period?"

**Procedure:**
- For each fold k, run Stage 1 vector shuffle permutation on that fold's data
- Aggregate per-fold p-values using **Fisher's combined test** or **harmonic mean p-value (HMP)**:
  - Fisher: `χ² = -2 × Σ ln(p_k)` — `χ²` ~ chi-squared with `2k` df under null
  - HMP: `k / Σ(1/p_k)` — more conservative; controls FWER under arbitrary dependence

| Pattern | Interpretation |
|---|---|
| p ≤ 0.10 in all folds | Strong, consistent signal |
| p ≤ 0.10 in majority | Signal present, some regime sensitivity |
| p ≤ 0.10 in 1 fold only | Regime-concentrated — treat with caution |
| p > 0.10 in all folds | No signal; feature should fail Stage 1 |

> [!note] Compute cost
> K-fold multiplies Stage 1 compute by K. Apply Stage 2 (pipeline/candle shuffle) k-fold only for features of particular interest — not by default.

---

## Phase 2 Stage 3 — IS Stability Formalization

**Procedure:**
- Split IS period into K non-overlapping folds (K = 5 or 8)
- For each fold: trim boundary bars by label horizon, compute raw + neighbor-smoothed metric for all params, record which params fall in the top-K selected set
- Stability criterion: param p is **IS-stable** if it appears in selected set across ≥ `min_folds_stable` folds (default 3)

---

## Choosing K

| IS data length | Recommended K | Rationale |
|---|---|---|
| 10–15 years daily | K = 4–5 | ~3-year folds; stable metric estimates |
| 15–24 years daily | K = 5–8 | ~3–4.6-year folds |
| > 24 years daily | K = 8–12 | keep folds ≥ 2 years |

- **K too small (2–3):** little resolution on stability spectrum
- **K too large (15+):** each fold has too few bars → noisy per-fold permutation tests

---

## Limitations

1. **Reduced power per fold** — each fold has 1/k of IS data; fold-level p-values are noisier
2. **Correlated folds** — adjacent folds share macro/vol regimes; Fisher and HMP handle this better than naive combination
3. **Not a live-trading simulation** — when fold k is OOS, training may include data from after fold k; this does not mirror production. Use sequential WF (Phase 3) for that.

---

## Related

- [[pipeline]] — master pipeline; Phases 2 and 3
- [[permutation_testing]] — Stage 1+2+3 permutation specs
- [[cpcv]] — combinatorial alternative with more paths for Stage 3
- [[param_stability]] — neighbor smoothing theory used inside folds
