# Walk-Forward Validation

> **Role:** Phase 3 — primary OOS robustness gate; Phase 4 — WF permutation test
> See [[pipeline]] for the full feature selection pipeline.

---

## What It Is

Sequential walk-forward validation: always trains on past data, tests on future data. The only CV scheme that accurately emulates production behavior.

> [!important] Key principle
> Every fold trains only on data available before the test period. No future information ever enters the training set. This is not true of [[kfold]] or [[cpcv]] paths.

---

## Phase 3 — Walk-Forward Validation

### Fold Structure

- **Expanding window** (default): training window grows with each fold
- **Rolling window** (alternative): fixed-length training window rolls forward
- **Initial training period:** 2000–2015 (15 years IS)
- **Test step:** 1 year per fold (~252 trading days)
- **Number of folds:** 8 (test period 2015–2023)

```
Fold 1: Train 2000–2015 │ Test 2015–2016
Fold 2: Train 2000–2016 │ Test 2016–2017
Fold 3: Train 2000–2017 │ Test 2017–2018
...
Fold 8: Train 2000–2022 │ Test 2022–2023
```

### Per-Fold Workflow

For each fold:
1. **Fit ALL param combos** on training candles (no pre-filtering by IS pass/fail)
2. Compute raw metric + **neighbor-smoothed metric** across the param grid
3. Identify the **stable region** — neighborhood-consistent params using pre-committed selection rule
4. Form ensemble from stable region params
5. **OOS eval:** generate forecasts on test period, compute Sharpe and drawdown

> [!note] Why fit all params?
> Neighbor smoothing requires the complete grid for correct neighborhood structure. Filtering to IS passers would distort the smoothed surface and bias stable region selection.

### Phase 3 Gate

A feature passes Phase 3 if its **stable region** (neighborhood-consistent param selection) is reproducible across enough folds:

- **Criterion:** stable region appears in ≥ `min_folds_stable` folds (default **3 of 8**)
- Aggregate OOS Sharpe across all 8 test folds is the primary point estimate

### Stability Outcomes Per Fold

| Outcome | Description | Action |
|---|---|---|
| Stable | Same params selected, consistent metric | Pass — contribute to ensemble |
| Drifting | Selected params shift gradually across folds | Acceptable — still contributes |
| Unstable | Wildly different params each fold, no coherent region | Fold excluded from ensemble; feature may fail gate if too many folds unstable |

---

## Phase 4 — Walk-Forward Permutation Test

**Purpose:** Confirm the WF Sharpe is not explained by spurious correlations or overfitting to the param search process. Three sub-stages in escalating cost and null strength; early stopping applies (fail sub-stage N → skip N+1).

---

### Sub-stage 1 — OOS Return Shuffle

**What:** For each replicate, shuffle only the OOS fold returns. Feature signals and selected params are fixed.

**Null:** The strategy's OOS performance is consistent with randomly ordered returns of the same distribution — no genuine alignment between signals and OOS future returns.

**Procedure:**
1. Run full WF on real data → observe aggregate OOS Sharpe `S_obs`
2. Repeat N = 500 times:
   - For each fold: shuffle that fold's OOS return series in place; IS data and signals unchanged
   - Compute aggregate OOS metric across all folds → `S_perm_i`
3. `p = fraction of S_perm_i ≥ S_obs`; **Gate:** p ≤ α = 0.05

**Cost:** Cheapest — no refitting, no feature recomputation. Early stopping: if this fails, skip sub-stages 2 and 3.

---

### Sub-stage 2 — IS Return Shuffle (Full Grid)

**What:** For each replicate, shuffle IS fold returns; OOS fold returns remain real. Run the full stability algorithm on shuffled IS data → select params → evaluate on real OOS.

**Null:** IS fitting on completely random data can identify params that generalise to real OOS returns — i.e., IS fitting provides no genuine signal for OOS prediction.

**Procedure:**
1. Repeat N = 500 times:
   - For each fold: shuffle that fold's IS return series in place; OOS returns unchanged
   - Run stability algorithm on shuffled IS → select best-region params (or score = 0 if no stable region / no params clear absolute metric threshold)
   - Evaluate selected params on real OOS fold → record fold metric
   - Aggregate across folds → `S_perm_i`
2. `p = fraction of S_perm_i ≥ S_obs`; **Gate:** p ≤ α = 0.05

> [!important] **Do not restrict the param grid for this sub-stage.** If you restrict to the known stable region, the stability algorithm running on shuffled IS data will still select from within that neighbourhood — and all those params are similar, so the null distribution is inflated. The full grid is required so that random IS fitting can select from anywhere in param space, producing a realistic null of "what does random IS fitting actually pick."

**Cost:** Medium — stability algorithm runs per replicate but no bar reconstruction. Apply absolute metric threshold (e.g. Sharpe ≥ 0.1) before floor-based selection to prevent spurious stable regions near zero.

---

### Sub-stage 3 — Candle Shuffle (Full WF Period)

**What:** Shuffle the full WF candle series as one pool; recompute the feature from scratch; run the complete walkforward procedure on the shuffled stream.

**Null:** The strategy's observed WF performance is consistent with a completely random market — the entire pipeline (price dynamics → feature → IS fitting → OOS evaluation) could have produced this result by chance from any randomly ordered market with the same bar-structure statistics.

**Procedure:**
1. Run full WF on real data → observe `S_obs` (same as sub-stage 1)
2. Repeat N = 500 times:
   - Shuffle the full WF candle series using [[candle_permutation]] algorithm
   - Fold boundaries are preserved **by position** (bar count), not calendar date
   - Feature recomputed from scratch on the shuffled stream
   - Full Phase 3 WF runs on shuffled data → stability algorithm → OOS eval → `S_perm_i`
3. `p = fraction of S_perm_i ≥ S_obs`; **Gate:** p ≤ α = 0.05

> [!note] Run sub-stage 3 on the **stable region only** (3–10 params identified in Phase 3), not the full param grid. The stable region restriction is appropriate here because the candle shuffle already tests the strongest null (random market); restricting the param space does not inflate the null as it would in sub-stage 2.

> [!warning] Compute cost
> N=500 full WF runs with candle reconstruction and full param grid per fold. Parallelise across permutations (`n_jobs=-1`) using existing `PermutationEngine` infrastructure. Restricting to the stable region dramatically reduces per-replicate cost.

---

### Null Hierarchy Summary

| Sub-stage | OOS returns | IS data | Feature | Cost | Tests |
|---|---|---|---|---|---|
| 1 — OOS return shuffle | Shuffled | Real | Fixed (real) | Cheap | Is OOS performance genuine given fixed strategy? |
| 2 — IS return shuffle | Real | Shuffled | Fixed (real, applied to shuffled IS) | Medium | Does IS fitting genuinely identify predictive params? |
| 3 — Candle shuffle | Shuffled (via candle reconstruct) | Shuffled (via candle reconstruct) | Recomputed from scratch | Expensive | Could a random market produce this result end-to-end? |

### Interpreting the Results

- All three sub-stages pass (p ≤ 0.05): strong evidence of genuine signal at every level
- Sub-stage 1 fails: OOS returns not aligned with signals — feature may be noise or regime-specific
- Sub-stage 2 fails after sub-stage 1 passes: IS fitting does not reliably identify predictive params (selection bias or overfitting to IS noise)
- Sub-stage 3 fails after 1–2 pass: price process structure may be exploitable but fragile to market-regime changes

---

## API Reference

```python
class FeatureValidator:
    def walkforward_test(
        portfolio: Portfolio,
        candles_df: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_step: int = 252,       # ~1 year
        num_steps: int = 10,
        target_col: str = 'log_return',
        verbose: bool = True
    ) -> Tuple[List[Dict], pd.DataFrame, Dict[str, float]]:
        # Returns: (fold_results, summary_df, aggregate_metrics)

    def walkforward_permutation_test(
        portfolio: Portfolio,
        candles_df: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        nreps: int = 500,
        alpha: float = 0.05,
        shuffle_target: bool = False,   # True = shuffle labels; False = shuffle features
        n_jobs: int = -1
    ) -> pd.DataFrame:
        # Returns results with: feature, original_metric, pval, significant
```

**Portfolio agnostic:** `FeatureValidator` accepts any `Portfolio` instance (single base model or full multi-ensemble). It copies and re-fits the portfolio on each fold's training data.

---

## Key Implementation Details

- **WalkForwardSplitter** (`feature_selection/walkforward/walkforward_model.py`): handles all splitting; filters folds with insufficient data (min 100 train, 10 test samples)
- **PortfolioTester** (`ensemble/portfolio_tester.py`): computes fold-level metrics (Sharpe, Sortino, drawdown, win rate)
- **FeaturePermutationStrategy** (`utils/evaluation/permutation_test/permutation_engine.py`): already supports `train_windows` parameter for two-region shuffling

---

## Related

- [[pipeline]] — master pipeline; where Phase 3 and 4 sit in the full sequence
- [[param_stability]] — neighbor smoothing and stable region selection used inside each WF fold
- [[cpcv]] — supplementary distributional view of OOS Sharpe across 28 paths; complements but does not replace sequential WF
