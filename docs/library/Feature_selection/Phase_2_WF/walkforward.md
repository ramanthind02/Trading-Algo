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

**Purpose:** confirm the WF Sharpe is not explained by spurious correlations or overfitting to the param search process.

### Method: Two-Region Shuffle

Shuffling globally across all data would break temporal structure and create an unrealistic null. Instead, features are shuffled independently within two temporal regions:

```
Region 1: [train_start, train_end)   ← initial training window shuffled as unit
Region 2: [train_end, data_end)      ← all remaining data shuffled as unit
```

This prevents information leakage between regions while preserving the walkforward structure.

### Procedure

1. Run full WF (Phase 3) on **real data** → observe aggregate Sharpe `S_obs`
2. Repeat N = 500 times:
   - Shuffle features within each of the two regions independently
   - Run full Phase 3 WF on shuffled data → `S_perm_i`
3. Compute p-value: `p = fraction of S_perm_i ≥ S_obs`
4. **Gate:** p ≤ α = 0.05 to pass

> [!warning] Compute cost
> N=500 full WF runs — each run refits all param combos across all folds. Parallelise across permutations (`n_jobs=-1`) using existing `PermutationEngine` infrastructure.

### Interpreting the Result

- `p ≤ 0.05`: WF Sharpe is unlikely under the null of no predictive signal → feature passes Phase 4
- `p > 0.05`: observed WF performance is consistent with chance → feature rejected
- Phase 4 failure after Phase 3 pass signals that param search inflated apparent performance (selection bias)

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
