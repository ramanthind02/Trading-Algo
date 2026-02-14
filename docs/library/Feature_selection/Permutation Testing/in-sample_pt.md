# In-Sample Permutation Testing (Spec)

In-sample permutation tests check whether a feature's performance is better than chance. The user specifies a single **objective metric** (e.g. Sortino, Sharpe) used to evaluate performance in **all** tests. Tests are run in sequence: features that pass earlier stages are passed to later ones. A single feature has **multiple parameter combinations**; each param combo is tested independently, so some params may pass and others fail.

Ensemble formation is **deferred to the end** of the pipeline. Stages 1–2 validate individual param combos. Stage 3 assesses temporal stability across walkforward folds. The researcher then manually forms an ensemble from the validated, stable set.

---

## User Input (Global)

| Input | Description |
|-------|-------------|
| **Objective metric** | The metric used to evaluate performance in every permutation test (e.g. Sortino ratio, Sharpe ratio). Same metric for null comparison and for ranking. |
| **Significance level (α)** | Pass threshold: original must beat the **(1 − α)** quantile of the null. Default **α = 0.1** (i.e. 90th percentile; lax to reduce false negatives across multiple layers). Stricter option: α = 0.05. |
| **Replicate count** | Number of permutation replicates (e.g. **500–1000**) to build the null distribution. |
| **Continuous null (optional)** | For pipeline permutation on **continuous** features: **shuffle_feature** (quick screen for stage 1) or **shuffle_candles** (recommended for stage 2). Candle-based permutation uses a **stronger null** that destroys temporal structure (see §2a). |
| **Metric threshold** | Pre-specified threshold for the selection metric (e.g. Sharpe > 0.5). Must be set BEFORE seeing any data to avoid data snooping. Same threshold used for original and all permutations. See [Continuous_binning](../feature_types/Continuous_binning.md) for recommended defaults. |

---

## 1. Shuffling Permutation Test (Vector-Level)

**Purpose:** Quick filter using only vectors. Destroy the relationship between feature and target by shuffling the feature, while keeping the same **marginal distribution** of the feature (same frequency of values).

### Procedure

- **Input:** The **feature vector** from a fitted model — either the **binned continuous** output (position multipliers or bin assignments) or the **rule-based** output (-1, 0, 1). Same length as target.
- **Shuffle:** Randomly permute the feature vector (same values, new order). This preserves the exact frequency of each value but breaks time alignment with the target.
- **Evaluate:** Compute the user's **objective metric** on the strategy returns: (target × shuffled_feature) where non-zero, or equivalent for position multipliers.
- **Compare:** Run many permutation replicates (see User Input). Original feature **passes** if the **original** objective is better than the **(1 − α)** quantile of the null (e.g. 90th percentile when α = 0.1), i.e. we reject the null that the feature has no edge.

### Scope

- Applied **per parameter combination**. A feature (e.g. RSI) has many param combos (e.g. lookback 2, 3, 4, 5). Each combo is tested separately; some may pass, some fail.
- **Output:** Only param combos that **pass** this test are passed to the next stage. No pipeline re-run — vectors only.
- **Early stopping:** Param combos that fail Stage 1 are **excluded** from Stage 2 testing (saves computation).

### Reference

- Base model concept (ensemble of binning models per param): [base_model](../base_models/base_model.md).

---

## 2. Pipeline Permutation Test

**Purpose:** Test the **entire pipeline** under the null, not just the final vector. More robust than vector shuffle because it respects how the signal is produced. Differs by feature type.

### 2a. Continuous Features

Two permutation modes (user choice; **shuffle_candles** is **recommended** as a stronger null):

**Option A — Shuffle raw feature (quick screen)**
- **Shuffle:** Shuffle the **raw bias-node feature vector** (the continuous series **before** binning), e.g. RSI values. This breaks the link between feature and target while keeping the same marginal distribution of the feature.
- **Pipeline:** Run the full [Continuous_binning](../feature_types/Continuous_binning.md) pipeline on the shuffled feature (same target): quantile binning → per-bin stats → contiguous regions → **threshold filter** → position multipliers.
- **Null:** "No relationship between feature values and target."
- **Use case:** Fast initial filter (stage 1 vector shuffle or early stage 2 screening).

**Option B — Shuffle candles (stronger null, recommended)**
- **Shuffle:** **Shuffle the bars (candles)** in time. Recompute the **bias-node feature** from the shuffled candle stream (so the feature series is derived from randomly ordered bars). Then run the full continuous binning pipeline on that feature (with the target aligned to the shuffled order, or as defined by the user's target alignment).
- **Pipeline:** Same as above: full binning pipeline; fixed pre-specified threshold; no valid bins → 0.
- **Null:** "No genuine temporal structure in price/volume that the feature exploits." This destroys both feature–target alignment and any **serial/temporal structure** in the feature (e.g. momentum runs, mean reversion patterns). Passing this test is a stricter requirement.
- **Use case:** Rigorous test for autocorrelated features (momentum, trend indicators). More computationally expensive but more conservative.

**Fixed threshold (both options):** Use the **pre-specified metric threshold** (set before seeing any data, see User Input). The threshold must be the same for original data and all permutations to avoid data snooping. If **no** bins satisfy the threshold **and** the minimum consecutive-bins (region width) rule, assign **0** for all timestamps (that permutation takes no trades; objective = 0 or undefined).

**Evaluate:** Compute the objective metric on the pipeline's output (position multiplier × target where non-zero). Compare to the original pipeline's metric over many replicates.

### 2b. Rule-Based Features

- **Why not shuffle the feature:** Rule-based outputs (-1, 0, 1) are **serially correlated** (e.g. breakouts persist). Shuffling the feature vector would give unrealistic sequences (e.g. rapid flipping that could not occur in real bars).
- **Shuffle bars:** Instead, **shuffle the bars (candles)** in time. Feed the **shuffled candle stream** into **all** param-combo rule-based models. Each model produces a rule output series; then apply the rule-based binning (Sharpe / adjusted Sharpe → position multiplier) as in [rule_based](../feature_types/rule_based.md).
- **Evaluate:** For each param combo, compute the objective metric on (target aligned to shuffled data × model output). Compare to original over many replicates.

So we ask: "If the order of bars had been random, would our rule still produce an edge?"

### Pass/fail

- Same idea as in §1: run many replicates, build null distribution; original **passes** if it beats the **(1 − α)** quantile of the null. Applied **per param combo** for both continuous and rule-based.
- **Early stopping:** Only param combos that passed Stage 1 are tested in Stage 2. Param combos that fail Stage 2 are **excluded** from Stage 3 (walkforward stability).

---

## 3. Walkforward Stability Analysis

**Purpose:** Assess whether the same region of parameter space is consistently "best" across different time periods. This is a **temporal stability analysis**, not a permutation test. It answers: "Is this feature's optimal param region stable over time, or does it jump around (suggesting noise or regime-dependence)?"

### Why stability analysis, not walkforward permutation

For **rule-based features**, there is no fitting step — the param IS the model. Traditional walkforward (train on fold K, predict fold K+1) is meaningless because there's nothing to train. Selecting the best param combo is itself the fitting process.

For **continuous features**, the binning pipeline is fitted, but the primary question here is the same: is the same param region consistently good across time periods?

We therefore reframe walkforward as a **diagnostic stability test** that informs the researcher's manual ensemble decision (§4).

### Fold structure

- Divide in-sample data into **non-overlapping walkforward folds** (e.g. 2-year chunks: 2015-2016, 2017-2018, 2019-2020, 2021-2022).
- Each fold represents a somewhat independent time period.
- Fold structure is defined by the researcher before running the analysis.

### Input: all params

Use **ALL parameter combinations** for the walkforward stability analysis — not just those that passed §1–§2. Reasons:

1. **Neighbor smoothing needs the full grid.** If only 5 of 20 params survived permutation testing and they're scattered, you can't compute meaningful neighbor averages.
2. **Stability is about relative ranking.** We need the full landscape to see whether the "good region" is consistent.
3. **Permutation tests already filtered.** The researcher will only select from validated params (see §4), but the full landscape informs the assessment.

### Procedure

**For each walkforward fold independently:**

1. **Compute objective metric** for ALL param combos on that fold's data:
   - **Continuous features:** Fit the full binning pipeline on that fold's data, evaluate on the same fold. (In-sample for the binning, but we're measuring relative param ranking, not absolute performance.)
   - **Rule-based features:** Compute rule output on that fold's data, evaluate. No fitting needed — just apply each param and measure performance.

2. **Compute smoothed neighbor metric** for each param:
   ```
   smoothed_obj(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])
   ```
   Same neighbor definition as [Grid-Aware Neighbor Averaging](../../../to-do/grid_neighbor_smoothing_specs.md): 1-step axis-aligned neighbors in the ordered parameter grid.

3. **Select top K param combos** by smoothed objective (recommended K = 3).

### Comparing across folds

Record the top-K selections per fold and assess consistency.

**Stable (good) — feature worth deploying:**
```
RSI lookback grid: [2, 3, 4, 5, 10, 14, 20]

Fold 1 (2015-2016): top 3 = {4, 5, 3}      → center ≈ 4
Fold 2 (2017-2018): top 3 = {5, 4, 10}     → center ≈ 5
Fold 3 (2019-2020): top 3 = {4, 5, 3}      → center ≈ 4
Fold 4 (2021-2022): top 3 = {5, 10, 4}     → center ≈ 5

→ Consistent neighborhood around 4–5. Slight variation is expected and healthy.
```

**Unstable (bad) — feature is noise or regime-dependent:**
```
Fold 1 (2015-2016): top 3 = {4, 5, 3}      → center ≈ 4
Fold 2 (2017-2018): top 3 = {20, 14, 10}   → center ≈ 14–20
Fold 3 (2019-2020): top 3 = {3, 2, 4}      → center ≈ 3
Fold 4 (2021-2022): top 3 = {14, 10, 20}   → center ≈ 14

→ Jumping between short and long lookbacks. Likely noise or regime-dependent.
```

### Overlay with permutation test results

The walkforward stability analysis runs on ALL params, but the researcher should **overlay which params passed permutation testing (§1, §2)**. A param that is consistently in the top-K across folds AND passed permutation tests is a strong candidate for the ensemble. A param that is in the top-K but failed permutation testing should not be selected.

### Continuous vs rule-based: key difference

- **Rule-based:** Walkforward is purely a temporal stability test. There is no model to generalize — the param IS the model. The fold-by-fold analysis asks: "Is the same param region good in different time periods?"
- **Continuous:** Walkforward serves a **dual purpose**: (1) assess param stability across folds (same as rule-based), and (2) observe whether binning models fitted on one fold's data produce reasonable param rankings. The binning is refitted per fold, so each fold's rankings come from an independent fit.

### Output

Stability report containing:
- Per-fold top-K param selections (with smoothed objectives)
- Per-fold smoothed objective landscape (table for 1D grids, heatmap for 2D+ grids)
- Overlay: which of the top-K params also passed permutation tests (§1, §2)
- Summary: is the feature stable (consistent param region) or unstable (jumping around)?

The researcher uses this report to make the ensemble decision (§4).

---

## 4. Researcher Ensemble Formation

**Purpose:** The researcher manually forms the ensemble from individually-validated, temporally-stable param combos. This replaces automated ensemble selection.

### Inputs available to researcher

1. **Permutation test results (§1, §2):** Which param combos passed individual permutation tests on the full in-sample data.
2. **Walkforward stability report (§3):** Which params were consistently selected as top-K across folds, the smoothed objective landscape per fold, and the overall stability assessment.

### Researcher decision

The researcher selects ensemble members that satisfy **both** criteria:
1. **Passed permutation tests** (§1 and §2) — statistically validated edge
2. **Show temporal stability** (§3) — consistently in or near the top-K across multiple folds

**Typical ensemble:** 2–10 members from a consistent parameter neighborhood.

**Example:**
```
RSI lookback grid: [2, 3, 4, 5, 10, 14, 20]

Params that passed permutation tests: {3, 4, 5, 10, 14}
Walkforward stability: consistent top-K around {4, 5, 3} across folds

→ Researcher selects ensemble: {3, 4, 5}
→ Drops 10 and 14 (passed permutation tests but not in the stable region)
```

### Pre-committed stability criteria (recommended)

Before running the walkforward stability analysis, the researcher should define what "stable" means. This prevents post-hoc rationalization.

**Examples of pre-committed criteria:**
- "At least 3 of 5 folds select params within a 3-step neighborhood of each other"
- "The best smoothed param is within 2 grid steps across all folds"
- "The top-3 selections overlap by at least 2 params across consecutive folds"

This is analogous to pre-specifying the metric threshold in §1–§2.

### Deployment param selection

The walkforward stability analysis also informs **which specific params to deploy**:

- **If stable across folds** (same region selected every time): Use full in-sample data to compute the final smoothed objective landscape. Select the ensemble from the stable region. Full data gives the most reliable estimate.
- **If showing a gradual drift** (optimal param shifting over time, e.g., shortening lookback): Recency bias is justified. Weight the most recent fold's selection more heavily, or use the most recent fold's top params as the ensemble.
- **If unstable** (params jumping around): The feature likely should not be deployed. Revisit the hypothesis.

### Ensemble output

- **Aggregation:** `final_signal = mean(member_1_output, member_2_output, ...)`
- **No weighting** by performance — simple average.
- **High correlation within ensemble is expected and intentional** — members from a parameter neighborhood (e.g. RSI 3, 4, 5) will be highly correlated. This is parameter smoothing for robustness, not redundancy.

---

## 5. Lock Ensemble for Out-of-Sample

The ensemble from §4 is **fixed**. Deploy on the **hold-out test set** (most recent data, never touched during §1–§4) for final validation before production.

- **No re-optimisation** of which params are in the ensemble.
- **No re-tuning** of thresholds or binning parameters.
- Membership stays fixed. Optional: member weights may be updated (e.g. by past performance); membership itself does not change.

---

## Summary Flow

1. **User:** Specifies **objective metric**, **α** (default 0.1), **replicate count** (e.g. 500–1000), and **pre-specified metric threshold** (e.g. Sharpe > 0.5) for all tests.

2. **Shuffling permutation (vector, §1):** Shuffle feature vector (binned continuous or rule-based) → compare objective to null. **Per param combo;** only **passing** params proceed to Stage 2. Fast initial filter.

3. **Pipeline permutation (§2):**
   - **Continuous:** **Quick screen** with shuffle raw feature, then **rigorous test** with shuffle candles (recommended) → run full [Continuous_binning](../feature_types/Continuous_binning.md) with **pre-specified threshold**; no valid bins → 0.
   - **Rule-based:** **Shuffle bars** → feed to all param-combo rule models → evaluate.
   Per param combo; only passing params proceed to Stage 3. **Early stopping:** params that failed Stage 1 are excluded.

4. **Walkforward stability analysis (§3):** Evaluate ALL params on each walkforward fold independently (including those that failed §1–§2 for neighbor smoothing). Compute smoothed neighbor metric per fold. Select top K per fold. Compare selections across folds. Output: stability report.

5. **Researcher ensemble formation (§4):** Researcher reviews permutation test results (§1–§2) + stability report (§3). Manually selects ensemble from params that **passed permutation tests AND show temporal stability**. Defines ensemble members and locks them.

6. **Lock for OOS (§5):** Ensemble is fixed. Deploy on hold-out test set. No re-tuning.

**Pipeline (with early stopping):**
- Stage 1 (all params) → Stage 2 (only Stage 1 passers) → Stage 3 (all params for smoothing) → researcher selects from (Stage 1 + 2 passers that are also stable in Stage 3) → lock for OOS.

---

## Considerations and Implementation Notes

### Funnel and cost

- **Stages §1–§2** are the permutation testing funnel with **early stopping**: many param combos enter Stage 1, only statistically validated ones proceed to Stage 2.
- **Computational savings from early stopping:**
  - If 50% of params fail Stage 1, Stage 2 runs on only 50% of params → ~50% reduction in Stage 2 compute
  - Example: 20 param combos, 1000 replicates each
    - Without early stopping: 20 × 1000 (Stage 1) + 20 × 1000 (Stage 2) = 40,000 evaluations
    - With early stopping (50% fail): 20 × 1000 (Stage 1) + 10 × 1000 (Stage 2) = 30,000 evaluations
- **Stage §3** (walkforward stability) runs on ALL params (needed for neighbor smoothing), but is computationally cheaper than permutation testing (no replicates — just one evaluation per fold per param).
- **Stage §4** (researcher ensemble formation) is a manual step with no computational cost.
- Total permutation replicates are concentrated in §1–§2. The walkforward stability analysis adds only `n_folds × n_params` evaluations.

### Correlation across features (redundancy)

**Context:** This framework is designed for testing **one feature family at a time** (e.g. RSI with various lookback params, or a single rule with various threshold params). In this context, high correlation within the ensemble is **expected and desirable** — it represents averaging over a "good neighborhood" of parameter space for regularization.

**Intra-family correlation (single feature tested):**
- **Do NOT filter** high correlation within a feature family (e.g. RSI_4 and RSI_5 at correlation 0.95+)
- This is **intentional averaging** for robustness, not redundancy
- The researcher limits ensemble size (typically 2–10 members), preventing excessive correlation

**Inter-family correlation (multiple features combined):**
- **If** you are combining multiple feature families into one portfolio (e.g. RSI_3-5 + MACD_24-28 + Momentum_10-15), apply a correlation filter **between families**, not within families.
- **When:** After each feature family has been individually tested and graduated.
- **What:** Compute correlation between the **aggregate signals** of each family (mean of RSI ensemble vs mean of MACD ensemble).
- **Rule:** If correlation(family_i_aggregate, family_j_aggregate) > **τ** (e.g. **0.85**), the families are redundant features. Keep the family with better objective and drop (or reduce weight of) the other.
- **Result:** Portfolio contains decorrelated feature families, but each family internally may have high correlation (which is fine).

**Summary:**
- **Single feature testing (current scope):** No correlation filter needed.
- **Multi-feature portfolio (future extension):** Filter between families, not within families.
- **Asset pooling:** When training on multiple assets in aggregate, consider checking that the feature family passes permutation tests on individual assets (robustness check), not just pooled data.

### Neighbor smoothing as diagnostic tool

The **smoothed neighbor metric** (used in §3) is the same concept as [Grid-Aware Neighbor Averaging](../../../to-do/grid_neighbor_smoothing_specs.md). In this pipeline, it serves as a **diagnostic tool** for the researcher — it highlights stable regions of parameter space and helps distinguish genuine signal (broad good region) from noise (isolated spikes). It is NOT used for automated ensemble selection.
