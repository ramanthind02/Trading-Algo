# Enhanced Top-K Ensemble Selection

> **Status:** Library specification
> **Date:** 2026-02-19
> **Extends:** [`param_selection_rule.md`](param_selection_rule.md) — replaces Steps 4–6 (Rank, Cap, K_min) with a three-objective selection algorithm
> **Master pipeline reference:** [`../pipeline_overview.md`](../pipeline_overview.md)

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Limitations of the Current Algorithm](#2-limitations-of-the-current-algorithm)
3. [Three Selection Objectives](#3-three-selection-objectives)
4. [Full Algorithm](#4-full-algorithm)
   - 4.1 [Hard Filters (Pre-Screening)](#41-hard-filters-pre-screening)
   - 4.2 [Stability Score](#42-stability-score)
   - 4.3 [Robustness Score (N-Block CV)](#43-robustness-score-n-block-cv)
   - 4.4 [Signal Correlation Matrix](#44-signal-correlation-matrix)
   - 4.5 [Composite Quality Score](#45-composite-quality-score)
   - 4.6 [Greedy Diversity-Aware Selection](#46-greedy-diversity-aware-selection)
5. [Pre-Committed Parameters](#5-pre-committed-parameters)
6. [Worked Example](#6-worked-example)
7. [Design Rationale](#7-design-rationale)
8. [Integration Points](#8-integration-points)
9. [Related Docs](#9-related-docs)

---

## 1. Overview and Motivation

This document specifies the **Enhanced Top-K Ensemble Selection algorithm**, which replaces the naive rank-and-cap step (Steps 4–6) in [`param_selection_rule.md`](param_selection_rule.md) with a three-objective selection procedure.

The existing algorithm selects the top-k parameter combinations by sorting on the smoothed objective (neighbor-averaged Sharpe/Sortino) and taking the highest-ranked k. This is correct for filtering isolated peaks (via the stability gate) and ensuring minimum quality (via the metric gate), but the top-k selection step itself ignores two critical dimensions:

- **Diversity**: the top-k by smoothed objective may all cluster in the same sub-region of parameter space, producing a highly correlated ensemble with no diversification benefit.
- **Robustness**: the smoothed objective is computed over the full training window; a parameter combo may look excellent in aggregate but perform well only in specific market regimes and fail in others.

This algorithm explicitly optimises all three dimensions simultaneously.

---

## 2. Limitations of the Current Algorithm

**Current Steps 4–6 in param_selection_rule.md:**

```
Step 4: Sort qualifying params by smoothed_objective descending
Step 5: Cap at K_max
Step 6: K_min check → return empty or ensemble
```

**What this misses:**

**Diversity failure:** If the parameter space has a single strong region (e.g., RSI lookbacks 4–6 all outperform), the top-k will be filled entirely by that cluster. All selected params produce nearly identical signals. The ensemble has the statistical properties of a single parameter, not k diverse ones.

**Regime concentration failure:** A parameter combination that earns Sharpe 1.2 in one market regime and -0.1 in others looks like Sharpe ~0.4 in aggregate — the same as a parameter that earns Sharpe 0.4 consistently. The current algorithm cannot distinguish these; it will select the volatile-regime combo if its aggregate smoothed score is higher.

**Frequency blindness:** A parameter that trades extremely rarely (e.g., 1% of bars) can produce inflated Sharpe statistics from a small sample. The current algorithm does not filter by trade frequency.

---

## 3. Three Selection Objectives

The enhanced algorithm explicitly balances three objectives, each targeting a distinct failure mode:

### Objective 1: Stability

**Target failure:** Isolated parameter peaks (overfitting to noise in parameter space).

**Measure:** Neighbor-smoothed objective and stability ratio. Already captured by Gate A (`stability_ratio ≥ 0.80`) from the existing algorithm — parameter combos that fail this gate are eliminated before scoring. The smoothed objective value then forms the **Stability Score** for surviving combos.

**Why smoothed, not raw:** A parameter with smoothed Sharpe 0.7 that has neighbors at Sharpe 0.65–0.72 is genuinely stable. A parameter with raw Sharpe 0.9 but smoothed Sharpe 0.5 is an isolated peak. The smoothed value encodes both performance and stability simultaneously.

### Objective 2: Robustness

**Target failure:** Regime-specific overfitting (parameter performs in one period, fails in others).

**Measure:** N-Block Cross-Validation on the training window. The training data is split into N equal-duration blocks; the objective metric is computed independently on each block. A combo that performs consistently across all blocks is more robust than one that earns the same aggregate score from a single good block.

**Formula:** `robustness_score = mean_block / (1 + cv_block)` where `cv_block = std(block_metrics) / |mean_block|` is the coefficient of variation across blocks. This penalises high variance across blocks while rewarding high mean.

### Objective 3: Diversity

**Target failure:** Ensemble members producing correlated signals (ensemble degenerates to a single signal).

**Measure:** Pairwise signal correlation between training-period return series. Two parameter combos that produce identical or nearly identical signals on the training data provide no diversification benefit even if they are in different areas of parameter space.

**Method:** Greedy max-marginal selection. Rather than precomputing a diversity score per combo, diversity is enforced by the selection procedure: each successive selection is penalised multiplicatively by its maximum absolute correlation with any already-selected combo.

---

## 4. Full Algorithm

This algorithm replaces Steps 4–6 of [`param_selection_rule.md`](param_selection_rule.md). Steps 1–3 (fit all params, compute smoothed objective, apply Gate A and Gate B) remain unchanged as the input to this procedure.

```
Enhanced Top-K Selection
Input:  qualifying_params    — params that passed Gate A (stability) and Gate B (quality)
                               from param_selection_rule.md Steps 1–3
        training_data        — full training window data (candles + target)
        config               — pre-committed config: k, trade_freq_min,
                               stability_ratio_min, n_blocks, diversity_weight,
                               weight_stability, weight_robustness
Output: selected_ensemble    — ordered list of up to k param labels
                               (empty = no position this period)

Step 1: Hard filter — trade frequency
   For each P in qualifying_params:
       signal_series(P) = evaluate_param(training_data, P)
       trade_freq(P) = fraction of bars where signal(P) ≠ 0
       Remove P if trade_freq(P) < trade_freq_min

Step 2: Hard filter — stability ratio
   stability_ratio(P) = smoothed_objective(P) / raw_objective(P)
   Remove P if stability_ratio(P) < stability_ratio_min
   [Note: this is Gate A from param_selection_rule.md, re-confirmed here]

Step 3: Compute robustness score for each surviving P
   Divide training_data into N equal-duration blocks: B_1, ..., B_N
   For each block B_i:
       block_metric(P, i) = objective_metric(evaluate_param(B_i, P))
   mean_block(P) = mean(block_metric(P, 1..N))
   cv_block(P)   = std(block_metric(P, 1..N)) / |mean_block(P)|
                   (set cv = 0 if mean_block = 0 to avoid division by zero)
   robustness_score(P) = mean_block(P) / (1 + cv_block(P))

Step 4: Normalize component scores to [0, 1] across surviving params
   stability_score_norm(P)  = normalize(smoothed_objective(P))
   robustness_score_norm(P) = normalize(robustness_score(P))
   where normalize(x) = (x - min(x)) / (max(x) - min(x))
                        [set to 0 if all values equal]

Step 5: Compute composite quality score
   quality(P) = weight_stability * stability_score_norm(P)
              + weight_robustness * robustness_score_norm(P)

Step 6: Compute signal correlation matrix
   For each pair (P, Q) in surviving params:
       corr(P, Q) = Pearson correlation of signal_series(P) and signal_series(Q)
                    on the full training window
   [signal_series computed in Step 1, no additional model fitting needed]

Step 7: Greedy diversity-aware selection
   selected = []
   candidates = sorted(surviving_params, by quality descending)

   Select first: selected.append(highest-quality candidate)

   While len(selected) < k AND candidates remain:
       For each remaining candidate P:
           max_corr(P) = max(|corr(P, S)| for S in selected)
           adjusted_score(P) = quality(P) * (1 - diversity_weight * max_corr(P))
       Select P with highest adjusted_score → add to selected

Step 8: K_min check
   If len(selected) < K_min:
       return []  # no position this period
   return selected
```

---

### 4.1 Hard Filters (Pre-Screening)

Two hard filters eliminate params before scoring:

**Trade frequency filter:** `trade_freq(P) ≥ trade_freq_min`

A parameter that trades less than 5% of bars produces signals from a thin slice of data. Its Sharpe statistic is unreliable regardless of its magnitude. Params below the frequency floor are eliminated entirely — they do not receive quality scores and cannot be selected.

Trade frequency = fraction of training-window bars where the fitted model produces a non-zero position. For continuous (binned) features, this is the fraction of bars where the binned signal maps to a long or short region. For rule-based features, this is the fraction of bars where the rule fires (+1 or -1).

**Stability ratio filter:** `stability_ratio(P) ≥ stability_ratio_min`

This is Gate A from `param_selection_rule.md`, re-applied here to the surviving qualifying set (should already be satisfied by all params entering this stage, but enforced explicitly).

---

### 4.2 Stability Score

For each surviving param P:

```
stability_score(P) = smoothed_objective(P)
```

The smoothed objective is computed by `add_smoothed_objective` (in `utils/grid_smoothing.py`) using axis-aligned 1-step neighbor averaging. It already encodes both performance level and neighborhood consistency. No additional computation is needed; the value is an output of Steps 1–3 in `param_selection_rule.md`.

This is normalized to [0, 1] across surviving params in Step 4.

---

### 4.3 Robustness Score (N-Block CV)

**Block construction:** The training window is divided into N equal-duration, non-overlapping, temporally ordered blocks. Each block is a contiguous slice of calendar time. Blocks are not shuffled — temporal ordering is preserved.

```
N = n_robustness_blocks  (default: 5)

If training window spans T total trading days:
    Block i covers days [i * (T/N), (i+1) * (T/N))
    (last block absorbs any remainder)
```

**Per-block metric:** For each block B_i and param P, the full evaluation pipeline is run on B_i's data. No fitting is done within the block — the model is fitted on the full training window (as in Step 1 of `param_selection_rule.md`) and then evaluated on each block's subset.

**Why evaluate, not refit?** Refitting within each block would make the block metric reflect parameter-level fitting ability, not signal robustness. We want to know: does this param's signal hold up across different market regimes? The regime diversity is provided by the different time blocks; the signal is fixed from the full-window fit.

**Block metric aggregation:**

```
mean_block(P)  = (1/N) * sum(block_metric(P, i) for i in 1..N)
std_block(P)   = sample std of {block_metric(P, 1), ..., block_metric(P, N)}
cv_block(P)    = std_block(P) / |mean_block(P)|

robustness_score(P) = mean_block(P) / (1 + cv_block(P))
```

The `1 + cv` denominator penalises high variance across blocks without ever producing a negative score from the variance term alone. A param with mean_block = 0.6 and cv = 0.5 scores 0.6 / 1.5 = 0.40. A param with mean_block = 0.6 and cv = 0.0 (perfectly consistent) scores 0.60.

**Edge cases:**
- If `mean_block(P) = 0`: set `robustness_score(P) = 0` (no edge, regardless of variance).
- If `mean_block(P) < 0`: the param has negative expected performance across blocks. Eliminate it with a hard filter (`mean_block ≥ 0` gate, enforced before scoring).
- If N = 1: `cv_block = 0` by definition, robustness degenerates to `mean_block`. The minimum useful value is N = 3.

---

### 4.4 Signal Correlation Matrix

The signal series computed in Step 1 (trade frequency evaluation) is reused here — no additional model fitting is required.

```
corr_matrix[P, Q] = Pearson(signal_series_train(P), signal_series_train(Q))
```

Where `signal_series_train(P)` is the return series produced by param P on the full training window (the same series used to measure trade frequency).

**Pearson vs rank correlation:** Pearson is appropriate here because we care about linear co-movement in position sizing, not rank order. Two params that are both long/short at the same time and in similar magnitudes have high linear correlation.

**Interpretation of the matrix:**
- `|corr| > 0.9`: Near-identical signals — selecting both provides almost no diversification. The greedy algorithm will heavily penalise the second such param.
- `|corr| = 0.5`: Moderate correlation — the greedy penalty is partial.
- `|corr| < 0.2`: Near-independent signals — minimal penalty.

---

### 4.5 Composite Quality Score

```
quality(P) = w_stab * stability_score_norm(P) + w_rob * robustness_score_norm(P)

where:
    stability_score_norm(P)  = (smoothed_obj(P) - min_smoothed) / (max_smoothed - min_smoothed)
    robustness_score_norm(P) = (robustness_score(P) - min_rob) / (max_rob - min_rob)
    w_stab = weight_stability     (default: 0.5)
    w_rob  = weight_robustness    (default: 0.5)
```

**Normalization is across surviving params within this fold only.** This is a within-fold normalization, not a cross-fold normalization. Its purpose is to put the two components on the same scale (both in [0, 1]) before combining with weights.

**Equal weights:** Both components receive weight 0.5 by default. The weights are pre-committed and do not vary by feature. Tuning them per feature would re-introduce the selection bias the algorithm is designed to prevent.

**Raw objective not included:** The raw objective is already incorporated in the stability score (via smoothed objective, which is a function of raw objective). Adding it explicitly would double-count it. The robustness score also contains the raw performance dimension (mean_block reflects overall magnitude). The composite quality score therefore contains performance information from both angles without redundant weighting.

---

### 4.6 Greedy Diversity-Aware Selection

```python
def greedy_select(quality, corr_matrix, k, diversity_weight):
    """
    quality: dict[param_label -> float], all values in [0, 1]
    corr_matrix: symmetric DataFrame indexed and columned by param_label
    k: target ensemble size
    diversity_weight: float in [0, 1]
    returns: list of param_labels in selection order
    """
    selected = []
    remaining = sorted(quality, key=quality.__getitem__, reverse=True)

    # Always take the highest-quality param unconditionally
    selected.append(remaining.pop(0))

    while len(selected) < k and remaining:
        best_score = -inf
        best_p = None

        for p in remaining:
            max_corr = max(abs(corr_matrix.loc[p, s]) for s in selected)
            adjusted = quality[p] * (1 - diversity_weight * max_corr)
            if adjusted > best_score:
                best_score = adjusted
                best_p = p

        selected.append(best_p)
        remaining.remove(best_p)

    return selected
```

**Why multiplicative penalty:** `quality * (1 - w * max_corr)` is bounded in [0, quality_max]. A param with `max_corr = 1.0` (perfectly correlated with a selected param) scores `quality * (1 - w)`, not zero. It remains selectable if its quality advantage is large enough. A param with `max_corr = 0.0` (uncorrelated) is not penalised at all. The penalty scales linearly with correlation.

**Why max correlation, not mean:** The ensemble diversification benefit is destroyed most by the single most-correlated pair. A new param that is 90% correlated with one selected param but uncorrelated with all others offers almost no diversification for the overlapping pair. `max_corr` targets this correctly.

**First selection is unconditional:** The highest-quality param is always selected first, with no diversity penalty. Diversity is a property of a set, not an individual param — the first selection has no set to be diverse with.

**K_min check after greedy selection:** After the greedy procedure terminates, the selected set is checked against `K_min`. If fewer than `K_min` params were selected (because the param grid is too small or all candidates are highly correlated with each other), the ensemble returns empty — no position this period.

---

## 5. Pre-Committed Parameters

All parameters must be committed before any walkforward fold is run. Changing them after observing fold results invalidates the walkforward.

| Parameter | Default | Description |
|---|---|---|
| `k` | 5 | Target ensemble size |
| `trade_freq_min` | 0.05 | Minimum fraction of bars that must have a non-zero signal |
| `stability_ratio_min` | 0.80 | Minimum `smoothed / raw` ratio (from Gate A in `param_selection_rule.md`) |
| `n_robustness_blocks` | 5 | Number of equal-duration blocks to split the training window into for block CV |
| `diversity_weight` | 0.40 | Weight on the diversity penalty in the greedy selection step (0 = no diversity enforcement, 1 = correlation completely dominates quality) |
| `weight_stability` | 0.50 | Weight on stability component in composite quality score |
| `weight_robustness` | 0.50 | Weight on robustness component in composite quality score |
| `K_min` | 2 | Minimum ensemble size; fewer → empty ensemble (no position) |
| `K_max` | 7 | Maximum ensemble size; caps selection if more than k pass all filters |

**Guideline for `diversity_weight`:** A value of 0.4 means a param that is 100% correlated with an already-selected param has its quality score reduced to 60% of its standalone value. It remains selectable only if its quality is sufficiently higher than alternatives. Values above 0.6 may over-enforce diversity at the expense of quality; values below 0.2 provide minimal diversity benefit.

**Guideline for `n_robustness_blocks`:** Should be set so each block contains at least one meaningful market regime. For a 15-year training window, N=5 gives 3-year blocks — enough to include distinct regime examples. For shorter training windows, reduce N. Minimum useful value is 3.

---

## 6. Worked Example

**Setup:** RSI lookbacks [2, 3, 4, 5, 6, 7, 8, 9, 10], training window 2000–2015, k=5, n_blocks=5.

**After Steps 1–3 of `param_selection_rule.md`** (smoothed objective, Gate A, Gate B), suppose the surviving qualifying params are:

| Lookback | Raw Sharpe | Smoothed Sharpe | Stability Ratio |
|---|---|---|---|
| 4 | 0.72 | 0.68 | 0.944 |
| 5 | 0.81 | 0.76 | 0.938 |
| 6 | 0.78 | 0.74 | 0.949 |
| 7 | 0.65 | 0.63 | 0.969 |
| 8 | 0.58 | 0.57 | 0.983 |

**Step 1 — Trade frequency filter:** All pass (all > 5%).

**Step 3 — Robustness scores** (5 blocks, 3 years each):

| Lookback | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | mean_block | cv_block | robustness |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.90 | 0.65 | 0.75 | 0.60 | 0.50 | 0.68 | 0.24 | 0.68/1.24 = **0.548** |
| 5 | 0.80 | 0.75 | 0.70 | 0.78 | 0.72 | 0.75 | 0.05 | 0.75/1.05 = **0.714** |
| 6 | 0.70 | 0.72 | 0.74 | 0.76 | 0.73 | 0.73 | 0.03 | 0.73/1.03 = **0.709** |
| 7 | 0.50 | 0.80 | 0.55 | 0.65 | 0.70 | 0.64 | 0.18 | 0.64/1.18 = **0.542** |
| 8 | 0.60 | 0.58 | 0.55 | 0.60 | 0.57 | 0.58 | 0.04 | 0.58/1.04 = **0.558** |

**Observation:** Lookback 4 has the highest raw and smoothed Sharpe but the worst robustness (high CV — it earned most of its performance in Block 1 only). Lookbacks 5 and 6 are more consistent.

**Step 5 — Composite quality scores** (normalized within fold):

| Lookback | Stability norm | Robustness norm | Quality (0.5+0.5) |
|---|---|---|---|
| 4 | (0.68-0.57)/(0.76-0.57) = 0.579 | (0.548-0.542)/(0.714-0.542) = 0.035 | **0.307** |
| 5 | (0.76-0.57)/(0.76-0.57) = 1.000 | (0.714-0.542)/(0.714-0.542) = 1.000 | **1.000** |
| 6 | (0.74-0.57)/(0.76-0.57) = 0.895 | (0.709-0.542)/(0.714-0.542) = 0.971 | **0.933** |
| 7 | (0.63-0.57)/(0.76-0.57) = 0.316 | (0.542-0.542)/(0.714-0.542) = 0.000 | **0.158** |
| 8 | (0.57-0.57)/(0.76-0.57) = 0.000 | (0.558-0.542)/(0.714-0.542) = 0.093 | **0.047** |

**Observation:** Despite having the highest raw Sharpe (0.81), lookback 5 now scores highest on the composite because it is both the most stable *and* the most consistent across regimes. Lookback 4 drops from first to third due to poor robustness.

**Step 6 — Signal correlation matrix** (hypothetical):

| | LB4 | LB5 | LB6 | LB7 | LB8 |
|---|---|---|---|---|---|
| LB4 | 1.00 | 0.95 | 0.91 | 0.80 | 0.70 |
| LB5 | 0.95 | 1.00 | 0.96 | 0.85 | 0.75 |
| LB6 | 0.91 | 0.96 | 1.00 | 0.88 | 0.78 |
| LB7 | 0.80 | 0.85 | 0.88 | 1.00 | 0.92 |
| LB8 | 0.70 | 0.75 | 0.78 | 0.92 | 1.00 |

**Step 7 — Greedy selection** (k=5, diversity_weight=0.40):

| Round | Selected | Candidates | Adjusted scores |
|---|---|---|---|
| 1 | LB5 (quality=1.00, unconditional) | LB4, LB6, LB7, LB8 | — |
| 2 | — | LB4: 0.307*(1-0.4*0.95)=**0.190**; LB6: 0.933*(1-0.4*0.96)=**0.575**; LB7: 0.158*(1-0.4*0.85)=**0.104**; LB8: 0.047*(1-0.4*0.75)=**0.033** | → Select LB6 |
| 3 | — | LB4: 0.307*(1-0.4*max(0.95,0.91))=**0.190**; LB7: 0.158*(1-0.4*max(0.85,0.88))=**0.103**; LB8: 0.047*(1-0.4*max(0.75,0.78))=**0.032** | → Select LB4 |
| 4 | — | LB7: 0.158*(1-0.4*max(0.85,0.88,0.80))=**0.103**; LB8: 0.047*(1-0.4*max(0.75,0.78,0.70))=**0.032** | → Select LB7 |
| 5 | — | LB8: 0.047*(1-0.4*max(0.75,0.78,0.70,0.92))=**0.029** | → Select LB8 |

**Selected ensemble:** [LB5, LB6, LB4, LB7, LB8]

**Interpretation:** Despite LB4 having the highest raw Sharpe and LB5/LB6 being highly correlated (0.96), all five are selected because there are only five candidates. In a larger grid where more distinct regions exist, the diversity penalty would exclude some of the tightly-clustered short lookbacks in favour of more distinct params.

---

## 7. Design Rationale

### Why not optimise directly?

An exact formulation of the three-objective problem is a combinatorial optimisation over all C(N, k) subsets of N surviving params. For small grids this is tractable, but for larger grids (N=50+) it is not. The greedy algorithm approximates the optimal solution in O(N×k) time with well-understood behaviour.

Greedy selection is also more auditable: for any fold, the researcher can trace exactly why each param was selected in which order, and what its adjusted score was.

### Why signal correlation, not parameter-space distance?

Two params can be numerically far apart in parameter space but produce nearly identical signals (e.g., RSI-9 and RSI-10 on daily data). Conversely, two params can be adjacent in parameter space but produce decorrelated signals (e.g., a lookback straddling a regime break). Signal correlation measures the property we actually care about: whether the two params contribute independent information to the ensemble.

### Why multiplicative, not additive, diversity penalty?

An additive penalty (`quality - w * max_corr`) can produce negative adjusted scores, making the ordering of low-quality candidates meaningless. A multiplicative penalty is bounded in [0, quality] and preserves relative ordering among candidates with similar correlation but different quality. It also has a natural interpretation: the adjusted score is the quality discounted by the redundancy fraction.

### Why the first selection is unconditional?

The diversity penalty requires a reference set of already-selected params. With an empty reference set, `max_corr = 0` for all candidates, so the penalty has no effect anyway. The algorithm makes this explicit: the first selection always picks the highest-quality candidate.

### Why equal weights (0.5 / 0.5)?

Unequal weights would require justification specific to each feature class, which would introduce per-feature tuning. Equal weights are the maximum-entropy prior when neither component has demonstrated dominance. If empirical evidence accumulates that stability or robustness is systematically more predictive for a specific feature class, weights can be adjusted — but must be committed before examining results, not after.

---

## 8. Integration Points

### In the Walkforward Runner

This algorithm is applied once per training fold, during `_build_fold_scores()` in `feature_research/walkforward/runner.py`:

```
Current:
  param_grid → evaluate on test fold → smooth → rank by smoothed → take top-k

Enhanced:
  param_grid → evaluate on training window → hard filters →
  robustness blocks → quality score → correlation matrix → greedy select → K_min check
```

The selected ensemble replaces the naive `top_k_features` in the fold summary.

New columns added to `fold_scores_df`:
- `trade_frequency`: fraction of bars with non-zero signal
- `stability_ratio`: smoothed / raw objective
- `robustness_score`: mean_block / (1 + cv_block)
- `quality_score`: composite quality score [0, 1]
- `selected_by_diversity`: bool — was this param selected by the greedy procedure?

New field in `WalkforwardRunReport.selection_summary_df`:
- `selected_ensemble_diversity`: list of selected labels from the greedy procedure (replaces `top_k_features`)

### In Production

The same algorithm runs at each production refit, using the full expanding training window as the "training data". The block structure for robustness is applied to the full training window (e.g., a 25-year window → 5 blocks of 5 years each).

### Config Additions to `WalkforwardResearchConfig`

```python
# New fields to add to feature_research/walkforward/config.py
use_enhanced_selection: bool = False        # opt-in; False = legacy top-k
trade_freq_min: float = 0.05
n_robustness_blocks: int = 5
diversity_weight: float = 0.40
weight_stability: float = 0.50
weight_robustness: float = 0.50
```

The `use_enhanced_selection` flag allows the legacy behaviour to remain the default until the new algorithm is validated. New research runs should opt in explicitly.

---

## 9. Related Docs

| Document | Relationship |
|---|---|
| [`param_selection_rule.md`](param_selection_rule.md) | Parent spec — Steps 1–3 (fit, smooth, gates) are unchanged; this doc replaces Steps 4–6 |
| [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md) | Neighbor smoothing theory — provides the stability score input to this algorithm |
| [`../Walkforward/walkforward.md`](../Walkforward/walkforward.md) | Describes the walkforward fold structure in which this selection runs |
| [`../pipeline_overview.md`](../pipeline_overview.md) | Master pipeline — Section 5 (Pre-Committed Param Selection Rule) |

---

*End of specification.*
