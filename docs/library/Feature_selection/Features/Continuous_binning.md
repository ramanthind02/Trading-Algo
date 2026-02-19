# Binning — Continuous Features (Simplified Grid Search Approach)

Binning turns continuous features into **trading signals** by identifying feature ranges where expected return reliably differs from zero. This spec describes a simplified grid search approach that automatically detects tail vs hump patterns through granularity selection.

---

## Purpose

- **Goal:** Exploit feature regions (tails or humps) where expected return is statistically significant.
- **Design principle:** Simple, transparent, and defensible — minimize complexity, maximize robustness.
- **Output:** Coverage-adjusted Sharpe ratio as a continuous position multiplier.

---

## Key Design Changes (Simplified Approach)

### Old Approach
- Fixed bin count (e.g., 15)
- Complex region detection and merging
- Multi-bin selection per feature
- Adjusted Sharpe with James-Stein shrinkage

### New Approach
- **Grid search** over bin counts (e.g., [10, 8, 5, 3])
- **Single best bin per direction** based on t-statistic
- **Coverage-adjusted Sharpe** as output
- **Automatic tail vs hump detection** (fine bins favor tails, coarse bins favor humps)

---

## User-Facing API

The user configures:

| Parameter | Description | Default |
|-----------|-------------|---------|
| **`bin_counts`** | List of bin counts to test (e.g., [10, 8, 5, 3]). Grid search selects best. | `[10, 8, 5, 3]` |
| **`strategy`** | Trading mode: `long`, `short`, or `long_short` | `"long"` |
| **`selection_metric`** | Metric for bin ranking: `sharpe`, `mean`, `t_stat`, `sortino` | `"sharpe"` |
| **`metric_threshold`** | Minimum metric value (pre-specified, not optimized) | `0.0` |
| **`t_threshold`** | Minimum \|t-stat\| for significance | `2.0` |
| **`use_coverage_bonus`** | Enable coverage-adjusted Sharpe | `False` |
| **`coverage_bonus_per_10pct`** | Bonus per 10% coverage above 10% floor | `0.02` |
| **`max_coverage_bonus`** | Maximum coverage bonus cap | `0.2` |

The model receives:

- **Feature vector** (continuous, e.g., RSI values)
- **Target vector** (vol-scaled returns, e.g., `log_return / EWSD`)

---

## Pipeline (Fully Automated)

### Step 1 — Grid Search Over Bin Counts

For each bin count in `bin_counts`:
1. Create quantile bins (equal sample count per bin)
2. Compute per-bin statistics
3. Select best bin per direction
4. Compute coverage-adjusted Sharpe for selected bins
5. Score the configuration (highest |t-stat|)

After testing all bin counts, keep the configuration with the highest score.

**Why grid search?**
- Larger bin counts (e.g., 10) capture **tail effects** (narrow, extreme regions)
- Smaller bin counts (e.g., 3) capture **humps** (broad, mid-range regions)
- Let the data decide which granularity is most significant

### Step 2 — Per-Bin Statistics

For each bin, compute:
- `mean_return` — average target return in bin
- `volatility` — standard deviation of returns
- `sharpe` — `mean_return / volatility` (not annualized)
- `t_stat` — `mean_return / (volatility / sqrt(N))`
- `selection_metric` — user-chosen metric for ranking

### Step 3 — Select Best Bin Per Direction

- **Long:** Bin with highest positive t-statistic
- **Short:** Bin with most negative t-statistic
- **Long_short:** Both (top long + top short)

Only **one bin per direction** is selected — simplicity over multi-region complexity.

### Step 4 — Coverage-Adjusted Sharpe (Optional)

If `use_coverage_bonus=True`:

```
coverage_pct = (bin_count / total_observations) * 100
bonus = 0.0 if coverage_pct <= 10 else ((coverage_pct - 10) / 10) * coverage_bonus_per_10pct
bonus = min(bonus, max_coverage_bonus)
adjusted_sharpe = sharpe + bonus
```

**Why coverage bonus?**
- Rewards features that work across more of the distribution
- Balances sharp-but-narrow edges vs stable-but-wide edges
- In portfolio sizing, low-coverage edges get scaled up (divide by `sqrt(coverage)`), so bonus partially offsets

### Step 5 — Output Position Multiplier

For observations in the selected bin:
- **Long:** `coverage_adjusted_sharpe` (positive)
- **Short:** `-coverage_adjusted_sharpe` (negative)
- **Long_short:** Both bins active (one positive, one negative)

For observations **outside** the selected bin: `0.0` (flat)

---

## Interface: fit / predict

**`fit(feature_data, target_data, normalization_data=None)`**
- Grid search over bin counts
- Select best bin per direction based on t-stat
- Store winning bin count, selected bins, and coverage-adjusted Sharpe

**`predict(feature_data, strategy='long'|'short'|'long_short', ...)`**
- Assign observations to bins using fitted edges
- Output coverage-adjusted Sharpe for selected bins, 0.0 otherwise

---

## Shape Detection (Automatic)

**No manual configuration needed.** The grid search automatically discovers:

- **Tails:** Fine bins (10, 8) capture extreme regions (RSI < 20, RSI > 80)
- **Humps:** Coarse bins (3, 5) capture mid-range peaks (RSI 40–60)

The bin count with the highest t-stat wins.

---

## Permutation Testing

**Threshold setting:** Pre-specify `t_threshold` and `metric_threshold` **before** seeing data.

During permutation testing:
- Fit model on both original and permuted features
- No screening threshold (fit all)
- Apply production threshold (`t_threshold=2.0`) **after** permutation test
- Feature passes if: (1) original beats 95th percentile of permutations, AND (2) |t| >= 2.0

---

## Safe Defaults

| Parameter | Default | Notes |
|-----------|---------|--------|
| **bin_counts** | `[10, 8, 5, 3]` | Test 4 granularities |
| **strategy** | `"long"` | Or `"short"`, `"long_short"` |
| **selection_metric** | `"sharpe"` | Accounts for volatility |
| **t_threshold** | `2.0` | Statistical significance filter |
| **metric_threshold** | `0.0` | No Sharpe floor by default |
| **use_coverage_bonus** | `False` | Disabled by default |
| **coverage_bonus_per_10pct** | `0.02` | Small incremental bonus |
| **max_coverage_bonus** | `0.2` | Cap at +0.2 Sharpe |

---

## Example Usage

```python
from feature_selection.base_models.continuous_binning import ContinuousBinningModel

model = ContinuousBinningModel(
    bin_counts=[10, 8, 5, 3],
    strategy="long_short",
    use_coverage_bonus=True,
    t_threshold=2.0,
)

model.fit(feature_series, target_series)

# Check winning configuration
print(f"Selected bin count: {model.n_bins}")
print(f"Long bin: {model.selected_bins_['long']}, Sharpe: {model.selected_bins_['long_sharpe']:.3f}")
print(f"Short bin: {model.selected_bins_['short']}, Sharpe: {model.selected_bins_['short_sharpe']:.3f}")

# Predict
predictions = model.predict(feature_series, strategy="long_short")
```

---

## Summary

- **User sets:** `bin_counts`, `strategy`, thresholds, coverage bonus
- **Model does:** Grid search → select best bin per direction → output coverage-adjusted Sharpe
- **Philosophy:** **Simplicity over complexity.** Single-bin selection, transparent scoring, automatic tail/hump detection.

Pipeline in one line: **Grid search → select → validate** — not "detect → merge → optimize."
