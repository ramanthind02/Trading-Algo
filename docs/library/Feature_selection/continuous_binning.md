# Continuous Binning

> [!summary] What it is
> Grid search over bin counts to find the **single best bin per direction** with the highest t-statistic. Automatically detects tail vs hump effects — no manual shape configuration.

Inputs: continuous feature vector + vol-scaled target (see [[base_feature]]).

---

## Old vs New Approach

| Old | New (current) |
|-----|---------------|
| Fixed bin count (e.g. 15) | **Grid search** over bin counts (e.g. `[10, 8, 5, 3]`) |
| Complex region detection + merging | **Single best bin** per direction |
| Multi-bin region selection | One winner per long / short |
| James-Stein shrinkage | Coverage-adjusted Sharpe (optional) |

---

## User-Facing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_counts` | `[10, 8, 5, 3]` | Granularities to search |
| `strategy` | `"long"` | `long`, `short`, or `long_short` |
| `selection_metric` | `"sharpe"` | Bin ranking metric (`sharpe`, `mean`, `t_stat`, `sortino`) |
| `metric_threshold` | `0.0` | Minimum metric value (pre-specified) |
| `t_threshold` | `2.0` | Minimum \|t-stat\| for significance |
| `use_coverage_bonus` | `False` | Enable coverage-adjusted Sharpe |
| `coverage_bonus_per_10pct` | `0.02` | Bonus per 10% coverage above 10% floor |
| `max_coverage_bonus` | `0.2` | Cap on total coverage bonus |

---

## Pipeline

### Step 1 — Grid Search Over Bin Counts

For each bin count in `bin_counts`:
1. Create **quantile bins** (equal sample count per bin)
2. Compute per-bin statistics
3. Select best bin per direction (highest |t-stat|)
4. Score the configuration

Keep the configuration (bin count) with the overall highest score.

### Step 2 — Per-Bin Statistics

For each bin:
- `mean_return` — average vol-scaled return in bin
- `volatility` — std of returns in bin
- `sharpe` = `mean_return / volatility`
- `t_stat` = `mean_return / (volatility / sqrt(N))`

### Step 3 — Select Best Bin Per Direction

- **Long:** bin with highest positive t-stat
- **Short:** bin with most negative t-stat
- **Long_short:** one long bin + one short bin

> [!important] Single bin only
> Only **one bin per direction** — no multi-region merging.

### Step 4 — Shape Detection (Automatic)

| Bin count | Pattern captured |
|-----------|-----------------|
| Fine (10, 8) | **Tails** — narrow extreme regions (e.g. RSI < 20, RSI > 80) |
| Coarse (3, 5) | **Humps** — broad mid-range regions (e.g. RSI 40–60) |

The bin count with the highest t-stat wins. No manual configuration needed.

### Step 5 — Coverage-Adjusted Sharpe (Optional)

When `use_coverage_bonus=True`:

```
coverage_pct = (bin_count / total_observations) * 100
bonus = 0.0                              if coverage_pct <= 10
bonus = ((coverage_pct - 10) / 10) * coverage_bonus_per_10pct  otherwise
bonus = min(bonus, max_coverage_bonus)
adjusted_sharpe = sharpe + bonus
```

Rewards features with broader distributional coverage; partially offsets portfolio-level scaling by `sqrt(coverage)`.

### Step 6 — Output Position Multiplier

- Observations **in** selected bin → coverage-adjusted Sharpe (long: positive, short: negative)
- Observations **outside** selected bin → `0.0` (flat)

---

## Fit / Predict

**`fit(feature_data, target_data, normalization_data=None)`**
- Grid search → select best bin per direction → store winning bin count, edges, and adjusted Sharpe

**`predict(feature_data, strategy='long'|'short'|'long_short', ...)`**
- Assign observations to bins using fitted edges
- Return coverage-adjusted Sharpe for selected bins, `0.0` otherwise

---

## Permutation Testing

> [!warning] Threshold must be pre-specified before seeing data
> Set `t_threshold` and `metric_threshold` before fitting.

- Fit on both original and permuted features (no screening during permutation)
- Apply production threshold (`t_threshold=2.0`) **after** permutation test
- Feature passes if: original beats 95th percentile of permuted scores **AND** |t| >= 2.0

See [[permutation_testing]] for full procedure.

---

> [!info] See also
> - [[base_feature]] — shared Sharpe statistics and position multiplier formula
> - [[permutation_testing]] — permutation test procedure and thresholds
