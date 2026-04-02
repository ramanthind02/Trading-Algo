# Feature Model

> [!summary] What it is
> A feature is a technical indicator (bias node) with a parameter set that generates a continuous or discrete signal. The feature model converts this signal into a position multiplier (0–2.0 range) via statistical analysis. Two types exist:
> - **Continuous** (e.g., RSI, momentum): require Phase 2 binning analysis (grid search → single best bin per direction)
> - **Rule-based** (e.g., breakouts, discrete signals): already discrete (-1/0/+1) → Phase 2 skipped, use three fixed levels

If you intentionally **avoid Phase 2–style learned quantile binning in production**—by freezing domain-meaningful absolute cutpoints and bin policy at research time—use the target architecture in [[Feature_selection/domain_discrete_signals]]. Legacy continuous features may still use the Phase 2 path below until migrated.

See [[base_feature]] for shared interface and Sharpe formula.

---

## Shared Components (Both Types)

### Inputs

| Input | Description |
|-------|-------------|
| **Feature** | Continuous vector (e.g. RSI 0–100) OR discrete vector (-1/0/+1) |
| **Target** | Vol-scaled returns: `return / EWSD` or `return / ATR`. Unitless, cross-asset comparable. |

See [[base_feature]] for target construction.

### Fit Behavior — Per-Level Statistics

Whether quantile bins (continuous) or fixed 3 levels (rule-based), compute for each bin/level:

- `mean_return` — average vol-scaled return
- `volatility` — std of returns
- `sharpe` = `mean_return / volatility`
- **Adjusted Sharpe** = `Sharpe × sqrt(N) / (sqrt(N) + k)` — shrinkage constant k (default 20)

### Position Scaling Formula (Same for Both Types)

Maps per-level/per-bin adjusted Sharpe to position multiplier:

| Level/Bin | Formula | Output range |
|-----------|---------|--------------|
| **Long (+1)** | `clip(1 + adjusted_Sharpe_long, 0.5, 2.0)` | [0.5, 2.0] |
| **Short (-1)** | `-clip(1 + adjusted_Sharpe_short, 0.5, 2.0)` | [-2.0, -0.5] |
| **Flat (0)** | `0.0` | 0 |

- **Long-short mode** (both ±1 active): output range is **[-2, 2]**
- **Short-only mode**: output range is [-2, -0.5]
- **Baseline exposure** = 1.0; **max tilt** = 2.0

### Predict

Returns continuous position multiplier in [-2, 2]:
- Observations in active region/level → scaled multiplier
- Observations outside → 0.0 (flat)

If `scaled=True`: apply portfolio-level volatility scaling (EWSD/ATR) and clipping.

---

## Phase 2: Continuous Binning (Continuous Features Only)

**Rule-based features skip this phase entirely** — use fixed 3 levels instead.

### Grid Search Over Bin Counts

For each bin count in `bin_counts` (default `[10, 8, 5, 3]`):
1. Create **quantile bins** (equal sample count per bin)
2. Compute per-bin statistics (mean_return, volatility, sharpe, t_stat)
3. Select best bin per direction (highest positive t-stat for long, most negative for short)
4. Score the configuration

Keep the configuration with the overall highest score.

### User-Facing Parameters

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

### Shape Detection (Automatic)

| Bin count | Pattern captured |
|-----------|-----------------|
| Fine (10, 8) | **Tails** — narrow extreme regions (e.g. RSI < 20, RSI > 80) |
| Coarse (3, 5) | **Humps** — broad mid-range regions (e.g. RSI 40–60) |

The bin count with the highest t-stat wins. No manual configuration needed.

### Coverage-Adjusted Sharpe (Optional)

When `use_coverage_bonus=True`:

```
coverage_pct = (bin_count / total_observations) * 100
bonus = 0.0                              if coverage_pct <= 10
bonus = ((coverage_pct - 10) / 10) * coverage_bonus_per_10pct  otherwise
bonus = min(bonus, max_coverage_bonus)
adjusted_sharpe = sharpe + bonus
```

Rewards features with broader distributional coverage; partially offsets portfolio-level scaling by `sqrt(coverage)`.

### Output

- Observations **in** selected bin → coverage-adjusted Sharpe (long: positive, short: negative)
- Observations **outside** selected bin → `0.0` (flat)

---

## Phase 2: Rule-Based (Rule-Based Features Only)

**Continuous features do the above binning analysis; rule-based features use this simpler path.**

### Fit Behavior

Treat the three levels (-1, 0, +1) as three fixed "bins." For each level, compute statistics on vol-scaled target and store per-level stats. No binning search, no quantile creation — only the shared scaling step.

### Predict

- Map each observation's rule value (-1, 0, +1) to the corresponding multiplier from fit
- If `scaled=True`: apply volatility scaling (EWSD/ATR) and clipping per existing pipeline

---

## Fit / Predict API (Both Types)

**`fit(feature_data, target_data, normalization_data=None)`**
- **Continuous**: grid search over bin_counts → select best bin per direction → store winning bin count, edges, and adjusted Sharpe
- **Rule-based**: compute per-level stats → store per-level adjusted Sharpes

**`predict(feature_data, strategy='long'|'short'|'long_short', ...)`**
- Assign observations to bins/levels using fitted edges
- Return position multiplier for each observation

---

## Permutation Testing

> [!warning] Threshold must be pre-specified before seeing data
> Set `t_threshold` and `metric_threshold` before fitting.

Both feature types use the same permutation test procedure:

1. **Fit** on both original and permuted features (no screening during permutation)
2. **Apply production threshold** (`t_threshold=2.0`) **after** permutation test
3. **Feature passes** if: original beats 95th percentile of permuted scores **AND** |t| >= 2.0

**Permutation modes:**
- **Continuous**: vector shuffle or candle shuffle (full pipeline reruns)
- **Rule-based**: candle shuffle only (vector shuffle not meaningful for rule-based)

See [[permutation_testing]] for full procedure.

---

> [!info] See also
> - [[base_feature]] — shared target definition and Sharpe formula details
> - [[permutation_testing]] — permutation test procedure and thresholds
> - [[pipeline]] — where Phase 2 fits in the full validation flow
