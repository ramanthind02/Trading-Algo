# EDA - Phase 1 Exploratory Data Analysis

> [!note] Status: Library reference - Phase 1 of [[pipeline]]
> No gate. Informational only. Researchers review exported outputs before committing to permutation testing.

## Purpose

Understand feature behavior, parameter landscape, and binning quality before any statistical testing. Structured diagnostics feed researcher review and downstream Power BI dashboards before moving to Phase 2.

---

## Common EDA Outputs (All Feature Types)

| Output | What to Look For |
|--------|------------------|
| Descriptive stats (min/max/mean/std/skew/kurt) | Extreme skew or kurtosis may need transformation |
| Missing data count | High NaN% in early history may require trimming the lookback window |
| Feature-target correlation (Pearson, Spearman, Kendall) | Any consistent direction? Lagged correlations? |
| Rolling correlation summary (default 252-day window) | Drift or sign flips imply temporal instability risk |
| Rolling objective metric summary (Sharpe, Sortino, etc.) | Periods of degradation imply regime sensitivity |

- Separate `EDAReport` generated per parameter combination, for example RSI lookbacks `[10, 14, 20]` produce three reports.
- Implementation: `eda/eda_runner.py`

---

## Continuous Feature EDA (Additional)

For features like RSI, momentum, and volatility ratios that require binning (see [[continuous_binning]]).

- **Decile summary table** (default 15 quantile bins): mean return, Sharpe, t-stat per bin
  - Look for monotonic trends or clear tail/hump structure
  - Monotonicity test: Kendall's tau; high absolute tau implies a cleaner signal
- **Distribution summaries**: histogram-ready bin counts, quantiles, and moments
  - Heavy tails or bimodality can signal regime mixing

### Binning Diagnostics

Fit `QuantileBinningModel` on IS data and inspect the `BinningDiagnosticsReport`:

- Contiguous regions exceeding Sharpe and t-stat thresholds
- **Shape**: tail (touches extreme bin) vs hump (surrounded by neutral bins)
- **Coverage**: percent of feature distribution in tradeable zones
- Red flags: only isolated bins passing implies likely noise; no regions implies the feature lacks structure

---

## Rule-Based Feature EDA (Additional)

For discrete signals `(-1, 0, +1)` - see [[rule_based]].

- **Per-level stats**: mean return, vol, Sharpe, adjusted Sharpe for each level
- Bootstrap 95% confidence intervals on mean returns; wide intervals imply small sample sizes per level
- **Transition matrix**: how often does the signal flip? Rapid flipping is suspicious

> [!note] No binning phase for rule-based features. They are already discrete, so binning diagnostics do not apply.

---

## Parameter Grid: Stability Landscape

Applies to both feature types. Input: grid-search results across parameter combinations.

- **Neighbor-smoothed metric**: `smoothed(P) = mean(objective(P) + objective(neighbors))`
- **Stability ratio**: `smoothed / raw`
  - `> 0.8` implies a stable region
  - `< 0.5` implies an isolated peak and likely overfit
- **2D parameter table**: `smoothed`, `raw`, `stability_ratio`, `delta`
- **1D parameter table**: raw vs smoothed values by parameter point

Full theory and selection rule: [[param_stability]]
Implementation: `eda/parameter_analysis.py` (`ParameterAnalyzer`) and data-only exports consumed by Power BI.

---

## Vault Integration

After the researcher is satisfied with EDA and permutation screening results:

1. Fit `BaseModel` on full IS data for each selected parameter combination.
2. Save to [[vault]]: `base_model.save_to_vault(ensemble_dir)`.
3. Vault structure: `vault/{timeframe}/{ensemble_name}_{direction}/features/{feature_column}.json`

> [!warning] Save only after the parameter selection rule ([[param_stability]]) has been applied. Do not save parameters that fail stability criteria.

---

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Quantile bins | 15 | User-customizable |
| Rolling metric window | 252 (daily), 52 (weekly) | Adjust per timeframe |
| t-stat threshold | 2.0 | Pre-specify before seeing data |
| Stability ratio threshold | 0.8 | For neighbor-smoothed regions |
| Significance alpha | 0.10 (exploratory), 0.05 (production) | Used in Phase 2+ |

> [!warning] All thresholds must be pre-specified before analysis. No per-feature retuning - that would be data snooping.

---

## Related Docs

- [[pipeline]] - Full 5-phase pipeline context
- [[continuous_binning]] - Quantile binning pipeline detail
- [[rule_based]] - Rule-based feature handling
- [[base_feature]] - Base feature interface
- [[param_stability]] - Neighbor smoothing theory and parameter selection rule
- [[permutation_testing]] - Phase 2 IS permutation screening
- [[vault]] - Feature persistence
