# EDA — Phase 1 Exploratory Data Analysis

> [!note] Status: Library reference — Phase 1 of [[pipeline]]
> No gate. Informational only. Researcher reviews outputs before committing to permutation testing.

## Purpose

Understand feature behavior, parameter landscape, and binning quality before any statistical testing. Rich diagnostics → researcher decides whether to proceed to Phase 2.

---

## Common EDA Outputs (All Feature Types)

| Output | What to Look For |
|--------|-----------------|
| Descriptive stats (min/max/mean/std/skew/kurt) | Extreme skew or kurtosis → may need transformation |
| Missing data count | High NaN% in early history → trim lookback window |
| Feature-target correlation (Pearson, Spearman, Kendall) | Any consistent direction? Lagged correlations? |
| Rolling correlation plot (default 252-day window) | Drift or sign flips → temporal instability risk |
| Rolling objective metric plot (Sharpe, Sortino, etc.) | Periods of degradation → regime sensitivity |

- Separate `EDAReport` generated **per param combo** — e.g., RSI lookbacks [10, 14, 20] → 3 reports
- Implementation: `eda/eda_runner.py`

---

## Continuous Feature EDA (Additional)

For features like RSI, momentum, vol ratios that require binning (see [[continuous_binning]]).

- **Decile plot** (default 15 quantile bins): mean return, Sharpe, t-stat per bin
  - Look for monotonic trend or clear tail/hump structure
  - Monotonicity test: Kendall's tau — high |τ| → cleaner signal
- **Distribution plots**: histogram with quantile overlays, Q-Q plot, KDE
  - Heavy tails or bimodality can signal regime mixing

### Binning Diagnostics

Fit `QuantileBinningModel` on IS data; inspect the `BinningDiagnosticsReport`:

- Contiguous regions exceeding Sharpe + t-stat thresholds
- **Shape**: tail (touches extreme bin) vs hump (surrounded by neutral bins)
- **Coverage**: % of feature distribution in tradeable zones
- Red flags: only single isolated bins passing → likely noise; no regions → feature lacks structure

---

## Rule-Based Feature EDA (Additional)

For discrete signals (-1, 0, +1) — see [[rule_based]].

- **Per-level stats**: mean return, vol, Sharpe, adjusted Sharpe for each level
- Bootstrap 95% CIs on mean returns — wide intervals → small sample per level
- **Level plot**: bar chart of performance by level
- **Transition matrix**: how often does the signal flip? Rapid flipping → suspicious

> [!note] No binning phase for rule-based features. They are already discrete; binning diagnostics do not apply.

---

## Parameter Grid: Stability Landscape

Applies to both feature types. Input: grid-search results across param combos.

- **Neighbor-smoothed metric**: `smoothed(P) = mean(objective(P) + objective(neighbors))`
- **Stability ratio**: `smoothed / raw`
  - `> 0.8` → stable region (genuine signal)
  - `< 0.5` → isolated peak (likely overfit)
- **2D heatmap** (interactive): layers for `smoothed`, `raw`, `stability_ratio`, `delta`
  - Overlay options: stable-region markers, stability contours
- **1D line plot**: raw vs smoothed with shaded stable regions

Full theory and selection rule: [[param_stability]]
Implementation: `eda/parameter_analysis.py` (`ParameterAnalyzer`), `metrics/plotting/parameter_plots.py`

---

## Vault Integration

After researcher is satisfied with all EDA and permutation screening results:

1. Fit `BaseModel` on full IS data for each selected param combo
2. Save to [[vault]]: `base_model.save_to_vault(ensemble_dir)`
3. Vault structure: `vault/{timeframe}/{ensemble_name}_{direction}/features/{feature_column}.json`

> [!warning] Save only after the param selection rule ([[param_stability]]) has been applied. Do not save params that fail stability criteria.

---

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Quantile bins | 15 | User-customizable |
| Rolling metric window | 252 (daily), 52 (weekly) | Adjust per timeframe |
| t-stat threshold | 2.0 | Pre-specify before seeing data |
| Stability ratio threshold | 0.8 | For neighbor-smoothed regions |
| Significance α | 0.10 (exploratory), 0.05 (production) | Used in Phase 2+ |

> [!warning] All thresholds must be pre-specified before analysis. No tuning per feature — this is data snooping.

---

## Related Docs

- [[pipeline]] — Full 5-phase pipeline context
- [[continuous_binning]] — Quantile binning pipeline detail
- [[rule_based]] — Rule-based feature handling
- [[base_feature]] — Base feature interface
- [[param_stability]] — Neighbor smoothing theory and param selection rule
- [[permutation_testing]] — Phase 2 IS permutation screening
- [[vault]] — Feature persistence
