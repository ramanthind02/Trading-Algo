# T002 — Continuous Feature EDA

## Goal
Build continuous-specific EDA infrastructure that performs decile analysis, distribution diagnostics, and monotonicity testing to validate feature-target relationships for raw continuous indicators.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Lines 91-108 (Continuous Feature EDA specification)
- `feature_selection/base_models/quantile_binning.py` — QuantileBinningModel for bin creation
- `docs/library/Feature_selection/features/Continuous_binning.md` — Continuous binning specification
- `feature_selection/eda/common_eda.py` — Common EDA infrastructure (T001 dependency)

## Scope
In scope:
- Decile/quantile binning (default: 15 bins, user-customizable)
- Per-bin statistics computation:
  - Mean target return
  - Volatility (std of returns in bin)
  - Sharpe ratio: `mean / std`
  - t-statistic: `mean / (std / sqrt(N))`
  - Sample count per bin
- Decile plot generation (visualize mean return, Sharpe, t-stat across bins)
- Monotonicity testing using Kendall's tau
- Distribution diagnostics:
  - Histogram with quantile overlays
  - Q-Q plot (compare feature to normal distribution)
  - Kernel density estimate (KDE)
- Data structures: `ContinuousEDAStats`, `ContinuousEDAPlots` dataclasses

Out of scope:
- Common EDA (descriptive stats, temporal stability, correlations) → T001
- Rule-based feature EDA → T003
- Binning model fitting for production → Phase 2 (separate from EDA)
- Report generation and aggregation → T004

## Interfaces (must match)
- Add: `feature_selection/eda/continuous_eda.py` — Core module for continuous feature EDA
  - `compute_decile_analysis(feature: pd.Series, target: pd.Series, n_bins: int = 15) -> DecileAnalysis`
  - `compute_monotonicity_test(bin_means: np.ndarray) -> MonotonicityTest`
  - `compute_distribution_diagnostics(feature: pd.Series) -> DistributionDiagnostics`
  - `create_continuous_eda_plots(feature: pd.Series, target: pd.Series, decile_stats: DecileAnalysis, dist_diag: DistributionDiagnostics) -> ContinuousEDAPlots`

- Add: `feature_selection/eda/eda_dataclasses.py` — Extend with continuous-specific structures
  - `@dataclass(frozen=True) class DecileBinStats` — bin_edges, mean_return, volatility, sharpe, t_stat, sample_count (all as arrays)
  - `@dataclass(frozen=True) class DecileAnalysis` — bin_stats: DecileBinStats, overall_trend: str (e.g., "monotonic_increasing", "U-shaped", "flat")
  - `@dataclass(frozen=True) class MonotonicityTest` — kendall_tau: float, p_value: float, is_monotonic: bool (|tau| > 0.5 and p < 0.05)
  - `@dataclass(frozen=True) class DistributionDiagnostics` — skewness, kurtosis, normality_test_stat, normality_p_value, is_normal: bool
  - `@dataclass(frozen=True) class ContinuousEDAPlots` — decile_plot_fig, histogram_fig, qq_plot_fig, kde_fig (matplotlib.figure.Figure)

## Data Contracts
- **Input schema:**
  - `feature: pd.Series` — Index: DatetimeIndex, Values: float (continuous values, e.g., RSI)
  - `target: pd.Series` — Index: DatetimeIndex (aligned with feature), Values: float (returns)
  - `n_bins: int` — Number of quantile bins (default: 15)

- **Output schema:**
  - `DecileBinStats.bin_edges`: array of length `n_bins + 1` (bin boundaries)
  - `DecileBinStats.mean_return`: array of length `n_bins` (mean return per bin)
  - `DecileBinStats.sharpe`: array of length `n_bins` (Sharpe ratio per bin)
  - `DecileBinStats.t_stat`: array of length `n_bins` (t-statistic per bin)
  - `DecileBinStats.sample_count`: array of length `n_bins` (count per bin)
  - All arrays aligned by bin index

- **Alignment expectations:**
  - Feature and target must have identical index before binning
  - Bins created using `pd.qcut()` with equal sample counts (quantile binning)
  - NaN handling: drop NaNs before binning, report dropped count

## Dependencies
- `pandas` — Series operations, `pd.qcut()` for quantile binning
- `scipy.stats` — Kendall's tau, normality tests (Shapiro-Wilk or Anderson-Darling), Q-Q plot
- `matplotlib` — Decile plots, histograms, Q-Q plots
- `seaborn` — KDE plots (optional, or use scipy.stats.gaussian_kde)
- `numpy` — Array operations for bin statistics

## Invariants / Constraints
- Deterministic: same inputs → same outputs (quantile bin edges may vary slightly with ties, use stable sorting)
- No lookahead: binning uses full sample but doesn't leak future information (EDA phase)
- Immutability: all dataclasses frozen
- Bin validation: `n_bins >= 2`, `n_bins <= len(feature.dropna()) / 10` (at least 10 samples per bin)
- Sharpe computation: if `volatility == 0` for a bin, set Sharpe to NaN
- t-stat computation: if `sample_count < 2`, set t-stat to NaN
- Monotonicity criterion: |kendall_tau| > 0.5 and p < 0.05 → is_monotonic = True

## Acceptance tests
1. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_decile_analysis_rsi` — Verify decile analysis for RSI (lookback=5, TimeFrame.D) produces 15 bins with expected statistics
2. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_monotonicity_test_synthetic` — Create synthetic monotonic feature, verify monotonicity test detects it
3. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_distribution_diagnostics` — Validate distribution diagnostics (skew, kurtosis, normality test) for RSI feature
4. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_decile_plot_generation` — Smoke test that decile plot creation succeeds (mean return, Sharpe, t-stat subplots)
5. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_histogram_with_quantile_overlays` — Verify histogram overlays show bin edges correctly
6. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_edge_case_few_bins` — Test with n_bins=3 (minimum viable) and n_bins=30 (many bins)
7. `pytest tests/integration/feature_validator/eda/test_continuous_eda.py::test_sharpe_zero_volatility_bin` — Bin with constant returns → volatility=0 → Sharpe=NaN (no crash)

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/eda/test_continuous_eda.py`
- [ ] Implementation in `feature_selection/eda/continuous_eda.py`
- [ ] Dataclasses extended in `feature_selection/eda/eda_dataclasses.py`
- [ ] Docs updated in `docs/api/feature_selection.md`
- [ ] `pytest tests/integration/feature_validator/eda/test_continuous_eda.py -q` passes
- [ ] Type hints pass strict mypy/pyright checks
- [ ] All dataclasses are frozen and immutable

## Notes
- Test feature: RSI with `lookback=[2,3,4,5,6,7,8,9,10]`, `TimeFrame.D`
- Default n_bins: 15 (per specification)
- Decile plot: use 3 subplots stacked vertically (mean return, Sharpe, t-stat) with x-axis = bin number
- Quantile overlays on histogram: show bin edges as vertical lines with different colors
- Q-Q plot: compare to theoretical normal distribution (scipy.stats.probplot)
- KDE: use scipy.stats.gaussian_kde or seaborn.kdeplot
- Edge case: if feature has < 30 samples, warn that decile analysis may be unreliable
- Monotonicity interpretation: positive tau → higher feature values → higher returns (monotonic increasing), negative tau → inverse relationship
