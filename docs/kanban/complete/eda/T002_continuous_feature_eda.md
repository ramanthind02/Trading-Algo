# T002 — Continuous Feature EDA

## Goal
Build continuous-specific EDA infrastructure that performs decile analysis, distribution diagnostics, and monotonicity testing to validate feature-target relationships for raw continuous indicators.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Lines 91-108 (Continuous Feature EDA specification)
- `feature_selection/base_models/quantile_binning.py` — QuantileBinningModel for bin creation
- `docs/library/Feature_selection/features/Continuous_binning.md` — Continuous binning specification
- `feature_selection/eda/common_eda.py` — Common EDA infrastructure (T001 dependency)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

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

**Unit tests** (location: `tests/validators/eda/test_continuous_eda.py`):
- `test_decile_analysis_known_values()` — synthetic 150-sample Series with deterministic bin membership, verify bin counts and mean returns match expected values exactly
- `test_monotonicity_test_monotonic_increasing()` — synthetic feature with strictly increasing bin means, verify `is_monotonic=True` and positive `kendall_tau`
- `test_monotonicity_test_flat()` — synthetic feature with flat bin means, verify `is_monotonic=False`
- `test_distribution_diagnostics_known_skew()` — normally distributed synthetic data (np.random.seed(42)), verify `is_normal=True` and skew near zero
- `test_decile_plot_smoke()` — synthetic data, verify three-subplot Figure created without errors
- `test_histogram_quantile_overlays_smoke()` — synthetic data, verify histogram Figure created with bin edge lines
- `test_edge_case_n_bins_2()` — n_bins=2 (minimum), verify two bins computed without error
- `test_edge_case_n_bins_30()` — n_bins=30 with 300 samples, verify 30 bins computed
- `test_sharpe_zero_volatility_bin()` — synthetic bin where all returns are identical (std=0), verify Sharpe=NaN, no crash
- `test_n_bins_too_large_raises()` — n_bins > len(feature)/10 → raises ValueError

**Integration tests** (location: `tests/integration/feature_validator/test_eda_pipeline.py`):
- Covered by `test_common_eda_continuous()` in `tests/integration/feature_validator/test_eda_pipeline.py`
- Uses default config: RSI lookback 5, ES daily, 2020-2023 (from `data/ohlc_data/`)
- Feature extracted via `extract_features_for_bias_node(bias_module='rsi', param_name='lookback', param_value=5, ticker=Ticker.ES, timeframe=TimeFrame.D)`
- Verifies: 15 bins produced, all bin stats non-null (except edge bins), monotonicity tau computed, all plots saved to output directory
- Customizable: `bias_module`, `param_name`, `param_value`, `ticker`, `timeframe`, `n_bins` exposed as parameters for researcher exploration with any continuous bias node
- Cache policy: `USE_CACHE=True`; if cache missing, skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback 5, ES, D, 2020-2023
- Researcher manual verification:
  - Inspect terminal output for per-bin Sharpe and t-stat — check for any bins with notably high signal
  - Review decile plot visually — expect mild monotonic or U-shaped pattern for RSI on ES
  - Check histogram — verify RSI distribution is roughly uniform (as expected for quantile binning)
  - Inspect Q-Q plot — RSI values may deviate from normal at tails
  - Note `monotonicity_tau` value — guide for whether feature warrants further investigation

## Definition of done
- [ ] Unit tests added under `tests/validators/eda/test_continuous_eda.py`
- [ ] Integration test covered by `tests/integration/feature_validator/test_eda_pipeline.py`
- [ ] Implementation in `feature_selection/eda/continuous_eda.py`
- [ ] Dataclasses extended in `feature_selection/eda/eda_dataclasses.py`
- [ ] Docs updated in `docs/api/feature_selection.md`
- [ ] `pytest tests/validators/eda/test_continuous_eda.py -q` passes
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
