# T001 — Common EDA Infrastructure

## Goal
Build shared EDA infrastructure that computes descriptive statistics, temporal stability metrics, correlation analysis, and rolling objective metrics for both continuous and rule-based features.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Lines 67-88 (Common EDA specification)
- `eda/eda_runner.py` — Existing EDA infrastructure to extend/refactor
- `utils/models.py` — Candle Pydantic model with OHLCV data
- `utils/enums.py` — TimeFrame, Ticker, Direction enums
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

## Scope
In scope:
- Descriptive statistics computation (min, max, mean, median, std, skew, kurtosis) for feature and target
- Missing data analysis (count and percentage of NaNs)
- Sample size computation per regime/period
- Temporal stability: time series plots (feature and target over time)
- Rolling correlation computation (feature-target correlation in rolling windows)
- Correlation analysis: Pearson, Spearman, Kendall coefficients
- Lagged correlation analysis (test for lead/lag relationships)
- Rolling objective metric computation with customizable window size (default: 252 days)
- Rolling objective plot visualization for regime change detection
- Common data structures (`CommonEDAStats`, `CommonEDAPlots` dataclasses)

Out of scope:
- Continuous-specific EDA (decile analysis, distribution plots) → T002
- Rule-based-specific EDA (per-level statistics, bootstrap CIs) → T003
- Report generation and aggregation → T004
- Integration with binning models or permutation testing

## Interfaces (must match)
- Add: `feature_selection/eda/common_eda.py` — Core module for common EDA computation
  - `compute_descriptive_stats(feature: pd.Series, target: pd.Series) -> DescriptiveStats`
  - `compute_temporal_stability(feature: pd.Series, target: pd.Series, timestamps: pd.DatetimeIndex, rolling_window: int) -> TemporalStability`
  - `compute_correlation_analysis(feature: pd.Series, target: pd.Series, max_lag: int = 5) -> CorrelationAnalysis`
  - `compute_rolling_objective(signals: pd.Series, returns: pd.Series, objective_fn: Callable, window: int) -> pd.Series`
  - `create_common_eda_plots(feature: pd.Series, target: pd.Series, timestamps: pd.DatetimeIndex, rolling_corr: pd.Series, rolling_obj: pd.Series) -> CommonEDAPlots`

- Add: `feature_selection/eda/eda_dataclasses.py` — Structured data contracts
  - `@dataclass(frozen=True) class DescriptiveStats` — min, max, mean, median, std, skew, kurtosis, nan_count, nan_pct, sample_size
  - `@dataclass(frozen=True) class TemporalStability` — rolling_correlation series, structural_breaks (list of timestamps)
  - `@dataclass(frozen=True) class CorrelationAnalysis` — pearson, spearman, kendall, lagged_correlations (dict[int, float])
  - `@dataclass(frozen=True) class CommonEDAPlots` — time_series_fig, rolling_corr_fig, rolling_obj_fig (matplotlib.figure.Figure)

## Data Contracts
- **Input schema:**
  - `feature: pd.Series` — Index: DatetimeIndex, Values: float or int
  - `target: pd.Series` — Index: DatetimeIndex (aligned with feature), Values: float (returns)
  - `timestamps: pd.DatetimeIndex` — Must align with feature/target indices
  - `objective_fn: Callable[[pd.Series, pd.Series], float]` — Takes (signals, returns) → scalar metric

- **Output schema:**
  - All dataclasses frozen (immutable)
  - NaN handling: stats compute with `.dropna()`, report NaN counts separately
  - Rolling metrics return `pd.Series` with same index as input (first `window-1` values are NaN)

- **Alignment expectations:**
  - Feature, target, and timestamps must have identical index
  - No lookahead: all computations use aligned indices only
  - Rolling windows use `.rolling()` with `min_periods=window` to avoid partial windows

## Dependencies
- `pandas` — DataFrame/Series operations, rolling windows, correlation
- `scipy.stats` — skew, kurtosis, correlation coefficients
- `matplotlib` — Time series and rolling metric plots
- `numpy` — Numerical operations
- `typing` — Type hints (Callable, Protocol)

## Invariants / Constraints
- Deterministic: same inputs → same outputs (no random seed needed)
- No lookahead: rolling windows only use past data at each timestamp
- Immutability: all dataclasses frozen, return new objects
- NaN handling: explicitly report missing data, use `.dropna()` for statistics
- Rolling window validation: `window <= len(feature)`, raise ValueError if violated
- Index alignment: validate that feature and target indices match exactly before computation

## Acceptance tests

**Unit tests** (location: `tests/validators/eda/test_common_eda.py`):
- `test_descriptive_stats_known_values()` — synthetic Series with known mean/std/skew, verify stats match expected values exactly
- `test_temporal_stability_no_lookahead()` — handcrafted time series, assert rolling correlation at timestamp t only uses data up to t
- `test_correlation_analysis_lagged_synthetic()` — construct synthetic feature with known 1-lag lead structure, verify lagged correlation is highest at lag=1
- `test_rolling_objective_sharpe_manual()` — small synthetic returns series (10 observations), verify rolling Sharpe matches manual calculation
- `test_nan_handling_explicit()` — Series with known NaN positions, verify nan_count and nan_pct fields are correct and dropna is applied before stats
- `test_index_misalignment_raises()` — feature and target with different DatetimeIndex → raises ValueError
- `test_rolling_window_exceeds_length_raises()` — window > len(feature) → raises ValueError
- `test_common_eda_plots_smoke()` — synthetic data, verify all three Figure objects are created without errors

**Integration tests** (location: `tests/integration/feature_validator/test_eda_pipeline.py`):
- Covered by `test_common_eda_continuous()` in `tests/integration/feature_validator/test_eda_pipeline.py`
- Uses default config: RSI lookback 5, ES daily, 2020-2023 (from `data/ohlc_data/`)
- Feature extracted via `extract_features_for_bias_node(bias_module='rsi', param_name='lookback', param_value=5, ticker=Ticker.ES, timeframe=TimeFrame.D)`
- Verifies: descriptive stats non-null, rolling correlation computed for full date range, lagged correlations dict has keys 1-5, plots saved to output directory
- Customizable: `bias_module`, `param_name`, `param_value`, `ticker`, `timeframe` exposed as parameters for researcher exploration with any bias node
- Cache policy: `USE_CACHE=True`; if cache missing, skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback 5, ES, D, 2020-2023
- Researcher manual verification:
  - Inspect terminal output for descriptive stats (mean, std, skew, NaN count)
  - Check rolling correlation plot visually — expect stable or trending pattern for RSI on ES
  - Review lagged correlations — expect near-zero for RSI (no strong lead/lag structure expected)
  - Confirm all three plots saved to output directory

## Definition of done
- [ ] Unit tests added under `tests/validators/eda/test_common_eda.py`
- [ ] Integration test covered by `tests/integration/feature_validator/test_eda_pipeline.py`
- [ ] Implementation in `feature_selection/eda/common_eda.py`
- [ ] Dataclasses in `feature_selection/eda/eda_dataclasses.py`
- [ ] Docs updated in `docs/api/feature_selection.md` (or create if needed)
- [ ] `pytest tests/validators/eda/test_common_eda.py -q` passes
- [ ] Type hints pass strict mypy/pyright checks
- [ ] All dataclasses are frozen and immutable

## Notes
- Test feature: RSI with `lookback=[2,3,4,5,6,7,8,9,10]`, `TimeFrame.D`
- Default rolling window: 252 days (1 trading year for daily data)
- Lagged correlations: test lags 1-5 (default), allow user to customize `max_lag`
- Plot styling: use consistent matplotlib style for all EDA plots (define in central config)
- Objective functions: support Sharpe, Sortino, mean return, Calmar, etc. (passed as callable)
- Edge case: empty series, all NaN series, misaligned indices → raise informative errors
