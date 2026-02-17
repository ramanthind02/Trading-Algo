# T004 — EDA Report Generation

## Goal
Build EDA report generation infrastructure that aggregates common, continuous, and rule-based EDA components into structured reports with diagnostic flags and output persistence for per-parameter-combination inspection.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Lines 120-131 (EDA Output specification)
- `feature_selection/eda/common_eda.py` — Common EDA infrastructure (T001)
- `feature_selection/eda/continuous_eda.py` — Continuous feature EDA (T002)
- `feature_selection/eda/rule_based_eda.py` — Rule-based feature EDA (T003)
- `eda/eda_runner.py` — Existing EDA infrastructure to extend
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

## Scope
In scope:
- `EDAReport` dataclass aggregating all EDA components:
  - Common EDA stats and plots (from T001)
  - Feature-type-specific EDA (continuous from T002 OR rule-based from T003)
  - Metadata (feature name, parameter combination, timeframe, ticker)
  - Diagnostic flags (warnings, red flags)
- Report generation orchestration:
  - `run_eda_for_continuous_feature()` — orchestrates T001 + T002
  - `run_eda_for_rule_based_feature()` — orchestrates T001 + T003
  - Per-parameter-combination execution (independent reports)
- Diagnostic flag computation:
  - Warnings: high NaN percentage (>10%), low sample size (<252), unstable rolling correlation
  - Red flags: all NaN feature, zero variance, extreme skewness, no monotonicity (continuous)
- Report persistence:
  - Save EDA reports to disk (JSON metadata + pickled plots, or HTML output)
  - Directory structure: `eda_reports/{feature_name}/{param_combo}/`
- Report retrieval and inspection utilities

Out of scope:
- EDA computation logic (delegated to T001, T002, T003)
- Integration with binning models or permutation testing → Phases 2-4
- Multi-parameter aggregation or comparison → handled by researcher manually
- Web-based dashboard (nice-to-have for future)

## Interfaces (must match)
- Add: `feature_selection/eda/eda_reporter.py` — Report generation and persistence
  - `run_eda_for_continuous_feature(feature: pd.Series, target: pd.Series, timestamps: pd.DatetimeIndex, metadata: EDAMetadata, config: EDAConfig) -> ContinuousEDAReport`
  - `run_eda_for_rule_based_feature(feature: pd.Series, target: pd.Series, timestamps: pd.DatetimeIndex, metadata: EDAMetadata, config: EDAConfig) -> RuleBasedEDAReport`
  - `compute_diagnostic_flags(common_stats: CommonEDAStats, feature_stats: Union[ContinuousEDAStats, RuleBasedEDAStats]) -> DiagnosticFlags`
  - `save_eda_report(report: Union[ContinuousEDAReport, RuleBasedEDAReport], output_dir: Path) -> Path`
  - `load_eda_report(report_path: Path) -> Union[ContinuousEDAReport, RuleBasedEDAReport]`

- Add: `feature_selection/eda/eda_dataclasses.py` — Extend with report structures
  - `@dataclass(frozen=True) class EDAMetadata` — feature_name: str, param_combo: dict[str, Any], timeframe: TimeFrame, ticker: Ticker, timestamp: datetime
  - `@dataclass(frozen=True) class EDAConfig` — n_bins: int, rolling_window: int, objective_fn: Callable, max_lag: int, bootstrap_iterations: int, random_seed: int
  - `@dataclass(frozen=True) class DiagnosticFlags` — warnings: list[str], red_flags: list[str], is_viable: bool (no red flags)
  - `@dataclass(frozen=True) class ContinuousEDAReport` — metadata: EDAMetadata, common_stats: CommonEDAStats, continuous_stats: ContinuousEDAStats, common_plots: CommonEDAPlots, continuous_plots: ContinuousEDAPlots, diagnostics: DiagnosticFlags
  - `@dataclass(frozen=True) class RuleBasedEDAReport` — metadata: EDAMetadata, common_stats: CommonEDAStats, rule_stats: RuleBasedEDAStats, common_plots: CommonEDAPlots, rule_plots: RuleBasedEDAPlots, diagnostics: DiagnosticFlags

## Data Contracts
- **Input schema:**
  - `feature: pd.Series` — Index: DatetimeIndex, Values: float (continuous) or int (rule-based)
  - `target: pd.Series` — Index: DatetimeIndex (aligned), Values: float (returns)
  - `timestamps: pd.DatetimeIndex` — Must align with feature/target
  - `metadata: EDAMetadata` — Feature and parameter identification
  - `config: EDAConfig` — User-specified configuration for EDA computation

- **Output schema:**
  - `ContinuousEDAReport` / `RuleBasedEDAReport`: frozen dataclass containing all EDA components
  - `DiagnosticFlags.warnings`: list of human-readable warning strings
  - `DiagnosticFlags.red_flags`: list of human-readable red flag strings
  - `DiagnosticFlags.is_viable`: True if `len(red_flags) == 0`

- **Persistence format:**
  - Directory: `{output_dir}/{feature_name}/{param_combo_hash}/`
  - Files:
    - `metadata.json` — EDAMetadata serialized to JSON
    - `common_stats.json` — CommonEDAStats serialized to JSON
    - `feature_stats.json` — ContinuousEDAStats or RuleBasedEDAStats serialized
    - `diagnostics.json` — DiagnosticFlags serialized
    - `plots/` subdirectory with matplotlib figures saved as PNG
    - `report.html` (optional): HTML report combining all components

## Dependencies
- `pandas` — Series operations
- `pathlib` — Path operations for file persistence
- `json` — Metadata serialization
- `pickle` or `joblib` — Plot object serialization (or save as PNG directly)
- `matplotlib` — Plot generation (via T001, T002, T003)
- `dataclasses` — Report structures
- `typing` — Union, Callable, Protocol
- `hashlib` — Hash parameter combo for directory naming

## Invariants / Constraints
- Deterministic: same inputs → same outputs (delegate to T001-T003 determinism)
- No lookahead: orchestration doesn't introduce lookahead (delegate to T001-T003)
- Immutability: all report dataclasses frozen
- Parameter isolation: each parameter combination gets independent report (no cross-contamination)
- Diagnostic thresholds (configurable, pre-specified):
  - High NaN warning: >10% missing data
  - Low sample size warning: <252 observations
  - Zero variance red flag: feature std == 0
  - Extreme skewness red flag: |skew| > 5
  - No monotonicity warning (continuous): |kendall_tau| < 0.3
- Report versioning: include EDA framework version in metadata for reproducibility

## Acceptance tests

**Unit tests** (location: `tests/validators/eda/test_eda_reporter.py`):
- `test_diagnostic_flags_high_nan_warning()` — mock `CommonEDAStats` with `nan_pct=0.15`, verify "high NaN" warning present in `DiagnosticFlags.warnings`
- `test_diagnostic_flags_low_sample_warning()` — mock stats with `sample_size=100`, verify "low sample size" warning present
- `test_diagnostic_flags_zero_variance_red_flag()` — mock stats with `feature_std=0.0`, verify red flag present and `is_viable=False`
- `test_diagnostic_flags_extreme_skewness_red_flag()` — mock stats with `feature_skew=6.0`, verify red flag present and `is_viable=False`
- `test_diagnostic_flags_clean_data_is_viable()` — mock clean stats, verify `warnings=[]`, `red_flags=[]`, `is_viable=True`
- `test_report_persistence_and_reload()` — build `ContinuousEDAReport` from mocked sub-components, save to `tempfile.TemporaryDirectory`, reload, verify all JSON metadata fields round-trip correctly
- `test_report_directory_structure()` — after `save_eda_report()`, verify directory contains `metadata.json`, `common_stats.json`, `feature_stats.json`, `diagnostics.json`, and `plots/` subdirectory
- `test_report_overwrite_false_raises()` — call `save_eda_report()` twice with `overwrite=False` → second call raises error
- `test_rule_based_report_smoke()` — build `RuleBasedEDAReport` from mocked sub-components, verify dataclass fields populated without errors
- `test_param_combo_hash_deterministic()` — same `param_combo` dict always produces the same directory hash

**Integration tests** (location: `tests/integration/feature_validator/test_eda_pipeline.py`):
- Covered by `test_common_eda_continuous()` in `tests/integration/feature_validator/test_eda_pipeline.py`
- Uses default config: RSI lookback 5, ES daily, 2020-2023 (from `data/ohlc_data/`)
- Feature extracted via `extract_features_for_bias_node(bias_module='rsi', param_name='lookback', param_value=5, ticker=Ticker.ES, timeframe=TimeFrame.D)`
- Verifies: full `ContinuousEDAReport` generated without errors, `DiagnosticFlags.is_viable` is True for clean RSI data, all output files saved to `tempfile.TemporaryDirectory`, plots present in `plots/` subdirectory
- Customizable: `bias_module`, `param_name`, `param_value`, `ticker`, `timeframe` exposed as parameters; researcher can run with any continuous bias node to generate a full EDA report for inspection
- Cache policy: `USE_CACHE=True`; if cache missing, skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback 5, ES, D, 2020-2023
- Researcher manual verification:
  - Inspect terminal output for `DiagnosticFlags` — confirm no unexpected warnings or red flags for RSI data
  - Open `metadata.json` to confirm feature name, param combo, and date range are recorded correctly
  - Open `plots/` directory and review decile plot and rolling correlation plot visually
  - Confirm `is_viable=True` is printed to terminal for a clean feature
  - For multiple param combos (e.g., lookback=[2,5,10]): verify 3 independent subdirectories created with distinct param hashes

## Definition of done
- [ ] Unit tests added under `tests/validators/eda/test_eda_reporter.py`
- [ ] Integration test covered by `tests/integration/feature_validator/test_eda_pipeline.py`
- [ ] Implementation in `feature_selection/eda/eda_reporter.py`
- [ ] Dataclasses extended in `feature_selection/eda/eda_dataclasses.py`
- [ ] Docs updated in `docs/api/feature_selection.md`
- [ ] `pytest tests/validators/eda/test_eda_reporter.py -q` passes
- [ ] Type hints pass strict mypy/pyright checks
- [ ] All dataclasses are frozen and immutable

## Notes
- Test feature: RSI with `lookback=[2,3,4,5,6,7,8,9,10]`, `TimeFrame.D` → generates 9 independent reports
- Parameter combo hash: use `hashlib.md5(json.dumps(param_combo, sort_keys=True).encode()).hexdigest()[:8]` for directory naming
- Default EDAConfig values:
  - `n_bins=15` (continuous only)
  - `rolling_window=252` (daily), adjust for weekly/monthly
  - `objective_fn=sharpe_ratio` (default)
  - `max_lag=5` (lagged correlations)
  - `bootstrap_iterations=1000` (rule-based only)
  - `random_seed=42` (bootstrap)
- Diagnostic flag thresholds: expose in EDAConfig for user customization
- HTML report generation (optional): use Jinja2 template to combine stats tables and plot images
- Report retrieval: `load_eda_report()` reconstructs dataclass from JSON + plots from PNG (or pickled figures)
- Edge case: if report already exists at path, overwrite or raise error? → Add `overwrite: bool` parameter (default: False, raise if exists)
- Logging: log report generation progress (which param combo, which phase) for debugging
