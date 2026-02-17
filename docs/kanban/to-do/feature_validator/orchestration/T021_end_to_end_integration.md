# T021 — End-to-End Integration Tests for Feature Validator

## Goal
Provide comprehensive end-to-end integration tests that validate the complete FeatureValidator pipeline (EDA → Binning → ParamSens → PermTest → Vault) using realistic test scenarios with RSI on ES daily data from 2000-2024.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Full specification (lines 1-533)
- `tests/integration/` — Existing integration test structure
- T018 — FeatureValidator API
- T019 — ValidationConfig
- T020 — Vault integration
- RSI default config: lookback=5, TimeFrame.D, ES, 2020-2023
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards and template

## Scope
In scope:
- Full pipeline smoke tests (continuous and rule-based features)
- Multi-parameter grid end-to-end validation
- Vault integration workflow tests
- Report serialization and persistence tests
- Error handling and validation tests
- Determinism and reproducibility tests
- Performance benchmarking (basic)

Out of scope:
- Unit tests for individual components (separate tasks for EDA, Binning, ParamSens, PermTest)
- Phase-specific implementation (T018, T019, T020 handle those)
- Production deployment testing (separate deployment task)
- Live trading integration (separate task)

## Interfaces (must match)

### Add: `tests/integration/feature_validator/test_feature_validator_e2e.py`

**Core integration tests:**
```python
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime

from feature_selection.feature_validator import FeatureValidator
from feature_selection.validation_config import ValidationConfig
from utils.enums import TimeFrame, Ticker, Direction
from ensemble.vault_manager import get_ensemble_summary

# Fixtures

@pytest.fixture(scope='module')
def es_daily_candles() -> pd.DataFrame:
    """Load ES daily candles 2000-2024 for integration tests.

    Returns:
        DataFrame with OHLCV data, ~6000 rows
    """
    # Load from test data directory or generate synthetic
    ...

@pytest.fixture(scope='module')
def es_daily_target(es_daily_candles: pd.DataFrame) -> pd.Series:
    """Compute forward returns for ES daily data.

    Returns:
        Series with next-bar log returns, aligned with candles
    """
    ...

@pytest.fixture
def temp_vault_dir(tmp_path: Path) -> Path:
    """Create temporary vault directory for tests."""
    vault_dir = tmp_path / 'vault'
    vault_dir.mkdir()
    return vault_dir

# Continuous Feature Tests

class TestContinuousFeatureEndToEnd:
    """End-to-end tests for continuous features (RSI example)."""

    def test_full_pipeline_rsi_single_param(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        tmp_path: Path
    ):
        """Test full pipeline with single RSI parameter.

        Setup: RSI lookback=14, ES daily, 2000-2024
        Validates: Pipeline runs without errors, all phases complete
        Checks:
        - ValidationReport returned
        - All four phases executed (EDA, Binning, ParamSens, PermTest)
        - Reports contain expected data structures
        - Output directory has plots and report JSON
        """
        ...

    def test_full_pipeline_rsi_multi_param_grid(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        tmp_path: Path
    ):
        """Test full pipeline with RSI parameter grid.

        Setup: RSI lookback=[2,3,4,5,6,7,8,9,10], ES daily, 2000-2024
        Validates: Multi-parameter grid processing, early stopping funnel
        Checks:
        - 9 parameter combinations processed
        - EDA reports for all 9 params
        - Binning diagnostics for all 9 params
        - Parameter sensitivity analysis covers full grid
        - Permutation test funnel (some params may fail Stage 1/2)
        - Final validated_params is subset of original grid
        """
        ...

    def test_early_stopping_permutation_funnel(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test that early stopping correctly excludes failed params.

        Setup: RSI grid with synthetic data designed to fail some params
        Validates: Funnel logic (failed Stage 1 → excluded from Stage 2)
        Checks:
        - len(params_passed_stage1) < total_params (some fail)
        - len(params_passed_stage2) <= len(params_passed_stage1)
        - Stage 2 only tests params that passed Stage 1
        - Logs show exclusion messages
        """
        ...

    def test_stable_region_identification(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test parameter sensitivity identifies stable regions.

        Setup: RSI lookback=[2,3,4,5,6,7,8,9,10]
        Validates: Grid-aware neighbor smoothing finds stable neighborhoods
        Checks:
        - ParameterSensitivityReport contains stability_ratios
        - stable_regions list identifies contiguous param neighborhoods
        - Stability ratio > 0.8 for at least some params
        - Visualization plots generated
        """
        ...

    def test_walkforward_stability_consistency(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test walkforward stability analysis identifies consistent params.

        Setup: RSI grid, 2-year folds over 2000-2024
        Validates: Top-K parameter selection consistency across folds
        Checks:
        - WalkforwardStabilityReport contains per-fold top-K
        - Some params appear in top-K across multiple folds
        - Consistency metrics computed
        - Final params_passed_walkforward identified
        """
        ...

    def test_vault_integration_full_workflow(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        temp_vault_dir: Path
    ):
        """Test complete workflow from validation to vault persistence.

        Setup: RSI grid, validation, stable neighborhood selection, vault save
        Validates: End-to-end researcher workflow
        Checks:
        - Validation completes successfully
        - Stable params identified
        - create_ensemble_from_validated_params() creates vault structure
        - Feature JSON files saved correctly
        - get_ensemble_summary() returns correct metadata
        - Vault directory structure matches spec
        """
        ...

    def test_report_serialization_and_reload(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        tmp_path: Path
    ):
        """Test that validation reports can be saved and reloaded.

        Setup: Run validation, save report to JSON, reload
        Validates: Report persistence and deserialization
        Checks:
        - ValidationReport.to_json() succeeds
        - ValidationReport.from_json() succeeds
        - Reloaded report matches original (excluding timestamp)
        - Plots referenced by file paths are valid
        """
        ...

    def test_determinism_with_fixed_seed(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test that pipeline is fully deterministic with fixed seed.

        Setup: Run pipeline twice with same config and seed
        Validates: Identical results (permutation tests, validated params)
        Checks:
        - report1.validated_params == report2.validated_params
        - Permutation p-values match
        - Stability ratios match
        - All numeric results identical
        """
        ...

# Rule-Based Feature Tests

class TestRuleBasedFeatureEndToEnd:
    """End-to-end tests for rule-based features."""

    def test_full_pipeline_rule_based_feature(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        tmp_path: Path
    ):
        """Test full pipeline with rule-based feature.

        Setup: Example rule-based feature (e.g., breakout filter)
        Validates: Rule-based pipeline (no binning phase)
        Checks:
        - ValidationReport.binning_reports is None
        - EDA includes per-level statistics
        - Parameter sensitivity works for rule params
        - Permutation tests use candle shuffle only
        - Validated params identified
        """
        ...

    def test_rule_based_permutation_candle_shuffle_only(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test that rule-based features use candle shuffle for Stage 2.

        Setup: Rule-based feature
        Validates: Stage 2 permutation uses candle shuffle (not feature shuffle)
        Checks:
        - PermutationTestReport.stage2_mode == 'candle_shuffle'
        - Rule output preserves serial correlation structure in null
        """
        ...

# Configuration Tests

class TestConfigurationManagement:
    """Test configuration presets and customization."""

    def test_exploratory_config_preset(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test exploratory config preset (relaxed thresholds).

        Setup: Use ValidationConfig.exploratory()
        Validates: Relaxed thresholds allow more params to pass
        Checks:
        - metric_threshold = 0.4
        - significance_level = 0.10
        - permutation_replicates = 500
        - More params survive validation vs production config
        """
        ...

    def test_production_config_preset(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test production config preset (strict thresholds).

        Setup: Use ValidationConfig.production()
        Validates: Strict thresholds filter out marginal params
        Checks:
        - metric_threshold = 0.6
        - significance_level = 0.05
        - permutation_replicates = 1000
        - Fewer params survive vs exploratory config
        """
        ...

    def test_custom_config_validation(self):
        """Test that custom config validates thresholds.

        Setup: Try to create configs with invalid values
        Validates: ValueError raised for invalid thresholds
        Checks:
        - metric_threshold <= 0 raises ValueError
        - significance_level not in (0,1) raises ValueError
        - n_bins < 3 raises ValueError
        """
        ...

# Error Handling Tests

class TestErrorHandling:
    """Test error handling and validation."""

    def test_missing_data_handling(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test pipeline handles missing data gracefully.

        Setup: Introduce NaNs in candles/target
        Validates: Pipeline detects and reports missing data
        Checks:
        - EDA reports missing data counts
        - Warning logged for high missing data percentage
        - Pipeline completes or raises informative error
        """
        ...

    def test_misaligned_candles_target(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test error handling for misaligned data.

        Setup: Candles and target with different indices
        Validates: ValueError raised with clear message
        Checks:
        - Error mentions index mismatch
        - No silent failures
        """
        ...

    def test_invalid_date_range(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series
    ):
        """Test error handling for invalid date ranges.

        Setup: date_range outside candle data range
        Validates: ValueError raised
        Checks:
        - Error mentions date range mismatch
        - Suggests valid date range
        """
        ...

    def test_vault_strategy_direction_mismatch(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        temp_vault_dir: Path
    ):
        """Test vault save rejects strategy/direction mismatch.

        Setup: Validate with long binning, try to save to short ensemble
        Validates: ValueError raised before any files written
        Checks:
        - Error mentions strategy/direction mismatch
        - Vault directory remains clean (no partial writes)
        """
        ...

    def test_vault_timeframe_mismatch(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        temp_vault_dir: Path
    ):
        """Test vault save rejects timeframe mismatch.

        Setup: Daily feature, try to save to Weekly ensemble
        Validates: ValueError raised
        Checks:
        - Error mentions timeframe mismatch
        - No files written
        """
        ...

# Performance Tests

class TestPerformance:
    """Basic performance benchmarking."""

    def test_pipeline_runtime_single_param(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        benchmark
    ):
        """Benchmark full pipeline runtime with single parameter.

        Setup: RSI lookback=14, ES daily 2000-2024
        Validates: Pipeline completes in reasonable time
        Checks:
        - Total runtime < 2 minutes (example threshold)
        - Permutation tests dominate runtime (expected)
        """
        ...

    def test_pipeline_runtime_multi_param(
        self,
        es_daily_candles: pd.DataFrame,
        es_daily_target: pd.Series,
        benchmark
    ):
        """Benchmark full pipeline runtime with parameter grid.

        Setup: RSI lookback=[2,3,4,5,6,7,8,9,10]
        Validates: Multi-param grid scales reasonably
        Checks:
        - Runtime approximately linear in number of params (for Stage 1)
        - Early stopping reduces Stage 2/3 runtime
        - Total runtime < 20 minutes (example threshold for 9 params)
        """
        ...
```

## Data Contracts

**Test data requirements:**
- ES daily candles: 2000-01-01 to 2024-12-31 (~6000 bars)
- Columns: open, high, low, close, volume, ticker, timeframe, date
- No missing data in critical periods (or explicitly test missing data handling)
- Forward returns aligned with candles (no lookahead)

**Test fixtures:**
- Shared fixtures for candles and target (module scope for performance)
- Temporary directories for output (function scope, auto-cleanup)
- Temporary vault directories (function scope, auto-cleanup)

**Assertion patterns:**
- Check for expected data structures (dicts, dataframes, lists)
- Verify numeric thresholds (e.g., stability_ratio > 0.8)
- Validate file existence and structure
- Use approximate comparisons for floats (pytest.approx)

## Dependencies
- `pytest` — test framework
- `pytest-benchmark` — performance benchmarking
- `feature_selection/feature_validator.py` — FeatureValidator API (T018)
- `feature_selection/validation_config.py` — ValidationConfig (T019)
- `ensemble/vault_manager.py` — Vault utilities (T020)
- `utils/enums.py` — TimeFrame, Ticker, Direction
- `pandas` — data manipulation
- `pathlib` — file operations

## Invariants / Constraints

**Test isolation:**
- Each test uses independent temp directories
- No shared mutable state between tests
- Cleanup after each test (automatic with tmp_path fixture)

**Determinism:**
- Tests using permutation tests must set fixed seed
- Same seed => same results (verify in determinism test)
- Floating-point comparisons use pytest.approx

**Data integrity:**
- No lookahead in test data (forward returns computed correctly)
- Candles and target aligned (same DatetimeIndex)
- No data leakage across folds (walkforward tests)

**Performance:**
- Tests should complete in reasonable time (< 30 minutes total suite)
- Use smaller parameter grids for smoke tests (e.g., 3 params instead of 9)
- Full grids only for specific end-to-end tests

## Acceptance tests

T021 is the **critical end-to-end integration test** for the entire Feature Validator system. It is the single test that validates the complete research workflow: EDA → Binning → Parameter Sensitivity → Permutation Testing → Vault persistence. All other integration tests (T018-T020) cover narrower integration scenarios; T021 is the authoritative end-to-end check.

**Unit tests:**

Unit tests are not required for T021 itself — T021's scope is end-to-end integration. Any pure logic in test helpers (e.g., fixture construction, assertion helpers) should be kept minimal. Unit-level coverage of individual components belongs to their respective tasks (EDA, Binning, ParamSens, PermTest).

**Integration tests:**

Primary test file: `tests/integration/feature_validator/test_feature_validator_e2e.py`

This is the main end-to-end integration test covering orchestration tasks T018-T021.

**Default integration configuration (use for all tests unless overridden):**
```python
DEFAULT_INTEGRATION_CONFIG = {
    'bias_module': 'rsi',
    'param_name': 'lookback',
    'param_value': 5,
    'ticker': Ticker.ES,
    'timeframe': TimeFrame.D,
    'date_range': ('2020-01-01', '2023-12-31'),
    'direction': Direction.LONG
}
```

**Primary test — full EDA → Binning → ParamSens → PermTest → Vault workflow:**
```python
@pytest.mark.integration
def test_end_to_end_validation_workflow(
    bias_module: str = 'rsi',
    param_name: str = 'lookback',
    param_value: int = 5,
    ticker: Ticker = Ticker.ES,
    timeframe: TimeFrame = TimeFrame.D,
    date_range: tuple = ('2020-01-01', '2023-12-31'),
    direction: Direction = Direction.LONG,
    use_cache: bool = True
):
    """End-to-end integration test for the complete Feature Validator pipeline.

    Covers T018 (API), T019 (config), T020 (vault), T021 (orchestration).
    Default: RSI lookback 5, ES daily, 2020-2023.
    Customizable for any bias node/param/ticker for researcher exploration.

    Pipeline phases verified:
    1. EDA — descriptive stats, decile analysis, report generated
    2. Binning diagnostics — region detection, heatmap, coverage metric
    3. Parameter sensitivity — stability ratios, smoothed objectives, stable regions
    4. Permutation testing — Stage 1 (vector shuffle), Stage 2 (pipeline perm),
                             Stage 3 (walkforward stability)
    5. Vault persistence — ensemble directory created, feature JSON saved, reload works
    """
    print("=" * 60)
    print("Integration Test: Feature Validator End-to-End")
    print("=" * 60)

    # Load real data from cache
    cache_manager = CacheManager()
    bias_spec = {
        'module_name': bias_module,
        'params': {param_name: param_value},
        'timeframes': [timeframe]
    }
    features = cache_manager.load_cache(bias_spec, ticker, timeframe)
    if features is None:
        pytest.skip("Cache not populated. Run CacheManager.populate_cache() first.")

    print(f"Bias: {bias_module}_signal_{timeframe.value}_{param_name}_{param_value}")
    print(f"Ticker: {ticker.value}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Samples: {len(features)} observations")

    # Build validator with exploratory config
    config = ValidationConfig.exploratory()
    validator = FeatureValidator(
        feature_type='continuous',
        bias_node_spec={
            'module_name': bias_module,
            'timeframes': [timeframe],
            'params': {param_name: [param_value]}
        },
        ticker=ticker,
        config=config
    )

    # Run full pipeline
    report = validator.run_full_pipeline(
        candles=candles,
        target=target,
        date_range=date_range,
        output_dir=output_dir
    )

    # Phase 1: EDA assertions
    assert report.eda_reports is not None
    assert len(report.eda_reports) >= 1
    print(f"\nEDA: {len(report.eda_reports)} parameter combination(s) analyzed")

    # Phase 2: Binning assertions
    assert report.binning_reports is not None
    print(f"Binning: reports generated for {len(report.binning_reports)} param(s)")

    # Phase 3: Parameter sensitivity assertions
    assert report.param_sensitivity_report is not None
    assert report.param_sensitivity_report.stability_ratios is not None
    print(f"ParamSens: stability ratios computed")

    # Phase 4: Permutation test assertions
    assert report.permutation_test_report is not None
    assert report.permutation_test_report.stage1_results is not None
    assert report.permutation_test_report.stage2_results is not None
    print(f"PermTest: Stage 1 and Stage 2 results present")

    # Phase 5: Vault assertions
    summary = get_ensemble_summary(ensemble_dir)
    assert summary['num_feature_columns'] >= 1
    print(f"Vault: {summary['num_feature_columns']} feature(s) saved")

    print(f"\nPlots saved to: {output_dir}")
    print("PASS: End-to-end validation workflow successful")
```

**Additional integration tests in the same file:**

1. `test_e2e_rule_based_feature()` — rule-based feature (no binning phase); verify `report.binning_reports is None`; verify Stage 2 permutation uses candle shuffle
2. `test_e2e_early_stopping_funnel()` — use RSI grid with at least 3 param values (e.g., lookback [3, 5, 14]); run full pipeline; verify `params_passed_stage2 <= params_passed_stage1 <= total_params`; verify Stage 3 only receives params that passed Stage 2
3. `test_e2e_determinism()` — run `test_end_to_end_validation_workflow` twice with same seed; assert `report1.validated_params == report2.validated_params` and all p-values and stability ratios match
4. `test_e2e_report_serialization()` — run pipeline; call `report.to_json(path)`; reload with `ValidationReport.from_json(path)`; assert reloaded report matches original
5. `test_e2e_exploratory_vs_production_config()` — run pipeline with `ValidationConfig.exploratory()` and with `ValidationConfig.production()`; assert exploratory config passes >= as many params as production config (relaxed thresholds)

**Customization:** All tests accept keyword arguments for `bias_module`, `param_name`, `param_value`, `ticker`, `timeframe`, `date_range`, enabling researchers to run with any bias node by modifying the call site.

**Cache policy:**
- Default: `USE_CACHE=True`
- If cache missing: `pytest.skip("Cache not populated. Run CacheManager.populate_cache() first.")`
- Cache spec: RSI lookback [5], ES, D, 2020-2023
- Vault output: use `tmp_path` pytest fixture for all vault writes (auto-cleaned, never writes to repo vault)

**Researcher manual verification steps:**

After running `pytest tests/integration/feature_validator/test_feature_validator_e2e.py -v -s`:

1. **Terminal summary block** — confirm printed header shows correct bias node, ticker, date range, and sample count
2. **EDA phase** — verify printed decile analysis: Sharpe ratios and t-stats for each bin look plausible (not all zero, not all NaN); skew and kurtosis of feature distribution printed
3. **Binning phase** — verify printed region summary: number of regions identified (expect 1-2 for RSI), coverage percentage (expect 20-50%), region boundaries look reasonable given RSI distribution
4. **Parameter sensitivity phase** — verify stability ratios printed; for a single param (lookback=5), expect stability ratio = 1.0 (no neighbors to smooth over)
5. **Permutation test phase** — verify p-values printed for Stage 1 and Stage 2; check whether RSI lookback 5 passes at exploratory alpha=0.10
6. **Vault phase** — verify summary: `num_feature_columns=1`, `feature_columns=['rsi_signal_D_lookback_5']`, model ID present
7. **Plots** — run with `SAVE_INTEGRATION_OUTPUTS=1` env var to persist plots; open EDA decile plot, binning heatmap, and parameter sensitivity heatmap and visually inspect for coherent structure
8. **Report JSON** — open saved report JSON from `output_dir`; confirm all phase results present, no null fields, `validated_params` field is a non-empty list (or empty with clear reason printed)

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/test_feature_validator_e2e.py`
- [ ] Test fixtures load real candle data from `data/ohlc_data/` via `CacheManager` (default: RSI lookback 5, ES daily, 2020-2023)
- [ ] `test_end_to_end_validation_workflow()` passes: all five phases complete, report fields populated, vault file created
- [ ] `test_e2e_rule_based_feature()` passes: binning phase skipped, candle shuffle permutation confirmed
- [ ] `test_e2e_early_stopping_funnel()` passes: funnel logic verified for multi-param grid
- [ ] `test_e2e_determinism()` passes: identical results with same seed
- [ ] `test_e2e_report_serialization()` passes: JSON round-trip preserves all fields
- [ ] All tests accept customization parameters (bias module, param, ticker, timeframe)
- [ ] All tests print terminal summary: config, key metrics, pass/fail verdict
- [ ] Cache policy explicit in each test: skip with message if cache missing
- [ ] Vault writes use `tmp_path` (never write to repo vault)
- [ ] Docs updated in `docs/testing/integration_tests.md`
- [ ] `pytest tests/integration/feature_validator/test_feature_validator_e2e.py -v -s` passes with readable terminal output
- [ ] Full suite completes in < 30 minutes

## Notes

**Test data generation:**
- Option 1: Load real ES daily data from existing cache/database
- Option 2: Generate synthetic OHLCV data with known statistical properties
- Option 3: Use fixture data committed to repo (ensure license compliance)
- Recommended: Start with synthetic data (deterministic, fast), add real data later

**Test organization:**
- Group tests by feature type (continuous vs rule-based)
- Separate configuration tests from pipeline tests
- Error handling tests in dedicated class
- Performance tests marked with `@pytest.mark.slow` (optional)

**Parameterization:**
- Use `@pytest.mark.parametrize` for testing multiple configs
- Example: Test with exploratory, production, conservative configs
- Avoids code duplication

**Mock vs real components:**
- Integration tests should use real components (not mocks)
- Only mock external dependencies (e.g., file I/O if necessary)
- Validate actual behavior, not test doubles

**Debugging failed tests:**
- Save intermediate outputs to tmp_path for inspection
- Use `pytest -vv -s` for verbose output with print statements
- Add logging to FeatureValidator for debugging
- Consider interactive debugger: `pytest --pdb`

**CI/CD integration:**
- These tests should run in CI pipeline before merge
- Consider splitting into smoke tests (fast) and full tests (slow)
- Fast tests: single param, basic validation
- Full tests: multi-param grids, walkforward stability

**Future enhancements:**
- Parallel test execution (pytest-xdist)
- Test coverage reporting (pytest-cov)
- Property-based testing (hypothesis) for edge cases
- Mutation testing to validate test quality
- Integration with real vault deployment
