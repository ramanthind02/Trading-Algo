# T018 — FeatureValidator API and Pipeline Orchestration

## Goal
Provide a researcher-facing public API (`FeatureValidator` class) that orchestrates the four validation phases (EDA, Binning, ParamSens, PermTest) for both continuous and rule-based features with early stopping and composable report generation.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — full specification (lines 1-533)
- `docs/library/Feature_selection/permutation_testing/in-sample_pt.md` — three-stage permutation testing
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — grid-aware smoothing
- `feature_selection/base_models/base_model.py` — `BinningModelBase` interface
- `eda/eda_runner.py` — existing EDA infrastructure
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`

## Scope
In scope:
- `FeatureValidator` class as main public API
- Pipeline orchestration logic (sequential phase execution with early stopping)
- Feature type routing (continuous vs rule-based pipeline branching)
- Report aggregation and persistence
- Per-parameter-combination execution with funnel logic (failed params don't proceed)
- Integration hooks for each phase (EDA, Binning, ParamSens, PermTest)

Out of scope:
- Individual phase implementations (separate tasks: EDA, Binning, ParamSens, PermTest)
- Configuration management (T019)
- Vault integration (T020)
- End-to-end integration tests (T021)

## Interfaces (must match)

### Add: `feature_selection/feature_validator.py`

**Main API class:**
```python
from dataclasses import dataclass
from typing import Protocol, Dict, List, Optional
from pathlib import Path
import pandas as pd

from utils.enums import TimeFrame, Ticker
from utils.models import Candle

class FeatureValidator:
    """Main API for feature validation pipeline.

    Orchestrates four phases:
    1. EDA (per-param exploratory diagnostics)
    2. Binning diagnostics (continuous features only)
    3. Parameter sensitivity (grid-aware neighbor smoothing)
    4. Permutation testing (3-stage with early stopping)

    Example usage:
        validator = FeatureValidator(
            feature_type='continuous',
            bias_node_spec={
                'module_name': 'rsi',
                'timeframes': [TimeFrame.D],
                'params': {'lookback': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
            },
            ticker=Ticker.ES,
            config=ValidationConfig(...)  # from T019
        )

        # Run full pipeline
        reports = validator.run_full_pipeline(
            candles=candles_df,
            target=target_series,
            date_range=('2000-01-01', '2024-12-31')
        )

        # Or run phases individually
        eda_reports = validator.run_eda(candles, target)
        binning_reports = validator.run_binning_diagnostics(candles, target)
        param_report = validator.run_parameter_sensitivity(candles, target)
        perm_reports = validator.run_permutation_tests(candles, target)
    """

    def __init__(
        self,
        feature_type: str,  # 'continuous' or 'rule_based'
        bias_node_spec: Dict,  # module_name, timeframes, params (with grid)
        ticker: Ticker,
        config: 'ValidationConfig'  # from T019
    ) -> None:
        """Initialize validator with feature spec and configuration."""
        ...

    def run_full_pipeline(
        self,
        candles: pd.DataFrame,
        target: pd.Series,
        date_range: tuple[str, str],
        output_dir: Optional[Path] = None
    ) -> 'ValidationReport':
        """Run all four phases sequentially with early stopping.

        Pipeline flow:
        1. Run EDA per param combo (all params)
        2. Run binning diagnostics per param combo (continuous only)
        3. Run parameter sensitivity (all params, uses grid smoothing)
        4. Run permutation tests (3-stage funnel with early stopping):
           - Stage 1 (vector shuffle): filter out failed params
           - Stage 2 (pipeline perm): only params that passed Stage 1
           - Stage 3 (walkforward): only params that passed Stage 1+2

        Args:
            candles: OHLCV data (must include columns from Candle model)
            target: Forward returns aligned with candles
            date_range: (start_date, end_date) for in-sample period
            output_dir: Optional directory to save reports and plots

        Returns:
            ValidationReport with all phase results and surviving params
        """
        ...

    def run_eda(
        self,
        candles: pd.DataFrame,
        target: pd.Series
    ) -> Dict[tuple, 'EDAReport']:
        """Run EDA for all parameter combinations.

        Returns:
            Dict mapping param_combo (e.g., (lookback=14,)) to EDAReport
        """
        ...

    def run_binning_diagnostics(
        self,
        candles: pd.DataFrame,
        target: pd.Series
    ) -> Dict[tuple, 'BinningDiagnosticsReport']:
        """Run binning diagnostics (continuous features only).

        Raises ValueError if called on rule-based features.

        Returns:
            Dict mapping param_combo to BinningDiagnosticsReport
        """
        ...

    def run_parameter_sensitivity(
        self,
        candles: pd.DataFrame,
        target: pd.Series
    ) -> 'ParameterSensitivityReport':
        """Run grid-aware parameter sensitivity analysis.

        Computes raw and smoothed objectives for full grid,
        identifies stable regions (stability ratio > 0.8).

        Returns:
            ParameterSensitivityReport with stability analysis
        """
        ...

    def run_permutation_tests(
        self,
        candles: pd.DataFrame,
        target: pd.Series,
        param_filter: Optional[List[tuple]] = None
    ) -> 'PermutationTestReport':
        """Run 3-stage permutation testing with early stopping.

        Stage 1: Vector shuffle (quick filter, α=0.10)
        Stage 2: Pipeline permutation (feature/candle shuffle, α=0.10)
        Stage 3: Walkforward stability (temporal consistency check)

        Args:
            candles: OHLCV data
            target: Forward returns
            param_filter: Optional subset of params to test (from sensitivity analysis)

        Returns:
            PermutationTestReport with per-stage results and surviving params
        """
        ...

    def get_validated_params(self) -> List[tuple]:
        """Get list of parameter combinations that passed all validation stages.

        Criteria:
        - Passed permutation test Stage 1 and 2
        - In stable region (sensitivity analysis stability ratio > 0.8)
        - Appeared in top-K across 3+ folds (walkforward stability)

        Returns:
            List of param combos (tuples) that meet all criteria
        """
        ...
```

**Report dataclasses:**
```python
@dataclass(frozen=True)
class ValidationReport:
    """Aggregated results from full validation pipeline."""
    feature_type: str  # 'continuous' or 'rule_based'
    bias_node_spec: Dict
    ticker: Ticker
    date_range: tuple[str, str]

    # Phase results
    eda_reports: Dict[tuple, 'EDAReport']
    binning_reports: Optional[Dict[tuple, 'BinningDiagnosticsReport']]  # None for rule-based
    param_sensitivity_report: 'ParameterSensitivityReport'
    permutation_test_report: 'PermutationTestReport'

    # Summary
    total_params_tested: int
    params_passed_stage1: int
    params_passed_stage2: int
    params_passed_walkforward: int
    validated_params: List[tuple]  # final list meeting all criteria

    # Metadata
    timestamp: str
    config: 'ValidationConfig'

@dataclass(frozen=True)
class EDAReport:
    """Per-parameter EDA diagnostics (defined in EDA task)."""
    param_combo: tuple
    # ... detailed fields in EDA task

@dataclass(frozen=True)
class BinningDiagnosticsReport:
    """Binning validation results (defined in Binning task)."""
    param_combo: tuple
    # ... detailed fields in Binning task

@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Grid-aware parameter stability analysis (defined in ParamSens task)."""
    raw_objectives: pd.DataFrame  # param_combo x objective
    smoothed_objectives: pd.DataFrame
    stability_ratios: pd.DataFrame
    stable_regions: List[tuple]  # param combos with stability ratio > 0.8
    # ... detailed fields in ParamSens task

@dataclass(frozen=True)
class PermutationTestReport:
    """3-stage permutation test results (defined in PermTest task)."""
    stage1_results: Dict[tuple, 'VectorShuffleResult']
    stage2_results: Dict[tuple, 'PipelinePermutationResult']
    stage3_results: 'WalkforwardStabilityResult'

    params_passed_stage1: List[tuple]
    params_passed_stage2: List[tuple]
    params_passed_walkforward: List[tuple]
    # ... detailed fields in PermTest task
```

## Data Contracts

**Input candles DataFrame:**
- Columns: `['open', 'high', 'low', 'close', 'volume', 'ticker', 'timeframe', 'date']`
- Index: DatetimeIndex
- Alignment: Must match `utils.models.Candle` schema
- No lookahead: All feature computations use bar data up to and including bar timestamp

**Target Series:**
- Index: DatetimeIndex (aligned with candles)
- Values: Forward returns (e.g., next-bar log return)
- No lookahead: Target at time t uses return from t to t+1

**Parameter grid specification:**
- Dict format: `{'param_name': [val1, val2, ...], ...}`
- Example: `{'lookback': [2, 3, 4, 5, 6, 7, 8, 9, 10]}`
- Grid expansion: Cartesian product of all param lists

## Dependencies
- `feature_selection/base_models/` — base model interface
- `eda/eda_runner.py` — EDA phase (extend in separate task)
- `eda/parameter_analysis.py` — parameter sensitivity phase (extend in separate task)
- `utils/permutation_test/permutation_engine.py` — permutation testing phase (extend in separate task)
- `nodes/` — bias node feature computation
- `utils/enums.py` — TimeFrame, Ticker, Direction
- `utils/models.py` — Candle

## Invariants / Constraints

**Determinism:**
- Same inputs (candles, target, config, seed) => identical reports
- All random operations (permutation tests) use config-specified seed

**Early stopping (permutation tests):**
- Parameter combos failing Stage 1 are excluded from Stage 2
- Parameter combos failing Stage 2 are excluded from Stage 3
- Funnel progression: many params → fewer params → validated params

**Feature type routing:**
- Continuous features: run all four phases (EDA → Binning → ParamSens → PermTest)
- Rule-based features: skip binning phase (EDA → ParamSens → PermTest)
- Binning diagnostics raises ValueError if called on rule-based features

**No data snooping:**
- All thresholds (metric threshold, t-stat threshold, significance level) set in config before pipeline runs
- Same thresholds applied to original and permuted data
- No mid-pipeline threshold adjustments

**Report immutability:**
- All report dataclasses are frozen
- Reports can be serialized to JSON for persistence

## Acceptance tests

1. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_full_pipeline_continuous_rsi -v`
   - Setup: RSI lookback [2, 3, 4, 5, 6, 7, 8, 9, 10], TimeFrame.D, ES, dates 2000-2024
   - Validates: Full pipeline runs without errors, returns ValidationReport
   - Checks: All four phases executed, reports populated, validated_params list present

2. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_full_pipeline_rule_based -v`
   - Setup: Rule-based feature with parameter grid
   - Validates: Binning phase skipped, other phases run correctly
   - Checks: binning_reports is None, EDA/ParamSens/PermTest results present

3. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_early_stopping_funnel -v`
   - Setup: Synthetic data where some params should fail Stage 1
   - Validates: Early stopping works correctly (failed params don't proceed)
   - Checks: len(params_passed_stage2) < len(params_passed_stage1)

4. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_determinism -v`
   - Setup: Run pipeline twice with same seed
   - Validates: Identical reports (all metrics, plots, validated params)
   - Checks: report1 == report2 (excluding timestamp)

5. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_individual_phases -v`
   - Setup: RSI continuous feature
   - Validates: Each phase can be run independently
   - Checks: run_eda(), run_binning_diagnostics(), run_parameter_sensitivity(), run_permutation_tests() all work

6. `pytest tests/integration/feature_validator/test_api_orchestration.py::test_get_validated_params -v`
   - Setup: Run full pipeline, then call get_validated_params()
   - Validates: Returns only params meeting all criteria (perm tests + stability + walkforward)
   - Checks: validated_params is subset of original grid, all criteria verified

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/test_api_orchestration.py`
- [ ] `FeatureValidator` class implemented in `feature_selection/feature_validator.py`
- [ ] Report dataclasses defined with frozen=True
- [ ] Early stopping logic verified (permutation test funnel)
- [ ] Feature type routing verified (continuous vs rule-based)
- [ ] Determinism verified (same seed => same results)
- [ ] Docs updated in `docs/api/feature_selection.md`
- [ ] `pytest tests/integration/feature_validator/test_api_orchestration.py -q` passes

## Notes

**Early stopping implementation:**
- Permutation testing should maintain three lists: `params_to_test_stage1`, `params_to_test_stage2`, `params_to_test_stage3`
- After Stage 1, filter out failed params before Stage 2
- After Stage 2, filter out failed params before Stage 3
- Log progression: "Stage 1: 9/9 params tested, 6 passed. Stage 2: 6/6 params tested, 4 passed. Stage 3: 4 params evaluated."

**Report persistence:**
- ValidationReport should be serializable to JSON
- Save to output_dir if specified: `{output_dir}/validation_report_{timestamp}.json`
- Include all plots as file paths (saved separately in `{output_dir}/plots/`)

**Progress logging:**
- Use logging module to track pipeline progress
- Log phase transitions: "Starting EDA phase...", "Completed EDA for 9 parameter combinations"
- Log early stopping decisions: "Excluding 3 params from Stage 2 (failed Stage 1)"

**Future extensions:**
- Parallel execution of per-param EDA and binning diagnostics (currently sequential)
- Interactive web-based report viewer (Dash/Streamlit)
- Automatic parameter grid refinement based on sensitivity analysis
