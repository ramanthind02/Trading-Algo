# Feature Validator Integration Testing Specification

> **Purpose:** Define integration testing standards for all Feature Validator tasks to ensure proper distinction from unit tests and enable manual researcher verification.
> **Audience:** Implementation agents and researchers working on Feature Validator components.
> **Last updated:** 2026-02-16

---

## Core Principles

### Unit vs Integration Test Distinction

**Unit Tests:**
- Test isolated logic with synthetic/mocked data
- Use handcrafted fixtures (e.g., small DataFrames with known values)
- Fast execution, no external dependencies
- Location: `tests/validators/`, `tests/base_models/`, or other unit-focused folders
- **NOT** in `tests/integration/`

**Integration Tests:**
- Test real end-to-end pipeline behavior
- Use **persisted repository data** from `data/ohlc_data/`
- Run through real extraction/model/validator paths
- Location: **MUST** be in `tests/integration/feature_validator/`
- Purpose: Enable researcher manual verification through terminal outputs, plots, and reports

---

## Integration Test Data Standards

### Default Configuration

**When only one parameter configuration is needed:**
```python
# Default integration test configuration
DEFAULT_INTEGRATION_CONFIG = {
    'bias_module': 'rsi',
    'param_name': 'lookback',
    'param_value': 5,
    'ticker': Ticker.ES,
    'timeframe': TimeFrame.D,
    'date_range': ('2020-01-01', '2023-12-31'),  # 4 years for stability
    'direction': Direction.LONG
}
```

**Data source:**
- MUST load candles from `data/ohlc_data/` (repository-backed)
- Use `CacheManager.populate_cache(...)` if cache doesn't exist
- Default to cache-backed execution (`USE_CACHE=True`)

**Feature extraction:**
- MUST use real bias node extraction: `extract_features_for_bias_node(...)`
- MUST use real target computation: aligned forward returns
- NO synthetic feature data in integration tests

---

### Flexibility and Customization

**All integration tests MUST be parameterizable:**
```python
@pytest.mark.integration
def test_binning_diagnostics_integration(
    bias_module: str = 'rsi',
    param_name: str = 'lookback',
    param_value: int = 5,
    ticker: Ticker = Ticker.ES,
    timeframe: TimeFrame = TimeFrame.D,
    **kwargs
):
    """Integration test for binning diagnostics pipeline.

    Defaults to RSI lookback 5, but can be customized for any bias node
    and parameter configuration to enable researcher exploration.
    """
    # Implementation uses provided parameters
    ...
```

**Customization points:**
- Bias module (rsi, ewmac, momentum, volatility, etc.)
- Parameter name and value (lookback, span, threshold, etc.)
- Ticker (ES, NQ, CL, GC, etc.)
- Timeframe (D, W, M)
- Date range (default 4 years for walkforward stability)
- Direction (LONG, SHORT)
- Validation thresholds (Sharpe > 0.5, t-stat > 2.0, etc.)

**Why flexibility matters:**
- Researchers need to test across different features
- Parameter sensitivity analysis requires grid configurations
- Different markets may require different date ranges
- Validation thresholds may vary by research context

---

## Integration Test Scope and Grouping

### Grouping Strategy

**Single integration test can cover multiple related tasks when:**
1. Tasks belong to the same pipeline phase (e.g., all binning tasks)
2. Tasks share the same data dependencies
3. Combined test doesn't exceed ~200 LOC or become hard to debug
4. Natural execution flow connects the tasks (e.g., diagnostics → plots → report)

**Separate integration tests when:**
1. Tasks belong to different pipeline phases (EDA vs permutation testing)
2. Different data requirements (continuous vs rule-based features)
3. Performance isolation needed (walkforward stability is slow)
4. Independent verification paths (parameter sensitivity vs binning)

### Recommended Integration Test Structure

```
tests/integration/feature_validator/
├── test_eda_pipeline.py                    # Covers EDA tasks (T001-T004)
│   ├── test_common_eda_continuous()         # Common + continuous EDA
│   └── test_rule_based_eda()                # Rule-based EDA
├── test_binning_diagnostics.py             # Covers binning tasks (T005-T008)
│   └── test_binning_full_pipeline()         # Infrastructure → detection → plots → report
├── test_parameter_sensitivity.py           # Covers param sens tasks (T009-T012)
│   └── test_parameter_sensitivity_pipeline() # Smoothing → stability → viz → report
├── test_permutation_testing.py             # Covers permutation tasks (T013-T017)
│   ├── test_vector_shuffle_stage1()
│   ├── test_pipeline_permutation_stage2()
│   ├── test_walkforward_stability_stage3()
│   └── test_early_stopping_orchestration()
└── test_feature_validator_e2e.py           # Covers orchestration (T018-T021)
    └── test_end_to_end_validation_workflow() # Full EDA → binning → param → perm → vault
```

**Total: ~8-10 integration test functions covering 20 tasks.**

---

## Not Every Task Needs Integration Tests

### Tasks That REQUIRE Integration Tests

**Pipeline validation tasks:**
- End-to-end workflows (EDA, binning, parameter sensitivity, permutation testing)
- Feature extraction integration
- Vault persistence
- Multi-phase orchestration

**Criteria:** Task involves cross-module interaction or data pipeline behavior.

---

### Tasks That DON'T Need Integration Tests

**Pure logic/algorithm tasks:**
- Grid neighbor smoothing algorithm (T009) — unit test with synthetic grid
- Stability ratio computation (T010) — unit test with known metrics
- Report dataclass construction — unit test with mock data

**Criteria:** Task is self-contained logic that can be fully tested with synthetic inputs.

**When in doubt:** Provide BOTH unit and integration tests. Unit tests verify correctness, integration tests verify real-world behavior.

---

## Integration Test Output and Verification

### Terminal Output Requirements

**Every integration test MUST print:**
1. **Configuration summary:**
   ```
   ═══════════════════════════════════════════
   Integration Test: Binning Diagnostics
   ═══════════════════════════════════════════
   Bias: rsi_signal_D_lookback_5
   Ticker: ES
   Date range: 2020-01-01 to 2023-12-31
   Samples: 1,008 observations
   ```

2. **Key metrics:**
   ```
   Binning Results:
   - Regions identified: 2
   - Region 1: bins [0, 1, 2] → mean Sharpe = 0.68, t-stat = 3.2
   - Region 2: bins [13, 14] → mean Sharpe = 0.52, t-stat = 2.4
   - Coverage: 38.5% of feature distribution
   - Shape: tail (long), hump (short)
   ```

3. **Pass/fail verdict:**
   ```
   ✅ PASS: Binning diagnostics successful
   ```

**Why:** Researcher can review terminal output to manually verify results without opening files.

---

### Plot and Report File Output

**Save artifacts to temporary directory:**
```python
import tempfile
from pathlib import Path

@pytest.mark.integration
def test_binning_diagnostics_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Run pipeline, save plots and reports
        report = run_binning_diagnostics(
            ...,
            output_dir=output_dir
        )

        # Verify files exist
        assert (output_dir / 'binning_heatmap.png').exists()
        assert (output_dir / 'region_boundaries.png').exists()
        assert (output_dir / 'binning_report.json').exists()

        # Print file locations for manual inspection
        print(f"\nPlots saved to: {output_dir}")
        print("  - binning_heatmap.png")
        print("  - region_boundaries.png")
```

**Optional: Persist to fixed location for manual review:**
```python
# For manual researcher inspection, optionally save to known location
MANUAL_REVIEW_DIR = Path('tests/integration/outputs/')
if os.getenv('SAVE_INTEGRATION_OUTPUTS'):
    shutil.copytree(output_dir, MANUAL_REVIEW_DIR / test_name)
    print(f"\n📊 Plots saved for manual review: {MANUAL_REVIEW_DIR / test_name}")
```

**Why:** Researcher can visually inspect plots/reports to verify correctness.

---

## Cache Policy for Integration Tests

### Default: Use Existing Cache

```python
@pytest.mark.integration
def test_eda_integration(use_cache: bool = True):
    if use_cache:
        # Attempt to load from cache
        features = cache_manager.load_cache(bias_spec, ticker, timeframe)
        if features is None:
            pytest.skip("Cache not populated. Run CacheManager.populate_cache() first.")
    else:
        # Populate cache if needed
        features = extract_features_for_bias_node(...)
        cache_manager.save_cache(features, bias_spec, ticker, timeframe)
```

### When to Populate Cache

**In the test itself:**
- If test is the FIRST integration test for a component (e.g., EDA)
- If cache is small and fast to compute (< 10 seconds)

**Via setup fixture:**
```python
@pytest.fixture(scope='module')
def feature_validator_integration_data():
    """Populate cache once for all feature validator integration tests."""
    bias_spec = {'module_name': 'rsi', 'params': {'lookback': 5}, ...}
    cache_manager.populate_cache(
        bias_specs=[bias_spec],
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        date_range=('2020-01-01', '2023-12-31')
    )
    return cache_manager
```

**When to skip:**
- If cache is large and slow (> 30 seconds)
- If test is optional/exploratory

**Explicit in task spec:** Each task must state cache policy in "Acceptance tests" section.

---

## Example Integration Test Template

```python
"""Integration tests for Feature Validator EDA pipeline.

Uses real repository data from data/ohlc_data/ and real bias node extraction.
Default configuration: RSI lookback 5 on ES daily data (2020-2023).
"""

import pytest
from pathlib import Path
from utils.enums import Ticker, TimeFrame, Direction
from feature_selection.validators.eda import run_eda_pipeline
from utils.cache_manager import CacheManager

# Default integration test configuration
DEFAULT_CONFIG = {
    'bias_module': 'rsi',
    'param_name': 'lookback',
    'param_value': 5,
    'ticker': Ticker.ES,
    'timeframe': TimeFrame.D,
    'date_range': ('2020-01-01', '2023-12-31'),
    'direction': Direction.LONG
}


@pytest.mark.integration
def test_eda_continuous_feature_integration(
    bias_module: str = DEFAULT_CONFIG['bias_module'],
    param_value: int = DEFAULT_CONFIG['param_value'],
    ticker: Ticker = DEFAULT_CONFIG['ticker'],
    timeframe: TimeFrame = DEFAULT_CONFIG['timeframe'],
):
    """Integration test for continuous feature EDA pipeline.

    Tests:
    - Real data loading from data/ohlc_data/
    - Real bias node feature extraction
    - Common EDA metrics + continuous-specific (decile analysis)
    - Report generation with plots

    Default: RSI lookback 5, ES daily, 2020-2023.
    Customizable for any bias node/param/ticker for researcher exploration.
    """
    print("=" * 60)
    print("Integration Test: Continuous Feature EDA")
    print("=" * 60)

    # Load real data
    cache_manager = CacheManager()
    bias_spec = {
        'module_name': bias_module,
        'params': {'lookback': param_value},
        'timeframes': [timeframe]
    }

    # Use cache or skip if not available
    features = cache_manager.load_cache(bias_spec, ticker, timeframe)
    if features is None:
        pytest.skip("Cache not populated. Run CacheManager.populate_cache() first.")

    print(f"Bias: {bias_module}_signal_{timeframe.value}_lookback_{param_value}")
    print(f"Ticker: {ticker.value}")
    print(f"Samples: {len(features)} observations")

    # Run EDA pipeline
    report = run_eda_pipeline(
        features=features,
        target=target,
        feature_type='continuous',
        output_dir=Path('/tmp/eda_integration')
    )

    # Verify report structure
    assert report.descriptive_stats is not None
    assert report.correlation_metrics is not None
    assert report.decile_analysis is not None  # Continuous-specific
    assert report.plots['decile_plot'] is not None

    # Print key metrics for manual verification
    print("\nDescriptive Statistics:")
    print(f"  Mean: {report.descriptive_stats.feature_mean:.4f}")
    print(f"  Std: {report.descriptive_stats.feature_std:.4f}")
    print(f"  Skew: {report.descriptive_stats.feature_skew:.4f}")

    print("\nDecile Analysis:")
    for i, bin_stats in enumerate(report.decile_analysis.bin_stats):
        print(f"  Bin {i}: Sharpe = {bin_stats.sharpe:.3f}, t-stat = {bin_stats.t_stat:.2f}")

    print(f"\n📊 Plots saved to: /tmp/eda_integration/")
    print("✅ PASS: EDA integration test successful")

    # Assertions for automated verification
    assert len(report.decile_analysis.bin_stats) == 15  # Default n_bins
    assert report.decile_analysis.monotonicity_tau is not None
```

---

## Task Specification Requirements

### Every Task Must Specify

**In "Acceptance tests" section:**

1. **Unit tests (if applicable):**
   ```markdown
   **Unit tests:**
   - `test_grid_neighbor_smoothing_1d()` — synthetic 1D grid with known neighbors
   - `test_grid_neighbor_smoothing_2d()` — synthetic 2D grid, verify boundary handling
   - `test_grid_neighbor_smoothing_empty()` — edge case: single point, no neighbors
   ```

2. **Integration tests (if applicable):**
   ```markdown
   **Integration tests:**
   - Covered by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
   - Uses default config: RSI lookback grid [3, 4, 5, 10, 14, 20], ES daily 2020-2023
   - Verifies: smoothed metrics computed, stability ratios > 0, plots generated
   - Researcher verifies: stability heatmap shows coherent regions, no isolated peaks
   ```

3. **Cache policy (integration only):**
   ```markdown
   **Cache policy:**
   - Use existing cache: `USE_CACHE=True`
   - If cache missing: skip with message "Run CacheManager.populate_cache() first"
   - Cache spec: RSI lookback [5], ES, D, 2020-2023
   ```

4. **Manual verification steps:**
   ```markdown
   **Researcher manual verification:**
   - Inspect terminal output for per-bin Sharpe ratios and t-stats
   - Review binning heatmap: expect 1-2 contiguous regions
   - Check region boundaries on feature distribution histogram
   - Verify coverage metric (expect 20-40% for typical RSI)
   ```

---

## Enforcement and Review

**Before moving task to `in-progress/`:**
- [ ] Task clearly separates unit vs integration test requirements
- [ ] Integration tests (if any) specify data source (`data/ohlc_data/`)
- [ ] Integration tests specify default config (use RSI-5 if single param needed)
- [ ] Integration tests are customizable (parameters exposed)
- [ ] Cache policy is explicit
- [ ] Manual verification steps are listed
- [ ] Integration test location specified (`tests/integration/feature_validator/...`)

**If task doesn't need integration tests:**
- [ ] Explicitly state: "Integration test: Not required (pure logic, covered by unit tests)"

---

**End of specification.**
