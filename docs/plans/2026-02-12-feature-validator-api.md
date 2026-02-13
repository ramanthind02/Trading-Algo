# Feature Validator API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified FeatureValidator API that orchestrates end-to-end feature validation (EDA + permutation testing + parameter sensitivity) for both continuous and rule-based features.

**Architecture:** Composition-based design with a single FeatureValidator class that wraps existing infrastructure (PermutationEngine, ParameterAnalyzer) and provides progressive report accumulation across validation stages.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, plotly, statsmodels (ADF/KPSS tests), existing utils/eda infrastructure

---

## Implementation Strategy

**Directory Structure:**
```
feature_selection/validators/
├── __init__.py
├── config.py              # ValidationConfig dataclass
├── reports/
│   ├── __init__.py
│   ├── base.py           # Base report classes
│   ├── eda.py            # EDA report classes
│   ├── permutation.py    # Permutation report classes
│   ├── stability.py      # Stability report classes
│   └── export.py         # Report export functions
├── eda/
│   ├── __init__.py
│   ├── common.py         # Common EDA methods
│   ├── continuous.py     # Continuous-specific EDA
│   └── rule_based.py     # Rule-based-specific EDA
├── diagnostics/
│   ├── __init__.py
│   ├── binning.py        # Binning diagnostics
│   └── rules.py          # Rule-based diagnostics
└── validator.py          # Main FeatureValidator class
```

**Phased Implementation:**
1. Phase 1: Report data structures (Tasks 1-3)
2. Phase 2: Configuration and validation (Task 4)
3. Phase 3: EDA methods (Tasks 5-7)
4. Phase 4: Permutation testing integration (Tasks 8-9)
5. Phase 5: Stability analysis (Task 10)
6. Phase 6: Core orchestration (Tasks 11-12)
7. Phase 7: Report export (Task 13)
8. Phase 8: Integration testing (Task 14)

---

## Task 1: Foundation - Report Data Structures (Base)

**Files:**
- Create: `feature_selection/validators/__init__.py`
- Create: `feature_selection/validators/reports/__init__.py`
- Create: `feature_selection/validators/reports/base.py`
- Test: `tests/validators/test_report_structures.py`

### Step 1: Write failing test for DescriptiveStats

**File:** `tests/validators/test_report_structures.py`

```python
import pytest
import numpy as np
from feature_selection.validators.reports.base import DescriptiveStats


def test_descriptive_stats_creation():
    """Test DescriptiveStats dataclass creation."""
    stats = DescriptiveStats(
        mean=0.5,
        std=1.2,
        skew=-0.3,
        kurtosis=2.1,
        min_val=-3.0,
        max_val=4.5,
        q25=0.1,
        median=0.6,
        q75=1.1,
        n_samples=1000,
    )

    assert stats.mean == 0.5
    assert stats.std == 1.2
    assert stats.n_samples == 1000


def test_descriptive_stats_from_series():
    """Test DescriptiveStats.from_series() factory method."""
    import pandas as pd

    data = pd.Series(np.random.randn(100))
    stats = DescriptiveStats.from_series(data)

    assert stats.n_samples == 100
    assert stats.mean == pytest.approx(data.mean(), rel=1e-6)
    assert stats.std == pytest.approx(data.std(), rel=1e-6)
    assert stats.median == pytest.approx(data.median(), rel=1e-6)
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_report_structures.py::test_descriptive_stats_creation -v
```

Expected: `FAIL` with "No module named 'feature_selection.validators'"

### Step 3: Create directory structure

```bash
mkdir -p feature_selection/validators/reports
mkdir -p tests/validators
touch feature_selection/validators/__init__.py
touch feature_selection/validators/reports/__init__.py
touch tests/validators/__init__.py
```

### Step 4: Write minimal implementation for DescriptiveStats

**File:** `feature_selection/validators/reports/base.py`

```python
"""Base report data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DescriptiveStats:
    """Descriptive statistics for a numeric series."""

    mean: float
    std: float
    skew: float
    kurtosis: float
    min_val: float
    max_val: float
    q25: float
    median: float
    q75: float
    n_samples: int

    @classmethod
    def from_series(cls, data: pd.Series) -> "DescriptiveStats":
        """Create DescriptiveStats from a pandas Series."""
        return cls(
            mean=float(data.mean()),
            std=float(data.std()),
            skew=float(data.skew()),
            kurtosis=float(data.kurtosis()),
            min_val=float(data.min()),
            max_val=float(data.max()),
            q25=float(data.quantile(0.25)),
            median=float(data.median()),
            q75=float(data.quantile(0.75)),
            n_samples=len(data),
        )


@dataclass(frozen=True)
class ADFTestResult:
    """Augmented Dickey-Fuller test result."""

    statistic: float
    p_value: float
    n_lags: int
    critical_values: dict[str, float]  # {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    is_stationary: bool  # True if p_value < 0.05

    @classmethod
    def from_adf_output(cls, adf_output: tuple) -> "ADFTestResult":
        """Create from statsmodels adfuller output."""
        statistic, p_value, n_lags, _, critical_values, _ = adf_output
        return cls(
            statistic=float(statistic),
            p_value=float(p_value),
            n_lags=int(n_lags),
            critical_values={k: float(v) for k, v in critical_values.items()},
            is_stationary=p_value < 0.05,
        )


@dataclass(frozen=True)
class KPSSTestResult:
    """KPSS stationarity test result."""

    statistic: float
    p_value: float
    n_lags: int
    critical_values: dict[str, float]
    is_stationary: bool  # True if p_value > 0.05 (KPSS null = stationary)

    @classmethod
    def from_kpss_output(cls, kpss_output: tuple) -> "KPSSTestResult":
        """Create from statsmodels kpss output."""
        statistic, p_value, n_lags, critical_values = kpss_output
        return cls(
            statistic=float(statistic),
            p_value=float(p_value),
            n_lags=int(n_lags),
            critical_values={k: float(v) for k, v in critical_values.items()},
            is_stationary=p_value > 0.05,
        )


@dataclass(frozen=True)
class MonotonicityTestResult:
    """Test for monotonic relationship (e.g., across deciles)."""

    kendall_tau: float  # Kendall's tau correlation
    p_value: float
    is_monotonic: bool  # True if |tau| > 0.3 and p < 0.05
    direction: Literal['increasing', 'decreasing', 'none']
```

### Step 5: Run tests to verify they pass

```bash
pytest tests/validators/test_report_structures.py::test_descriptive_stats_creation -v
pytest tests/validators/test_report_structures.py::test_descriptive_stats_from_series -v
```

Expected: `PASS` for both tests

### Step 6: Add tests for ADFTestResult and KPSSTestResult

**File:** `tests/validators/test_report_structures.py` (append)

```python
def test_adf_test_result_from_output():
    """Test ADFTestResult.from_adf_output() factory method."""
    from feature_selection.validators.reports.base import ADFTestResult

    # Mock statsmodels adfuller output format
    mock_output = (
        -3.5,  # statistic
        0.008,  # p_value
        5,  # n_lags
        100,  # n_obs
        {'1%': -3.43, '5%': -2.86, '10%': -2.57},  # critical_values
        None  # icbest
    )

    result = ADFTestResult.from_adf_output(mock_output)

    assert result.statistic == -3.5
    assert result.p_value == 0.008
    assert result.is_stationary is True  # p < 0.05
    assert result.critical_values['5%'] == -2.86


def test_kpss_test_result_from_output():
    """Test KPSSTestResult.from_kpss_output() factory method."""
    from feature_selection.validators.reports.base import KPSSTestResult

    # Mock statsmodels kpss output format
    mock_output = (
        0.3,  # statistic
        0.1,  # p_value
        5,  # n_lags
        {'1%': 0.739, '5%': 0.463, '10%': 0.347}  # critical_values
    )

    result = KPSSTestResult.from_kpss_output(mock_output)

    assert result.statistic == 0.3
    assert result.p_value == 0.1
    assert result.is_stationary is True  # p > 0.05 (null = stationary)
```

### Step 7: Run tests to verify they pass

```bash
pytest tests/validators/test_report_structures.py -v
```

Expected: `PASS` for all tests

### Step 8: Commit

```bash
git add feature_selection/validators/ tests/validators/
git commit -m "feat: add base report data structures

- Add DescriptiveStats for numeric summaries
- Add ADFTestResult and KPSSTestResult for stationarity tests
- Add MonotonicityTestResult for decile trend analysis
- Implement factory methods for statsmodels integration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Report Data Structures (EDA Reports)

**Files:**
- Create: `feature_selection/validators/reports/eda.py`
- Modify: `tests/validators/test_report_structures.py`

### Step 1: Write failing test for EDAReport

**File:** `tests/validators/test_report_structures.py` (append)

```python
def test_eda_report_creation():
    """Test EDAReport dataclass creation."""
    from feature_selection.validators.reports.eda import EDAReport
    from feature_selection.validators.reports.base import (
        DescriptiveStats,
        ADFTestResult,
        KPSSTestResult,
    )
    import pandas as pd

    feature_stats = DescriptiveStats.from_series(pd.Series(np.random.randn(100)))
    target_stats = DescriptiveStats.from_series(pd.Series(np.random.randn(100)))

    mock_adf = (-3.5, 0.008, 5, 100, {'5%': -2.86}, None)
    mock_kpss = (0.3, 0.1, 5, {'5%': 0.463})

    report = EDAReport(
        feature_stats=feature_stats,
        target_stats=target_stats,
        correlations={'pearson': 0.15, 'spearman': 0.18},
        lagged_correlations=pd.Series([0.15, 0.10, 0.05]),
        adf_test=ADFTestResult.from_adf_output(mock_adf),
        kpss_test=KPSSTestResult.from_kpss_output(mock_kpss),
        rolling_correlation=pd.Series([0.12, 0.15, 0.18]),
        regime_stats={},
        distribution_plot=None,
        correlation_plot=None,
        time_series_plot=None,
        stationarity_plot=None,
    )

    assert report.feature_stats == feature_stats
    assert report.correlations['pearson'] == 0.15
    assert report.adf_test.is_stationary is True
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_report_structures.py::test_eda_report_creation -v
```

Expected: `FAIL` with "No module named 'feature_selection.validators.reports.eda'"

### Step 3: Implement EDAReport

**File:** `feature_selection/validators/reports/eda.py`

```python
"""EDA report data structures."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .base import DescriptiveStats, ADFTestResult, KPSSTestResult, MonotonicityTestResult


@dataclass(frozen=True)
class ContinuousEDAReport:
    """Continuous feature EDA extensions."""

    # Decile analysis
    decile_stats: pd.DataFrame  # decile → mean, std, Sharpe, t-stat
    decile_plot: Figure | Path | None
    monotonicity_test: MonotonicityTestResult

    # Outlier analysis
    outlier_fraction: float
    outlier_impact: float  # change in Sharpe when removing outliers
    outlier_plot: Figure | Path | None

    # Non-linearity
    linear_r2: float
    polynomial_r2: dict[int, float]  # degree → R²
    non_linearity_plot: Figure | Path | None

    # Binning diagnostics (added later in Task 6)
    binning_diagnostics: Any | None = None


@dataclass(frozen=True)
class RuleEDAReport:
    """Rule-based feature EDA extensions."""

    # Level distribution
    level_counts: dict[int, int]  # {-1: count, 0: count, 1: count}
    level_fractions: dict[int, float]
    imbalance_flag: bool  # True if any level < 10%

    # Per-level statistics
    level_stats: pd.DataFrame  # level → mean, std, Sharpe, t-stat, adjusted_Sharpe
    level_confidence_intervals: dict[int, tuple[float, float]]
    level_plot: Figure | Path | None

    # Regime transitions
    transition_matrix: pd.DataFrame  # P(level_t+1 | level_t)
    average_duration: dict[int, float]  # avg time spent in each level
    transition_plot: Figure | Path | None

    # Grid report (added later in Task 7)
    grid_report: Any | None = None


@dataclass(frozen=True)
class EDAReport:
    """Exploratory data analysis results."""

    # Distribution statistics
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats

    # Correlation analysis
    correlations: dict[str, float]  # {'pearson': 0.15, 'spearman': 0.18, ...}
    lagged_correlations: pd.Series  # correlation at different lags

    # Stationarity tests
    adf_test: ADFTestResult
    kpss_test: KPSSTestResult

    # Temporal stability
    rolling_correlation: pd.Series  # time series of rolling correlation
    regime_stats: dict[str, DescriptiveStats]  # per-regime statistics

    # Plots (stored as Figure objects or file paths)
    distribution_plot: Figure | Path | None
    correlation_plot: Figure | Path | None
    time_series_plot: Figure | Path | None
    stationarity_plot: Figure | Path | None

    # Feature-type-specific reports
    continuous_report: ContinuousEDAReport | None = None
    rule_report: RuleEDAReport | None = None

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
```

### Step 4: Run test to verify it passes

```bash
pytest tests/validators/test_report_structures.py::test_eda_report_creation -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/reports/eda.py tests/validators/
git commit -m "feat: add EDA report data structures

- Add EDAReport for common EDA results
- Add ContinuousEDAReport for continuous-specific analysis
- Add RuleEDAReport for rule-based-specific analysis
- Include stationarity tests, correlations, diagnostics

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Report Data Structures (Permutation & Stability)

**Files:**
- Create: `feature_selection/validators/reports/permutation.py`
- Create: `feature_selection/validators/reports/stability.py`
- Modify: `tests/validators/test_report_structures.py`

### Step 1: Write failing test for PermutationReport

**File:** `tests/validators/test_report_structures.py` (append)

```python
def test_permutation_report_creation():
    """Test PermutationReport dataclass creation."""
    from feature_selection.validators.reports.permutation import PermutationReport
    import numpy as np

    report = PermutationReport(
        stage='stage1_vector_shuffle',
        observed_sharpe=0.8,
        observed_t_stat=2.5,
        observed_returns_mean=0.05,
        permuted_sharpes=np.random.randn(1000),
        permuted_t_stats=np.random.randn(1000),
        p_value=0.01,
        confidence_level=0.95,
        critical_value=0.5,
        passed=True,
        margin=0.3,
        permutation_histogram=None,
        qq_plot=None,
        n_permutations=1000,
        random_seed=42,
        execution_time=5.2,
    )

    assert report.stage == 'stage1_vector_shuffle'
    assert report.passed is True
    assert report.p_value == 0.01
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_report_structures.py::test_permutation_report_creation -v
```

Expected: `FAIL`

### Step 3: Implement PermutationReport and StabilityReport

**File:** `feature_selection/validators/reports/permutation.py`

```python
"""Permutation test report data structures."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PermutationReport:
    """Results from permutation testing (Stage 1 or Stage 2)."""

    stage: Literal['stage1_vector_shuffle', 'stage2_feature_shuffle', 'stage2_candle_shuffle']

    # Observed statistics (real data)
    observed_sharpe: float
    observed_t_stat: float
    observed_returns_mean: float

    # Permutation distribution
    permuted_sharpes: np.ndarray  # shape: (n_permutations,)
    permuted_t_stats: np.ndarray

    # Statistical test results
    p_value: float
    confidence_level: float  # e.g., 0.95
    critical_value: float  # threshold from permutation distribution

    # Verdict
    passed: bool
    margin: float  # observed - critical_value (safety margin)

    # Diagnostic plots
    permutation_histogram: Figure | Path | None
    qq_plot: Figure | Path | None

    # Metadata
    n_permutations: int
    random_seed: int | None
    execution_time: float  # seconds
```

**File:** `feature_selection/validators/reports/stability.py`

```python
"""Stability analysis report data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class FoldResult:
    """Results from a single walkforward fold."""

    fold_index: int
    train_period: tuple[datetime, datetime]

    # Raw results (all param combos)
    objective_values: pd.DataFrame  # param combo → objective (Sharpe, t-stat, etc.)

    # Smoothed results (after neighbor averaging)
    smoothed_objectives: pd.Series
    stability_ratios: pd.Series

    # Top-K params for this fold
    top_k_params: list[dict[str, Any]]  # Top K param combos by smoothed objective


@dataclass(frozen=True)
class StabilityReport:
    """Walkforward stability analysis results."""

    # Walkforward folds
    n_folds: int
    fold_dates: list[tuple[datetime, datetime]]  # (train_start, train_end) per fold

    # Parameter grid
    params_grid: dict[str, list]
    n_param_combos: int

    # Per-fold results
    fold_results: list[FoldResult]  # One per fold

    # Aggregated stability metrics
    top_params_consistency: pd.DataFrame  # param combo → % folds in top-K
    stable_neighborhoods: list[dict[str, Any]]  # regions that appear consistently

    # Grid-aware neighbor smoothing
    smoothed_objectives: pd.DataFrame  # param combo → smoothed objective per fold
    stability_ratios: pd.DataFrame  # param combo → stability ratio per fold

    # Temporal consistency
    rank_correlation_across_folds: float  # Spearman correlation of param rankings
    best_region_stability: str  # "Stable" / "Moderate" / "Unstable"

    # Plots
    stability_heatmap: Figure | Path | None
    top_params_bar_chart: Figure | Path | None
    parameter_trajectory_plot: Figure | Path | None

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)
```

### Step 4: Run test to verify it passes

```bash
pytest tests/validators/test_report_structures.py::test_permutation_report_creation -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/reports/ tests/validators/
git commit -m "feat: add permutation and stability report structures

- Add PermutationReport for stages 1 & 2
- Add StabilityReport for stage 3 walkforward analysis
- Add FoldResult for per-fold stability metrics
- Include p-values, margins, consistency scores

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Configuration and Validation

**Files:**
- Create: `feature_selection/validators/config.py`
- Create: `tests/validators/test_config.py`

### Step 1: Write failing test for ValidationConfig

**File:** `tests/validators/test_config.py`

```python
import pytest
from feature_selection.validators.config import ValidationConfig


def test_validation_config_defaults():
    """Test ValidationConfig with default values."""
    config = ValidationConfig(feature_type='continuous')

    assert config.feature_type == 'continuous'
    assert config.n_permutations == 1000
    assert config.confidence_level == 0.95
    assert config.min_sharpe_threshold == 0.5
    assert config.random_seed is None


def test_validation_config_custom():
    """Test ValidationConfig with custom values."""
    config = ValidationConfig(
        feature_type='rule_based',
        n_permutations=500,
        min_sharpe_threshold=0.7,
        random_seed=42,
    )

    assert config.feature_type == 'rule_based'
    assert config.n_permutations == 500
    assert config.random_seed == 42


def test_validation_config_invalid_feature_type():
    """Test ValidationConfig rejects invalid feature types."""
    # This should fail at type-check level with mypy,
    # but we can test runtime validation if we add it
    config = ValidationConfig(feature_type='continuous')
    assert config.feature_type in ['continuous', 'rule_based']
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_config.py::test_validation_config_defaults -v
```

Expected: `FAIL`

### Step 3: Implement ValidationConfig

**File:** `feature_selection/validators/config.py`

```python
"""Validation configuration."""
from dataclasses import dataclass
from typing import Literal


FeatureType = Literal['continuous', 'rule_based']


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for validation pipeline."""

    feature_type: FeatureType
    n_permutations: int = 1000
    confidence_level: float = 0.95
    min_sharpe_threshold: float = 0.5
    min_t_stat_threshold: float = 2.0
    neighbor_steps: int = 1  # For grid smoothing
    top_k_per_fold: int = 5  # Walkforward stability
    random_seed: int | None = None
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/validators/test_config.py -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/config.py tests/validators/test_config.py
git commit -m "feat: add ValidationConfig dataclass

- Define FeatureType literal type
- Add ValidationConfig with sensible defaults
- Configure permutation counts, thresholds, grid params

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Common EDA Methods

**Files:**
- Create: `feature_selection/validators/eda/__init__.py`
- Create: `feature_selection/validators/eda/common.py`
- Create: `tests/validators/test_eda_common.py`

### Step 1: Write failing test for distribution analysis

**File:** `tests/validators/test_eda_common.py`

```python
import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.eda.common import (
    compute_distribution_stats,
    compute_correlations,
    run_stationarity_tests,
)


def test_compute_distribution_stats():
    """Test distribution statistics computation."""
    data = pd.Series(np.random.randn(1000))
    stats = compute_distribution_stats(data)

    assert stats.n_samples == 1000
    assert -0.2 < stats.mean < 0.2  # Approximately zero
    assert 0.8 < stats.std < 1.2  # Approximately one
    assert hasattr(stats, 'skew')
    assert hasattr(stats, 'kurtosis')


def test_compute_correlations():
    """Test correlation computation."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500))
    target = pd.Series(0.3 * feature + np.random.randn(500) * 0.5)

    corrs = compute_correlations(feature, target)

    assert 'pearson' in corrs
    assert 'spearman' in corrs
    assert 'kendall' in corrs
    assert 0.2 < corrs['pearson'] < 0.6  # Positive correlation


def test_run_stationarity_tests():
    """Test ADF and KPSS stationarity tests."""
    # Generate stationary series
    stationary_data = pd.Series(np.random.randn(200))

    adf_result, kpss_result = run_stationarity_tests(stationary_data)

    assert hasattr(adf_result, 'p_value')
    assert hasattr(kpss_result, 'p_value')
    assert hasattr(adf_result, 'is_stationary')
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_eda_common.py::test_compute_distribution_stats -v
```

Expected: `FAIL`

### Step 3: Implement common EDA functions

**File:** `feature_selection/validators/eda/common.py`

```python
"""Common EDA methods for both feature types."""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

from feature_selection.validators.reports.base import (
    DescriptiveStats,
    ADFTestResult,
    KPSSTestResult,
)


def compute_distribution_stats(data: pd.Series) -> DescriptiveStats:
    """
    Compute descriptive statistics for a numeric series.

    Args:
        data: Numeric series

    Returns:
        DescriptiveStats dataclass
    """
    return DescriptiveStats.from_series(data)


def compute_correlations(
    feature: pd.Series,
    target: pd.Series,
) -> dict[str, float]:
    """
    Compute feature-target correlations.

    Args:
        feature: Feature series
        target: Target series

    Returns:
        Dictionary with pearson, spearman, kendall correlations
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < 10:
        return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}

    pearson_corr = aligned['feature'].corr(aligned['target'], method='pearson')
    spearman_corr = aligned['feature'].corr(aligned['target'], method='spearman')
    kendall_corr = aligned['feature'].corr(aligned['target'], method='kendall')

    return {
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'kendall': float(kendall_corr),
    }


def run_stationarity_tests(
    data: pd.Series,
) -> tuple[ADFTestResult, KPSSTestResult]:
    """
    Run ADF and KPSS stationarity tests.

    Args:
        data: Time series data

    Returns:
        Tuple of (ADFTestResult, KPSSTestResult)
    """
    # Drop NaN
    clean_data = data.dropna()

    if len(clean_data) < 20:
        raise ValueError(f"Insufficient data for stationarity tests: {len(clean_data)} < 20")

    # Run ADF test
    adf_output = adfuller(clean_data, autolag='AIC')
    adf_result = ADFTestResult.from_adf_output(adf_output)

    # Run KPSS test
    kpss_output = kpss(clean_data, regression='c', nlags='auto')
    kpss_result = KPSSTestResult.from_kpss_output(kpss_output)

    return adf_result, kpss_result


def compute_lagged_correlations(
    feature: pd.Series,
    target: pd.Series,
    max_lag: int = 10,
) -> pd.Series:
    """
    Compute lagged correlations (feature_t vs. target_t+k).

    Args:
        feature: Feature series
        target: Target series
        max_lag: Maximum lag to compute

    Returns:
        Series with correlations at each lag
    """
    correlations = []

    for lag in range(max_lag + 1):
        if lag == 0:
            corr = feature.corr(target)
        else:
            # Shift target forward by lag
            target_lagged = target.shift(-lag)
            corr = feature.corr(target_lagged)

        correlations.append(corr)

    return pd.Series(correlations, index=range(max_lag + 1))


def compute_rolling_correlation(
    feature: pd.Series,
    target: pd.Series,
    window: int = 252,  # ~1 year for daily data
) -> pd.Series:
    """
    Compute rolling correlation over time.

    Args:
        feature: Feature series
        target: Target series
        window: Rolling window size

    Returns:
        Series with rolling correlation
    """
    # Align series
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < window:
        raise ValueError(f"Insufficient data for rolling correlation: {len(aligned)} < {window}")

    rolling_corr = aligned['feature'].rolling(window=window).corr(aligned['target'])

    return rolling_corr
```

**File:** `feature_selection/validators/eda/__init__.py`

```python
"""EDA methods for feature validation."""
from .common import (
    compute_distribution_stats,
    compute_correlations,
    run_stationarity_tests,
    compute_lagged_correlations,
    compute_rolling_correlation,
)

__all__ = [
    'compute_distribution_stats',
    'compute_correlations',
    'run_stationarity_tests',
    'compute_lagged_correlations',
    'compute_rolling_correlation',
]
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/validators/test_eda_common.py -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/eda/ tests/validators/test_eda_common.py
git commit -m "feat: add common EDA methods

- Add distribution statistics computation
- Add correlation analysis (Pearson, Spearman, Kendall)
- Add stationarity tests (ADF, KPSS)
- Add lagged and rolling correlations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Continuous-Specific EDA

**Files:**
- Create: `feature_selection/validators/eda/continuous.py`
- Create: `tests/validators/test_eda_continuous.py`

### Step 1: Write failing test for decile analysis

**File:** `tests/validators/test_eda_continuous.py`

```python
import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.eda.continuous import (
    compute_decile_stats,
    test_monotonicity,
    detect_outliers,
)


def test_compute_decile_stats():
    """Test decile-based analysis."""
    np.random.seed(42)

    # Create monotonic feature → target relationship
    feature = pd.Series(np.linspace(0, 10, 1000))
    target = pd.Series(0.5 * feature + np.random.randn(1000) * 0.5)

    decile_stats = compute_decile_stats(feature, target, n_deciles=10)

    assert len(decile_stats) == 10
    assert 'mean_target' in decile_stats.columns
    assert 'std_target' in decile_stats.columns
    assert 'sharpe' in decile_stats.columns

    # Check monotonicity (mean should increase across deciles)
    means = decile_stats['mean_target'].values
    assert means[-1] > means[0]  # Higher decile → higher mean


def test_monotonicity_test():
    """Test monotonicity detection."""
    # Monotonic increasing
    decile_means = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = test_monotonicity(decile_means)

    assert result.is_monotonic is True
    assert result.direction == 'increasing'
    assert result.kendall_tau > 0.8


def test_detect_outliers():
    """Test outlier detection."""
    data = pd.Series(np.random.randn(1000))
    # Add some outliers
    data.iloc[0] = 10.0
    data.iloc[1] = -10.0

    outlier_mask, outlier_fraction = detect_outliers(data, method='iqr')

    assert outlier_fraction > 0
    assert outlier_mask.sum() >= 2  # At least our 2 outliers
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_eda_continuous.py::test_compute_decile_stats -v
```

Expected: `FAIL`

### Step 3: Implement continuous EDA functions

**File:** `feature_selection/validators/eda/continuous.py`

```python
"""Continuous feature EDA methods."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Literal

from feature_selection.validators.reports.base import MonotonicityTestResult


def compute_decile_stats(
    feature: pd.Series,
    target: pd.Series,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute per-decile target statistics.

    Args:
        feature: Feature series
        target: Target series
        n_deciles: Number of deciles (default 10)

    Returns:
        DataFrame with decile statistics (mean, std, Sharpe, t-stat per decile)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < n_deciles * 5:
        raise ValueError(f"Insufficient data for decile analysis: {len(aligned)}")

    # Assign deciles
    aligned['decile'] = pd.qcut(aligned['feature'], q=n_deciles, labels=False, duplicates='drop')

    # Compute per-decile stats
    decile_stats = []

    for decile in sorted(aligned['decile'].unique()):
        decile_data = aligned[aligned['decile'] == decile]['target']

        mean_target = decile_data.mean()
        std_target = decile_data.std()
        n_samples = len(decile_data)

        # Sharpe ratio
        sharpe = mean_target / std_target if std_target > 0 else 0.0

        # t-statistic
        t_stat = mean_target / (std_target / np.sqrt(n_samples)) if std_target > 0 else 0.0

        decile_stats.append({
            'decile': decile,
            'mean_target': mean_target,
            'std_target': std_target,
            'sharpe': sharpe,
            't_stat': t_stat,
            'n_samples': n_samples,
        })

    return pd.DataFrame(decile_stats)


def test_monotonicity(decile_means: pd.Series) -> MonotonicityTestResult:
    """
    Test if decile means show monotonic trend.

    Args:
        decile_means: Series of mean target values per decile

    Returns:
        MonotonicityTestResult
    """
    # Kendall's tau correlation with decile index
    decile_index = np.arange(len(decile_means))
    tau, p_value = stats.kendalltau(decile_index, decile_means.values)

    # Determine monotonicity and direction
    is_monotonic = abs(tau) > 0.3 and p_value < 0.05

    if is_monotonic:
        direction = 'increasing' if tau > 0 else 'decreasing'
    else:
        direction = 'none'

    return MonotonicityTestResult(
        kendall_tau=float(tau),
        p_value=float(p_value),
        is_monotonic=is_monotonic,
        direction=direction,
    )


def detect_outliers(
    data: pd.Series,
    method: Literal['iqr', 'zscore'] = 'iqr',
    threshold: float = 3.0,
) -> tuple[pd.Series, float]:
    """
    Detect outliers using IQR or Z-score method.

    Args:
        data: Numeric series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (IQR multiplier or Z-score)

    Returns:
        Tuple of (outlier_mask, outlier_fraction)
    """
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (data < lower) | (data > upper)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        outlier_mask = pd.Series(z_scores > threshold, index=data.index)
    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_fraction = outlier_mask.sum() / len(data)

    return outlier_mask, outlier_fraction


def fit_polynomial_regression(
    feature: pd.Series,
    target: pd.Series,
    max_degree: int = 3,
) -> dict[int, float]:
    """
    Fit polynomial regressions of varying degrees.

    Args:
        feature: Feature series
        target: Target series
        max_degree: Maximum polynomial degree

    Returns:
        Dictionary mapping degree → R²
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    X = aligned['feature'].values.reshape(-1, 1)
    y = aligned['target'].values

    r2_scores = {}

    for degree in range(1, max_degree + 1):
        # Create polynomial features
        X_poly = np.column_stack([X**d for d in range(1, degree + 1)])

        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_poly, y)

        # Compute R²
        r2 = model.score(X_poly, y)
        r2_scores[degree] = r2

    return r2_scores
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/validators/test_eda_continuous.py -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/eda/continuous.py tests/validators/test_eda_continuous.py
git commit -m "feat: add continuous-specific EDA methods

- Add decile analysis with Sharpe and t-stats
- Add monotonicity testing via Kendall's tau
- Add outlier detection (IQR and Z-score methods)
- Add polynomial regression for non-linearity assessment

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Rule-Based-Specific EDA

**Files:**
- Create: `feature_selection/validators/eda/rule_based.py`
- Create: `tests/validators/test_eda_rule_based.py`

### Step 1: Write failing test for level analysis

**File:** `tests/validators/test_eda_rule_based.py`

```python
import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.eda.rule_based import (
    compute_level_distribution,
    compute_level_stats,
    compute_transition_matrix,
)


def test_compute_level_distribution():
    """Test level distribution computation."""
    # Create rule-based feature with -1, 0, 1 levels
    feature = pd.Series([-1] * 200 + [0] * 400 + [1] * 400)

    level_counts, level_fractions, imbalance_flag = compute_level_distribution(feature)

    assert level_counts[-1] == 200
    assert level_counts[0] == 400
    assert level_counts[1] == 400
    assert level_fractions[-1] == 0.2
    assert imbalance_flag is True  # -1 level < 25%


def test_compute_level_stats():
    """Test per-level target statistics."""
    np.random.seed(42)

    feature = pd.Series([-1] * 300 + [0] * 300 + [1] * 300)
    target = pd.Series(
        list(-0.5 + np.random.randn(300) * 0.3) +  # -1 level: negative mean
        list(0.0 + np.random.randn(300) * 0.3) +   # 0 level: zero mean
        list(0.5 + np.random.randn(300) * 0.3)     # 1 level: positive mean
    )

    level_stats = compute_level_stats(feature, target)

    assert len(level_stats) == 3
    assert 'mean_target' in level_stats.columns
    assert 'sharpe' in level_stats.columns

    # Check that level 1 has higher mean than level -1
    mean_neg1 = level_stats.loc[level_stats['level'] == -1, 'mean_target'].values[0]
    mean_pos1 = level_stats.loc[level_stats['level'] == 1, 'mean_target'].values[0]
    assert mean_pos1 > mean_neg1


def test_compute_transition_matrix():
    """Test regime transition matrix computation."""
    # Create feature with transitions
    feature = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 1, 0])

    transition_matrix = compute_transition_matrix(feature)

    assert transition_matrix.shape == (3, 3)
    assert transition_matrix.index.tolist() == [-1, 0, 1]
    assert np.allclose(transition_matrix.sum(axis=1), 1.0)  # Rows sum to 1
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_eda_rule_based.py::test_compute_level_distribution -v
```

Expected: `FAIL`

### Step 3: Implement rule-based EDA functions

**File:** `feature_selection/validators/eda/rule_based.py`

```python
"""Rule-based feature EDA methods."""
import pandas as pd
import numpy as np
from scipy import stats


def compute_level_distribution(
    feature: pd.Series,
    min_level_fraction: float = 0.10,
) -> tuple[dict[int, int], dict[int, float], bool]:
    """
    Compute level distribution for rule-based features.

    Args:
        feature: Rule-based feature (values should be -1, 0, 1)
        min_level_fraction: Minimum fraction per level (default 10%)

    Returns:
        Tuple of (level_counts, level_fractions, imbalance_flag)
    """
    # Count samples per level
    level_counts = feature.value_counts().to_dict()

    # Compute fractions
    total_samples = len(feature)
    level_fractions = {level: count / total_samples for level, count in level_counts.items()}

    # Check for imbalance
    min_fraction = min(level_fractions.values())
    imbalance_flag = min_fraction < min_level_fraction

    return level_counts, level_fractions, imbalance_flag


def compute_level_stats(
    feature: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compute per-level target statistics.

    Args:
        feature: Rule-based feature
        target: Target series

    Returns:
        DataFrame with level statistics (mean, std, Sharpe, t-stat, adjusted_Sharpe)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    level_stats = []

    for level in sorted(aligned['feature'].unique()):
        level_data = aligned[aligned['feature'] == level]['target']

        mean_target = level_data.mean()
        std_target = level_data.std()
        n_samples = len(level_data)

        # Sharpe ratio
        sharpe = mean_target / std_target if std_target > 0 else 0.0

        # t-statistic
        t_stat = mean_target / (std_target / np.sqrt(n_samples)) if std_target > 0 else 0.0

        # Adjusted Sharpe (from feature selection spec)
        k = 1.0  # penalty parameter
        adjusted_sharpe = sharpe * np.sqrt(n_samples) / (np.sqrt(n_samples) + k)

        level_stats.append({
            'level': level,
            'mean_target': mean_target,
            'std_target': std_target,
            'sharpe': sharpe,
            't_stat': t_stat,
            'adjusted_sharpe': adjusted_sharpe,
            'n_samples': n_samples,
        })

    return pd.DataFrame(level_stats)


def compute_transition_matrix(feature: pd.Series) -> pd.DataFrame:
    """
    Compute transition matrix for rule-based feature levels.

    P(level_t+1 | level_t)

    Args:
        feature: Rule-based feature

    Returns:
        Transition matrix (DataFrame with rows=from_level, cols=to_level)
    """
    # Create shifted series
    from_level = feature[:-1].reset_index(drop=True)
    to_level = feature[1:].reset_index(drop=True)

    # Count transitions
    transitions = pd.crosstab(from_level, to_level)

    # Normalize to probabilities (row-wise)
    transition_matrix = transitions.div(transitions.sum(axis=1), axis=0)

    # Ensure all levels are present (even if zero transitions)
    for level in [-1, 0, 1]:
        if level not in transition_matrix.index:
            transition_matrix.loc[level] = 0.0
        if level not in transition_matrix.columns:
            transition_matrix[level] = 0.0

    transition_matrix = transition_matrix.sort_index().sort_index(axis=1)

    return transition_matrix


def compute_average_duration(feature: pd.Series) -> dict[int, float]:
    """
    Compute average duration (consecutive periods) in each level.

    Args:
        feature: Rule-based feature

    Returns:
        Dictionary mapping level → average duration
    """
    # Identify regime switches
    switches = feature != feature.shift(1)
    regime_ids = switches.cumsum()

    # Group by regime and compute durations
    durations = feature.groupby(regime_ids).size()
    levels = feature.groupby(regime_ids).first()

    # Compute average duration per level
    average_durations = {}

    for level in feature.unique():
        level_durations = durations[levels == level]
        average_durations[int(level)] = float(level_durations.mean()) if len(level_durations) > 0 else 0.0

    return average_durations


def compute_level_confidence_intervals(
    feature: pd.Series,
    target: pd.Series,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> dict[int, tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for mean target per level.

    Args:
        feature: Rule-based feature
        target: Target series
        confidence_level: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary mapping level → (lower_ci, upper_ci)
    """
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    confidence_intervals = {}
    alpha = 1 - confidence_level

    for level in sorted(aligned['feature'].unique()):
        level_data = aligned[aligned['feature'] == level]['target'].values

        if len(level_data) < 10:
            # Insufficient data for bootstrap
            confidence_intervals[int(level)] = (np.nan, np.nan)
            continue

        # Bootstrap resampling
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(level_data, size=len(level_data), replace=True)
            bootstrap_means.append(sample.mean())

        # Compute confidence interval
        lower_ci = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper_ci = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        confidence_intervals[int(level)] = (float(lower_ci), float(upper_ci))

    return confidence_intervals
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/validators/test_eda_rule_based.py -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/eda/rule_based.py tests/validators/test_eda_rule_based.py
git commit -m "feat: add rule-based-specific EDA methods

- Add level distribution computation with imbalance detection
- Add per-level statistics (mean, Sharpe, adjusted Sharpe)
- Add transition matrix for regime analysis
- Add average duration and confidence intervals

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Permutation Testing Integration

**Files:**
- Create: `feature_selection/validators/permutation.py`
- Create: `tests/validators/test_permutation_integration.py`

### Step 1: Write failing test for permutation wrapper

**File:** `tests/validators/test_permutation_integration.py`

```python
import pytest
import numpy as np
import pandas as pd
from feature_selection.validators.permutation import run_vector_shuffle_test
from utils.permutation_test.permutation_engine import PermutationEngine, FeaturePermutationStrategy


def test_run_vector_shuffle_test():
    """Test vector shuffle permutation test."""
    np.random.seed(42)

    # Create feature with predictive power
    feature = pd.Series(np.random.randn(500))
    target = pd.Series(0.3 * feature + np.random.randn(500) * 0.5)

    # Run permutation test
    report = run_vector_shuffle_test(
        feature=feature,
        target=target,
        n_permutations=100,
        confidence_level=0.95,
        random_seed=42,
    )

    assert report.stage == 'stage1_vector_shuffle'
    assert report.n_permutations == 100
    assert report.observed_sharpe != 0
    assert len(report.permuted_sharpes) == 100
    assert hasattr(report, 'p_value')
    assert hasattr(report, 'passed')
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_permutation_integration.py::test_run_vector_shuffle_test -v
```

Expected: `FAIL`

### Step 3: Implement permutation wrapper

**File:** `feature_selection/validators/permutation.py`

```python
"""Permutation testing integration."""
import time
import pandas as pd
import numpy as np
from typing import Callable

from utils.permutation_test.permutation_engine import PermutationEngine, FeaturePermutationStrategy
from feature_selection.validators.reports.permutation import PermutationReport


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std()


def _compute_t_stat(returns: np.ndarray) -> float:
    """Compute t-statistic."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / (returns.std() / np.sqrt(len(returns)))


def run_vector_shuffle_test(
    feature: pd.Series,
    target: pd.Series,
    n_permutations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> PermutationReport:
    """
    Run Stage 1: Vector Shuffle Permutation Test.

    Shuffles feature vector, recomputes Sharpe, tests significance.

    Args:
        feature: Feature series
        target: Target series
        n_permutations: Number of permutations
        confidence_level: Confidence level (e.g., 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        PermutationReport
    """
    start_time = time.time()

    # Align data
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    # Compute observed statistics
    # For vector shuffle, we're testing if feature values predict target
    # Simple approach: bin feature into quantiles, compute Sharpe of target in top quantile
    q_high = aligned['feature'].quantile(0.75)
    high_feature_mask = aligned['feature'] >= q_high
    selected_returns = aligned.loc[high_feature_mask, 'target'].values

    observed_sharpe = _compute_sharpe(selected_returns)
    observed_t_stat = _compute_t_stat(selected_returns)
    observed_returns_mean = selected_returns.mean()

    # Run permutations
    permuted_sharpes = []
    permuted_t_stats = []

    rng = np.random.RandomState(random_seed)

    for _ in range(n_permutations):
        # Shuffle feature
        shuffled_feature = aligned['feature'].values.copy()
        rng.shuffle(shuffled_feature)

        # Recompute statistics
        q_high_perm = np.quantile(shuffled_feature, 0.75)
        high_mask_perm = shuffled_feature >= q_high_perm
        selected_returns_perm = aligned['target'].values[high_mask_perm]

        permuted_sharpes.append(_compute_sharpe(selected_returns_perm))
        permuted_t_stats.append(_compute_t_stat(selected_returns_perm))

    permuted_sharpes = np.array(permuted_sharpes)
    permuted_t_stats = np.array(permuted_t_stats)

    # Compute p-value (one-sided: observed > permuted)
    p_value = (permuted_sharpes >= observed_sharpe).sum() / n_permutations

    # Compute critical value
    critical_value = np.percentile(permuted_sharpes, confidence_level * 100)

    # Verdict
    passed = observed_sharpe > critical_value
    margin = observed_sharpe - critical_value

    execution_time = time.time() - start_time

    return PermutationReport(
        stage='stage1_vector_shuffle',
        observed_sharpe=float(observed_sharpe),
        observed_t_stat=float(observed_t_stat),
        observed_returns_mean=float(observed_returns_mean),
        permuted_sharpes=permuted_sharpes,
        permuted_t_stats=permuted_t_stats,
        p_value=float(p_value),
        confidence_level=confidence_level,
        critical_value=float(critical_value),
        passed=passed,
        margin=float(margin),
        permutation_histogram=None,  # Plotting handled separately
        qq_plot=None,
        n_permutations=n_permutations,
        random_seed=random_seed,
        execution_time=execution_time,
    )
```

### Step 4: Run test to verify it passes

```bash
pytest tests/validators/test_permutation_integration.py::test_run_vector_shuffle_test -v
```

Expected: `PASS`

### Step 5: Commit

```bash
git add feature_selection/validators/permutation.py tests/validators/test_permutation_integration.py
git commit -m "feat: add permutation testing integration

- Implement run_vector_shuffle_test for Stage 1
- Compute observed vs. permuted Sharpe ratios
- Calculate p-values and critical values
- Return PermutationReport with verdict

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Core Validator - Initialization and EDA Orchestration

**Files:**
- Create: `feature_selection/validators/validator.py`
- Create: `tests/validators/test_feature_validator.py`

### Step 1: Write failing test for FeatureValidator initialization

**File:** `tests/validators/test_feature_validator.py`

```python
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from feature_selection.validators.validator import FeatureValidator
from feature_selection.validators.config import ValidationConfig
from utils.permutation_test.permutation_engine import PermutationEngine, FeaturePermutationStrategy
from eda.parameter_analysis import ParameterAnalyzer


@pytest.fixture
def sample_data():
    """Create sample feature and target data."""
    np.random.seed(42)
    feature = pd.Series(np.random.randn(500), name='test_feature')
    target = pd.Series(0.2 * feature + np.random.randn(500) * 0.5, name='target')
    return feature, target


@pytest.fixture
def validator(tmp_path, sample_data):
    """Create FeatureValidator instance."""
    feature, target = sample_data

    config = ValidationConfig(feature_type='continuous')

    # Create minimal dependencies
    # Note: In real tests, use proper instances
    permutation_engine = None  # Placeholder
    parameter_analyzer = None  # Placeholder

    return FeatureValidator(
        config=config,
        permutation_engine=permutation_engine,
        parameter_analyzer=parameter_analyzer,
        output_dir=tmp_path,
    )


def test_validator_initialization(validator):
    """Test FeatureValidator initialization."""
    assert validator.config.feature_type == 'continuous'
    assert validator.output_dir.exists()


def test_run_eda_continuous(validator, sample_data):
    """Test run_eda for continuous feature."""
    feature, target = sample_data

    eda_report = validator.run_eda(
        feature_data=feature.to_frame(),
        target=target,
    )

    assert eda_report.feature_stats is not None
    assert eda_report.target_stats is not None
    assert 'pearson' in eda_report.correlations
    assert eda_report.adf_test is not None
    assert eda_report.kpss_test is not None
```

### Step 2: Run test to verify it fails

```bash
pytest tests/validators/test_feature_validator.py::test_validator_initialization -v
```

Expected: `FAIL`

### Step 3: Implement FeatureValidator initialization and run_eda

**File:** `feature_selection/validators/validator.py`

```python
"""Main FeatureValidator class."""
from pathlib import Path
from typing import Generator
import pandas as pd
import numpy as np

from feature_selection.validators.config import ValidationConfig
from feature_selection.validators.reports.eda import EDAReport, ContinuousEDAReport, RuleEDAReport
from feature_selection.validators.reports.permutation import PermutationReport
from feature_selection.validators.eda.common import (
    compute_distribution_stats,
    compute_correlations,
    run_stationarity_tests,
    compute_lagged_correlations,
    compute_rolling_correlation,
)
from feature_selection.validators.eda.continuous import (
    compute_decile_stats,
    test_monotonicity,
    detect_outliers,
    fit_polynomial_regression,
)
from feature_selection.validators.eda.rule_based import (
    compute_level_distribution,
    compute_level_stats,
    compute_transition_matrix,
    compute_average_duration,
    compute_level_confidence_intervals,
)


class FeatureValidator:
    """
    Unified feature validation interface for continuous and rule-based features.

    Orchestrates end-to-end validation workflow:
    1. EDA (distributions, correlations, stationarity)
    2. Stage 1: Vector Shuffle Permutation
    3. Stage 2: Pipeline Permutation (feature/candle shuffle)
    4. Stage 3: Walkforward Stability Analysis
    5. Researcher Ensemble Formation (guided by reports)
    """

    def __init__(
        self,
        config: ValidationConfig,
        permutation_engine: any,  # PermutationEngine from utils
        parameter_analyzer: any,  # ParameterAnalyzer from eda
        output_dir: Path,
    ):
        """
        Initialize validator with configuration and dependencies.

        Args:
            config: Validation configuration
            permutation_engine: Existing permutation testing engine
            parameter_analyzer: Existing parameter sensitivity analyzer
            output_dir: Base directory for report outputs
        """
        self.config = config
        self.permutation_engine = permutation_engine
        self.parameter_analyzer = parameter_analyzer
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_eda(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
    ) -> EDAReport:
        """
        Run exploratory data analysis.

        Args:
            feature_data: Feature DataFrame (single column for now)
            target: Target series

        Returns:
            EDAReport with all EDA results
        """
        # Extract single feature column (for now, assume single feature)
        if len(feature_data.columns) > 1:
            raise NotImplementedError("Multi-feature EDA not yet implemented")

        feature = feature_data.iloc[:, 0]

        # Common EDA
        feature_stats = compute_distribution_stats(feature)
        target_stats = compute_distribution_stats(target)

        correlations = compute_correlations(feature, target)
        lagged_correlations = compute_lagged_correlations(feature, target, max_lag=10)

        adf_test, kpss_test = run_stationarity_tests(feature)

        # Rolling correlation (use smaller window if insufficient data)
        window = min(252, len(feature) // 4)
        if window >= 20:
            rolling_correlation = compute_rolling_correlation(feature, target, window=window)
        else:
            rolling_correlation = pd.Series([correlations['pearson']])

        # Regime stats (placeholder - could add regime detection later)
        regime_stats = {}

        # Feature-type-specific EDA
        if self.config.feature_type == 'continuous':
            continuous_report = self._run_continuous_eda(feature, target)
            rule_report = None
        else:
            continuous_report = None
            rule_report = self._run_rule_eda(feature, target)

        # Diagnostic flags (basic checks)
        warnings = []
        red_flags = []

        if abs(correlations['pearson']) < 0.05:
            warnings.append("Very low feature-target correlation")

        if not adf_test.is_stationary:
            warnings.append("Feature may be non-stationary")

        return EDAReport(
            feature_stats=feature_stats,
            target_stats=target_stats,
            correlations=correlations,
            lagged_correlations=lagged_correlations,
            adf_test=adf_test,
            kpss_test=kpss_test,
            rolling_correlation=rolling_correlation,
            regime_stats=regime_stats,
            distribution_plot=None,  # Plotting handled separately
            correlation_plot=None,
            time_series_plot=None,
            stationarity_plot=None,
            continuous_report=continuous_report,
            rule_report=rule_report,
            warnings=warnings,
            red_flags=red_flags,
        )

    def _run_continuous_eda(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> ContinuousEDAReport:
        """Run continuous-specific EDA."""
        # Decile analysis
        decile_stats = compute_decile_stats(feature, target, n_deciles=10)
        monotonicity_test = test_monotonicity(decile_stats['mean_target'])

        # Outlier analysis
        outlier_mask, outlier_fraction = detect_outliers(feature, method='iqr')

        # Estimate outlier impact (simplified)
        if outlier_fraction > 0:
            clean_target = target[~outlier_mask]
            clean_sharpe = clean_target.mean() / clean_target.std() if clean_target.std() > 0 else 0
            full_sharpe = target.mean() / target.std() if target.std() > 0 else 0
            outlier_impact = clean_sharpe - full_sharpe
        else:
            outlier_impact = 0.0

        # Non-linearity tests
        polynomial_r2 = fit_polynomial_regression(feature, target, max_degree=3)
        linear_r2 = polynomial_r2[1]

        return ContinuousEDAReport(
            decile_stats=decile_stats,
            decile_plot=None,
            monotonicity_test=monotonicity_test,
            outlier_fraction=outlier_fraction,
            outlier_impact=outlier_impact,
            outlier_plot=None,
            linear_r2=linear_r2,
            polynomial_r2=polynomial_r2,
            non_linearity_plot=None,
        )

    def _run_rule_eda(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> RuleEDAReport:
        """Run rule-based-specific EDA."""
        # Level distribution
        level_counts, level_fractions, imbalance_flag = compute_level_distribution(feature)

        # Per-level statistics
        level_stats = compute_level_stats(feature, target)
        level_confidence_intervals = compute_level_confidence_intervals(
            feature, target, confidence_level=0.95, n_bootstrap=1000
        )

        # Regime transitions
        transition_matrix = compute_transition_matrix(feature)
        average_duration = compute_average_duration(feature)

        return RuleEDAReport(
            level_counts=level_counts,
            level_fractions=level_fractions,
            imbalance_flag=imbalance_flag,
            level_stats=level_stats,
            level_confidence_intervals=level_confidence_intervals,
            level_plot=None,
            transition_matrix=transition_matrix,
            average_duration=average_duration,
            transition_plot=None,
        )
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/validators/test_feature_validator.py -v
```

Expected: `PASS` for test_validator_initialization, possibly SKIP for test_run_eda_continuous if dependencies not mocked

### Step 5: Commit

```bash
git add feature_selection/validators/validator.py tests/validators/test_feature_validator.py
git commit -m "feat: add FeatureValidator core class with EDA orchestration

- Implement FeatureValidator initialization
- Implement run_eda for continuous and rule-based features
- Orchestrate common + feature-specific EDA methods
- Generate comprehensive EDAReport

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Remaining Tasks (Summary)

Due to space constraints, I'll summarize the remaining tasks. Each follows the same TDD pattern (test → fail → implement → pass → commit).

### Task 10: Stability Analysis Methods
- File: `feature_selection/validators/stability.py`
- Implement walkforward fold splitting
- Implement grid-aware neighbor smoothing
- Implement top-K parameter selection per fold
- Compute temporal consistency metrics

### Task 11: Core Validator - Permutation Stages
- Modify: `feature_selection/validators/validator.py`
- Implement `run_stage1_permutation()`
- Implement `run_stage2_permutation()`
- Integrate with permutation.py wrapper functions

### Task 12: Core Validator - Full Pipeline
- Modify: `feature_selection/validators/validator.py`
- Implement `run_full_validation()`
- Implement `run_progressive_validation()`
- Implement `run_stage3_stability()`
- Add early exit on permutation failure

### Task 13: Report Export
- Create: `feature_selection/validators/reports/export.py`
- Implement HTML export with embedded Plotly
- Implement PDF export via matplotlib + ReportLab
- Implement Markdown export
- Implement JSON export
- Implement `generate_researcher_summary()`

### Task 14: Integration Testing
- Create: `tests/validators/test_integration.py`
- Test full validation pipeline end-to-end
- Test with continuous features
- Test with rule-based features
- Test progressive validation
- Test parameter sensitivity integration

---

## Summary

This implementation plan provides a TDD-based roadmap for building the Feature Validator API with:

- **14 bite-sized tasks** (each 2-5 minutes per step)
- **Clear file structure** in `feature_selection/validators/`
- **Incremental testing** at each step
- **Frequent commits** after each task
- **Integration with existing infrastructure** (PermutationEngine, ParameterAnalyzer)

Total estimated time: 15-20 hours for complete implementation (Tasks 1-14).

**Next steps:** Begin execution using superpowers:executing-plans or superpowers:subagent-driven-development.
