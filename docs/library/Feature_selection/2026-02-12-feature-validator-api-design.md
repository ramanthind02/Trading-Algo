# Feature Validator API Design Specification

**Date:** 2026-02-12
**Status:** Approved
**Replaces:** `eda/feature_explorer.py`
**Integration:** `FeatureExtractor → FeatureValidator → Reports`

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Core Components & API Surface](#2-core-components--api-surface)
3. [EDA & Parameter Analysis Methods](#3-eda--parameter-analysis-methods)
4. [Binning Diagnostics & Feature-Type Specific Methods](#4-binning-diagnostics--feature-type-specific-methods)
5. [Report Object Structures](#5-report-object-structures)
6. [Data Flow & Integration Patterns](#6-data-flow--integration-patterns)

---

## 1. Overview & Architecture

### 1.1 Purpose

The **Feature Validator API** provides a unified interface for end-to-end feature analysis and validation in the Trading-Algo systematic trading framework. It replaces the existing `FeatureExplorer` class with a comprehensive validation system that:

- Integrates exploratory data analysis (EDA) and permutation-based validation
- Handles both continuous and rule-based features through a single interface
- Orchestrates the complete 3-stage validation pipeline (vector shuffle, pipeline permutation, walkforward stability)
- Supports rigorous parameter sensitivity testing (1D through 4D)
- Generates comprehensive reports for researcher decision-making
- Enforces hypothesis-driven testing principles (not data mining)

### 1.2 Design Philosophy

**Hypothesis-Driven Validation:**
- Pre-specified thresholds and criteria before validation
- No parameter optimization—only validation of pre-committed choices
- Permutation testing as statistical gatekeeping
- Researcher-driven ensemble formation based on stability analysis

**Progressive Workflow:**
- Accumulative reporting: each stage adds to a unified `ValidationReport`
- Flexibility: run full pipeline or individual stages
- Early exit on failure: stop when permutation tests fail
- Documentation: automatic capture of decisions and thresholds

**Adaptive Behavior:**
- Single unified class (`FeatureValidator`) with feature-type parameter
- Feature-type-specific logic encapsulated internally
- Consistent API surface regardless of feature type
- Type-safe parameter handling

### 1.3 Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FeatureValidator                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Core Orchestration                                   │  │
│  │  - feature_type: Literal['continuous', 'rule_based']  │  │
│  │  - ValidationReport accumulation                      │  │
│  │  - Stage sequencing & early exit                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     EDA      │  │  Validation  │  │  Parameter   │      │
│  │   Methods    │  │    Engine    │  │   Analysis   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                  ┌─────────▼──────────┐                      │
│                  │ ValidationReport   │                      │
│                  └────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   PermutationEngine   ParameterAnalyzer    PlottingFunctions
   (existing)          (existing)           (existing)
```

**Key Principles:**
- **Composition over inheritance:** Single class wraps existing components
- **Dependency injection:** Pass in PermutationEngine, ParameterAnalyzer as dependencies
- **Separation of concerns:** Validation logic vs. plotting vs. report generation
- **Type safety:** NewTypes for domain primitives, strict mypy compliance

---

## 2. Core Components & API Surface

### 2.1 Main Class: FeatureValidator

```python
from typing import Literal, Protocol
from dataclasses import dataclass
from pathlib import Path

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
        permutation_engine: PermutationEngine,
        parameter_analyzer: ParameterAnalyzer,
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
        ...

    # === High-Level Workflow Methods ===

    def run_full_validation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        params_grid: dict[str, list] | None = None,
    ) -> ValidationReport:
        """
        Execute complete validation pipeline: EDA → Stage 1 → Stage 2 → Stage 3.

        Returns accumulated ValidationReport. Stops early if permutation tests fail.
        """
        ...

    def run_progressive_validation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        params_grid: dict[str, list] | None = None,
    ) -> Generator[tuple[str, StageReport], None, None]:
        """
        Yield stage-by-stage results for interactive workflow.

        Yields:
            (stage_name, stage_report) tuples as each stage completes
        """
        ...

    # === Modular Stage Access ===

    def run_eda(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
    ) -> EDAReport:
        """
        Run exploratory data analysis only.

        Includes:
        - Distribution analysis (feature & target)
        - Correlation analysis (feature-target, inter-feature if multi-param)
        - Stationarity tests (ADF, KPSS)
        - Temporal stability plots
        - [Continuous only] Decile analysis
        - [Rule-based only] Level distribution & regime analysis
        """
        ...

    def run_stage1_permutation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
    ) -> PermutationReport:
        """
        Stage 1: Vector Shuffle Permutation Test.

        Quick filter: shuffle feature vector, recompute Sharpe, test significance.
        """
        ...

    def run_stage2_permutation(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        candles: pd.DataFrame,  # For candle_shuffle
        permutation_type: Literal['feature_shuffle', 'candle_shuffle'],
    ) -> PermutationReport:
        """
        Stage 2: Pipeline Permutation Test.

        - feature_shuffle: Shuffle feature, recompute through binning pipeline
        - candle_shuffle: Shuffle candles, recompute feature → binning
        """
        ...

    def run_stage3_stability(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        params_grid: dict[str, list],
        walkforward_folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    ) -> StabilityReport:
        """
        Stage 3: Walkforward Stability Analysis.

        - Fit all param combos on each fold
        - Apply grid-aware neighbor smoothing
        - Identify top-K params per fold
        - Report temporal consistency of best regions
        """
        ...

    # === Parameter Sensitivity Methods ===

    def analyze_parameter_sensitivity(
        self,
        feature_data: pd.DataFrame,
        target: pd.Series,
        params_grid: dict[str, list],
        dimensions: Literal[1, 2, 3, 4] = 1,
    ) -> ParameterReport:
        """
        Analyze parameter sensitivity in 1D through 4D.

        Wraps existing ParameterAnalyzer with appropriate visualizations:
        - 1D: Line plots with confidence bands
        - 2D: Heatmaps + contour plots
        - 3D: Interactive Plotly 3D surface
        - 4D: Interactive Plotly 4D scatter (color/size encoding)
        """
        ...

    def compute_robustness_metrics(
        self,
        params_grid: dict[str, list],
        objective_values: pd.DataFrame,
    ) -> RobustnessMetrics:
        """
        Compute parameter robustness metrics.

        Returns:
        - Variance contribution per parameter
        - Consistency score (% folds where param in top-K)
        - Risk score (max drawdown sensitivity)
        - Stability ratio (smoothed/raw objective)
        """
        ...
```

### 2.2 Report Generation & Export

```python
class FeatureValidator:
    # ... (continued)

    def export_report(
        self,
        report: ValidationReport,
        format: Literal['html', 'pdf', 'markdown', 'json'],
        output_path: Path,
    ) -> Path:
        """
        Export validation report to specified format.

        - HTML: Interactive with embedded Plotly charts
        - PDF: Static report via matplotlib + ReportLab
        - Markdown: Text-based with image links
        - JSON: Structured data for programmatic access
        """
        ...

    def generate_researcher_summary(
        self,
        report: ValidationReport,
    ) -> str:
        """
        Generate concise text summary for researcher decision-making.

        Example output:
        ```
        Feature: RSI_lookback_14
        Type: continuous

        ✓ Stage 1: Vector Shuffle PASSED (p=0.003)
        ✓ Stage 2: Pipeline Permutation PASSED (p=0.012)
        ✓ Stage 3: Walkforward Stability PASSED

        Top Stable Params (appear in 4/5 folds):
        - lookback=14: Sharpe=0.82, stability_ratio=0.91
        - lookback=13: Sharpe=0.79, stability_ratio=0.89
        - lookback=15: Sharpe=0.77, stability_ratio=0.88

        Recommended Ensemble: [13, 14, 15]
        Rationale: Tight cluster in param space, consistent across time
        ```
        """
        ...
```

---

## 3. EDA & Parameter Analysis Methods

### 3.1 Exploratory Data Analysis

**Common to Both Feature Types:**

```python
def run_eda(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
) -> EDAReport:
    """
    Comprehensive EDA covering:

    A) Distribution Analysis
       - Feature distribution (histogram, KDE, Q-Q plot)
       - Target distribution
       - Joint distribution (feature vs. target scatter with density)

    B) Correlation Analysis
       - Feature-target correlation (Pearson, Spearman, Kendall)
       - If multi-parameter: inter-feature correlation matrix
       - Lagged correlation (feature_t vs. target_t+k)

    C) Stationarity Tests
       - Augmented Dickey-Fuller (ADF) test
       - KPSS test (trend vs. level stationarity)
       - Rolling statistics (mean, std) over time

    D) Temporal Stability
       - Rolling correlation (feature-target) over time windows
       - Regime detection (identify market regimes, compute per-regime stats)
       - Temporal distribution (feature distribution by year/quarter)

    E) Time Series Visualization
       - Feature time series plot with target overlay
       - Residual plots (feature vs. target residuals after linear fit)
       - Event study plots (feature behavior around extreme target moves)

    Returns:
        EDAReport with all plots, statistics, and diagnostic flags
    """
    ...
```

**Continuous-Specific EDA:**

```python
def _run_continuous_eda(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
) -> ContinuousEDAReport:
    """
    Continuous feature EDA extensions:

    - Decile Analysis:
      * Sort feature into 10 deciles
      * Compute per-decile target statistics (mean, std, Sharpe, t-stat)
      * Plot decile bars with error bands
      * Monotonicity test (is there a trend across deciles?)

    - Outlier Analysis:
      * Identify outliers (±3 SD or IQR method)
      * Report % outliers, visualize outlier impact on target

    - Non-linearity Tests:
      * Fit linear regression (feature → target)
      * Fit polynomial regression (degree 2, 3)
      * Compare R² to assess non-linearity
      * Plot fitted curves

    Returns:
        ContinuousEDAReport with decile plots, outlier diagnostics
    """
    ...
```

**Rule-Based-Specific EDA:**

```python
def _run_rule_eda(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
) -> RuleEDAReport:
    """
    Rule-based feature EDA extensions:

    - Level Distribution:
      * Report % samples in each level (-1, 0, +1)
      * Flag imbalance if any level < 10% of samples

    - Per-Level Statistics:
      * Mean, std, Sharpe, t-stat per level
      * Confidence intervals for each level's target mean
      * Visual: bar chart with error bars

    - Regime Transition Analysis:
      * Transition matrix (prob of moving from level i to level j)
      * Average duration in each level
      * Identify "sticky" regimes (long stays in one level)

    - Grid Report Integration:
      * If params_grid provided, generate rule_based_grid_report
      * Per-metric threshold pass rates
      * Overall robustness verdict (ROBUST/FRAGILE)
      * Best/worst parameter permutations
      * Parameter sensitivity (variance contribution)

    Returns:
        RuleEDAReport with level stats, transition matrix, grid report
    """
    ...
```

### 3.2 Parameter Sensitivity Analysis

**Integration with Existing ParameterAnalyzer:**

```python
def analyze_parameter_sensitivity(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    params_grid: dict[str, list],
    dimensions: Literal[1, 2, 3, 4] = 1,
) -> ParameterReport:
    """
    Wraps existing ParameterAnalyzer with appropriate visualizations.

    1D Analysis (single parameter):
    - Line plot: parameter value vs. objective (Sharpe, t-stat, etc.)
    - Confidence bands (bootstrap or analytical)
    - Mark stable regions (where smoothed objective ~ raw objective)

    2D Analysis (two parameters):
    - Heatmap: parameter_1 vs. parameter_2, color = objective
    - Contour plot: level curves of objective
    - Mark stable regions (high stability ratio)

    3D Analysis (three parameters):
    - Plotly 3D surface: param_1, param_2, param_3 → objective
    - Interactive rotation, slicing
    - Color-code by stability ratio

    4D Analysis (four parameters):
    - Plotly 4D scatter: param_1, param_2, param_3, param_4
    - Encode using x, y, z, color, size
    - Filter by objective threshold to show only viable regions

    Delegates to:
    - ParameterAnalyzer.analyze_parameter() (1D)
    - ParameterAnalyzer.analyze_2d_parameters() (2D)
    - ParameterAnalyzer.analyze_nd_parameters() (3D, 4D)
    - parameter_plots.plot_parameter_sensitivity() (1D)
    - parameter_plots.plot_2d_parameter_surface() (2D)
    - parameter_plots.plot_3d_parameter_interactive() (3D)
    - parameter_plots.plot_4d_parameter_interactive() (4D)

    Returns:
        ParameterReport with plots, robustness metrics, stability analysis
    """
    ...
```

**Robustness Metrics:**

```python
def compute_robustness_metrics(
    self,
    params_grid: dict[str, list],
    objective_values: pd.DataFrame,
) -> RobustnessMetrics:
    """
    Compute parameter robustness metrics.

    Metrics:
    1. Variance Contribution:
       - Per-parameter variance explained in objective
       - Identifies which params are most influential

    2. Consistency Score:
       - % of walkforward folds where param combo in top-K
       - High consistency = stable over time

    3. Risk Score:
       - Max drawdown sensitivity to parameter changes
       - Low risk = robust to param perturbations

    4. Stability Ratio:
       - smoothed_objective / raw_objective
       - High ratio (>0.9) = stable neighborhood
       - Low ratio (<0.5) = isolated peak (suspicious)

    Delegates to existing ParameterAnalyzer.compute_robustness_metrics()

    Returns:
        RobustnessMetrics dataclass with per-param scores
    """
    ...
```

---

## 4. Binning Diagnostics & Feature-Type Specific Methods

### 4.1 Continuous Feature Binning Diagnostics

```python
def analyze_binning_quality(
    self,
    feature_data: pd.Series,
    target: pd.Series,
    n_bins: int = 15,
) -> BinningDiagnostics:
    """
    Analyze quality of quantile binning for continuous features.

    Diagnostics:
    1. Bin Boundaries & Sample Counts:
       - Report bin edges (quantiles)
       - Report samples per bin (should be roughly equal for quantile binning)
       - Flag bins with < 5% of samples

    2. Per-Bin Statistics:
       - Mean, std, Sharpe, t-stat per bin
       - Plot: bin index vs. target mean (with error bars)
       - Monotonicity test: is there a trend across bins?

    3. Bin Merging Analysis:
       - Show which bins get merged into contiguous regions
       - Report region statistics (Sharpe, adjusted Sharpe)
       - Visualize region boundaries overlaid on bin plot

    4. Position Multiplier Distribution:
       - After applying threshold filter + risk scaling
       - Plot histogram of final position multipliers
       - Report % of samples in each multiplier range

    5. Sensitivity to n_bins:
       - Vary n_bins (e.g., 10, 15, 20, 25)
       - Plot: n_bins vs. final Sharpe
       - Identify stable range (where Sharpe plateaus)

    Returns:
        BinningDiagnostics with plots, bin statistics, stability analysis
    """
    ...

def plot_binning_pipeline(
    self,
    feature_data: pd.Series,
    target: pd.Series,
    n_bins: int = 15,
) -> Figure:
    """
    Visualize complete binning pipeline:

    Panel 1: Raw feature distribution (histogram)
    Panel 2: Bin boundaries overlaid on feature histogram
    Panel 3: Per-bin target mean with error bars
    Panel 4: Contiguous regions after merging (color-coded)
    Panel 5: Final position multipliers (histogram)

    Returns:
        Matplotlib Figure with 5 subplots
    """
    ...
```

### 4.2 Rule-Based Feature Diagnostics

```python
def analyze_rule_levels(
    self,
    feature_data: pd.Series,
    target: pd.Series,
) -> RuleDiagnostics:
    """
    Analyze rule-based feature levels (-1, 0, +1).

    Diagnostics:
    1. Level Distribution:
       - Report % samples in each level
       - Flag imbalance (any level < 10%)
       - Plot bar chart

    2. Per-Level Performance:
       - Mean, std, Sharpe, t-stat, adjusted Sharpe per level
       - Confidence intervals (bootstrap)
       - Plot: level vs. target mean (with error bars)

    3. Threshold Filtering:
       - Apply t_threshold (e.g., 2.0) to filter out weak levels
       - Report which levels pass/fail
       - Show impact on final Sharpe

    4. Position Multiplier Scaling:
       - After applying risk scaling formula
       - Plot distribution of final position multipliers
       - Report % samples in each multiplier range

    5. Regime Analysis:
       - Identify market regimes (bull/bear/sideways)
       - Compute per-regime, per-level statistics
       - Flag regime-dependent performance

    Returns:
        RuleDiagnostics with plots, level statistics, regime analysis
    """
    ...

def analyze_rule_grid(
    self,
    params_grid: dict[str, list],
    results: pd.DataFrame,
) -> RuleGridReport:
    """
    Generate rule-based grid performance report.

    Per rule_based_grid_report_specs.md:

    1. Per-Metric Threshold Pass Rates:
       - For each metric (Sharpe, t-stat, adjusted_Sharpe, etc.)
       - Report % of param combos that pass threshold
       - Plot heatmap of pass rates across param space

    2. Overall Robustness Verdict:
       - ROBUST: >80% of combos pass all thresholds
       - MODERATE: 50-80% pass
       - FRAGILE: <50% pass

    3. Best/Worst Permutations:
       - Top 5 param combos by composite score
       - Bottom 5 param combos
       - Visualize in param space

    4. Parameter Sensitivity:
       - Variance contribution per parameter
       - Identify which params matter most
       - Plot sensitivity bars

    5. Low-Sample Handling:
       - Flag param combos with < min_samples in any level
       - Report % of combos flagged
       - Suggest increasing sample size or reducing param grid

    Returns:
        RuleGridReport with verdict, pass rates, sensitivity analysis
    """
    ...
```

### 4.3 Visualization Strategy

**Matplotlib (Statistical Plots):**
- Distribution plots (histograms, KDE, Q-Q)
- Correlation matrices (heatmaps)
- Time series plots (feature, target, residuals)
- Bin diagnostics (bar charts with error bars)
- Decile analysis (bar charts)
- Per-level statistics (bar charts)

**Plotly (Interactive Parameter Analysis):**
- 1D parameter sensitivity (line plots with hover)
- 2D parameter heatmaps (interactive hover, zoom)
- 3D parameter surfaces (rotate, slice)
- 4D parameter scatter (filter, color-code)
- Grid reports (interactive tables, filterable)

**Rationale:**
- Matplotlib: Publication-quality static plots, PDF export
- Plotly: Interactive exploration for researchers during validation

---

## 5. Report Object Structures

### 5.1 ValidationReport (Top-Level)

```python
@dataclass(frozen=True)
class ValidationReport:
    """
    Accumulated report from all validation stages.

    Progressive accumulation:
    - Start with EDAReport
    - Add Stage1Report (vector shuffle)
    - Add Stage2Report (pipeline permutation)
    - Add Stage3Report (walkforward stability)
    - Optionally add ParameterReport (sensitivity analysis)
    """

    feature_name: str
    feature_type: FeatureType
    timestamp: datetime

    # Stage reports (optional, added progressively)
    eda_report: EDAReport | None = None
    stage1_report: PermutationReport | None = None
    stage2_report: PermutationReport | None = None
    stage3_report: StabilityReport | None = None
    parameter_report: ParameterReport | None = None

    # Overall validation status
    validation_status: Literal['passed', 'failed', 'incomplete']
    failure_stage: str | None = None  # e.g., "Stage 1: Vector Shuffle"

    # Researcher notes (filled manually after review)
    researcher_notes: str = ""
    ensemble_decision: list[dict[str, Any]] | None = None  # Selected param combos

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        ...

    def to_markdown(self) -> str:
        """Generate markdown report."""
        ...

    def to_html(self) -> str:
        """Generate HTML report with embedded plots."""
        ...
```

### 5.2 EDAReport

```python
@dataclass(frozen=True)
class EDAReport:
    """
    Exploratory data analysis results.
    """

    # Distribution statistics
    feature_stats: DescriptiveStats  # mean, std, skew, kurtosis, etc.
    target_stats: DescriptiveStats

    # Correlation analysis
    correlations: dict[str, float]  # {'pearson': 0.15, 'spearman': 0.18, ...}
    lagged_correlations: pd.Series  # correlation at different lags

    # Stationarity tests
    adf_test: ADFTestResult  # statistic, p-value, critical_values
    kpss_test: KPSSTestResult

    # Temporal stability
    rolling_correlation: pd.Series  # time series of rolling correlation
    regime_stats: dict[str, DescriptiveStats]  # per-regime statistics

    # Plots (stored as Figure objects or file paths)
    distribution_plot: Figure | Path
    correlation_plot: Figure | Path
    time_series_plot: Figure | Path
    stationarity_plot: Figure | Path

    # Feature-type-specific reports
    continuous_report: ContinuousEDAReport | None = None
    rule_report: RuleEDAReport | None = None

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
```

### 5.3 PermutationReport (Stages 1 & 2)

```python
@dataclass(frozen=True)
class PermutationReport:
    """
    Results from permutation testing (Stage 1 or Stage 2).
    """

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
    permutation_histogram: Figure | Path  # histogram of permuted statistics
    qq_plot: Figure | Path  # Q-Q plot of permuted distribution

    # Metadata
    n_permutations: int
    random_seed: int | None
    execution_time: float  # seconds
```

### 5.4 StabilityReport (Stage 3)

```python
@dataclass(frozen=True)
class StabilityReport:
    """
    Walkforward stability analysis results.
    """

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
    stability_heatmap: Figure | Path  # params × folds heatmap
    top_params_bar_chart: Figure | Path  # consistency scores
    parameter_trajectory_plot: Figure | Path  # how best params change over folds

    # Diagnostic flags
    warnings: list[str] = field(default_factory=list)

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
```

### 5.5 ParameterReport

```python
@dataclass(frozen=True)
class ParameterReport:
    """
    Parameter sensitivity analysis results.
    """

    # Configuration
    params_grid: dict[str, list]
    dimensions: Literal[1, 2, 3, 4]

    # Objective values
    objective_values: pd.DataFrame  # param combo → objective

    # Robustness metrics
    robustness_metrics: RobustnessMetrics

    # Plots (dimension-specific)
    sensitivity_plots: list[Figure | Path]  # 1D: line plots, 2D: heatmaps, etc.

    # Best parameter recommendations
    recommended_params: list[dict[str, Any]]  # Top params by robustness score
    recommended_ensemble: list[dict[str, Any]]  # Suggested ensemble members

    # Diagnostic insights
    high_variance_params: list[str]  # Params with high variance contribution
    stable_regions: list[dict[str, Any]]  # Regions with stability_ratio > 0.9
    isolated_peaks: list[dict[str, Any]]  # Regions with stability_ratio < 0.5
```

### 5.6 Feature-Type-Specific Reports

**ContinuousEDAReport:**

```python
@dataclass(frozen=True)
class ContinuousEDAReport:
    """Continuous feature EDA extensions."""

    # Decile analysis
    decile_stats: pd.DataFrame  # decile → mean, std, Sharpe, t-stat
    decile_plot: Figure | Path
    monotonicity_test: MonotonicityTestResult

    # Outlier analysis
    outlier_fraction: float
    outlier_impact: float  # change in Sharpe when removing outliers
    outlier_plot: Figure | Path

    # Non-linearity
    linear_r2: float
    polynomial_r2: dict[int, float]  # degree → R²
    non_linearity_plot: Figure | Path

    # Binning diagnostics
    binning_diagnostics: BinningDiagnostics | None = None
```

**RuleEDAReport:**

```python
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
    level_plot: Figure | Path

    # Regime transitions
    transition_matrix: pd.DataFrame  # P(level_t+1 | level_t)
    average_duration: dict[int, float]  # avg time spent in each level
    transition_plot: Figure | Path

    # Grid report (if params_grid provided)
    grid_report: RuleGridReport | None = None
```

**RuleGridReport:**

```python
@dataclass(frozen=True)
class RuleGridReport:
    """Rule-based grid performance report (per specs)."""

    # Per-metric pass rates
    metric_pass_rates: dict[str, float]  # metric_name → % combos passing

    # Overall verdict
    robustness_verdict: Literal['ROBUST', 'MODERATE', 'FRAGILE']
    overall_pass_rate: float  # % combos passing all thresholds

    # Best/worst permutations
    best_params: list[dict[str, Any]]  # Top 5 param combos
    worst_params: list[dict[str, Any]]  # Bottom 5

    # Parameter sensitivity
    variance_contribution: dict[str, float]  # param_name → variance explained
    sensitivity_plot: Figure | Path

    # Low-sample warnings
    low_sample_fraction: float  # % combos with insufficient samples
    low_sample_params: list[dict[str, Any]]

    # Plots
    pass_rate_heatmap: Figure | Path
    best_params_plot: Figure | Path
```

---

## 6. Data Flow & Integration Patterns

### 6.1 Data Flow Diagram

```
┌──────────────────┐
│ FeatureExtractor │
│  (existing)      │
└────────┬─────────┘
         │
         │ feature_data (DataFrame)
         │ target (Series)
         │ candles (DataFrame)
         │ params_grid (dict)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              FeatureValidator                       │
│                                                     │
│  ┌──────────────────────────────────────┐          │
│  │  1. run_eda()                        │          │
│  │     ├─ distribution_analysis()       │          │
│  │     ├─ correlation_analysis()        │          │
│  │     ├─ stationarity_tests()          │          │
│  │     └─ temporal_stability()          │          │
│  │         ├─ [continuous] decile_analysis()       │
│  │         └─ [rule] level_analysis()              │
│  └──────────────────────────────────────┘          │
│                   │                                 │
│                   ▼                                 │
│  ┌──────────────────────────────────────┐          │
│  │  2. run_stage1_permutation()         │          │
│  │     → PermutationEngine.vector_shuffle()        │
│  └──────────────────────────────────────┘          │
│                   │                                 │
│                   ▼                                 │
│  ┌──────────────────────────────────────┐          │
│  │  3. run_stage2_permutation()         │          │
│  │     → PermutationEngine.feature_shuffle() OR    │
│  │       PermutationEngine.candle_shuffle()        │
│  └──────────────────────────────────────┘          │
│                   │                                 │
│                   ▼                                 │
│  ┌──────────────────────────────────────┐          │
│  │  4. run_stage3_stability()           │          │
│  │     ├─ For each fold:                │          │
│  │     │    ├─ fit all param combos     │          │
│  │     │    └─ compute objectives       │          │
│  │     ├─ apply_neighbor_smoothing()    │          │
│  │     ├─ identify_top_k_per_fold()     │          │
│  │     └─ assess_temporal_consistency() │          │
│  └──────────────────────────────────────┘          │
│                   │                                 │
│                   ▼                                 │
│  ┌──────────────────────────────────────┐          │
│  │  5. [Optional] analyze_parameter_sensitivity()  │
│  │     → ParameterAnalyzer (existing)              │
│  │     → parameter_plots.* (existing)              │
│  └──────────────────────────────────────┘          │
│                   │                                 │
│                   ▼                                 │
│         ValidationReport                            │
│  (accumulated across all stages)                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Report Export                      │
│  - export_report()                  │
│    → HTML / PDF / Markdown / JSON   │
│  - generate_researcher_summary()    │
│    → concise text for decisions     │
└─────────────────────────────────────┘
```

### 6.2 Typical Workflows

**Workflow 1: Quick Validation (Full Pipeline)**

```python
# Initialize validator
config = ValidationConfig(
    feature_type='continuous',
    n_permutations=1000,
    min_sharpe_threshold=0.5,
)
validator = FeatureValidator(
    config=config,
    permutation_engine=permutation_engine,
    parameter_analyzer=parameter_analyzer,
    output_dir=Path("reports/validation"),
)

# Run full validation
report = validator.run_full_validation(
    feature_data=rsi_features,  # DataFrame with RSI_14, RSI_15, etc.
    target=vol_scaled_returns,
    params_grid={'lookback': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
)

# Check status
if report.validation_status == 'passed':
    # Export HTML report
    validator.export_report(report, format='html', output_path=Path("rsi_validation.html"))

    # Get researcher summary
    summary = validator.generate_researcher_summary(report)
    print(summary)
else:
    print(f"Validation failed at {report.failure_stage}")
```

**Workflow 2: Progressive Validation (Stage-by-Stage)**

```python
# Run stages progressively, inspect after each
for stage_name, stage_report in validator.run_progressive_validation(
    feature_data=rule_features,
    target=vol_scaled_returns,
    params_grid={'threshold': [0.5, 0.6, 0.7, 0.8]},
):
    print(f"\n=== {stage_name} ===")

    if stage_name == "EDA":
        # Inspect EDA results
        print(f"Feature-target correlation: {stage_report.correlations['pearson']:.3f}")
        print(f"Stationarity (ADF p-value): {stage_report.adf_test.p_value:.3f}")

    elif stage_name.startswith("Stage"):
        # Check permutation results
        if not stage_report.passed:
            print(f"FAILED: p-value = {stage_report.p_value:.4f}")
            break
        else:
            print(f"PASSED: p-value = {stage_report.p_value:.4f}, margin = {stage_report.margin:.2f}")

    elif stage_name == "Stability":
        # Review stability
        print(f"Best region stability: {stage_report.best_region_stability}")
        print(f"Top params consistency:\n{stage_report.top_params_consistency.head()}")
```

**Workflow 3: Parameter Sensitivity Focus**

```python
# Run EDA + parameter analysis (skip permutation for now)
eda_report = validator.run_eda(feature_data=features, target=target)
param_report = validator.analyze_parameter_sensitivity(
    feature_data=features,
    target=target,
    params_grid={'lookback': range(5, 25), 'smoothing': [0.1, 0.2, 0.3]},
    dimensions=2,
)

# Inspect robustness
print(f"High variance params: {param_report.high_variance_params}")
print(f"Stable regions: {param_report.stable_regions}")
print(f"Isolated peaks (suspicious): {param_report.isolated_peaks}")

# Export parameter report
validator.export_report(
    ValidationReport(
        feature_name="RSI",
        feature_type='continuous',
        timestamp=datetime.now(),
        eda_report=eda_report,
        parameter_report=param_report,
        validation_status='incomplete',
    ),
    format='html',
    output_path=Path("rsi_param_sensitivity.html"),
)
```

**Workflow 4: Rule-Based Grid Report**

```python
# Configure for rule-based feature
config = ValidationConfig(
    feature_type='rule_based',
    n_permutations=1000,
)
validator = FeatureValidator(config, ...)

# Run full validation
report = validator.run_full_validation(
    feature_data=breakout_rule_features,
    target=vol_scaled_returns,
    params_grid={
        'lookback': [20, 30, 40, 50, 60],
        'threshold': [0.5, 0.75, 1.0, 1.25, 1.5],
    },
)

# Access rule-based grid report
grid_report = report.eda_report.rule_report.grid_report
print(f"Robustness verdict: {grid_report.robustness_verdict}")
print(f"Overall pass rate: {grid_report.overall_pass_rate:.1%}")
print(f"Parameter sensitivity:\n{grid_report.variance_contribution}")

# Export grid report separately
validator.export_report(report, format='pdf', output_path=Path("breakout_grid_report.pdf"))
```

### 6.3 Integration with Existing Infrastructure

**PermutationEngine (existing):**
- Located in `permutation_testing/` module
- Provides `vector_shuffle()`, `feature_shuffle()`, `candle_shuffle()`
- FeatureValidator wraps these methods, handles report generation

**ParameterAnalyzer (existing):**
- Located in `eda/parameter_analysis.py`
- Provides `analyze_parameter()`, `analyze_2d_parameters()`, `analyze_nd_parameters()`
- FeatureValidator delegates parameter sensitivity analysis to this class

**Plotting Functions (existing):**
- Located in `metrics/plotting/parameter_plots.py`
- Provides `plot_parameter_sensitivity()`, `plot_2d_parameter_surface()`, etc.
- FeatureValidator uses these for visualization generation

**BinningModelBase (existing):**
- Located in `feature_selection/base_models/base_model.py`
- Continuous binning pipeline: quantile bins → regions → threshold filter → position multipliers
- Rule-based binning: per-level stats → threshold filter → position multipliers
- FeatureValidator uses binning models internally for diagnostics

**Integration Pattern:**
- **Composition:** FeatureValidator owns instances of PermutationEngine, ParameterAnalyzer
- **Delegation:** Delegates specialized tasks to existing components
- **Orchestration:** Coordinates workflow, manages ValidationReport accumulation
- **Extension:** Adds EDA methods, report generation, export capabilities

### 6.4 Error Handling & Validation

**Input Validation:**

```python
def _validate_inputs(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    params_grid: dict[str, list] | None,
) -> None:
    """
    Validate inputs before running validation pipeline.

    Checks:
    - feature_data and target have same index
    - No missing values in target (feature NaNs handled by binning)
    - Target is vol-scaled (check for reasonable std range)
    - params_grid keys match expected feature parameters
    - Sufficient sample size (>= 100 observations)
    """
    ...
```

**Graceful Degradation:**
- If a plot fails to generate, log warning and continue
- If a permutation test fails (e.g., numerical instability), report error in ValidationReport
- If insufficient data for a stage, skip with warning rather than crash

**Early Exit Strategy:**
- Stop validation pipeline at first failed permutation test
- Set `validation_status='failed'` and `failure_stage` in ValidationReport
- Return partial report with all completed stages

### 6.5 Edge Cases & Robustness

**Low Sample Size:**
- Warn if < 100 observations total
- Flag bins/levels with < 5% of samples
- Adjust permutation test critical values for small samples

**Extreme Feature Values:**
- Outlier detection in EDA
- Robust quantile estimation for binning (handle outliers gracefully)
- Flag if >10% of samples are outliers

**Non-Stationary Features:**
- Detect non-stationarity in EDA (ADF, KPSS tests)
- Warn researcher if stationarity assumptions violated
- Suggest differencing or detrending

**High-Correlation Features (Multi-Parameter):**
- Detect multicollinearity in EDA
- Flag if inter-feature correlation > 0.95
- Note: High correlation is intentional for parameter ensembles (e.g., RSI 14 vs. RSI 15)

**Missing Data:**
- Target: Require no missing values (raise error if found)
- Feature: Allow NaNs, handle in binning (drop or forward-fill based on feature type)
- Report % missing and handling strategy in EDA

**Numerical Stability:**
- Handle division by zero in Sharpe calculations (return NaN or 0)
- Clip extreme position multipliers to [-2, 2] range
- Use robust covariance estimators for correlation matrices

---

## Summary

This design specification defines a comprehensive **Feature Validator API** that:

1. **Unifies** continuous and rule-based feature validation through a single interface
2. **Integrates** EDA, permutation testing, and parameter sensitivity analysis
3. **Orchestrates** the complete validation workflow (Stages 1-3) with modular access
4. **Reuses** existing infrastructure (PermutationEngine, ParameterAnalyzer, plotting functions)
5. **Generates** rich, progressive reports for researcher decision-making
6. **Supports** multiple export formats (HTML, PDF, Markdown, JSON)
7. **Enforces** hypothesis-driven testing principles (pre-specified thresholds, no optimization)

**Key Design Decisions:**
- Unified class with feature_type parameter (not inheritance)
- Progressive report accumulation (ValidationReport)
- Hybrid visualization strategy (Plotly for interactivity, Matplotlib for statistical plots)
- Full integration with existing components via composition/delegation
- Comprehensive error handling and validation

**Next Steps:**
1. ✅ Complete design specification (this document)
2. ⏭️ Write detailed implementation plan
3. Implement FeatureValidator class
4. Implement report objects and export functions
5. Integrate with existing PermutationEngine and ParameterAnalyzer
6. Write comprehensive tests
7. Migrate FeatureExplorer usage to FeatureValidator
8. Documentation and examples

---

**End of Design Specification**
