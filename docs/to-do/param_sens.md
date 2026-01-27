# Parameter Sensitivity Analysis Specification

## 1. Overview

This specification defines a comprehensive parameter sensitivity analysis system for trading features that supports 1-4 parameter dimensions with interactive visualizations and robustness metrics.

### Key Requirements

- **Multi-dimensional support**: Handle features with 1-4 parameters
- **Adaptive visualizations**: Different plot types based on parameter count
- **Robustness analysis**: Comprehensive evaluation of strategy stability
- **Standalone architecture**: Separate from `generate_summary_report()` for focused analysis
- **Existing parameter space**: Use only pre-computed feature combinations from metadata

### Current Implementation Status

**Known Issues**:
- ✅ **1D Parameter Analysis**: Working correctly (`plot_parameter_sensitivity()`)
- ❌ **2D Parameter Analysis**: **NOT WORKING** - `plot_2d_parameter_surface()` needs to be fixed
- ⏳ **3D Parameter Analysis**: Not yet implemented (NEW)
- ⏳ **4D Parameter Analysis**: Not yet implemented (NEW)

**Priority**: Fix 2D implementation before implementing 3D/4D features to ensure consistent architecture.

---

## 2. Architectural Design

### 2.1 Extension Points

The system extends the existing architecture in `eda/feature_explorer.py` and `eda/parameter_analysis.py`:

```
┌─────────────────────┐
│  FeatureExplorer    │
│                     │
│  - plot_parameter_  │
│    sensitivity      │  (1 param) ──> delegates ──> ParameterAnalyzer
│  - plot_2d_param_   │                               - analyze_parameter
│    surface          │  (2 params) ─> delegates ──> - analyze_2d_parameters
│  - plot_nd_param_   │                               
│    analysis [NEW]   │  (3-4 params) ─> delegates ─> - analyze_nd_parameters [NEW]
│                     │
│  - generate_param_  │
│    sensitivity_     │
│    report [NEW]     │
└─────────────────────┘
           │
           └──> metrics/plotting/parameter_plots.py
                - plot_parameter_sensitivity
                - plot_2d_parameter_surface
                - plot_nd_parameter_analysis [NEW]
```

### 2.2 Data Flow

```
Feature Metadata
    │
    ├─> Extract Param Grid
    │       │
    │       ├─> Compute Metrics (per combination)
    │       │       │
    │       │       └─> Results DataFrame
    │       │               │
    │       │               ├─> Generate Visualizations
    │       │               │       │
    │       │               │       ├─> 1D: Line plot
    │       │               │       ├─> 2D: Surface plot
    │       │               │       ├─> 3D: Interactive plot (1 dropdown/slider)
    │       │               │       └─> 4D: Interactive plot (2 dropdowns/sliders)
    │       │               │
    │       │               └─> Generate Summary Report
    │       │                       │
    │       │                       ├─> Robustness Scores
    │       │                       ├─> Parameter Sensitivity Ranking
    │       │                       └─> Text Report File
    │       │
    │       └─> Interactive Plots (HTML/Plotly)
```

---

## 3. Visualization Specifications

### 3.1 One Parameter (1D)

**Existing Implementation**: `plot_parameter_sensitivity()`

- **Plot Type**: Line graph with markers
- **X-axis**: Parameter value
- **Y-axis**: Objective metric (e.g., Sortino ratio)
- **Secondary Y-axis**: Sample size (bar chart, gray, 20% opacity)
- **Library**: Plotly

### 3.2 Two Parameters (2D)

**Existing Implementation**: `plot_2d_parameter_surface()`

⚠️ **STATUS: NOT WORKING** - This implementation needs to be fixed before proceeding with 3D/4D features.

- **Plot Type**: 3D surface/contour plot
- **X-axis**: Parameter 2 value
- **Y-axis**: Parameter 1 value
- **Z-axis**: Objective metric
- **Library**: Plotly
- **Options**: surface, scatter, heatmap, contour, lines

**Required Fixes**:
- Debug and fix `plot_2d_parameter_surface()` in `metrics/plotting/parameter_plots.py`
- Ensure `analyze_2d_parameters()` in `ParameterAnalyzer` works correctly
- Verify parameter grid extraction for 2D case
- Test with real feature data to identify root cause

### 3.3 Three Parameters (3D) - NEW

**Method**: `plot_3d_parameter_analysis()`

**Interactive Controls**:
- Dropdown menu to select fixed parameter (param1, param2, or param3)
- Slider to set value of fixed parameter
- Plot updates dynamically showing 2D slice of 3D parameter space

**Visualization**:
- **Base plot**: 3D surface/contour (same as 2D case)
- **X-axis**: First remaining parameter
- **Y-axis**: Second remaining parameter
- **Z-axis**: Objective metric
- **Title**: Dynamically updated to show fixed parameter value

**Implementation Strategy**:

```python
# Plotly updatemenus for dropdown + slider
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {'label': f'Fix {param}', 'method': 'update', ...}
            for param in [param1, param2, param3]
        ],
        'direction': 'down',
        'x': 0.1,
        'y': 1.15
    }],
    sliders=[{
        'steps': [{'args': [...], 'label': str(val)} for val in param_values],
        'active': 0,
        'y': -0.1
    }]
)
```

### 3.4 Four Parameters (4D) - NEW

**Method**: `plot_4d_parameter_analysis()`

**Interactive Controls**:
- **First dropdown**: Select first fixed parameter (param1, param2, param3, or param4)
- **First slider**: Set value of first fixed parameter
- **Second dropdown**: Select second fixed parameter (from remaining 3)
- **Second slider**: Set value of second fixed parameter
- Plot shows 2D slice of 4D parameter space (2 params fixed, 2 params visualized)

**Visualization**:
- **Base plot**: 3D surface/contour
- **X-axis**: First remaining parameter
- **Y-axis**: Second remaining parameter
- **Z-axis**: Objective metric
- **Title**: Shows both fixed parameter values

**Layout**:

```
[Dropdown 1: Select Fixed Param 1] [Slider 1: Value]
[Dropdown 2: Select Fixed Param 2] [Slider 2: Value]

[3D Surface Plot showing remaining 2 params vs metric]
```

---

## 4. Summary Report Specification

### 4.1 Report Structure

**Method**: `generate_parameter_sensitivity_report()`

**Output**: Text file with following sections:

```
======================================================================
Parameter Sensitivity Analysis Report
======================================================================
Feature: rsi_signal_D_lookback_[X]_threshold_[Y]...
Generated: 2025-01-26 10:30:45
======================================================================

PARAMETER SPACE
----------------------------------------------------------------------
Parameters Analyzed: 3
  - lookback: [2, 5, 10, 14, 20] (5 values)
  - threshold: [30, 40, 50, 60, 70] (5 values)  
  - smoothing: [1, 2, 3] (3 values)
Total Combinations: 75

PERFORMANCE SUMMARY
----------------------------------------------------------------------
Objective Metric: Sortino Ratio
Threshold: 1.0

Overall Statistics:
  - Mean Metric: 1.25 ± 0.35 (std)
  - Median Metric: 1.18
  - Min Metric: 0.45 (at lookback=2, threshold=70, smoothing=1)
  - Max Metric: 2.10 (at lookback=14, threshold=50, smoothing=2)
  - Range: 1.65

Performance Distribution:
  - Above Threshold (>1.0): 48/75 (64.0%)
  - Below Threshold (<1.0): 27/75 (36.0%)
  - Negative Performance (<0): 5/75 (6.7%)

ROBUSTNESS ANALYSIS
----------------------------------------------------------------------
Variance Stability:
  - Coefficient of Variation: 0.28 (Lower is better)
  - Interpretation: MODERATE stability across parameters
  
Consistency Score:
  - % Positive Returns: 93.3%
  - % Above Threshold: 64.0%
  - Consistency Rating: GOOD

Extreme Outcomes:
  - Max Drawdown (worst case): -0.15
  - Max Drawdown (average): -0.08
  - Risk Rating: LOW

OVERALL ROBUSTNESS SCORE: 7.2/10
  - Variance Component: 7.5/10
  - Consistency Component: 8.0/10
  - Risk Component: 6.0/10

Interpretation: ROBUST - Strategy performs well across most parameter 
combinations with moderate variance and low risk of extreme losses.

PARAMETER SENSITIVITY RANKING
----------------------------------------------------------------------
(Ranked by impact on metric variance)

1. threshold: Variance contribution = 45.2%
   - High sensitivity: Metric varies significantly with this parameter
   
2. lookback: Variance contribution = 38.5%
   - Moderate sensitivity: Notable impact on performance
   
3. smoothing: Variance contribution = 16.3%
   - Low sensitivity: Minimal impact on performance

RECOMMENDATIONS
----------------------------------------------------------------------
1. OPTIMAL REGION: lookback=[10-20], threshold=[40-60], smoothing=[2-3]
   - 12/15 combinations above threshold in this region (80%)
   - Average metric in region: 1.65

2. AVOID: lookback<5 with threshold>60
   - Only 1/6 combinations above threshold (17%)
   - High risk of poor performance

3. ROBUST CHOICES: 
   - smoothing=2 shows consistent performance (78% above threshold)
   - threshold=50 has lowest variance (CV=0.18)

======================================================================
```

### 4.2 Robustness Scoring Algorithm

**Component 1: Variance Stability (0-10)**

```python
cv = std(metric_values) / abs(mean(metric_values))
variance_score = max(0, 10 - (cv * 20))  # Penalize high CV
```

**Component 2: Consistency Score (0-10)**

```python
pct_above_threshold = sum(metric > threshold) / n_combinations
consistency_score = pct_above_threshold * 10
```

**Component 3: Risk Score (0-10)**

```python
max_drawdown_severity = abs(min(max_drawdowns))
risk_score = max(0, 10 - (max_drawdown_severity * 50))
```

**Overall Score**:

```python
robustness_score = (variance_score + consistency_score + risk_score) / 3
```

**Rating Thresholds**:
- 8.0-10.0: **HIGHLY ROBUST**
- 6.0-7.9: **ROBUST**
- 4.0-5.9: **MODERATELY ROBUST**
- 2.0-3.9: **FRAGILE**
- 0.0-1.9: **HIGHLY FRAGILE**

---

## 5. API Design

### 5.1 FeatureExplorer Methods

```python
def plot_nd_parameter_analysis(
    self,
    module_name: str,
    param_names: List[str],  # 3-4 parameter names
    target_col: str = 'log_return',
    metric: Optional[Any] = None,
    n_bins: int = 5,
    base_model: Optional[Any] = None,
    show_plot: bool = True,
    plot_type: str = 'surface',
    feature_name: Optional[str] = None
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyze and visualize parameter sensitivity for 3-4 parameters.
    
    Returns interactive Plotly figure with dropdowns/sliders to explore
    the parameter space by fixing 1-2 dimensions.
    
    **Integration Note**: This method works with dataframes containing many features
    and extracts parameters using the standardized column naming convention
    (e.g., `rsi_signal_D_lookback_14_threshold_50`). Parameters are extracted
    from feature metadata (if available) or by parsing column names.
    
    Parameters
    ----------
    module_name : str
        Name of the module (e.g., 'rsi', 'macd')
    param_names : List[str]
        List of 3-4 parameter names to analyze
    target_col : str, default='log_return'
        Target column to use for computing metrics
    metric : Optional[Any], default=None
        Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
        Must have a .compute() method. If None, defaults to SortinoRatio.
    n_bins : int, default=5
        Number of bins for model-specific binning
    base_model : Optional[Any], default=None
        Model instance with fit/predict methods. If None, uses QuantileBinningModel.
    show_plot : bool, default=True
        Whether to show the plot
    plot_type : str, default='surface'
        Type of plot to generate: 'surface', 'scatter', 'heatmap', 'contour', 'lines'
    feature_name : Optional[str], default=None
        Optional filter for specific feature type (e.g., 'signal')
        
    Returns
    -------
    Tuple[pd.DataFrame, go.Figure]
        - DataFrame with parameter values and computed metrics
        - Interactive Plotly figure
        
    Examples
    --------
    >>> from metrics.performance import SortinoRatio
    >>> metric = SortinoRatio(annualization_factor=252)
    >>> 
    >>> # 3D Parameter Analysis
    >>> df, fig = explorer.plot_nd_parameter_analysis(
    ...     module_name='rsi',
    ...     param_names=['lookback', 'threshold', 'smoothing'],
    ...     metric=metric,
    ...     show_plot=True
    ... )
    """
```

```python
def generate_parameter_sensitivity_report(
    self,
    module_name: str,
    param_names: Optional[List[str]] = None,  # None = all params for module
    target_col: str = 'log_return',
    metric: Optional[Any] = None,
    metric_threshold: float = 1.0,
    n_bins: int = 5,
    base_model: Optional[Any] = None,
    export_path: Optional[str] = None,
    feature_name: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive parameter sensitivity report with visualizations
    and robustness analysis.
    
    This is a STANDALONE method (not integrated into generate_summary_report)
    that provides focused analysis of parameter space exploration.
    
    **Integration Note**: This method works with dataframes containing many features
    and extracts parameters using the standardized column naming convention.
    It leverages the existing `FeatureExplorer._param_mapping` infrastructure
    to efficiently identify and group parameterized features.
    
    Parameters
    ----------
    module_name : str
        Name of the module (e.g., 'rsi', 'macd')
    param_names : Optional[List[str]], default=None
        List of parameter names to analyze. If None, analyzes all parameters
        for the module (automatically detected from metadata).
    target_col : str, default='log_return'
        Target column to use for computing metrics
    metric : Optional[Any], default=None
        Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
        Must have a .compute() method. If None, defaults to SortinoRatio.
    metric_threshold : float, default=1.0
        Threshold value for determining "good" vs "bad" performance.
        Used for robustness analysis (e.g., % strategies above threshold).
    n_bins : int, default=5
        Number of bins for model-specific binning
    base_model : Optional[Any], default=None
        Model instance with fit/predict methods. If None, uses QuantileBinningModel.
    export_path : Optional[str], default=None
        Path to export text report. If None, report is not saved to file.
    feature_name : Optional[str], default=None
        Optional filter for specific feature type (e.g., 'signal')
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'results_df': DataFrame with all parameter combinations and metrics
        - 'summary_stats': Dict with mean, std, min, max, percentiles, etc.
        - 'robustness_scores': Dict with variance, consistency, risk scores
        - 'figures': Dict with Plotly figures for each parameter dimensionality
        - 'parameter_sensitivity': Dict with variance contribution per parameter
        - 'report_path': Path to exported text report (if export_path provided)
        
    Examples
    --------
    >>> from metrics.performance import SortinoRatio
    >>> from feature_selection.base_models.quantile_binning import QuantileBinningModel
    >>> 
    >>> metric = SortinoRatio(annualization_factor=252)
    >>> model = QuantileBinningModel(n_bins=5, selection_metric='sortino')
    >>> 
    >>> # Comprehensive Report
    >>> report = explorer.generate_parameter_sensitivity_report(
    ...     module_name='rsi',
    ...     param_names=None,  # All params
    ...     metric_threshold=1.0,
    ...     metric=metric,
    ...     base_model=model,
    ...     export_path='research/reports/rsi_param_sensitivity.txt',
    ...     verbose=True
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Robustness Score: {report['robustness_scores']['overall_score']:.1f}/10")
    >>> print(f"Rating: {report['robustness_scores']['rating']}")
    >>> print(f"Best Params: {report['summary_stats']['max_config']}")
    """
```

### 5.2 ParameterAnalyzer Methods

```python
def analyze_nd_parameters(
    self,
    feature_grid: Dict[Tuple, List[str]],  # e.g., {(p1, p2, p3): [features]}
    target_col: str = 'log_return',
    metric: Optional[Any] = None,
    n_bins: int = 5,
    base_model: Optional[Any] = None
) -> pd.DataFrame:
    """
    Analyze parameter combinations for 3+ parameters.
    
    Extends analyze_2d_parameters to support higher dimensions.
    
    Parameters
    ----------
    feature_grid : Dict[Tuple, List[str]]
        Dictionary mapping parameter tuples to feature names.
        For 3 params: {(p1_val, p2_val, p3_val): [feature_names]}
        For 4 params: {(p1_val, p2_val, p3_val, p4_val): [feature_names]}
    target_col : str, default='log_return'
        Target column to use for computing metrics
    metric : Optional[Any], default=None
        Metric object from metrics.performance
    n_bins : int, default=5
        Number of bins for QuantileBinningModel
    base_model : Optional[Any], default=None
        Model instance with fit/predict methods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - param1_value, param2_value, param3_value, [param4_value]
        - metric_name (e.g., 'sortino')
        - mean, std, max_drawdown, n_samples
        
    Examples
    --------
    >>> # 3D parameter grid
    >>> feature_grid = {
    ...     (2, 30, 1): ['rsi_signal_D_lookback_2_threshold_30_smoothing_1'],
    ...     (2, 40, 1): ['rsi_signal_D_lookback_2_threshold_40_smoothing_1'],
    ...     # ... more combinations
    ... }
    >>> df = analyzer.analyze_nd_parameters(feature_grid, target_col='log_return')
    """
```

```python
def compute_robustness_metrics(
    self,
    results_df: pd.DataFrame,
    metric_col: str,
    metric_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Compute comprehensive robustness metrics from parameter analysis results.
    
    Calculates variance stability, consistency, and risk scores to evaluate
    how robust a strategy is across its parameter space.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from analyze_parameter, analyze_2d_parameters, or analyze_nd_parameters
        Must contain column specified by metric_col
    metric_col : str
        Name of the metric column to analyze (e.g., 'sortino', 'sharpe')
    metric_threshold : float, default=1.0
        Threshold value for determining "good" vs "bad" performance
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'variance_score': 0-10 score based on coefficient of variation
        - 'consistency_score': 0-10 score based on % above threshold
        - 'risk_score': 0-10 score based on max drawdown
        - 'overall_score': Average of component scores
        - 'rating': String rating (HIGHLY ROBUST, ROBUST, MODERATELY ROBUST, FRAGILE, HIGHLY FRAGILE)
        - 'statistics': Summary statistics dict (mean, std, cv, min, max, percentiles, etc.)
        - 'parameter_sensitivity': Variance contribution per parameter (if multiple params)
        
    Examples
    --------
    >>> df = analyzer.analyze_2d_parameters(feature_grid, target_col='log_return')
    >>> metrics = analyzer.compute_robustness_metrics(
    ...     results_df=df,
    ...     metric_col='sortino',
    ...     metric_threshold=1.0
    ... )
    >>> print(f"Robustness Score: {metrics['overall_score']:.1f}/10")
    >>> print(f"Rating: {metrics['rating']}")
    """
```

---

## 6. Implementation Details

### 6.1 New Files/Modifications Required

**File**: `metrics/plotting/parameter_plots.py`
- Add `plot_3d_parameter_interactive()`
- Add `plot_4d_parameter_interactive()`
- Add `_create_parameter_dropdown()`
- Add `_create_parameter_slider()`

**File**: `eda/parameter_analysis.py`
- Add `analyze_nd_parameters()`
- Add `compute_robustness_metrics()`
- Add `_compute_parameter_sensitivity_ranking()`

**File**: `eda/feature_explorer.py`
- Add `plot_nd_parameter_analysis()`
- Add `generate_parameter_sensitivity_report()`
- Add `_export_parameter_sensitivity_report()`

### 6.2 Data Structures

**Results DataFrame Structure (3D example)**:

```python
pd.DataFrame({
    'param1_value': [2, 2, 2, 5, 5, 5, ...],
    'param2_value': [30, 40, 50, 30, 40, 50, ...],
    'param3_value': [1, 1, 1, 1, 1, 1, ...],
    'sortino': [1.2, 1.5, 1.1, 1.3, 1.8, 1.4, ...],
    'mean': [0.01, 0.015, 0.009, ...],
    'std': [0.02, 0.018, 0.021, ...],
    'max_drawdown': [-0.08, -0.06, -0.10, ...],
    'n_samples': [150, 148, 152, ...]
})
```

**Robustness Metrics Structure**:

```python
{
    'variance_score': 7.5,
    'consistency_score': 8.0,
    'risk_score': 6.0,
    'overall_score': 7.2,
    'rating': 'ROBUST',
    'statistics': {
        'mean': 1.25,
        'std': 0.35,
        'cv': 0.28,
        'min': 0.45,
        'max': 2.10,
        'median': 1.18,
        'q25': 0.95,
        'q75': 1.52,
        'range': 1.65,
        'pct_above_threshold': 0.64,
        'pct_positive': 0.933,
        'pct_negative': 0.067,
        'min_config': {'param1': 2, 'param2': 70, 'param3': 1},
        'max_config': {'param1': 14, 'param2': 50, 'param3': 2}
    },
    'parameter_sensitivity': {
        'threshold': 0.452,
        'lookback': 0.385,
        'smoothing': 0.163
    }
}
```

### 6.3 Parameter Grid Extraction Algorithm

```python
def _extract_parameter_grid(
    self,
    module_name: str,
    param_names: List[str],
    feature_name: Optional[str] = None
) -> Dict[Tuple, List[str]]:
    """
    Extract parameter grid from feature metadata.
    
    Returns dict mapping parameter value tuples to feature names.
    """
    # Get all features for this module from metadata
    module_features = {
        feat: meta for feat, meta in self.feature_metadata.items()
        if meta.get('module', '').lower() == module_name.lower()
    }
    
    # Optional: filter by feature_name token
    if feature_name is not None:
        module_features = {
            feat: meta for feat, meta in module_features.items()
            if self._canonicalize_param_name(
                helpers.parse_feature_column_name(feat).get('feature', '')
            ) == self._canonicalize_param_name(feature_name)
        }
    
    # Build parameter grid
    param_grid = {}
    for feat, meta in module_features.items():
        params = meta.get('parameters', {})
        
        # Extract parameter values in order
        param_values = tuple(
            params.get(pname) for pname in param_names
        )
        
        # Skip if any parameter is missing
        if any(v is None for v in param_values):
            continue
        
        # Add to grid
        if param_values not in param_grid:
            param_grid[param_values] = []
        param_grid[param_values].append(feat)
    
    return param_grid
```

### 6.4 Integration with FeatureExplorer Pipeline

**Critical Requirement**: The parameter sensitivity analysis system must integrate seamlessly with the existing `FeatureExplorer` pipeline, which operates on dataframes containing many features with standardized column names.

#### Input Format

The system receives:
- **`features_df`**: DataFrame with many feature columns (potentially hundreds)
- **`targets_df`**: DataFrame with target columns (log_return, etc.)
- **`metadata`**: Optional dictionary containing feature metadata from `FeatureExtractor`

**Example Input DataFrame**:
```python
features_df.columns = [
    'rsi_signal_D_lookback_14',
    'rsi_signal_D_lookback_20',
    'rsi_signal_D_lookback_30',
    'rsi_signal_D_lookback_14_threshold_50',
    'rsi_signal_D_lookback_20_threshold_50',
    'ewmac_signal_D_spanFast_64_spanSlow_256',
    'ewmac_signal_D_spanFast_32_spanSlow_128',
    # ... many more features
]
```

#### Standardized Column Naming Convention

All features follow the standardized naming convention from `BiasNode`:

```
{module_name}_{feature}_{timeframe}_{param1}_{value1}_{param2}_{value2}...
```

**Examples**:
- `rsi_signal_D_lookback_14` → module='rsi', feature='signal', params={'lookback': 14}
- `rsi_signal_D_lookback_14_threshold_50` → module='rsi', feature='signal', params={'lookback': 14, 'threshold': 50}
- `ewmac_signal_D_spanFast_64_spanSlow_256` → module='ewmac', feature='signal', params={'spanFast': 64, 'spanSlow': 256}

#### Parameter Extraction Strategy

The system must extract parameters from column names using the existing infrastructure:

1. **Primary Method**: Use `FeatureExplorer._param_mapping` (from metadata)
   - If `metadata` is provided with `feature_metadata`, use it directly
   - Already parsed and typed correctly

2. **Fallback Method**: Parse column names using `helpers.parse_feature_column_name()`
   - Used when metadata is not available
   - Must handle all standardized naming patterns
   - Extract module, feature, and parameters correctly

**Implementation Pattern**:
```python
def _extract_parameter_grid(
    self,
    module_name: str,
    param_names: List[str],
    feature_name: Optional[str] = None
) -> Dict[Tuple, List[str]]:
    """
    Extract parameter grid from feature metadata OR column names.
    
    Priority:
    1. Use self.feature_metadata if available (from FeatureExtractor)
    2. Fall back to parsing column names using helpers.parse_feature_column_name()
    """
    # Method 1: Use metadata if available
    if self.feature_metadata:
        module_features = {
            feat: meta for feat, meta in self.feature_metadata.items()
            if meta.get('module', '').lower() == module_name.lower()
        }
    else:
        # Method 2: Parse from column names
        module_features = {}
        for col in self.feature_names:  # Already filtered to exclude 'ticker', etc.
            try:
                parsed = helpers.parse_feature_column_name(col)
                if parsed and parsed.get('module', '').lower() == module_name.lower():
                    module_features[col] = {
                        'module': parsed['module'],
                        'parameters': parsed['params'],
                        'feature': parsed.get('feature', 'signal')
                    }
            except Exception:
                continue
    
    # Build parameter grid from extracted features
    param_grid = {}
    for feat, meta in module_features.items():
        params = meta.get('parameters', {})
        param_values = tuple(params.get(pname) for pname in param_names)
        
        if any(v is None for v in param_values):
            continue
        
        if param_values not in param_grid:
            param_grid[param_values] = []
        param_grid[param_values].append(feat)
    
    return param_grid
```

#### Integration Points

1. **FeatureExplorer Initialization**:
   - System receives `features_df` and `targets_df` (already aligned by index)
   - `FeatureExplorer` already extracts `feature_names` and builds `_param_mapping`
   - Parameter sensitivity methods should leverage existing `_feature_groups` structure

2. **Parameter Grid Extraction**:
   - Must work with both metadata-based and column-name-based parsing
   - Should reuse `FeatureExplorer._initialize_parameter_mapping()` logic
   - Must handle features with varying parameter counts (1-4 params)

3. **Feature Filtering**:
   - Filter by `module_name` (e.g., 'rsi', 'ewmac')
   - Optional filter by `feature_name` token (e.g., 'signal' vs 'signalBool')
   - Exclude normalization columns (ATR/EWSD) automatically

4. **Data Validation**:
   - Ensure all requested parameter names exist in feature metadata/column names
   - Validate that parameter combinations exist in the dataframe
   - Provide clear error messages when parameters not found

#### Example Integration Flow

```python
# Step 1: FeatureExtractor creates features with standardized names
extractor = FeatureExtractor(ticker=Ticker.SPY)
result = extractor.extract(bias_node_specs=[...])
# result['features'] has columns like: 'rsi_signal_D_lookback_14', etc.
# result['feature_metadata'] contains parsed metadata

# Step 2: FeatureExplorer consumes the dataframes
explorer = FeatureExplorer(
    features_df=result['features'],
    targets_df=result['targets'],
    metadata={'feature_metadata': result['feature_metadata']}
)
# FeatureExplorer automatically builds _param_mapping from metadata

# Step 3: Parameter sensitivity analysis uses existing infrastructure
df, fig = explorer.plot_nd_parameter_analysis(
    module_name='rsi',
    param_names=['lookback', 'threshold'],
    # System automatically:
    # - Finds all 'rsi' features from _param_mapping
    # - Extracts parameter values from metadata or column names
    # - Builds parameter grid
    # - Analyzes each combination
)
```

#### Key Integration Requirements

- ✅ **Must work with dataframes containing many features** (not just single feature)
- ✅ **Must use standardized column naming convention** for parameter extraction
- ✅ **Must leverage existing `_param_mapping` and `_feature_groups`** in FeatureExplorer
- ✅ **Must support both metadata-based and column-name-based parsing**
- ✅ **Must handle features with 1-4 parameters** automatically
- ✅ **Must filter by module and feature type** correctly
- ✅ **Must provide clear errors** when parameters/features not found
- ✅ **Must be compatible with existing `plot_parameter_sensitivity()` and `plot_2d_parameter_surface()`** methods

#### Testing Integration

When implementing, ensure:
1. Test with dataframes containing 100+ features
2. Test with both metadata-provided and metadata-missing scenarios
3. Test parameter extraction for all standardized naming patterns
4. Test filtering by module_name and feature_name
5. Verify compatibility with existing 1D/2D methods

---

## 7. Example Usage

```python
from eda.feature_explorer import FeatureExplorer
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from metrics.performance import SortinoRatio

# Setup
explorer = FeatureExplorer(features_df, targets_df, metadata)
model = QuantileBinningModel(n_bins=5, selection_metric='sortino')
metric = SortinoRatio(annualization_factor=252)

# ============================================================
# Example 1: 3D Parameter Analysis
# ============================================================
df, fig = explorer.plot_nd_parameter_analysis(
    module_name='rsi',
    param_names=['lookback', 'threshold', 'smoothing'],
    metric=metric,
    base_model=model,
    show_plot=True
)

# Interact with dropdown to fix a parameter, then use slider
# to explore different values while viewing 2D surface of remaining params

# ============================================================
# Example 2: 4D Parameter Analysis
# ============================================================
df, fig = explorer.plot_nd_parameter_analysis(
    module_name='macd',
    param_names=['fast', 'slow', 'signal', 'threshold'],
    metric=metric,
    base_model=model,
    plot_type='surface'
)

# Use two dropdowns + sliders to fix 2 parameters,
# view 2D surface of remaining 2 parameters

# ============================================================
# Example 3: Comprehensive Report
# ============================================================
report = explorer.generate_parameter_sensitivity_report(
    module_name='rsi',
    param_names=None,  # All params for module
    metric_threshold=1.0,
    metric=metric,
    base_model=model,
    export_path='research/reports/rsi_param_sensitivity.txt',
    verbose=True
)

# Access results
print(f"Robustness Score: {report['robustness_scores']['overall_score']:.1f}/10")
print(f"Rating: {report['robustness_scores']['rating']}")
print(f"Best Configuration: {report['summary_stats']['max_config']}")
print(f"Optimal Region: {report['recommendations']['optimal_region']}")

# Access figures
if 'figures' in report:
    # Show appropriate figure based on parameter count
    n_params = len(report['param_names'])
    if n_params == 1:
        fig = report['figures']['1d_line']
    elif n_params == 2:
        fig = report['figures']['2d_surface']
    elif n_params == 3:
        fig = report['figures']['3d_interactive']
    elif n_params == 4:
        fig = report['figures']['4d_interactive']
    
    fig.show()

# ============================================================
# Example 4: Robustness Analysis Only (no plots)
# ============================================================
from eda.parameter_analysis import ParameterAnalyzer

analyzer = ParameterAnalyzer(features_df, targets_df)

# Get parameter grid
param_grid = explorer._extract_parameter_grid(
    module_name='rsi',
    param_names=['lookback', 'threshold']
)

# Analyze
df = analyzer.analyze_2d_parameters(
    feature_grid=param_grid,
    target_col='log_return',
    metric=metric,
    base_model=model
)

# Compute robustness metrics
robustness = analyzer.compute_robustness_metrics(
    results_df=df,
    metric_col='sortino',
    metric_threshold=1.0
)

print(f"Robustness Score: {robustness['overall_score']:.1f}/10")
print(f"Parameter Sensitivity Ranking:")
for param, contribution in sorted(
    robustness['parameter_sensitivity'].items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {param}: {contribution*100:.1f}%")
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Test File**: `tests/test_parameter_sensitivity.py`

```python
def test_compute_robustness_metrics():
    """Test robustness metric computation with known values."""
    
def test_parameter_sensitivity_ranking():
    """Test parameter variance contribution calculation."""
    
def test_extract_parameter_grid_3d():
    """Test parameter grid extraction for 3 parameters."""
    
def test_extract_parameter_grid_4d():
    """Test parameter grid extraction for 4 parameters."""
    
def test_robustness_rating_thresholds():
    """Test rating assignment based on score thresholds."""
```

### 8.2 Integration Tests

**Test File**: `tests/integration/test_parameter_analysis_integration.py`

```python
def test_3d_visualization_end_to_end():
    """Test 3D parameter analysis with real feature data."""
    
def test_4d_visualization_end_to_end():
    """Test 4D parameter analysis with real feature data."""
    
def test_report_generation_end_to_end():
    """Test full report generation from extraction to export."""
    
def test_interactive_controls():
    """Test that Plotly interactive controls work correctly."""
```

### 8.3 Visual Regression Tests

- Compare generated plots against reference images
- Verify interactive controls work correctly (manual testing)
- Test on multiple browsers/devices for interactive HTML exports

---

## 9. Success Criteria

- ✓ Support 1-4 parameter dimensions with appropriate visualizations
- ✓ Interactive dropdowns/sliders for 3-4D exploration
- ✓ Comprehensive robustness scoring (variance + consistency + risk)
- ✓ Human-readable text report with actionable insights
- ✓ Standalone method (not integrated into `generate_summary_report`)
- ✓ Use existing parameter combinations from metadata
- ✓ Maintain functional programming principles (pure functions where possible)
- ✓ Type hints for all function signatures
- ✓ Comprehensive docstrings with examples
- ✓ No external dependencies beyond existing stack (Plotly, pandas, numpy)

---

## 10. Future Enhancements (Out of Scope)

- Export interactive plots to standalone HTML files
- Parameter optimization recommendations using gradient-based methods
- Statistical significance testing for parameter effects (ANOVA)
- Parallel computation for large parameter spaces (>1000 combinations)
- Integration with walk-forward validation
- Parameter sensitivity across multiple timeframes/instruments
- Comparison reports (compare robustness of multiple modules)
- Automated parameter range suggestions based on historical robustness

---

## 11. Implementation Checklist

### Phase 0: Fix Existing 2D Implementation (PRIORITY)
- [ ] **Fix `plot_2d_parameter_surface()`** - Debug and resolve current issues
- [ ] **Fix `analyze_2d_parameters()`** - Ensure correct parameter grid processing
- [ ] **Test 2D visualization** - Verify with real feature data
- [ ] **Document root cause** - Identify and document what was broken

### Phase 1: Core Infrastructure
- [ ] Add `analyze_nd_parameters()` to `ParameterAnalyzer`
- [ ] Add `compute_robustness_metrics()` to `ParameterAnalyzer`
- [ ] Add `_extract_parameter_grid()` to `FeatureExplorer`
  - [ ] Support metadata-based parameter extraction
  - [ ] Support column-name-based parameter extraction (fallback)
  - [ ] Handle standardized column naming convention
  - [ ] Filter by module_name and feature_name correctly
- [ ] Write unit tests for core functions
- [ ] **Integration Test**: Verify parameter extraction from dataframes with many features

### Phase 2: Visualization
- [ ] Add `plot_3d_parameter_interactive()` to `parameter_plots.py`
- [ ] Add `plot_4d_parameter_interactive()` to `parameter_plots.py`
- [ ] Add `plot_nd_parameter_analysis()` to `FeatureExplorer`
- [ ] Test interactive controls manually

### Phase 3: Report Generation
- [ ] Add `_export_parameter_sensitivity_report()` to `FeatureExplorer`
- [ ] Add `generate_parameter_sensitivity_report()` to `FeatureExplorer`
- [ ] Test report generation end-to-end
- [ ] Verify text report formatting

### Phase 4: Documentation & Testing
- [ ] Write comprehensive docstrings with examples
- [ ] Add integration tests
- [ ] Create usage examples in notebooks
- [ ] Update README with new capabilities

---

## 12. Notes & Considerations

### Current Implementation Status
- **1D (1 parameter)**: ✅ Working - `plot_parameter_sensitivity()` is functional
- **2D (2 parameters)**: ❌ **BROKEN** - `plot_2d_parameter_surface()` needs to be fixed before implementing 3D/4D
- **3D/4D**: Not yet implemented - Blocked until 2D is fixed

**Action Required**: Debug and fix 2D implementation first to ensure consistent architecture and avoid propagating issues to higher dimensions.

### Integration with FeatureExplorer Pipeline
- **Input Format**: System receives dataframes with many features (potentially hundreds of columns)
- **Column Naming**: All features follow standardized naming convention: `{module}_{feature}_{tf}_{param1}_{val1}_{param2}_{val2}...`
- **Parameter Extraction**: Must work with both:
  - Metadata-based extraction (from `FeatureExtractor.feature_metadata`)
  - Column-name-based parsing (fallback using `helpers.parse_feature_column_name()`)
- **Existing Infrastructure**: Leverage `FeatureExplorer._param_mapping` and `_feature_groups` for efficiency
- **Compatibility**: Must integrate seamlessly with existing `plot_parameter_sensitivity()` and `plot_2d_parameter_surface()` methods

### Performance
- Large parameter spaces (>500 combinations) may be slow
- Consider caching results for repeated analysis
- Progress bars for long-running computations

### Edge Cases
- Handle constant features (zero variance)
- Handle missing parameter combinations gracefully
- Validate parameter names exist in metadata
- Handle features with insufficient samples

### User Experience
- Clear error messages when parameters not found
- Helpful warnings when parameter space is sparse
- Interactive plot tooltips with full configuration details
- Export HTML option for sharing interactive plots

### Code Quality
- Follow existing architecture patterns
- Use pure functions where possible
- Comprehensive type hints
- Detailed docstrings with examples
- Unit test coverage >80%
