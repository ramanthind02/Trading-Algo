Reviewing the current feature selector implementation to draft high-level specs for the in-sample and out-of-sample feature selectors.


[3 tools called]


High-level specifications for the in-sample and out-of-sample feature selectors:

## Specifications: In-Sample and Out-of-Sample Feature Selectors

### 1. File Renaming

**Task**: Rename `feature_selection/feature_selector.py` → `feature_selection/is_feature_selector.py`

**Changes**:
- Rename file
- Rename class `FeatureSelector` → `ISFeatureSelector` (In-Sample Feature Selector)
- Update all imports across the codebase
- Update docstrings to clarify this is for in-sample analysis

---

### 2. New Out-of-Sample Feature Selector (`os_feature_selector.py`)

**Purpose**: Out-of-sample validation framework for BaseModel features using walk-forward, cross-validation, and permutation tests. Supports multiple features simultaneously, following the same workflow as `FeatureExplorer`.

#### 2.1 Class Structure

**Class Name**: `OSFeatureSelector`

**Initialization**:
```python
def __init__(
    self,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Similarities to FeatureExplorer/ISFeatureSelector**:
- Same initialization pattern as `FeatureExplorer` (features_df, targets_df, metadata)
- Works with multiple features simultaneously (columns in `features_df`)
- Same internal data structure (`self.df` combining features and targets)
- Same basic validation and data alignment
- Same `results` dictionary for storing test outputs
- Supports `target_col` parameter in all methods (like `ISFeatureSelector`)

**Differences from ISFeatureSelector**:
- Accepts `features_df` (DataFrame) instead of single `feature_data` (Series)
- Supports multiple features in a single instance
- Focus on out-of-sample validation methods only
- No in-sample analysis methods (e.g., `plot_rolling_decile_whiskers`)
- All methods return structured results suitable for feature selection decisions
- Methods can operate on single features or all features at once

**Workflow**:
```python
# Step 1: Extract features using FeatureExtractor (same as FeatureExplorer)
from feature_extraction.feature_extractor_class import FeatureExtractor
extractor = FeatureExtractor(ticker=Ticker.SPY)
result = extractor.extract(bias_node_specs=[...])

# Step 2: Create OSFeatureSelector from extracted dataframes (same pattern)
from feature_selection.os_feature_selector import OSFeatureSelector
selector = OSFeatureSelector(
    features_df=result['features'],
    targets_df=result['targets'],
    metadata=result
)

# Step 3: Run out-of-sample tests
results = selector.walkforward_test(
    model=model,
    objective_metric=metric,
    feature_cols=['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],  # Multiple features
    target_col='log_return',  # Specify target column
    ...
)
```

---

#### 2.2 Core Methods

##### 2.2.1 Walk-Forward Test

**Method**: `walkforward_test()`

**Purpose**: Out-of-sample walk-forward validation using any BaseModel. Supports single or multiple features.

**Parameters**:
- `model: BaseModel` - Model instance (e.g., QuantileBinningModel)
- `objective_metric: ObjectiveMetric` - Custom metric (e.g., SortinoRatio, SharpeRatio)
- `feature_cols: Union[str, List[str]]` - Single feature column name or list of feature columns to test
- `train_start: datetime` - Initial training window start
- `train_end: datetime` - Initial training window end
- `test_step: int` - Days per test period (default: 252)
- `num_steps: int` - Number of walk-forward steps (default: 10)
- `strategy: str` - 'long' or 'short' (default: 'long')
- `target_col: str` - Target column for binning (default: 'log_return')
- `normalization_data: Optional[pd.DataFrame]` - Volatility data DataFrame if model requires it (columns match feature_cols)
- `verbose: bool` - Print progress (default: True)

**Returns**:
- If single feature (`feature_cols` is str): Returns same as current `ISFeatureSelector.walkforward_analysis()`
  - `pd.DataFrame` with step-level results
  - `List[Dict]` with detailed step info
  - `Optional[plt.Figure]` if plotting enabled
- If multiple features (`feature_cols` is List[str]): Returns `Dict[str, Tuple]`
  - Keys: feature column names
  - Values: Tuple of (results_df, step_info, fig) for each feature
- Aggregated results stored in `self.results['walkforward_test']`:
  - `results_by_feature`: Dict mapping feature names to their results
  - `summary_df`: DataFrame summarizing all features with columns:
    - `feature`: Feature name
    - `final_oos_metric`: Final metric on aggregated returns
    - `mean_oos_metric`: Mean metric across steps
    - `total_trades`: Total trades
    - Additional summary statistics

**Behavior**:
- Uses `WalkForwardSplitter` for data splitting
- For each feature: fit model on train, predict on test, compute metric
- Aggregates OOS returns across all steps for each feature
- Computes final metric on aggregated returns (not mean of step metrics)
- Handles normalization data if model requires it (can be DataFrame with multiple columns)
- Stores results in `self.results['walkforward_test']`
- If multiple features, processes them sequentially (or in parallel if `n_jobs > 1`)

---

##### 2.2.2 Cross-Validation Test

**Method**: `cv_test()`

**Purpose**: K-fold cross-validation for out-of-sample validation. Supports single or multiple features.

**Parameters**:
- `model: BaseModel` - Model instance
- `objective_metric: ObjectiveMetric` - Custom metric
- `feature_cols: Union[str, List[str]]` - Single feature column name or list of feature columns to test
- `n_splits: int` - Number of CV folds (default: 5)
- `strategy: str` - 'long' or 'short' (default: 'long')
- `target_col: str` - Target column (default: 'log_return')
- `normalization_data: Optional[pd.DataFrame]` - Volatility data DataFrame if needed (columns match feature_cols)
- `shuffle: bool` - Whether to shuffle before splitting (default: False, preserve time order)
- `verbose: bool` - Print progress (default: True)

**Returns**:
- If single feature (`feature_cols` is str): Returns same format as walkforward_test for single feature
- If multiple features (`feature_cols` is List[str]): Returns `Dict[str, Tuple]` with results for each feature
- Aggregated results stored in `self.results['cv_test']`:
  - `results_by_feature`: Dict mapping feature names to their results
  - `summary_df`: DataFrame summarizing all features

**Behavior**:
- Uses `sklearn.model_selection.KFold` (or time-series variant)
- For each feature: fit on train, predict on test, compute metric
- Aggregates OOS returns across folds for each feature
- Computes final metric on aggregated returns
- Handles normalization data (can be DataFrame with multiple columns)
- Stores results in `self.results['cv_test']`

---

##### 2.2.3 Permutation Tests

**Method**: `permutation_test()`

**Purpose**: Permutation-based robustness testing using feature shuffling to assess whether a feature's predictive power is statistically significant. Supports single or multiple features with modular, DRY design.

**Key Design Principle**: The permutation test should be **modular and DRY** by accepting an objective function that encapsulates the CV/walkforward method. The permutation test remains blissfully unaware of how the metric is computed.

**Parameters**:
- `feature_cols: Union[str, List[str]]` - Single feature column name or list of feature columns to test
- `objective_func: Callable` - **Objective function that includes the test method (CV/walkforward)**. This function:
  - Takes `(feature_data: pd.Series, target_data: pd.Series, normalization_data: Optional[pd.Series])` as input
  - Returns a float representing the metric value
  - Encapsulates the entire test logic (walkforward/CV) internally
  - Example: `lambda feat, tgt, norm: run_walkforward_test(feat, tgt, norm, model, metric, ...)`
- `permutation_mode: str` - Permutation strategy: 'full_shuffle' or 'fold_shuffle' (default: 'full_shuffle')
- `nreps: int` - Number of permutation replications (default: 100)
- `target_col: str` - Target column (default: 'log_return')
- `normalization_data: Optional[pd.DataFrame]` - Volatility data DataFrame if needed (columns match feature_cols)
- `alpha: float` - Significance level (default: 0.05)
- `random_seed: Optional[int]` - Random seed (default: None)
- `n_jobs: int` - Parallel jobs (-1 = all CPUs, default: -1)
- `verbose: bool` - Print progress (default: True)

**Returns**:
- `pd.DataFrame` with columns:
  - `feature`: Feature name (if multiple features tested)
  - `original_metric`: Original metric value (before permutation)
  - `pval`: Permutation p-value (fraction of permutations >= original)
  - `significant`: Boolean (pval <= alpha)
  - `n_valid_permutations`: Number of successful permutations
  - `permutation_mode`: Which strategy was used
  - Additional statistics (mean permuted metric, std, etc.)

**Behavior**:
- **Modular Design**: Uses `PermutationEngine` from `utils.permutation_test.permutation_engine`
- **DRY Principle**: The `objective_func` encapsulates all test logic (walkforward/CV), so permutation test doesn't need to know details
- For each feature:
  1. Extract feature data and target data from `self.df`
  2. Run original test via `objective_func` to get baseline metric
  3. For each permutation replication:
     - Apply permutation strategy (full_shuffle or fold_shuffle)
     - Re-run test via `objective_func` on permuted data
     - Compute metric on permuted results
     - Compare to original metric
- P-value = fraction of permutations where permuted metric >= original metric
- Low p-value (< alpha) indicates robustness (original metric is unlikely by chance)
- Supports parallel execution via `n_jobs` (both across features and across permutations)
- Stores results in `self.results['permutation_test']`

**Example Usage**:
```python
# Create objective function that encapsulates walkforward test
def create_walkforward_objective(model, objective_metric, train_start, train_end, 
                                 test_step, num_steps, strategy, target_col):
    """Factory function to create objective function for walkforward test."""
    def objective_func(feature_data, target_data, normalization_data=None):
        # Run walkforward test internally
        results = run_walkforward_test(
            model=model,
            feature_data=feature_data,
            target_data=target_data,
            normalization_data=normalization_data,
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps,
            strategy=strategy,
            target_col=target_col
        )
        # Return final metric from aggregated returns
        return results['final_oos_metric']
    return objective_func

# Use in permutation test
objective_func = create_walkforward_objective(
    model=model,
    objective_metric=metric,
    train_start=datetime(2000, 1, 1),
    train_end=datetime(2010, 1, 1),
    test_step=252,
    num_steps=10,
    strategy='long',
    target_col='log_return'
)

results = selector.permutation_test(
    feature_cols=['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
    objective_func=objective_func,
    permutation_mode='full_shuffle',
    target_col='log_return',
    nreps=100
)
```

**Refactoring Notes**:
- Refactor existing permutation code to use `PermutationEngine` from `utils.permutation_test.permutation_engine`
- Create helper functions to construct objective functions for walkforward and CV tests
- The permutation test should delegate all test-specific logic to the `objective_func`
- This makes the permutation test reusable for any test type (walkforward, CV, insample, etc.)

---

#### 2.2.3.1 Permutation Strategies for Walk-Forward Tests

**Two Approaches**:

##### Approach 1: Full Dataset Shuffle (`permutation_mode='full_shuffle'`)

**Algorithm**:
1. Run original walk-forward test to get baseline metric
2. For each permutation:
   - Shuffle the entire feature dataset (globally, preserving index alignment with targets)
   - Re-run the complete walk-forward test from scratch on shuffled data
   - Fit model on each training window, predict on each test window
   - Aggregate OOS returns and compute metric
   - Compare permuted metric to original metric

**Pros**:
- Tests the full pipeline: model fitting + prediction on shuffled data
- More conservative: Tests if feature has predictive power when completely randomized
- Realistic: Simulates what would happen if feature had no relationship to target
- Tests model robustness: Ensures model doesn't overfit to spurious patterns
- Better for detecting false discoveries: If feature is truly random, model should fail on shuffled data

**Cons**:
- Computationally expensive: Must re-run entire walk-forward test for each permutation (nreps × num_steps model fits)
- Slower: Each permutation requires full model training and prediction
- May be overly conservative: Could reject features that have weak but real predictive power
- Temporal structure destroyed: Global shuffle breaks all temporal relationships

**Use Case**: When you want to test if the feature has ANY predictive power, regardless of temporal structure.

---

##### Approach 2: OOS Fold Shuffle (`permutation_mode='fold_shuffle'`)

**Algorithm**:
1. Run original walk-forward test to get baseline metric and collect OOS returns from all test folds
2. For each permutation:
   - Shuffle only the OOS returns across test folds (preserve within-fold structure)
   - Re-compute metric on shuffled OOS returns
   - Compare permuted metric to original metric

**Pros**:
- Computationally efficient: Only requires one walk-forward test run, then simple shuffling
- Fast: No model refitting needed, just shuffling and metric computation
- Preserves model decisions: Tests if the model's signal selection is robust to return shuffling
- Tests temporal stability: Checks if good performance is due to consistent patterns across folds
- Better for detecting overfitting: If model is overfitting, shuffling returns should break the pattern

**Cons**:
- Less conservative: Doesn't test if feature itself is predictive, only if returns are well-distributed
- Doesn't test model fitting: Model is only fitted once, so doesn't test if model can learn from shuffled features
- May miss false discoveries: A feature could have no predictive power but still pass if returns happen to align well
- Assumes model decisions are correct: Only tests the aggregation of returns, not the feature-model relationship

**Use Case**: When you want to test if the model's performance is robust to return distribution, or when computational resources are limited.

---

#### 2.2.3.2 Permutation Strategies for Cross-Validation Tests

**Two Approaches**:

##### Approach 1: Full Dataset Shuffle (`permutation_mode='full_shuffle'`)

**Algorithm**:
1. Run original CV test to get baseline metric
2. For each permutation:
   - Shuffle the entire feature dataset (globally, preserving index alignment with targets)
   - Re-run the complete CV test from scratch on shuffled data
   - Fit model on each training fold, predict on each test fold
   - Aggregate OOS returns and compute metric
   - Compare permuted metric to original metric

**Pros**:
- Tests full pipeline: Model fitting + prediction on completely randomized data
- More conservative: Strong test of feature predictive power
- Realistic null hypothesis: Tests if feature has any relationship to target
- Better for feature selection: More likely to reject non-predictive features

**Cons**:
- Computationally expensive: Must re-run entire CV test for each permutation (nreps × n_splits model fits)
- Slower: Each permutation requires full model training
- May be overly conservative: Could reject weakly predictive features
- Temporal structure destroyed: Global shuffle breaks time-series relationships (if present)

**Use Case**: When you want a strong test of feature predictive power and have computational resources.

---

##### Approach 2: OOS Fold Shuffle (`permutation_mode='fold_shuffle'`)

**Algorithm**:
1. Run original CV test to get baseline metric and collect OOS returns from all test folds
2. For each permutation:
   - Shuffle only the OOS returns across CV folds (preserve within-fold structure)
   - Re-compute metric on shuffled OOS returns
   - Compare permuted metric to original metric

**Pros**:
- Computationally efficient: Only requires one CV test run, then simple shuffling
- Fast: No model refitting needed
- Preserves model decisions: Tests if model's predictions are robust to return shuffling
- Tests cross-fold consistency: Checks if good performance is consistent across folds
- Better for detecting overfitting: If model overfits to specific folds, shuffling should break it

**Cons**:
- Less conservative: Doesn't test if feature is predictive, only if returns are well-distributed
- Doesn't test model fitting: Model is only fitted once
- May miss false discoveries: Could pass even if feature has no predictive power
- Assumes model decisions are correct: Only tests aggregation, not feature-model relationship

**Use Case**: When you want to test cross-fold consistency or when computational resources are limited.

---

#### 2.2.3.3 Recommended Usage

**For Walk-Forward Tests**:
- **Use `full_shuffle`** when:
  - You want the strongest test of feature predictive power
  - Computational resources allow (can be slow with many steps)
  - You're doing initial feature screening
  - You want to detect false discoveries early
  
- **Use `fold_shuffle`** when:
  - Computational resources are limited
  - You've already validated features with other methods
  - You want to test temporal stability of returns
  - You're doing final validation on a large feature set

**For CV Tests**:
- **Use `full_shuffle`** when:
  - You want the strongest test of feature predictive power
  - You have computational resources
  - You're doing initial feature screening
  
- **Use `fold_shuffle`** when:
  - Computational resources are limited
  - You want to test cross-fold consistency
  - You're doing final validation

**Default**: `full_shuffle` (more conservative, better for feature selection)

---

#### 2.2.3.4 Implementation Details

**Full Shuffle Implementation**:
```python
# For each permutation:
permuted_feature = feature_data.sample(frac=1.0, random_state=seed).values
permuted_feature = pd.Series(permuted_feature, index=feature_data.index)
# Re-run complete test (walkforward/cv) with permuted_feature
results = run_test(model, permuted_feature, target_data, ...)
permuted_metric = objective_metric.compute(results['aggregated_oos_returns'])
```

**Fold Shuffle Implementation**:
```python
# Run original test once
original_results = run_test(model, feature_data, target_data, ...)
original_oos_returns = original_results['aggregated_oos_returns']

# For each permutation:
permuted_oos_returns = original_oos_returns.sample(frac=1.0, random_state=seed)
permuted_metric = objective_metric.compute(permuted_oos_returns)
```

**For Walk-Forward with Full Shuffle**:
- Shuffle feature globally, then re-run walk-forward test
- Each training window sees shuffled feature values
- Each test window sees predictions based on model trained on shuffled data

**For Walk-Forward with Fold Shuffle**:
- Run walk-forward test once, collect OOS returns from each test fold
- Shuffle the OOS returns across folds (preserve temporal order within each fold)
- Re-compute metric on shuffled returns

**For CV with Full Shuffle**:
- Shuffle feature globally, then re-run CV test
- Each training fold sees shuffled feature values
- Each test fold sees predictions based on model trained on shuffled data

**For CV with Fold Shuffle**:
- Run CV test once, collect OOS returns from each test fold
- Shuffle the OOS returns across folds
- Re-compute metric on shuffled returns

---

#### 2.3 Supporting Methods

##### 2.3.1 Results Access

**Method**: `get_results(test_name: str) -> Dict[str, Any]`
- Retrieve stored results for a specific test
- Returns the dictionary stored in `self.results[test_name]`

**Method**: `get_summary() -> pd.DataFrame`
- Aggregate summary across all tests run
- Columns: test_name, metric_value, pval (if permutation), significant, etc.

##### 2.3.2 Utility Methods

**Method**: `_extract_normalization_data(model: BaseModel, feature_col: str) -> Optional[pd.Series]`
- Extract normalization column from `self.df` based on model's `normalize_by` attribute
- Similar logic to ISFeatureSelector, but works with specific feature column
- Returns Series for single feature, or DataFrame for multiple features

**Method**: `_validate_model_and_metric(model: BaseModel, objective_metric: ObjectiveMetric) -> None`
- Validate that model implements BaseModel interface
- Validate that objective_metric implements ObjectiveMetric interface

**Method**: `_create_walkforward_objective_func(model, objective_metric, train_start, train_end, test_step, num_steps, strategy, target_col) -> Callable`
- Factory function to create objective function for walkforward permutation tests
- Returns a callable that takes `(feature_data, target_data, normalization_data)` and returns metric
- Encapsulates all walkforward logic internally

**Method**: `_create_cv_objective_func(model, objective_metric, n_splits, strategy, target_col) -> Callable`
- Factory function to create objective function for CV permutation tests
- Returns a callable that takes `(feature_data, target_data, normalization_data)` and returns metric
- Encapsulates all CV logic internally

---

#### 2.4 Integration with Existing Code

**Dependencies**:
- `feature_selection.base_models.base_model.BaseModel` - Base model interface
- `utils.objective_metric.*` - Objective metric classes
- `feature_selection.walkforward.walkforward_model.WalkForwardSplitter` - For walk-forward splitting
- `sklearn.model_selection.KFold` - For CV splitting
- `utils.permutation_test.permutation_engine.PermutationEngine` - **Core permutation infrastructure (REQUIRED)**
- `utils.permutation_test.permutation_engine.FeaturePermutationStrategy` - For feature shuffling
- `feature_extraction.feature_extractor_class.FeatureExtractor` - For feature extraction (same workflow as FeatureExplorer)

**Similarities to FeatureExplorer/ISFeatureSelector**:
- Same initialization pattern as `FeatureExplorer` (features_df, targets_df, metadata)
- Same workflow: Extract features → Create selector → Run tests
- Same data structure and validation
- Similar handling of normalization data (extended to support multiple features)
- Similar verbose output formatting
- Similar results storage pattern
- Supports `target_col` parameter in all methods (like `ISFeatureSelector`)

**Key Differences**:
- Accepts `features_df` (DataFrame) instead of single `feature_data` (Series)
- Supports multiple features simultaneously
- Focus on out-of-sample validation only
- All methods return structured results suitable for feature selection
- Permutation tests use modular, DRY design with objective functions
- No visualization methods (delegated to separate plotting modules)
- Results are quantitative and decision-oriented
- Uses `PermutationEngine` for all permutation tests (refactored from existing code)

---

### 3. Implementation Notes

#### 3.1 Permutation Test Implementation (Modular & DRY)

**Core Design**: Use `PermutationEngine` from `utils.permutation_test.permutation_engine` which is already modular and DRY.

**Objective Function Pattern**:
```python
# Objective function encapsulates the test method (walkforward/CV)
def objective_func(feature_data: pd.Series, target_data: pd.Series, 
                   normalization_data: Optional[pd.Series] = None) -> float:
    """
    Run test (walkforward/CV) and return metric.
    
    This function is passed to PermutationEngine, which will:
    1. Call it with original data to get baseline metric
    2. Permute the data
    3. Call it with permuted data for each replication
    4. Compare metrics
    """
    # Run walkforward or CV test internally
    results = run_walkforward_or_cv_test(
        feature_data=feature_data,
        target_data=target_data,
        normalization_data=normalization_data,
        model=model,
        objective_metric=objective_metric,
        # ... other test-specific params ...
    )
    return results['final_oos_metric']
```

**Using PermutationEngine**:
```python
from utils.permutation_test.permutation_engine import (
    PermutationEngine, 
    FeaturePermutationStrategy
)

# Create strategy
strategy = FeaturePermutationStrategy()

# Create engine
engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)

# Create criterion function that wraps objective_func
def criterion_func(data: pd.DataFrame, feature_col: str) -> float:
    """Extract feature and target, call objective_func."""
    feature_data = data[feature_col]
    target_data = data[target_col]
    normalization_data = data.get(normalization_col) if normalization_col else None
    return objective_func(feature_data, target_data, normalization_data)

# Run permutation test
results = engine.run_permutation_test(
    data=self.df,  # Combined features + targets DataFrame
    feature_cols=feature_cols,
    criterion_func=criterion_func,
    nreps=nreps,
    random_seed=random_seed,
    alpha=alpha,
    # Strategy-specific kwargs (e.g., train_windows for walkforward)
    **strategy_kwargs
)
```

**Full Shuffle Mode**:
- Uses `FeaturePermutationStrategy` with global shuffling
- For each permutation: shuffle feature values, call `objective_func` with permuted data
- `objective_func` internally runs complete test (walkforward/CV) on shuffled data

**Fold Shuffle Mode**:
- Requires custom permutation strategy or modification to `objective_func`
- `objective_func` should:
  1. Run test once and collect OOS returns
  2. For each permutation: shuffle OOS returns, recompute metric
- Alternative: Create `FoldShufflePermutationStrategy` that shuffles returns after test

**Walk-Forward Considerations**:
- Full shuffle: Shuffle feature globally, then re-run walk-forward (each step fits model on shuffled data)
- Fold shuffle: Run walk-forward once, collect OOS returns per step, shuffle returns across steps
- For walk-forward, pass `train_windows` to `FeaturePermutationStrategy.permute()` to shuffle within windows

**CV Considerations**:
- Full shuffle: Shuffle feature globally, then re-run CV (each fold fits model on shuffled data)
- Fold shuffle: Run CV once, collect OOS returns per fold, shuffle returns across folds

**Refactoring Existing Code**:
- Refactor `utils.permutation_test.permute_bars.py` to use `PermutationEngine` pattern
- Create helper functions in `os_feature_selector.py`:
  - `_create_walkforward_objective_func()` - Factory for walkforward objective
  - `_create_cv_objective_func()` - Factory for CV objective
- These factories encapsulate all test-specific logic, making permutation test DRY

#### 3.2 Objective Metric Interface

**Expected Interface**:
```python
class ObjectiveMetric:
    def compute(self, returns: pd.Series) -> float:
        """Compute metric on returns series."""
        pass
```

**Examples**: `SortinoRatio`, `SharpeRatio`, `ProfitFactor`, etc.

#### 3.3 Results Storage

**Structure**:
```python
self.results = {
    'walkforward_test': {
        'results_df': pd.DataFrame,
        'aggregated_results': Dict,
        'config': Dict  # Parameters used
    },
    'cv_test': {...},
    'permutation_test': {
        'results_df': pd.DataFrame,
        'config': Dict
    }
}
```

---

### 4. Testing Strategy

**Unit Tests**:
- Test initialization with `features_df` and `targets_df` (single and multiple features)
- Test data validation (matching indices, required columns)
- Test walk-forward test with single feature
- Test walk-forward test with multiple features
- Test CV test with single and multiple features
- Test permutation test with walkforward objective function
- Test permutation test with CV objective function
- Test permutation test with both `full_shuffle` and `fold_shuffle` modes
- Test results storage and retrieval (single and multiple features)
- Test `target_col` parameter in all methods
- Test error handling (insufficient data, invalid parameters, missing features)
- Test normalization data handling (single and multiple features)

**Integration Tests**:
- Test with real feature data from `FeatureExtractor`
- Test with different BaseModel types
- Test with different ObjectiveMetric types
- Test permutation test robustness with various objective functions
- Test workflow: FeatureExtractor → OSFeatureSelector → Results
- Test multiple features processed in parallel
- Test that permutation test is truly DRY (can swap objective functions)

---

### 5. Migration Plan

1. **Refactor Permutation Code** (if needed):
   - Ensure `PermutationEngine` in `utils.permutation_test.permutation_engine` supports the objective function pattern
   - Refactor any existing permutation code to use `PermutationEngine`
   - Create helper functions for constructing objective functions

2. **Rename FeatureSelector**:
   - Rename `feature_selector.py` → `is_feature_selector.py`
   - Update class name `FeatureSelector` → `ISFeatureSelector`
   - Update all imports across the codebase

3. **Create OSFeatureSelector**:
   - Create `os_feature_selector.py` with new class
   - Implement initialization with `features_df` and `targets_df` (following `FeatureExplorer` pattern)
   - Implement `walkforward_test()` with single and multiple feature support
   - Implement `cv_test()` with single and multiple feature support
   - Implement `permutation_test()` using `PermutationEngine` with objective function pattern
   - Add helper methods: `_create_walkforward_objective_func()`, `_create_cv_objective_func()`
   - Support `target_col` parameter in all methods

4. **Update Imports**:
   - Update any code that imports `FeatureSelector` to use `ISFeatureSelector`
   - Add imports for `OSFeatureSelector` where needed

5. **Add Unit Tests**:
   - Add comprehensive unit tests for `OSFeatureSelector`
   - Test single and multiple features
   - Test permutation test with different objective functions
   - Test `target_col` parameter

6. **Update Documentation**:
   - Update docstrings and examples
   - Document the objective function pattern for permutation tests
   - Document the workflow (FeatureExtractor → OSFeatureSelector)

---

### 6. Design Principles

- **Single Responsibility**: OSFeatureSelector focuses on out-of-sample validation
- **DRY (Don't Repeat Yourself)**: Permutation tests use objective functions to avoid code duplication
- **Modularity**: Permutation tests are modular - they don't need to know test implementation details
- **Flexibility**: Accepts any BaseModel and ObjectiveMetric; supports any test type via objective functions
- **Consistency**: Similar API to `FeatureExplorer` and `ISFeatureSelector` where appropriate
- **Multiple Features**: Supports single or multiple features in a single instance
- **Workflow Consistency**: Follows same workflow as `FeatureExplorer` (FeatureExtractor → Selector)
- **Extensibility**: Easy to add new test types or permutation strategies via objective functions
- **Performance**: Support parallel execution for permutation tests (across features and replications)
- **Clarity**: Clear separation between in-sample and out-of-sample analysis
- **Target Column Flexibility**: All methods support `target_col` parameter for flexibility

These specs provide a high-level blueprint for implementing the in-sample and out-of-sample feature selectors.