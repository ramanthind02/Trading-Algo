<!-- ff20bd66-27b3-43f0-89d7-9006d17310af 2eacb7b8-caef-464f-b603-2b48b9aa206a -->
# Time Series Feature Wrapper Node Implementation

## Overview

Create a new `TimeSeriesFeatureNode` class in `nodes/ts_feature.py` that wraps any BiasNode and applies time series feature extraction functions from functime.

## Implementation Details

### 1. Create TimeSeriesFeatureNode Class (`nodes/ts_feature.py`)

**Key Components:**

- Inherits from `BiasNode`
- Takes parameters:
  - `wrapped_node`: Another BiasNode instance
  - `lookback`: Rolling window size (int)
  - `transformation`: functime feature extraction function (e.g., `mean_abs_change`, `autocorrelation`)
  - `transformation_name`: String name for the transformation (for column naming)
  - `transformation_args`: Optional args for the transformation function

**State Management:**

- Maintain rolling windows for each output from wrapped node (one window per output value)
- Store outputs in Polars Series/lazy frames for efficient processing
- Track datetimes for alignment

**Column Naming:**

- Use pattern: `{originalFeatureName}_{transformationName}_{lookback}`
- Example: `rsi_signal_D_lookback14_meanAbsChange_60`
- Handle wrapped node's `get_column_names()` to get base feature names

**Lazy Computation:**

- During `add_candle()`: Call wrapped node, store outputs in windows, return placeholders (NaN or wrapped node outputs)
- Store windows as Polars lazy frames for efficiency
- Compute features only when needed (via a method that gets called at extraction time)

### 2. Integration Points

**During Backtest (`add_candle` method):**

- Call `wrapped_node.add_candle(candle)` to get outputs
- Append each output value to its respective rolling window
- Maintain windows as collections (will convert to Polars at computation time)
- Return wrapped node outputs directly (so existing pipeline works)

**At Feature Extraction:**

- Need to hook into feature extraction pipeline
- Options:
  - A) Add method `compute_features_from_windows()` that processes stored windows
  - B) Override how columns/outputs are handled
  - C) Post-process matrix_df after backtest completes

**Recommended Approach:**

- Store raw outputs during backtest
- Add a static method or utility function that post-processes the matrix_df
- Alternatively: Create a special extraction path that detects TimeSeriesFeatureNode and computes features before returning matrix_df

### 3. Polars Integration

**Window Storage:**

- Use collections (deque or list) during backtest for efficient appending
- Convert to Polars Series when computing features
- Use Polars lazy evaluation for feature computation

**Feature Computation:**

- For each output window:
  - Convert to Polars Series
  - Apply rolling window using Polars expressions
  - Call transformation function on window
  - Return computed feature value

### 4. Helper Functions

**In `utils/helpers.py` or new module:**

- Function to post-process matrix_df and compute TS features for TimeSeriesFeatureNode instances
- Function to identify and extract TimeSeriesFeatureNode instances from MLManager

### 5. Node Factory Integration

**Update `utils/helpers.py` `create_bias_node()`:**

- Handle TimeSeriesFeatureNode creation
- Support params like: `{'wrapped_module': 'rsi', 'wrapped_params': {...}, 'lookback': 60, 'transformation': 'mean_abs_change'}`

## Files to Create/Modify

1. **Create `nodes/ts_feature.py`**

   - TimeSeriesFeatureNode class
   - Integration with functime functions

2. **Modify `utils/helpers.py`**

   - Update `create_bias_node()` to handle TimeSeriesFeatureNode specs
   - Add helper function for post-processing TS features

3. **Update `nodes/__init__.py`**

   - Export TimeSeriesFeatureNode

4. **Update `feature_extraction/ml_manager.py`** (if needed)

   - Add hook for post-processing TimeSeriesFeatureNode outputs
   - Or add method to compute TS features before returning matrix_df

## Example Usage

```python
# In bias_node_specs:
{
    'module_name': 'ts_feature',
    'timeframes': [TimeFrame.D],
    'params': {
        'wrapped_module': 'rsi',
        'wrapped_params': {'lookback': 14},
        'lookback': 60,
        'transformation': 'mean_abs_change',
        'transformation_name': 'meanAbsChange'
    }
}
```

This would create: `rsi_signal_D_lookback14_meanAbsChange_60`

### To-dos

- [ ] Create nodes/ts_feature.py with TimeSeriesFeatureNode class that wraps another BiasNode and maintains rolling windows
- [ ] Implement window storage using collections during backtest, with methods to convert to Polars for lazy computation
- [ ] Implement column naming logic that appends _{transformationName}_{lookback} to base feature names
- [ ] Integrate functime transformation functions, handling both Series and Expr inputs
- [ ] Create helper function in utils/helpers.py to post-process matrix_df and compute TS features at extraction time
- [ ] Update utils/helpers.py create_bias_node() to handle TimeSeriesFeatureNode creation from specs
- [ ] Update nodes/__init__.py to export TimeSeriesFeatureNode
- [ ] Add hook in feature_extraction/ml_manager.py to trigger TS feature computation before returning matrix_df