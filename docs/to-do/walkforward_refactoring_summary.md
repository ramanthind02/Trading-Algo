# FeatureValidator Refactoring Summary

## Date: 2026-01-26

## Motivation

The user requested a simplification of the `FeatureValidator` API to follow the single responsibility principle more strictly. The original implementation mixed portfolio construction and validation responsibilities.

## Changes Made

### Before: Dual-Mode API

The original API supported two modes:
1. **Feature-level mode**: Accepted `feature_cols`, created portfolios internally
2. **Portfolio-level mode**: Accepted `portfolio`, used it directly

This created complexity:
- Feature selection logic in the validator
- Portfolio/ensemble construction in the validator
- Many parameters for binning models, bias nodes, strategies, etc.
- Unclear separation of concerns

### After: Simplified Portfolio-Agnostic API

The new API has a single mode:
- **Always accepts a Portfolio instance** (required parameter)
- User constructs the portfolio beforehand (single-feature or multi-ensemble)
- Validator just repeatedly calls `fit_from_candles()` and `predict_from_candles()`
- No feature selection or portfolio construction logic

## API Changes

### walkforward_test()

**Before:**
```python
def walkforward_test(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    feature_cols: Optional[Union[str, List[str]]] = None,  # REMOVED
    target_col: str = 'log_return',
    portfolio: Optional[Portfolio] = None,  # NOW REQUIRED
    candles_df: Optional[pd.DataFrame] = None,  # NOW REQUIRED
    bias_node_specs: Optional[List[Dict[str, Any]]] = None,  # REMOVED
    objective_metric: Optional[ObjectiveMetric] = None,  # REMOVED
    strategy: str = 'long',  # REMOVED
    target_volatility: float = 0.20,  # REMOVED
    binning_model_type: str = 'QuantileBinningModel',  # REMOVED
    binning_model_params: Optional[Dict[str, Any]] = None,  # REMOVED
    normalization_data: Optional[pd.DataFrame] = None,  # REMOVED
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
```

**After:**
```python
def walkforward_test(
    self,
    portfolio: Portfolio,  # REQUIRED, moved to first position
    candles_df: pd.DataFrame,  # REQUIRED, moved to second position
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    target_col: str = 'log_return',
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
```

### walkforward_permutation_test()

**Before:**
```python
def walkforward_permutation_test(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    feature_cols: Optional[Union[str, List[str]]] = None,  # REMOVED
    target_col: str = 'log_return',
    portfolio: Optional[Portfolio] = None,  # NOW REQUIRED
    candles_df: Optional[pd.DataFrame] = None,  # NOW REQUIRED
    nreps: int = 100,
    alpha: float = 0.05,
    random_seed: Optional[int] = 42,
    n_jobs: int = -1,
    shuffle_target: bool = False,
    exclude_features: Optional[List[str]] = None,
    verbose: bool = True,
    **walkforward_kwargs
) -> pd.DataFrame:
```

**After:**
```python
def walkforward_permutation_test(
    self,
    portfolio: Portfolio,  # REQUIRED, moved to first position
    candles_df: pd.DataFrame,  # REQUIRED, moved to second position
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    target_col: str = 'log_return',
    nreps: int = 100,
    alpha: float = 0.05,
    random_seed: Optional[int] = 42,
    n_jobs: int = -1,
    shuffle_target: bool = False,
    exclude_features: Optional[List[str]] = None,
    verbose: bool = True,
    **walkforward_kwargs
) -> pd.DataFrame:
```

## Removed Code

### Methods Removed

1. **`_select_best_feature_variation()`** (~50 lines)
   - Used to select best feature from multiple variations
   - User now handles this before creating portfolio

2. **`_create_portfolio_for_feature()`** (~100 lines)
   - Used to create Portfolio with single ensemble + base model
   - Complex logic with temp files, serialization, etc.
   - User now handles this before calling validator

3. **`_infer_bias_node_spec()`** (~20 lines)
   - Used to parse feature column names
   - No longer needed

### Imports Removed

- `DiversifiedEnsemble` - no longer creating ensembles
- `create_base_model_from_config` - no longer creating base models
- `TimeFrame`, `Ticker` - no longer serializing enums (except in _copy_portfolio)
- `ObjectiveMetric` - no longer doing feature selection
- `generate_tearsheet` - was unused

### Output Fields Removed

- `selected_feature` field removed from fold results and summary DataFrame
- This field tracked which feature was selected (for multi-variation testing)
- No longer applicable since user constructs portfolio

## Benefits

1. **Simpler API**: Fewer parameters, clearer responsibility
2. **Single Responsibility**: Validator only validates, doesn't construct
3. **More Flexible**: User has full control over portfolio construction
4. **Less Code**: ~170 lines of code removed
5. **Easier to Understand**: No complex branching between modes
6. **Better Separation of Concerns**: Portfolio construction is user's responsibility

## Migration Guide

**Old Usage (Feature-Level):**
```python
validator = FeatureValidator(features_df, targets_df)
fold_results, summary_df, metrics = validator.walkforward_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    feature_cols='rsi_signal_D_lookback_14',
    candles_df=candles_df,
    strategy='long',
    binning_model_type='QuantileBinningModel'
)
```

**New Usage:**
```python
# Step 1: Create portfolio (user's responsibility)
portfolio = create_single_feature_portfolio(
    feature_col='rsi_signal_D_lookback_14',
    strategy='long'
)

# Step 2: Run validation
validator = FeatureValidator(features_df, targets_df)
fold_results, summary_df, metrics = validator.walkforward_test(
    portfolio=portfolio,
    candles_df=candles_df,
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1)
)
```

**Old Usage (Portfolio-Level):**
```python
validator = FeatureValidator(features_df, targets_df)
fold_results, summary_df, metrics = validator.walkforward_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    portfolio=my_portfolio,
    candles_df=candles_df
)
```

**New Usage (unchanged except parameter order):**
```python
validator = FeatureValidator(features_df, targets_df)
fold_results, summary_df, metrics = validator.walkforward_test(
    portfolio=my_portfolio,  # Now first parameter (required)
    candles_df=candles_df,  # Now second parameter (required)
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1)
)
```

## Implementation Details

### Key Changes in Workflow

**Before:**
1. Check if portfolio provided
2. If no portfolio: select best feature, create portfolio
3. If portfolio: use it directly
4. Copy portfolio
5. Fit/predict on fold

**After:**
1. Copy portfolio (always provided)
2. Fit/predict on fold

### Simplified Per-Fold Logic

**Before:**
```python
if portfolio is not None:
    fold_portfolio = self._copy_portfolio(portfolio)
    selected_feature = None
else:
    selected_feature = self._select_best_feature_variation(...)
    bias_node_spec = self._infer_bias_node_spec(selected_feature)
    fold_portfolio = self._create_portfolio_for_feature(
        feature_column=selected_feature,
        bias_node_spec=bias_node_spec,
        ...
    )
```

**After:**
```python
fold_portfolio = self._copy_portfolio(portfolio)
```

## Test Updates

Tests were updated to create portfolios before calling `walkforward_test()`:

```python
def create_test_portfolio(features_df, feature_col='rsi_signal_D_lookback_14'):
    """Create a minimal test portfolio with single feature."""
    # ... create ensemble from control file ...
    portfolio = Portfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.D,
        target_volatility=0.20
    )
    return portfolio
```

All tests pass with the new API.

## Documentation Updates

Updated documentation files:
1. `docs/to-do/walkforward.md` - Full specification
2. `docs/to-do/walkforward_implementation_summary.md` - Implementation summary
3. `tests/test_feature_validator.py` - Test examples

## Backwards Compatibility

**Breaking Change**: This is a breaking API change.

Old code calling `walkforward_test()` with `feature_cols` will break. Users must:
1. Create portfolio beforehand
2. Pass portfolio as required parameter
3. Remove feature selection parameters

## Future Enhancements

With the simplified API, future enhancements are clearer:

1. **Helper Functions**: Create helper functions for common portfolio patterns
   - `create_single_feature_portfolio()`
   - `create_multi_feature_portfolio()`
   - `load_portfolio_from_vault()`

2. **Portfolio Builders**: Fluent API for portfolio construction
   - `PortfolioBuilder().add_ensemble(...).build()`

3. **Feature Selection**: Separate class for feature selection
   - `FeatureSelector` to select best features
   - Returns best feature name
   - User creates portfolio from selected feature

4. **Parallel Execution**: Add parallelization at fold level
   - Process multiple folds in parallel using joblib
   - Currently sequential

## Conclusion

The refactoring successfully simplified the `FeatureValidator` API by removing portfolio construction responsibilities and making it truly portfolio-agnostic. The class now has a single, clear purpose: validate portfolios using walkforward analysis.

The user is now responsible for portfolio construction, which provides more flexibility and follows the single responsibility principle more strictly.
