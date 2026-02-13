# FeatureValidator - Walkforward Testing Specifications

## Overview

The `FeatureValidator` class provides a comprehensive walkforward testing framework for validating portfolios. It reuses existing infrastructure (Portfolio, PortfolioTester, WalkForwardSplitter) to maintain DRY principles while providing a clean, portfolio-agnostic API.

### Design Philosophy

The validator is **agnostic to portfolio structure**:
- Always accepts a Portfolio instance (required parameter)
- User constructs the portfolio beforehand (single-feature or multi-ensemble)
- Validator just repeatedly calls `fit_from_candles()` and `predict_from_candles()` on the portfolio
- No feature selection or portfolio construction logic in the validator

### Use Cases

1. **Single-Feature Testing**: Test individual features
   - User creates minimal Portfolio with single ensemble + single base model
   - Validator fits/predicts on each fold
   - Useful for feature validation

2. **Multi-Ensemble Testing**: Test full portfolios
   - User creates Portfolio with multiple ensembles, custom weights, etc.
   - Validator fits/predicts on each fold
   - Useful for portfolio backtesting

## Design Principles

1. **Reuse Existing Infrastructure**: Leverage `Portfolio`, `PortfolioTester`, and `WalkForwardSplitter` classes
2. **DRY (Don't Repeat Yourself)**: Avoid duplicating code from `OSFeatureSelector` or walkforward logic
3. **Single Responsibility**: Focus on walkforward validation workflow, delegate everything else
   - Delegate portfolio construction to user
   - Delegate model fitting to Portfolio.fit_from_candles()
   - Delegate prediction to Portfolio.predict_from_candles()
   - Delegate metrics to PortfolioTester
4. **Portfolio Agnostic**: Work with any Portfolio instance
   - No assumptions about portfolio structure
   - No feature selection logic
   - No portfolio creation logic
5. **Simple API**: Minimal parameters, clear responsibility

## Class Structure

```python
class FeatureValidator:
    """
    Walkforward feature validation framework.
    
    Validates features using walkforward analysis by:
    1. Splitting data into train/test folds using WalkForwardSplitter
    2. For each fold:
       a. Fit a Portfolio with single ensemble + single base model on training data
       b. Generate predictions on test data
       c. Calculate returns and metrics using PortfolioTester
    3. Aggregate results across all folds
    
    Supports both single features and rule-based features with multiple variations.
    For rule-based features, selects best variation using objective metric on training data.
    """
```

## Input Parameters

### Required Parameters

- `portfolio: Portfolio`
  - Portfolio instance to test (required)
  - Can be single-feature or multi-ensemble
  - User constructs this beforehand
  - Will be copied and re-fitted on each fold

- `candles_df: pd.DataFrame`
  - OHLC candles data (required)
  - Used for fitting and prediction
  - Must contain columns: datetime, open, high, low, close, volume, ticker, timeframe

### Walkforward Parameters

- `train_start: datetime`
  - Start date for initial training window

- `train_end: datetime`
  - End date for initial training window

- `test_step: int = 252`
  - Number of days for test period (~1 year)
  - Also the roll-forward step size

- `num_steps: int = 10`
  - Number of walkforward steps to perform

### Optional Parameters

- `target_col: str = 'log_return'`
  - Target column name (for validation only, not used in fitting)
  - Options: 'raw_return', 'log_return', 'log_return_atr', 'log_return_ewsd'

- `verbose: bool = True`
  - Print progress information

## Output Structure

### Per-Fold Results (Nested Array)

Returns a list of dictionaries, one per fold:

```python
List[Dict[str, Any]]
```

Each fold dictionary contains:

- `fold: int` - Fold number (0-indexed)
- `train_start: datetime` - Training window start date
- `train_end: datetime` - Training window end date
- `test_start: datetime` - Test window start date
- `test_end: datetime` - Test window end date
- `n_train: int` - Number of training samples
- `n_test: int` - Number of test samples
- `selected_feature: str` - Feature column selected (for rule-based features)
- `positions_df: pd.DataFrame` - Position fractions for test period
  - Columns: ticker, datetime, forecast_score, position_fraction
- `strategy_returns: pd.Series` - Strategy returns for test period (indexed by datetime)
- `baseline_returns: pd.Series` - Baseline returns for test period (indexed by datetime)
- `metrics: Dict[str, float]` - Metrics computed on test returns
  - Keys: 'sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown', etc.
  - Computed using PortfolioTester.generate_tearsheet() with mode='metrics'

### Aggregated Results

Additionally returns a summary DataFrame:

```python
pd.DataFrame
```

Columns:
- `fold: int` - Fold number
- `train_start: datetime`
- `train_end: datetime`
- `test_start: datetime`
- `test_end: datetime`
- `selected_feature: str` - Feature column selected
- `sharpe_ratio: float`
- `sortino_ratio: float`
- `total_return: float`
- `max_drawdown: float`
- `n_trades: int` - Number of trades in test period
- `win_rate: float` - Win rate (fraction of positive returns)

## Implementation Details

### 1. Data Splitting

Reuse `WalkForwardSplitter` from `feature_selection/walkforward/walkforward_model.py`:

```python
from feature_selection.walkforward.walkforward_model import WalkForwardSplitter

splitter = WalkForwardSplitter(
    train_start=train_start,
    train_end=train_end,
    test_step=test_step,
    num_steps=num_steps
)

splits = splitter.split(features_df.index)
```

This returns a list of `(train_indices, test_indices)` tuples.

### 2. Portfolio Setup (Per Fold)

For each fold, either use provided portfolio or create a minimal portfolio structure:

**If portfolio is provided:**
```python
# Use provided portfolio (will be re-fitted on training data)
portfolio = provided_portfolio.copy()  # Create copy to avoid modifying original
# Note: Portfolio will be re-fitted via fit_from_candles() on each fold
```

**If portfolio is None (create new):**
```python
# Create single base model
base_model = create_base_model_from_config(
    config={
        'name': feature_column,
        'feature_column': feature_column,
        'model_type': binning_model_type,
        'strategy': strategy,
        'constructor_params': binning_model_params,
        'bias_node_spec': bias_node_spec  # From feature extraction
    },
    tickers=tickers  # Extract from features_df if available
)

# Create single ensemble with single base model
ensemble = DiversifiedEnsemble(
    target_volatility=target_volatility,
    control_file_path=None  # Create from scratch
)
ensemble.base_models[feature_column] = base_model

# Create portfolio with single ensemble
portfolio = Portfolio(
    ensembles=[ensemble],
    trading_timeframe=TimeFrame.D,  # Infer from data or parameter
    target_volatility=target_volatility
)
```

**Important**: When using an existing portfolio, it will be re-fitted on each fold's training data. This allows testing full portfolios (with multiple ensembles, custom weights, etc.) using walkforward validation.

### 3. Feature Selection for Rule-Based Features

When multiple feature variations are provided:

```python
if len(feature_cols) > 1 and bias_node_specs is not None:
    # Fit all variations on training data
    training_metrics = {}
    
    for feature_col, bias_spec in zip(feature_cols, bias_node_specs):
        # Create base model for this variation
        base_model = create_base_model_from_config(...)
        
        # Fit on training data
        train_features = features_df.iloc[train_idx][feature_col]
        train_targets = targets_df.iloc[train_idx][target_col]
        base_model.fit(train_features, train_targets)
        
        # Generate predictions on training data
        train_signals = base_model.predict(train_features, strategy=strategy)
        train_returns = train_targets[train_signals > 0]
        
        # Compute objective metric
        if len(train_returns) > 0:
            metric_value = objective_metric.compute(train_returns)
            training_metrics[feature_col] = metric_value
        else:
            training_metrics[feature_col] = -np.inf
    
    # Select best feature
    best_feature = max(training_metrics.items(), key=lambda x: x[1])[0]
else:
    # Single feature - use as-is
    best_feature = feature_cols[0] if isinstance(feature_cols, str) else feature_cols[0]
```

### 4. Per-Fold Workflow

For each fold:

```python
for fold, (train_idx, test_idx) in enumerate(splits):
    # 1. Select best feature (if multiple variations and portfolio is None)
    if portfolio is None:
        selected_feature = select_best_feature(...)
        # Create portfolio with selected feature
        portfolio = create_portfolio_for_feature(selected_feature, ...)
    else:
        # Use provided portfolio (no feature selection needed)
        selected_feature = None  # Not applicable for portfolio-level testing
        # Create copy to avoid modifying original
        portfolio = _copy_portfolio(portfolio)
    
    # 2. Get training candles and targets
    train_candles = extract_candles_from_features(features_df.iloc[train_idx], ...)
    train_targets = targets_df.iloc[train_idx][target_col]
    
    # 3. Fit portfolio on training data (re-fits all ensembles/base models)
    portfolio.fit_from_candles(train_candles, train_targets)
    
    # 4. Get test candles
    test_candles = extract_candles_from_features(features_df.iloc[test_idx], ...)
    
    # 5. Generate predictions
    positions_df = portfolio.predict_from_candles(test_candles)
    
    # 6. Calculate returns and metrics using PortfolioTester
    tester = PortfolioTester(portfolio)
    tester.positions_df = positions_df
    strategy_returns = tester.calculate_strategy_returns(test_candles)
    baseline_returns = tester.calculate_baseline_returns(test_candles)
    
    # 7. Compute metrics (extract from tearsheet)
    metrics = extract_metrics_from_tearsheet(
        strategy_returns, baseline_returns, mode='metrics'
    )
    
    # 8. Store fold results
    fold_results.append({
        'fold': fold,
        'train_start': features_df.index[train_idx].min(),
        'train_end': features_df.index[train_idx].max(),
        'test_start': features_df.index[test_idx].min(),
        'test_end': features_df.index[test_idx].max(),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'selected_feature': selected_feature,  # None for portfolio-level testing
        'positions_df': positions_df,
        'strategy_returns': strategy_returns,
        'baseline_returns': baseline_returns,
        'metrics': metrics
    })
```

### 5. Aggregation

After all folds:

```python
# Create summary DataFrame
summary_rows = []
for fold_result in fold_results:
    summary_rows.append({
        'fold': fold_result['fold'],
        'train_start': fold_result['train_start'],
        'train_end': fold_result['train_end'],
        'test_start': fold_result['test_start'],
        'test_end': fold_result['test_end'],
        'selected_feature': fold_result['selected_feature'],
        **fold_result['metrics']  # Unpack metrics dict
    })

summary_df = pd.DataFrame(summary_rows)

# Also compute aggregate metrics across all folds
all_strategy_returns = pd.concat([r['strategy_returns'] for r in fold_results])
all_baseline_returns = pd.concat([r['baseline_returns'] for r in fold_results])
aggregate_metrics = extract_metrics_from_tearsheet(
    all_strategy_returns, all_baseline_returns, mode='metrics'
)
```

## API Design

### Main Method

```python
def walkforward_test(
    self,
    portfolio: Portfolio,
    candles_df: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    target_col: str = 'log_return',
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
    """
    Perform walkforward validation on a portfolio.
    
    The validator is agnostic to portfolio structure - it works with any Portfolio
    instance (single base model or full multi-ensemble portfolio). For each fold,
    it copies the portfolio, fits on training data, predicts on test data, and
    computes metrics.
    
    Parameters
    ----------
    portfolio : Portfolio
        Portfolio instance to test (required). Can be:
        - Minimal portfolio with single ensemble + single base model
        - Full portfolio with multiple ensembles and custom configurations
    candles_df : pd.DataFrame
        Candles DataFrame with OHLC data (required for fitting/prediction)
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int, default=252
        Number of days for test period (~1 year)
    num_steps : int, default=10
        Number of walkforward steps
    target_col : str, default='log_return'
        Target column to use (only for validation, not used in fitting)
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    Tuple[List[Dict], pd.DataFrame, Dict]
        (fold_results, summary_df, aggregate_metrics)
        - fold_results: List of per-fold result dictionaries
        - summary_df: Summary DataFrame with one row per fold
        - aggregate_metrics: Aggregated metrics across all folds
    """
```

### Permutation Test Method

```python
def walkforward_permutation_test(
    self,
    portfolio: Portfolio,
    candles_df: pd.DataFrame,
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
    """
    Perform walkforward permutation test.
    
    Compares walkforward performance on real data vs permuted data.
    Uses two-region shuffling: first training window vs remaining data.
    """
```

### Helper Methods

```python
def _copy_portfolio(
    self,
    portfolio: Portfolio
) -> Portfolio:
    """Create a shallow copy of portfolio to avoid modifying original."""
    # Creates new Portfolio instance with same configuration
    # Portfolio will be re-fitted on training data

def _filter_candles(
    self,
    candles_df: pd.DataFrame,
    date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Filter candles DataFrame to date range."""

def _extract_metrics_from_returns(
    self,
    strategy_returns: pd.Series,
    baseline_returns: pd.Series
) -> Dict[str, float]:
    """Extract metrics from strategy and baseline returns using quantstats."""

def _create_walkforward_criterion_func(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    target_col: str,
    portfolio: Portfolio,
    candles_df: pd.DataFrame,
    verbose: bool,
    **walkforward_kwargs
) -> Callable:
    """Create criterion function for permutation test."""
```

## Edge Cases

### 1. Rule-Based Features with Multiple Variations

**Problem**: When a rule-based feature has multiple parameter variations (e.g., RSI with lookback=[5, 10, 14]), we need to select the best one in training data.

**Solution**:
- Accept `bias_node_specs` list matching `feature_cols`
- Fit all variations on training data
- Compute objective metric for each
- Select best variation for test period
- Store selected feature in fold results

### 2. Missing Candles Data

**Problem**: Features/targets DataFrames may not contain OHLC candle data needed for PortfolioTester.

**Solution**:
- Add `candles_df` parameter to `walkforward_test()` method
- If candles_df is provided, use it directly
- If None, attempt to extract from features_df/targets_df metadata
- Raise clear error if candles data cannot be found

**Note**: For portfolio-level testing, candles_df is typically available from the original data source.

### 3. Multi-Ticker Features

**Problem**: Features may span multiple tickers, requiring proper aggregation.

**Solution**:
- Extract tickers from features_df if 'ticker' column exists
- Pass tickers to base model creation
- Portfolio handles multi-ticker aggregation automatically

### 4. Insufficient Data in Fold

**Problem**: A fold may have insufficient training or test samples.

**Solution**:
- Use `WalkForwardSplitter` which already filters splits with insufficient data
- Minimum thresholds: 100 training samples, 10 test samples (from walkforward_model.py)
- Skip fold if insufficient data (log warning)

## Integration with Existing Code

### Reused Components

1. **WalkForwardSplitter** (`feature_selection/walkforward/walkforward_model.py`)
   - Handles all data splitting logic
   - Returns list of (train_idx, test_idx) tuples

2. **Portfolio** (`ensemble/portfolio.py`)
   - Handles model fitting and prediction
   - Applies risk management (IDM, instrument weights, position capping)

3. **PortfolioTester** (`ensemble/portfolio_tester.py`)
   - Calculates strategy returns from positions
   - Generates tearsheets with metrics
   - Handles baseline comparison

4. **DiversifiedEnsemble** (`ensemble/diversified_ensemble.py`)
   - Wraps base models
   - Handles volatility scaling

5. **BaseModel** (`feature_selection/base_models/feature_base_model.py`)
   - Owns bias nodes and binning models
   - Handles feature extraction and prediction

### New Components

1. **FeatureValidator** (new class)
   - Orchestrates walkforward workflow
   - Handles feature selection for rule-based features
   - Aggregates results across folds

## Example Usage

### Single-Feature Testing

```python
from feature_selection.feature_validator import FeatureValidator
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from datetime import datetime

# Step 1: Create portfolio with single feature
# (User constructs this beforehand - see helper function below)
portfolio = create_single_feature_portfolio(
    feature_col='rsi_signal_D_lookback_14',
    strategy='long'
)

# Step 2: Initialize validator
validator = FeatureValidator(
    features_df=features_df,
    targets_df=targets_df
)

# Step 3: Run walkforward test
fold_results, summary_df, aggregate_metrics = validator.walkforward_test(
    portfolio=portfolio,
    candles_df=candles_df,
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10
)

# Access results
print(summary_df)  # Summary per fold
print(aggregate_metrics)  # Aggregated metrics
print(fold_results[0]['strategy_returns'])  # Returns for first fold
```

### Multi-Ensemble Testing

```python
from ensemble.vault_manager import load_ensemble_from_vault

# Step 1: Load or create portfolio with multiple ensembles
portfolio = Portfolio(
    ensembles=[
        load_ensemble_from_vault('vault/D/rsi_long'),
        load_ensemble_from_vault('vault/D/momentum_long')
    ],
    trading_timeframe=TimeFrame.D,
    target_volatility=0.20
)

# Step 2: Initialize validator
validator = FeatureValidator(
    features_df=features_df,
    targets_df=targets_df
)

# Step 3: Run walkforward test
fold_results, summary_df, aggregate_metrics = validator.walkforward_test(
    portfolio=portfolio,
    candles_df=candles_df,
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10
)
```

### Permutation Testing

```python
# Run permutation test on portfolio
results = validator.walkforward_permutation_test(
    portfolio=portfolio,
    candles_df=candles_df,
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10,
    nreps=100,
    alpha=0.05,
    shuffle_target=False  # Shuffle features (default)
)

# Check significant features
significant = results[results['significant']]
print(f"Significant: {len(significant)}/{len(results)}")
```

## Walkforward Permutation Testing

### Overview

Walkforward permutation testing validates whether features/portfolios have genuine predictive power by comparing their walkforward performance against permuted (shuffled) data. This is a feature-based permutation test that shuffles feature values (or target values) while preserving the walkforward structure.

**Key Principle**: If a feature/portfolio performs better on real data than on permuted data, it suggests genuine predictive power rather than spurious correlations.

### Design: Two-Region Permutation

For walkforward permutation tests, we use a **two-region shuffling approach**:

1. **First Training Window**: Shuffled as a single unit
   - Region: `[train_start, train_end)`
   - All features shuffled independently within this region only

2. **Remaining Data**: Shuffled as another single unit
   - Region: `[train_end, data_end)`
   - Includes all subsequent training windows and test windows
   - All features shuffled independently within this region only

**Why Two Regions?**
- **Prevents data contamination**: First training window is isolated from future data
- **Maintains temporal structure**: No information leakage between regions
- **Realistic null hypothesis**: Tests if features predict within each temporal region
- **Walkforward integrity**: Each fold's training data is shuffled independently from its test data

**Alternative (NOT recommended)**: Global shuffling across all data would:
- ❌ Allow future data to contaminate first training window
- ❌ Break temporal structure of walkforward validation
- ❌ Create unrealistic null hypothesis

**Note**: This two-region approach is already implemented in `FeaturePermutationStrategy.permute()` via the `train_windows` parameter. We just need to pass the two regions!

### Reusing Existing Infrastructure

**✅ Can Reuse**: `FeaturePermutationStrategy` from `utils/permutation_test/permutation_engine.py`

The existing `FeaturePermutationStrategy.permute()` method already supports `train_windows` parameter:

```python
def permute(
    self,
    data: pd.DataFrame,
    random_seed: int,
    feature_cols: List[str],
    exclude_features: Optional[List[str]] = None,
    train_windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    **kwargs
) -> pd.DataFrame:
```

**How it works**:
- If `train_windows` is provided, shuffles features independently within each region
- Each region is shuffled separately (no cross-contamination)
- Perfect for walkforward permutation tests!

**Implementation Note**: We just need to pass two regions:
```python
train_windows = [
    (train_start, train_end),  # First training window
    (train_end, data_end)      # All remaining data
]
```

### API Design

```python
def walkforward_permutation_test(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    feature_cols: Optional[Union[str, List[str]]] = None,
    target_col: str = 'log_return',
    portfolio: Optional[Portfolio] = None,
    candles_df: Optional[pd.DataFrame] = None,
    nreps: int = 100,
    alpha: float = 0.05,
    random_seed: Optional[int] = 42,
    n_jobs: int = -1,
    shuffle_target: bool = False,
    exclude_features: Optional[List[str]] = None,
    **walkforward_kwargs
) -> pd.DataFrame:
    """
    Perform walkforward permutation test.
    
    Compares walkforward performance on real data vs permuted data.
    Uses two-region shuffling: first training window vs remaining data.
    
    Parameters
    ----------
    shuffle_target : bool, default=False
        If True, shuffle target values instead of feature values.
        If False, shuffle feature values (default).
    
    exclude_features : List[str], optional
        Features to NOT shuffle (e.g., ATR for normalization).
        Only used if shuffle_target=False.
    
    nreps : int, default=100
        Number of permutation replications
    
    alpha : float, default=0.05
        Significance level
    
    random_seed : int, optional
        Random seed for reproducibility
    
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all CPUs)
    
    **walkforward_kwargs
        Additional arguments passed to walkforward_test()
    
    Returns
    -------
    pd.DataFrame
        Results with columns:
        - feature: Feature/portfolio identifier
        - original_metric: Original walkforward metric value
        - pval: Permutation p-value
        - significant: Boolean (pval <= alpha)
        - n_valid_permutations: Number of successful permutations
    """
```

### Implementation Details

#### 1. Create Criterion Function

The criterion function runs walkforward_test on permuted data:

```python
def _create_walkforward_criterion_func(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    target_col: str,
    portfolio: Optional[Portfolio],
    candles_df: pd.DataFrame,
    **walkforward_kwargs
) -> Callable:
    """
    Create criterion function for permutation test.
    
    This function will be called by PermutationEngine for each permutation.
    It runs walkforward_test on the (potentially permuted) data and returns
    the aggregated metric.
    
    IMPORTANT: The data parameter is already permuted by PermutationEngine.
    We just need to extract features/targets and run walkforward_test.
    """
    def criterion_func(data: pd.DataFrame, feature_col: str) -> float:
        """
        Run walkforward test and return aggregated metric.
        
        Args:
            data: DataFrame with features and targets (ALREADY PERMUTED by engine)
            feature_col: Feature column name (or 'portfolio' for portfolio-level)
        
        Returns:
            Aggregated metric value (e.g., Sharpe ratio across all folds)
        """
        # Extract features and targets from permuted data
        # Note: data is already permuted by PermutationEngine
        if feature_col == 'portfolio':
            # Portfolio-level: use all feature columns (portfolio will extract what it needs)
            features_df = data.drop(columns=[target_col]).copy()
        else:
            # Feature-level: use specific feature column
            features_df = data[[feature_col]].copy()
        
        targets_df = data[[target_col]].copy()
        
        # Temporarily replace self.features_df and self.targets_df with permuted data
        # This allows walkforward_test() to use permuted data transparently
        original_features = self.features_df
        original_targets = self.targets_df
        
        try:
            self.features_df = features_df
            self.targets_df = targets_df
            
            # Run walkforward test on permuted data
            fold_results, summary_df, aggregate_metrics = self.walkforward_test(
                train_start=train_start,
                train_end=train_end,
                test_step=test_step,
                num_steps=num_steps,
                feature_cols=feature_col if feature_col != 'portfolio' else None,
                target_col=target_col,
                portfolio=portfolio,
                candles_df=candles_df,
                **walkforward_kwargs
            )
            
            # Return primary metric (e.g., Sharpe ratio)
            # Could also use Sortino, total return, etc.
            return aggregate_metrics.get('sharpe_ratio', 0.0)
        
        except Exception as e:
            # If walkforward test fails on permuted data, return 0.0
            # This will be counted as "permuted < original" in p-value calculation
            if verbose:
                print(f"  Warning: Walkforward test failed on permuted data: {e}")
            return 0.0
        
        finally:
            # Restore original data
            self.features_df = original_features
            self.targets_df = original_targets
    
    return criterion_func
```

**Note**: The criterion function temporarily replaces `self.features_df` and `self.targets_df` with permuted data. This allows `walkforward_test()` to work transparently without modification. The original data is restored in a `finally` block.

#### 2. Set Up Two-Region Permutation

```python
# Get data end date
data_end = max(features_df.index.max(), targets_df.index.max())

# Create two regions for permutation
train_windows = [
    (train_start, train_end),  # First training window
    (train_end, data_end)      # All remaining data
]

# Combine features and targets into single DataFrame
combined_df = pd.concat([features_df, targets_df], axis=1)
combined_df = combined_df.dropna()  # Remove any NaN rows
```

#### 3. Handle Target vs Feature Shuffling

```python
if shuffle_target:
    # Shuffle target instead of features
    # Strategy: Create a custom permutation that shuffles target within regions
    # Note: This requires a custom strategy or modification to FeaturePermutationStrategy
    # For now, we can shuffle target manually:
    strategy = FeaturePermutationStrategy()
    # Temporarily rename target_col to a feature name for shuffling
    combined_df_renamed = combined_df.rename(columns={target_col: '_temp_target'})
    permuted_df = strategy.permute(
        data=combined_df_renamed,
        random_seed=random_seed,
        feature_cols=['_temp_target'],
        train_windows=train_windows
    )
    permuted_df = permuted_df.rename(columns={'_temp_target': target_col})
else:
    # Shuffle features (default)
    strategy = FeaturePermutationStrategy()
    permuted_df = strategy.permute(
        data=combined_df,
        random_seed=random_seed,
        feature_cols=feature_cols,
        exclude_features=exclude_features,
        train_windows=train_windows
    )
```

#### 4. Run Permutation Test

```python
from utils.permutation_test.permutation_engine import PermutationEngine, FeaturePermutationStrategy

# Create strategy
strategy = FeaturePermutationStrategy()

# Create engine
engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)

# Create criterion function
criterion_func = self._create_walkforward_criterion_func(
    train_start=train_start,
    train_end=train_end,
    test_step=test_step,
    num_steps=num_steps,
    target_col=target_col,
    portfolio=portfolio,
    candles_df=candles_df,
    **walkforward_kwargs
)

# Run permutation test
results = engine.run_permutation_test(
    data=combined_df,
    feature_cols=feature_cols if not shuffle_target else [target_col],
    criterion_func=criterion_func,
    nreps=nreps,
    random_seed=random_seed,
    alpha=alpha,
    train_windows=train_windows,  # Pass to strategy via strategy_kwargs
    exclude_features=exclude_features if not shuffle_target else None
)
```

### Complete Implementation Flow

```python
def walkforward_permutation_test(
    self,
    train_start: datetime,
    train_end: datetime,
    test_step: int = 252,
    num_steps: int = 10,
    feature_cols: Optional[Union[str, List[str]]] = None,
    target_col: str = 'log_return',
    portfolio: Optional[Portfolio] = None,
    candles_df: Optional[pd.DataFrame] = None,
    nreps: int = 100,
    alpha: float = 0.05,
    random_seed: Optional[int] = 42,
    n_jobs: int = -1,
    shuffle_target: bool = False,
    exclude_features: Optional[List[str]] = None,
    verbose: bool = True,
    **walkforward_kwargs
) -> pd.DataFrame:
    """
    Perform walkforward permutation test.
    """
    # 1. Validate inputs
    if portfolio is None and feature_cols is None:
        raise ValueError("Either portfolio or feature_cols must be provided")
    
    # 2. Prepare combined DataFrame
    combined_df = pd.concat([self.features_df, self.targets_df], axis=1)
    combined_df = combined_df.dropna()
    
    # 3. Determine data end date
    data_end = combined_df.index.max()
    
    # 4. Create two-region train_windows
    train_windows = [
        (train_start, train_end),  # First training window
        (train_end, data_end)      # All remaining data
    ]
    
    # 5. Determine what to shuffle
    if shuffle_target:
        shuffle_cols = [target_col]
        exclude_features = None  # Not applicable for target shuffling
    else:
        if isinstance(feature_cols, str):
            shuffle_cols = [feature_cols]
        else:
            shuffle_cols = feature_cols
        # exclude_features already set by parameter
    
    # 6. Create criterion function
    criterion_func = self._create_walkforward_criterion_func(
        train_start=train_start,
        train_end=train_end,
        test_step=test_step,
        num_steps=num_steps,
        target_col=target_col,
        portfolio=portfolio,
        candles_df=candles_df,
        feature_cols=feature_cols,
        **walkforward_kwargs
    )
    
    # 7. Create permutation strategy and engine
    strategy = FeaturePermutationStrategy()
    engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
    
    # 8. Run permutation test
    results = engine.run_permutation_test(
        data=combined_df,
        feature_cols=shuffle_cols,
        criterion_func=criterion_func,
        nreps=nreps,
        random_seed=random_seed,
        alpha=alpha,
        train_windows=train_windows,  # Passed to strategy via strategy_kwargs
        exclude_features=exclude_features
    )
    
    # 9. Add metadata
    results['permutation_mode'] = 'walkforward_two_region'
    results['shuffle_target'] = shuffle_target
    results['n_valid_permutations'] = nreps  # Assuming all successful
    
    return results
```

### Example Usage

#### Feature Shuffling (Default)

```python
# Run walkforward permutation test with feature shuffling
results = validator.walkforward_permutation_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10,
    feature_cols='rsi_signal_D_lookback_14',
    target_col='log_return',
    candles_df=candles_df,
    nreps=100,
    alpha=0.05,
    shuffle_target=False  # Shuffle features (default)
)

# Check significant features
significant = results[results['significant']]
print(f"Significant features: {len(significant)}/{len(results)}")
print(significant[['feature', 'original_metric', 'pval']])
```

#### Target Shuffling

```python
# Run walkforward permutation test with target shuffling
results = validator.walkforward_permutation_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10,
    feature_cols='rsi_signal_D_lookback_14',
    target_col='log_return',
    candles_df=candles_df,
    nreps=100,
    shuffle_target=True  # Shuffle target instead of features
)
```

#### Portfolio-Level Permutation Test

```python
# Test full portfolio with permutation
results = validator.walkforward_permutation_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    test_step=252,
    num_steps=10,
    portfolio=my_portfolio,  # Full portfolio
    candles_df=candles_df,
    target_col='log_return',
    nreps=100,
    alpha=0.05
)

# Results will have 'portfolio' as feature name
print(f"Portfolio p-value: {results.loc[0, 'pval']:.4f}")
print(f"Significant: {results.loc[0, 'significant']}")
```

### Notes on Reusing Existing Code

**✅ Fully Reusable**:
- `FeaturePermutationStrategy` - Already supports `train_windows` parameter
- `PermutationEngine` - Handles parallel execution and p-value computation
- Two-region shuffling logic - Already implemented in `FeaturePermutationStrategy.permute()`

**⚠️ Minor Modifications Needed**:
- Criterion function needs to call `walkforward_test()` instead of simple metric computation
- Need to handle portfolio-level testing (feature_col='portfolio')
- Target shuffling may need custom handling (or extend FeaturePermutationStrategy)

**Implementation Strategy**:
1. Create criterion function that wraps `walkforward_test()`
2. Use existing `FeaturePermutationStrategy` with `train_windows` parameter
3. Use existing `PermutationEngine` for parallel execution
4. Return results in same format as `OSFeatureSelector.permutation_test()`

### Edge Cases

#### 1. Portfolio-Level Testing

**Problem**: When testing a portfolio, there's no single `feature_col` to shuffle.

**Solution**:
- Use `feature_col='portfolio'` as identifier
- Shuffle all features used by the portfolio's ensembles
- Or shuffle target values instead (set `shuffle_target=True`)

#### 2. Multiple Feature Variations

**Problem**: When testing rule-based features with multiple variations, which to shuffle?

**Solution**:
- Shuffle all feature variations independently
- Criterion function selects best variation in training data (same as walkforward_test)
- Compare aggregated metric across all variations

#### 3. Insufficient Permutations

**Problem**: Some permutations may fail (e.g., insufficient data in a fold).

**Solution**:
- Track `n_valid_permutations` in results
- Criterion function should return 0.0 or NaN on failure
- PermutationEngine already handles exceptions gracefully

## Future Enhancements

1. **Cross-Validation Support**: Add `cv_test()` method similar to `OSFeatureSelector`
2. **Bar Permutation Testing**: Add support for bar-based permutation (shuffle price bars and re-extract features)
   - Can reuse `BarPermutationStrategy` with `train_windows` parameter
   - More realistic but computationally intensive
3. **Parallel Execution**: Use joblib to parallelize fold processing within walkforward_test
4. **Caching**: Cache feature extraction results across folds
5. **Visualization**: Add plotting methods for fold-by-fold performance
6. **Export Results**: Save fold results to disk for later analysis
7. **Multiple Metrics**: Support multiple metrics in permutation test (not just Sharpe ratio)

## Migration from OSFeatureSelector

The `FeatureValidator` replaces `OSFeatureSelector` for walkforward testing:

**Old (OSFeatureSelector)**:
```python
selector = OSFeatureSelector(features_df, targets_df)
results_df, step_info, fig = selector.walkforward_test(
    model=model,
    objective_metric=metric,
    feature_cols='rsi_signal_D',
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1)
)
```

**New (FeatureValidator)**:
```python
validator = FeatureValidator(features_df, targets_df)
fold_results, summary_df, aggregate_metrics = validator.walkforward_test(
    train_start=datetime(2010, 1, 1),
    train_end=datetime(2015, 1, 1),
    feature_cols='rsi_signal_D',
    target_col='log_return'
)
```

**Benefits**:
- Reuses Portfolio/PortfolioTester infrastructure
- Consistent with production pipeline
- Better integration with ensemble system
- More comprehensive metrics via tearsheets
- **Supports portfolio-level backtesting**: Can test full portfolios (multiple ensembles, custom weights, etc.) using walkforward validation
- **Flexible**: Works with any Portfolio instance, not just single-feature portfolios

