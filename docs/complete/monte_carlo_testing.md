# Robustness Testing Function Specification

## Overview

Create a specification document for a robustness testing function that performs statistical testing on return series using probabilistic resampling methods. The function supports three resampling approaches:

1. **Monte Carlo (Permutation)**: Random shuffle without replacement - tests if order matters (null hypothesis: returns are IID)
2. **Bootstrap**: Sampling with replacement - quantifies variance of performance metrics and tests robustness
3. **Block Bootstrap**: Block-based sampling with replacement - preserves serial correlation (autocorrelation) in returns

All methods generate multiple resampled series and visualize their equity curves (cumulative returns) to assess the robustness of trading strategies.

## File Location

- **Path**: `docs/to-do/monte_carlo_testing.md`

## Function Requirements

### Function Signature

```python
def robustness_test(
    returns: pd.Series,
    n_samples: int,
    method: ResamplingMethod = ResamplingMethod.MONTE_CARLO,
    random_seed: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_original: bool = True,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    verbose: bool = True,
    # Bootstrap-specific parameters
    block_size: Optional[int] = None,  # Required for block bootstrap
) -> Tuple[plt.Figure, List[pd.Series], Dict[str, Any]]
```

### Input Parameters

- `returns` (pd.Series): Series of returns with datetime index
- `n_samples` (int): Number of resampled series to generate
- `method` (ResamplingMethod): Resampling method to use (default: `ResamplingMethod.MONTE_CARLO`)
  - `ResamplingMethod.MONTE_CARLO`: Random permutation/shuffle without replacement
  - `ResamplingMethod.BOOTSTRAP`: Simple bootstrap (sampling with replacement)
  - `ResamplingMethod.BLOCK_BOOTSTRAP`: Block bootstrap (preserves serial correlation)
- `random_seed` (Optional[int]): Random seed for reproducibility
- `figsize` (Tuple[int, int]): Figure size for plot (default: (14, 8))
- `show_original` (bool): Whether to highlight original series on plot (default: True)
- `alpha` (float): Transparency for resampled equity curves (default: 0.05)
- `save_path` (Optional[str]): Path to save figure
- `verbose` (bool): Whether to print progress
- `block_size` (Optional[int]): Block size for block bootstrap. Required when `method=ResamplingMethod.BLOCK_BOOTSTRAP`, ignored otherwise (default: None)

### Outputs

1. **Plot** (plt.Figure): Distribution plot showing:
   - Equity curves (cumulative returns) for each resampled series
   - Original equity curve highlighted (if `show_original=True`)
   - Percentile bands (5th, 25th, 50th, 75th, 95th) as shaded regions or lines
   - Statistical summary text box with key metrics
   - Proper axis labels and title

2. **Array of Series** (List[pd.Series]): List containing each resampled series as a pandas Series, preserving original index structure. Each series contains the resampled returns (not cumulative).

3. **Statistics Dictionary** (Dict[str, Any]): Summary statistics including:
   - `resampling_method`: Which method was used (str)
   - `original_cumulative_return`: Original cumulative return (final value)
   - `mean_cumulative_return`: Mean of final cumulative returns across resamples
   - `median_cumulative_return`: Median of final cumulative returns
   - `std_cumulative_return`: Standard deviation of final cumulative returns
   - `percentiles`: Dict with keys `p5`, `p25`, `p50`, `p75`, `p95` for final cumulative returns
   - `p_value`: Fraction of resamples with final cumulative return >= original
   - `n_valid_samples`: Number of valid resamples
   - `original_mean_return`: Original series mean return
   - `original_volatility`: Original series standard deviation
   - `autocorr_lag1`: Original series autocorrelation at lag 1
   - `block_size`: Block size used (if block bootstrap, otherwise None)
   - `preserved_autocorr`: Whether autocorrelation was preserved (for block bootstrap, compares resampled autocorr to original)

### Resampling Methods

#### Monte Carlo (Permutation)

- **Method**: Random shuffle without replacement
- **Use Case**: Tests if order matters (null hypothesis: returns are IID)
- **Implementation**: 
  - For each resample, randomly shuffle the return values
  - Preserve original index structure
  - Each return value appears exactly once per resample (no duplicates)
- **When to Use**: When testing whether the temporal ordering of returns matters for strategy performance

#### Bootstrap

- **Method**: Sampling with replacement (independent draws)
- **Use Case**: Quantifies variance of performance metrics, tests robustness of risk measures (Sharpe, Calmar, drawdown)
- **Implementation**:
  - For each resample, randomly sample return values with replacement
  - Preserve original index structure
  - Each return value can appear multiple times or not at all in a resample
  - Resampled series length matches original series length
- **When to Use**: When you want to assess the distribution of performance metrics under the assumption that returns are IID

#### Block Bootstrap

- **Method**: Sample blocks of consecutive returns with replacement
- **Use Case**: Preserves serial correlation (autocorrelation) in returns while still resampling
- **Implementation**:
  - Divide returns into blocks of consecutive values (non-overlapping blocks)
  - Sample blocks with replacement
  - Concatenate sampled blocks to create resampled series
- **Block Size**:
  - `block_size` parameter is required when using `ResamplingMethod.BLOCK_BOOTSTRAP`
  - User must specify the block size (typically based on autocorrelation analysis or domain knowledge)
  - Common choices: 5-20 for daily returns, larger for weekly/monthly data
- **When to Use**: When returns exhibit serial correlation and you want to preserve this structure in resampling

### Additional Features

- Input validation:
  - Check that returns is a pandas Series
  - Check for numeric dtype
  - Check for non-empty series
  - Validate datetime index
  - Validate `method` is a valid `ResamplingMethod` enum value
  - If `method=ResamplingMethod.BLOCK_BOOTSTRAP`, validate that `block_size` is provided and >= 1
- Progress tracking for large n_samples (print every 10% or 100 iterations)
- Error handling for failed resampling iterations (skip and continue)
- Preserve original datetime index in all output series
- Handle NaN values appropriately (drop or fill before resampling)

### Implementation Notes

#### Metrics Library Usage (CRITICAL)

**All metric, equity, and risk calculations MUST use the centralized `metrics/` library:**

- **Equity Curves**: Use `metrics.equity.cumulative_returns()` or `metrics.equity.equity_curve()` for computing cumulative returns
- **Drawdown Calculations**: Use `metrics.risk.max_drawdown()` and `metrics.risk.drawdown_series()` for all drawdown metrics
- **Performance Metrics**: Use `metrics.performance` for any performance metrics (Sharpe, Sortino, etc.) if needed
- **Equity Tracking**: Use `metrics.equity.equity_peak()` and `metrics.equity.equity_tracking()` for peak tracking

**If functionality is missing from the metrics library:**
1. **First**: Add the missing functionality to the appropriate `metrics/` submodule
2. **Then**: Use it from the metrics library in the implementation
3. **Never**: Implement metric/equity/risk logic directly in the robustness_test function or strategy classes

**Plotting**:
- Use matplotlib for plotting (consistent with existing codebase)
- Follow existing plotting patterns from `plotting/distribution.py` and `plotting/cumulative_plot.py`
- Plotting functions should be in `plotting/` directory, not in the main robustness_test function
- If metric-specific plotting utilities are needed, add them to `metrics/plotting/` first, then use them

#### Architecture & Design

- Follow existing strategy pattern from `utils/permutation_test/permutation_engine.py` for consistency
- Use `Protocol` or `ABC` for strategy interface (consistent with codebase style)
- Strategy classes should ONLY handle resampling logic, NOT metric calculations
- Main function should delegate all metric/equity calculations to metrics library
- Compute cumulative returns only for plotting purposes (not stored in output series)
- Use appropriate color scheme: gray/light colors for resampled curves, distinct color for original
- Add percentile bands as shaded regions or distinct colored lines
- Include statistical summary in text box (similar to `plot_feature_distribution`)
- Include method name in plot title (e.g., "Monte Carlo Robustness Test", "Bootstrap Robustness Test")
- Consider performance: for large n_samples, may want to sample a subset for plotting while computing all statistics
- For block bootstrap, use non-overlapping blocks for simplicity
- Preserve original datetime index in all resampled series

### Example Usage

```python
import pandas as pd
import numpy as np
from robustness_test import robustness_test
from utils.enums import ResamplingMethod
from metrics.equity import cumulative_returns  # Use metrics library for calculations

# Create sample returns
dates = pd.date_range('2020-01-01', periods=252, freq='D')
returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

# Example 1: Monte Carlo (Permutation) - default method
fig, resampled_series, stats = robustness_test(
    returns=returns,
    n_samples=1000,
    method=ResamplingMethod.MONTE_CARLO,
    random_seed=42,
    show_original=True
)

# Access results
print(f"Method: {stats['resampling_method']}")
print(f"Original cumulative return: {stats['original_cumulative_return']:.4f}")
print(f"P-value: {stats['p_value']:.4f}")
print(f"Number of resamples: {len(resampled_series)}")

# Example 2: Bootstrap (sampling with replacement)
fig, resampled_series, stats = robustness_test(
    returns=returns,
    n_samples=1000,
    method=ResamplingMethod.BOOTSTRAP,
    random_seed=42
)

# Example 3: Block Bootstrap (preserves serial correlation)
fig, resampled_series, stats = robustness_test(
    returns=returns,
    n_samples=1000,
    method=ResamplingMethod.BLOCK_BOOTSTRAP,
    block_size=20,  # Required: specify block size
    random_seed=42
)

# Access block bootstrap specific statistics
print(f"Block size used: {stats['block_size']}")
print(f"Autocorrelation preserved: {stats['preserved_autocorr']}")

# User can compute cumulative returns from resampled series if needed
# Use metrics library for consistency
from metrics.equity import cumulative_returns

for i, resampled_returns in enumerate(resampled_series[:5]):  # First 5 examples
    cum_returns = cumulative_returns(resampled_returns)
    print(f"Resample {i} final cumulative: {cum_returns.iloc[-1]:.4f}")
```

## Architecture

### Resampling Strategy Pattern

Following the existing `PermutationStrategy` pattern in `utils/permutation_test/permutation_engine.py`, the implementation uses an abstract base class hierarchy:

```
ResamplingStrategy (ABC)
├── MonteCarloStrategy (random shuffle without replacement)
├── BootstrapStrategy (sampling with replacement)
└── BlockBootstrapStrategy (block-based sampling with replacement)
```

Each strategy implements:
- `resample(returns: pd.Series, random_seed: Optional[int]) -> pd.Series`: Generate a single resampled series
- `validate_params(**kwargs) -> None`: Validate method-specific parameters

**Important**: Strategy classes should ONLY handle resampling logic. All metric/equity calculations should be delegated to the metrics library in the main `robustness_test()` function, not in the strategy classes.

### ResamplingMethod Enum

The `ResamplingMethod` enum is defined in `utils/enums.py`:

```python
class ResamplingMethod(Enum):
    MONTE_CARLO = "monte_carlo"  # Random permutation/shuffle
    BOOTSTRAP = "bootstrap"  # Simple bootstrap (with replacement)
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Block bootstrap (preserves correlation)
```

## Implementation Details

### Algorithm

1. **Input Validation**:
   - Verify `returns` is a pandas Series
   - Check for numeric dtype
   - Ensure non-empty series
   - Validate datetime index exists
   - Validate `method` is a valid `ResamplingMethod` enum value
   - Validate bootstrap-specific parameters if applicable

2. **Data Preparation**:
   - Drop or handle NaN values
   - Store original index for preservation
   - Compute original cumulative return for comparison
   - Compute original series statistics (mean, volatility, autocorrelation)

3. **Method Selection & Strategy Initialization**:
   - Determine which `ResamplingStrategy` to use based on `method` parameter
   - Validate that `block_size` is provided if `method=ResamplingMethod.BLOCK_BOOTSTRAP`
   - Create appropriate strategy instance:
     - `MonteCarloStrategy()` for `MONTE_CARLO`
     - `BootstrapStrategy()` for `BOOTSTRAP`
     - `BlockBootstrapStrategy(block_size)` for `BLOCK_BOOTSTRAP`

4. **Resampling Generation**:
   - For each resample (1 to n_samples):
     - Set random seed (if provided): `np.random.seed(random_seed + i)`
     - Call strategy's `resample()` method to generate resampled returns
     - Create new Series with resampled values and original index
     - Store in list

5. **Statistics Computation**:
   - For each resampled series:
     - Use `metrics.equity.cumulative_returns()` to compute equity curve
     - Extract final cumulative return value from equity curve
   - Compute statistics across all resamples:
     - Mean, median, std of final cumulative returns
     - Percentiles (5th, 25th, 50th, 75th, 95th) of final cumulative returns
     - P-value: fraction where resampled >= original
   - For block bootstrap, compute autocorrelation of resampled series and compare to original
   - **All statistics calculations should use metrics library functions where applicable**

6. **Plotting**:
   - Create figure with appropriate size
   - For each resampled series, use `metrics.equity.cumulative_returns()` to compute equity curve for plotting
   - Use `metrics.equity.cumulative_returns()` for original series equity curve
   - Plot all resampled equity curves with low alpha (transparency)
   - Plot original equity curve with distinct color and higher linewidth
   - Add percentile bands (5th, 25th, 50th, 75th, 95th) as shaded regions or distinct colored lines
   - Add statistical summary text box with key metrics
   - Format axes, labels, title (include method name), grid
   - **Plotting logic should be in a separate function in `plotting/` directory, not inline in robustness_test**

7. **Return Results**:
   - Return figure, list of resampled series, statistics dictionary

### Resampling Method Implementation Details

#### Monte Carlo Strategy

```python
class MonteCarloStrategy(ResamplingStrategy):
    def resample(self, returns: pd.Series, random_seed: Optional[int]) -> pd.Series:
        # Shuffle return values randomly
        values = returns.values.copy()
        np.random.seed(random_seed)
        np.random.shuffle(values)
        return pd.Series(values, index=returns.index)
```

#### Bootstrap Strategy

```python
class BootstrapStrategy(ResamplingStrategy):
    def resample(self, returns: pd.Series, random_seed: Optional[int]) -> pd.Series:
        # Sample with replacement
        n = len(returns)
        np.random.seed(random_seed)
        indices = np.random.choice(n, size=n, replace=True)
        return pd.Series(returns.values[indices], index=returns.index)
```

#### Block Bootstrap Strategy

```python
class BlockBootstrapStrategy(ResamplingStrategy):
    def __init__(self, block_size: int):
        if block_size is None:
            raise ValueError("block_size is required for BlockBootstrapStrategy")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
    
    def resample(self, returns: pd.Series, random_seed: Optional[int]) -> pd.Series:
        n = len(returns)
        block_size = self.block_size
        
        # Create blocks (non-overlapping)
        n_blocks = (n + block_size - 1) // block_size  # Ceiling division
        n_samples_needed = n_blocks
        
        # Sample blocks with replacement
        np.random.seed(random_seed)
        block_indices = np.random.choice(n_blocks, size=n_samples_needed, replace=True)
        
        # Concatenate sampled blocks
        resampled_values = []
        for block_idx in block_indices:
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, n)
            resampled_values.extend(returns.values[start_idx:end_idx])
        
        # Trim to original length
        resampled_values = resampled_values[:n]
        return pd.Series(resampled_values, index=returns.index)
```

### Error Handling

- If input validation fails, raise appropriate exceptions (TypeError, ValueError)
- If `block_size` is not provided when `method=ResamplingMethod.BLOCK_BOOTSTRAP`, raise `ValueError`
- If `block_size` is provided but is < 1, raise `ValueError`
- If resampling fails for a single iteration, log warning and continue (don't fail entire function)
- If plotting fails, return None for figure but still return data

### Performance Considerations

- For very large n_samples (e.g., > 1000), consider:
  - Plotting only a subset of curves (e.g., every Nth resample)
  - Computing all statistics but plotting sample
  - Using progress bars for user feedback

### Testing Requirements

- Test all three methods with same return series to compare results
- Test with various return series (positive, negative, zero mean, autocorrelated)
- Test with different n_samples (10, 100, 1000)
- Test with NaN values in input
- Test with different random seeds (reproducibility)
- Test edge cases (empty series, single value, all zeros)
- Test block bootstrap with:
  - Different block sizes (small, medium, large)
  - Edge cases: block_size=1, block_size >= series length
- Verify autocorrelation preservation in block bootstrap:
  - Compare original autocorrelation to resampled autocorrelation
  - Should be similar for block bootstrap, different for regular bootstrap/Monte Carlo
- Test that Monte Carlo produces unique permutations (no duplicates in a single resample)
- Test that Bootstrap can produce duplicates (sampling with replacement)
- Test that Block Bootstrap preserves local structure (consecutive blocks)

## Metrics Library Dependencies

This implementation requires the following from the centralized `metrics/` library:

### Required Functions

- `metrics.equity.cumulative_returns()` - Compute equity curves from returns (used for plotting and statistics)
- `metrics.equity.equity_curve()` - Alias for cumulative_returns (for clarity)
- `metrics.risk.drawdown_series()` - Compute drawdown series (if needed for analysis)
- `metrics.risk.max_drawdown()` - Compute maximum drawdown (if needed for analysis)

### Usage Pattern

```python
# In robustness_test() function:
from metrics.equity import cumulative_returns
from metrics.risk import max_drawdown  # If needed

# For statistics computation:
for resampled_returns in resampled_series:
    equity_curve = cumulative_returns(resampled_returns)
    final_value = equity_curve.iloc[-1]
    # ... use final_value for statistics

# For plotting:
original_equity = cumulative_returns(returns)
for resampled_returns in resampled_series:
    resampled_equity = cumulative_returns(resampled_returns)
    # ... plot equity curves
```

**If any required functionality is missing from the metrics library, it MUST be added there first before implementing robustness_test().**

