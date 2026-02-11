# Prop Firm Challenge Simulator Specification

## Overview

Create a specification document for a prop firm challenge simulator that evaluates trading strategies against prop firm rules using two distinct simulation methods. The simulator accepts daily returns as a pandas Series indexed by datetime and tracks equity, drawdowns, costs, and challenge outcomes.

**Overall Goal**: Get a feel for how well the strategy can pass prop firm challenges by simulating multiple challenge attempts and analyzing pass/fail probabilities, costs, and performance distributions.

## File Location

- **Path**: `docs/to-do/prop_firm_challenge_simulator.md`

## Core Requirements

### Input Data

- **Primary Input**: `pd.Series` of daily returns indexed by datetime
- Returns should be in decimal form (e.g., 0.01 for 1%)
- Datetime index must be timezone-aware or timezone-naive (consistent handling)

### Challenge Rules (All in Percentage Terms)

1. **Max Drawdown** (required): Maximum allowed drawdown from equity peak
2. **Max Daily Drawdown** (optional): Maximum allowed single-day drawdown
3. **Trailing Drawdown** (optional): Drawdown threshold that follows equity peak. Note: Some firms have no trailing drawdown, only max drawdown
4. **Profit Target** (required): Target profit percentage to pass challenge
5. **Minimum Trading Days** (required): Minimum number of trading days before challenge can pass

### Special Rules

- If profit target is reached before minimum trading days, equity must remain above profit target until minimum days are satisfied
- Trailing drawdown is calculated using end-of-day (EOD) returns
- All drawdown calculations are relative to equity peaks
- All equity values are tracked in percentage terms (starting at 0 = 0%), abstracting away dollar amounts for different account sizes

### Cost Structure

1. **Monthly Fee** (optional): Recurring fee paid each month (e.g., $100/month)
2. **One-Time Fee** (optional): Upfront fee paid at challenge start (e.g., $100)
3. **Reset Fee** (required): Fee paid when challenge fails and resets (e.g., $50)
4. **Fee Tracking**: Track calendar months and apply monthly fees on the same day each month
5. **Constraint**: Either `monthly_fee` OR `one_time_fee` must be specified (at least one required)

### Output Statistics (Required)

1. **Pass Probability**: Fraction of simulations that pass
2. **Fail Probability**: Fraction of simulations that fail
3. **Distribution of Days-to-Pass**: Statistical distribution (mean, median, percentiles) of days required to pass
4. **Distribution of Max Equity Peak Before Failure**: For failed simulations, distribution of peak equity reached
5. **Expected Number of Resets**: Average number of resets across all simulations
6. **Expected Total Evaluation Cost**: Average total cost (fees + resets) across all simulations
7. Support for more stats in the future 


## Simulation Method 1: Historical Walk-Forward Simulation

### Purpose

Measure performance under real historical sequencing. Preserves true market structure and regimes.

### Method

1. **Start Index Selection**: Randomly select start indices from the historical daily return series
2. **Day-by-Day Simulation**: Step forward day-by-day using actual historical returns
3. **Tracking Variables**:
   - Cumulative equity (starting from 1.0 = 100%)
   - Running equity peak (highest equity seen so far)
   - Trailing drawdown threshold (peak * (1 - trailing_drawdown_pct))
   - Trading day count
   - Current drawdown from peak
   - Daily drawdown (if max_daily_drawdown is specified)

### Termination Conditions

**FAIL** when:
- Equity breaches trailing drawdown threshold (if enabled)
- Equity breaches max drawdown from peak
- Single-day drawdown exceeds max daily drawdown (if enabled)

**PASS** when:
- Equity reaches profit target AND minimum trading days are satisfied
- Equity remains above profit target until minimum days are met

### Limitations

- Limited to historical sample paths available in the data
- Cannot generate paths beyond historical data range
- Used as realism check, not tail-risk estimator

### Implementation Notes

- Preserve chronological order of returns
- Track calendar dates for monthly fee calculations
- Handle edge cases (insufficient data, early termination)

## Simulation Method 2: Monte Carlo Path Simulation (Block Bootstrap)

### Purpose

Estimate structural pass/fail probabilities under path uncertainty. Simulates entire equity paths, not summary statistics.

### Key Principle

Simulate complete equity paths using block bootstrap to preserve:
- Return distribution
- Local autocorrelation
- Volatility clustering
- Drawdown sequencing

### Return Generation Method

**Block Bootstrap Algorithm**:

1. **Block Length**: User-specified block length `L` (e.g., 5, 10, 20 days)
2. **Block Sampling**:
   - Sample contiguous blocks of length `L` days from historical returns
   - Blocks are sampled with replacement
   - Blocks maintain chronological order within themselves
3. **Path Construction**:
   - Concatenate sampled blocks sequentially
   - Continue until target path length is reached
   - Truncate final block to exact target length if needed

### Advantages

- Preserves short-term dependencies (autocorrelation)
- Maintains volatility clustering patterns
- Allows generation of paths longer than historical data
- Better tail-risk estimation than simple bootstrap

### Implementation Notes

- Block length should be configurable (default: 10-20 days)
- Handle edge cases (block length > available data)
- Ensure sufficient blocks for path generation
- Track calendar progression for fee calculations (approximate based on trading days)

## Data Structures & Type Safety

### Configuration (Frozen Dataclass)

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class SimulationMethod(Enum):
    HISTORICAL_WALKFORWARD = "historical_walkforward"
    MONTE_CARLO_BLOCK = "monte_carlo_block"

@dataclass(frozen=True)
class ChallengeRules:
    max_drawdown_pct: float  # Required: e.g., 0.10 for 10%
    profit_target_pct: float  # Required: e.g., 0.08 for 8%
    min_trading_days: int  # Required
    max_daily_drawdown_pct: Optional[float] = None  # Optional
    trailing_drawdown_pct: Optional[float] = None  # Optional (some firms have none)

@dataclass(frozen=True)
class ChallengeCosts:
    reset_fee: float  # Required: e.g., 50.0
    monthly_fee: Optional[float] = None  # Optional: e.g., 100.0
    one_time_fee: Optional[float] = None  # Optional: e.g., 100.0
    # Validation: Either monthly_fee OR one_time_fee must be provided

@dataclass(frozen=True)
class SimulationConfig:
    method: SimulationMethod
    rules: ChallengeRules
    costs: ChallengeCosts
    n_simulations: int  # Number of simulations to run
    block_length: Optional[int] = None  # For Monte Carlo: block size (default: 10)
    random_seed: Optional[int] = None  # For reproducibility
    # Note: initial_balance is always 1.0 (100%) - equity tracked in percentage terms
```

### Result Types (ADTs)

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class ChallengePass:
    days_to_pass: int
    final_equity: float  # Percentage (e.g., 1.08 = 108%)
    max_equity_peak: float  # Percentage
    total_cost: float  # Dollar amount
    n_resets: int
    equity_curve: pd.Series  # Equity over time (percentage)

@dataclass(frozen=True)
class ChallengeFail:
    days_to_failure: int
    failure_reason: str  # "max_drawdown", "trailing_drawdown", "max_daily_drawdown"
    max_equity_peak: float  # Percentage
    equity_at_failure: float  # Percentage
    total_cost: float  # Dollar amount
    n_resets: int
    equity_curve: pd.Series  # Equity over time (percentage)

ChallengeResult = Union[ChallengePass, ChallengeFail]
```

### Output Statistics (Frozen Dataclass)

```python
@dataclass(frozen=True)
class SimulationStatistics:
    pass_probability: float
    fail_probability: float
    days_to_pass_distribution: dict[str, float]  # mean, median, p5, p25, p75, p95
    max_equity_peak_failure_distribution: dict[str, float]  # mean, median, percentiles
    expected_n_resets: float
    expected_total_cost: float
    n_simulations: int
    n_passed: int
    n_failed: int
```

## Function Signatures

### Main Simulation Function

```python
def simulate_prop_firm_challenge(
    daily_returns: pd.Series,
    config: SimulationConfig,
    start_date: Optional[datetime] = None  # For historical walkforward: specific start
) -> tuple[list[ChallengeResult], SimulationStatistics]:
    """
    Run prop firm challenge simulation using specified method.
    
    Args:
        daily_returns: Series of daily returns (decimal) indexed by datetime
        config: Simulation configuration (method, rules, costs)
        start_date: Optional start date for historical walkforward (if None, random starts)
    
    Returns:
        Tuple of (list of individual results, aggregate statistics)
    """
```

### Visualization Functions

**All plotting functions MUST use metrics library for calculations:**

```python
def plot_equity_curves(
    results: list[ChallengeResult],
    figsize: tuple[int, int] = (14, 8),
    show_percentiles: bool = True,
    alpha: float = 0.1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot equity curves for all simulations.
    
    Implementation Notes:
    - Equity curves are already computed and stored in ChallengeResult.equity_curve
    - If additional equity calculations are needed, use metrics.equity functions
    - Percentile calculations should use numpy/pandas, not custom implementations
    
    Shows:
    - Individual equity curves (transparent)
    - Percentile bands (5th, 25th, 50th, 75th, 95th)
    - Pass vs Fail curves (different colors)
    - Profit target and drawdown thresholds as horizontal lines
    """

def plot_days_to_pass_distribution(
    results: list[ChallengeResult],
    figsize: tuple[int, int] = (12, 6),
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of days to pass for successful challenges.
    
    Shows:
    - Histogram of days to pass
    - KDE overlay
    - Statistical summary (mean, median, percentiles)
    """

def plot_max_equity_peak_distribution(
    results: list[ChallengeResult],
    figsize: tuple[int, int] = (12, 6),
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of max equity peak before failure for failed challenges.
    
    Shows:
    - Histogram of max equity peaks
    - KDE overlay
    - Statistical summary
    - Profit target line for reference
    """

def plot_simulation_summary(
    results: list[ChallengeResult],
    stats: SimulationStatistics,
    figsize: tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive summary dashboard with multiple subplots.
    
    Subplots:
    1. Equity curves (all simulations)
    2. Days to pass distribution
    3. Max equity peak distribution (failures)
    4. Cost analysis (total cost distribution)
    5. Statistics summary table
    """
```

### Helper Functions

**All helper functions MUST use metrics library for calculations:**

```python
def run_historical_walkforward_simulation(
    daily_returns: pd.Series,
    rules: ChallengeRules,
    costs: ChallengeCosts,
    start_idx: int,
    random_seed: Optional[int] = None
) -> ChallengeResult:
    """
    Run single historical walkforward simulation from start index.
    
    Implementation Notes:
    - Use metrics.equity.equity_curve() to compute equity from returns
    - Use metrics.equity.equity_peak() to track running peak
    - Use metrics.risk.drawdown_series() to compute drawdown series
    - Use metrics.risk.max_drawdown() to check max drawdown breaches
    - Do NOT implement equity/drawdown calculations directly
    """

def run_monte_carlo_block_simulation(
    daily_returns: pd.Series,
    rules: ChallengeRules,
    costs: ChallengeCosts,
    path_length: int,
    block_length: int,
    random_seed: Optional[int] = None
) -> ChallengeResult:
    """
    Run single Monte Carlo block bootstrap simulation.
    
    Implementation Notes:
    - Use block bootstrap resampling (can reuse from robustness_test implementation)
    - Use metrics.equity.equity_curve() to compute equity from generated returns
    - Use metrics.equity.equity_peak() to track running peak
    - Use metrics.risk.drawdown_series() to compute drawdown series
    - Use metrics.risk.max_drawdown() to check max drawdown breaches
    - Do NOT implement equity/drawdown calculations directly
    """

def compute_statistics(
    results: list[ChallengeResult]
) -> SimulationStatistics:
    """
    Compute aggregate statistics from simulation results.
    
    Implementation Notes:
    - Use metrics library functions where applicable
    - Statistics are computed from ChallengeResult objects (which already use metrics library)
    - Percentile calculations should use numpy/pandas, not custom implementations
    """
```

## Implementation Details

### Metrics Library Usage (CRITICAL)

**All metric, equity, and risk calculations MUST use the centralized `metrics/` library:**

- **Equity Curves**: Use `metrics.equity.cumulative_returns()` or `metrics.equity.equity_curve()` for computing equity from returns
- **Equity Tracking**: Use `metrics.equity.equity_peak()` and `metrics.equity.equity_tracking()` for peak tracking
- **Drawdown Calculations**: Use `metrics.risk.max_drawdown()` and `metrics.risk.drawdown_series()` for all drawdown metrics
- **Performance Metrics**: Use `metrics.performance` for any performance metrics (Sharpe, Sortino, Calmar) if needed

**If functionality is missing from the metrics library:**
1. **First**: Add the missing functionality to the appropriate `metrics/` submodule
   - Example: If trailing drawdown calculation is needed, add `trailing_drawdown()` to `metrics/risk/drawdown.py`
   - Example: If daily drawdown calculation is needed, add `daily_drawdown()` to `metrics/risk/drawdown.py`
2. **Then**: Use it from the metrics library in the simulation functions
3. **Never**: Implement metric/equity/risk logic directly in simulation functions or result classes

**Plotting**:
- All plotting functions should be in `plotting/` directory
- If metric-specific plotting utilities are needed, add them to `metrics/plotting/` first, then use them
- Plotting functions should use metrics library for all calculations (equity curves, drawdowns, etc.)

### Equity Tracking

**Implementation MUST use metrics library:**

1. **Initial Balance**: Start with `0.0` (0% - percentage terms)
2. **Daily Update**: Use `metrics.equity.equity_curve()` or `metrics.equity.cumulative_returns()` to compute equity from returns
3. **Peak Tracking**: Use `metrics.equity.equity_peak()` to track running peak
4. **Drawdown Calculation**: Use `metrics.risk.drawdown_series()` to compute drawdown series
5. **Trailing Drawdown**: 
   - Use `metrics.equity.equity_peak()` to get current peak
   - Calculate threshold: `trailing_threshold = equity_peak * (1 - trailing_drawdown_pct)`
   - If trailing drawdown calculation is not in metrics library, add `trailing_drawdown_threshold()` to `metrics/risk/drawdown.py` first
6. **Profit Target Check**: `equity[t] >= (0.0 + profit_target_pct)`
7. **Max Drawdown Check**: Use `metrics.risk.max_drawdown()` to check if max drawdown is breached

### Cost Tracking

1. **Initialization**: Apply one-time fee (if specified) at start
2. **Monthly Fees**: 
   - Track calendar months from start date
   - Apply monthly fee on same day each month (e.g., if start is Feb 5, apply on Mar 5, Apr 5, etc.)
   - For Monte Carlo: approximate calendar progression based on trading days (assume ~21 trading days per month)
3. **Reset Fees**: Apply when challenge fails, then reset equity to 0.0 (0%)
4. **Cost Accumulation**: Track total cost (fees + resets) throughout simulation

### Validation Rules

1. **Input Validation**:
   - Verify `daily_returns` is pd.Series with datetime index
   - Check for numeric dtype
   - Validate non-empty series
   - Ensure returns are in reasonable range (e.g., -2.0 to 2.0)
2. **Config Validation**:
   - All percentages must be positive and reasonable (0 < pct < 1.0 typically)
   - `min_trading_days` must be positive integer
   - `n_simulations` must be positive
   - `block_length` must be positive if using Monte Carlo method
   - Either `monthly_fee` OR `one_time_fee` must be provided (not both None)
3. **Edge Case Handling**:
   - Insufficient data for requested path length
   - Block length > available data
   - Early termination conditions

## Visualization Requirements

**All plotting functions MUST be in `plotting/` directory and use metrics library:**

### Equity Curves Plot

- **Location**: `plotting/prop_firm_plots.py` or similar
- **Individual Curves**: Plot all simulation equity curves with low alpha (transparency)
  - Equity curves are already computed using `metrics.equity.equity_curve()` in simulation functions
  - Do NOT recompute equity curves in plotting function - use the equity_curve from ChallengeResult
- **Color Coding**: 
  - Green for passed challenges
  - Red for failed challenges
- **Percentile Bands**: Show 5th, 25th, 50th, 75th, 95th percentiles as shaded regions or distinct lines
  - Use numpy/pandas for percentile calculations, not custom implementations
- **Reference Lines**:
  - Profit target (horizontal line)
  - Max drawdown threshold (horizontal line, if applicable)
  - Trailing drawdown threshold (dynamic line following peak)
- **Formatting**: Proper axis labels, title, grid, legend

### Distribution Plots

- **Location**: `plotting/prop_firm_plots.py` or similar
- **Histogram + KDE**: For days-to-pass and max-equity-peak distributions
  - Follow patterns from `plotting/distribution.py`
- **Statistical Summary**: Text box with mean, median, percentiles
  - Use numpy/pandas for statistics, not custom implementations
- **Reference Lines**: Profit target line for max equity peak plot
- **Formatting**: Consistent with existing plotting patterns in `plotting/distribution.py`

### Summary Dashboard

- **Location**: `plotting/prop_firm_plots.py` or similar
- **Multi-panel Layout**: Use subplots to show all key visualizations together
- **Statistics Table**: Summary statistics in text format
- **Consistent Styling**: Follow existing codebase plotting conventions
- **All calculations**: Use metrics library or numpy/pandas, never custom metric implementations

## Example Usage

```python
import pandas as pd
import numpy as np
from datetime import datetime
from prop_firm_simulator import (
    simulate_prop_firm_challenge,
    SimulationConfig,
    SimulationMethod,
    ChallengeRules,
    ChallengeCosts,
    plot_equity_curves,
    plot_simulation_summary
)

# Prepare returns data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

# Configure challenge
rules = ChallengeRules(
    max_drawdown_pct=0.10,  # 10% max drawdown
    profit_target_pct=0.08,  # 8% profit target
    min_trading_days=30,
    trailing_drawdown_pct=0.05,  # 5% trailing drawdown
    max_daily_drawdown_pct=0.05  # 5% max daily drawdown
)

costs = ChallengeCosts(
    monthly_fee=100.0,  # OR one_time_fee=100.0
    reset_fee=50.0
)

config = SimulationConfig(
    method=SimulationMethod.MONTE_CARLO_BLOCK,
    rules=rules,
    costs=costs,
    n_simulations=1000,
    block_length=10,
    random_seed=42
)

# Run simulation
results, stats = simulate_prop_firm_challenge(
    daily_returns=returns,
    config=config
)

# Access statistics
print(f"Pass probability: {stats.pass_probability:.2%}")
print(f"Expected cost: ${stats.expected_total_cost:.2f}")
print(f"Average days to pass: {stats.days_to_pass_distribution['mean']:.1f}")

# Create visualizations
fig_curves = plot_equity_curves(results, show_percentiles=True)
fig_summary = plot_simulation_summary(results, stats)
```

## Testing Requirements

1. **Unit Tests**:
   - Individual simulation runs (both methods)
   - Cost calculation logic
   - Drawdown calculations
   - Termination conditions
   - Equity tracking (percentage-based)

2. **Integration Tests**:
   - Full simulation runs with various configs
   - Statistics computation accuracy
   - Edge cases (early pass, immediate failure)
   - Visualization generation

3. **Validation Tests**:
   - Input validation
   - Config validation (monthly_fee OR one_time_fee requirement)
   - Error handling

## Performance Considerations

- For large `n_simulations` (e.g., > 10,000), consider:
  - Parallel processing for independent simulations
  - Progress tracking (tqdm or similar)
  - Memory-efficient result storage
  - Option to store only summary statistics, not full equity curves
- Block bootstrap can be computationally intensive for long paths
- Consider caching block samples if running multiple simulations with same parameters
- For visualizations, consider sampling subset of curves for plotting while computing all statistics

## Future Enhancements

- **Payout Simulation**: Model passing challenges and receiving payouts (different rules, model separately)
- Additional simulation methods (e.g., parametric bootstrap)
- Risk-adjusted metrics: Use `metrics.performance.SharpeRatio` and `metrics.performance.CalmarRatio` (add CalmarRatio to metrics library if not present)
- Multiple challenge phases (e.g., evaluation phase → funded phase)

## Metrics Library Dependencies

This implementation requires the following from the centralized `metrics/` library:

### Required Functions

- `metrics.equity.cumulative_returns()` - Compute equity curves from returns
- `metrics.equity.equity_curve()` - Alias for cumulative_returns (for clarity)
- `metrics.equity.equity_peak()` - Track running equity peak
- `metrics.equity.equity_tracking()` - Track equity, peak, and drawdown simultaneously
- `metrics.risk.drawdown_series()` - Compute drawdown series from equity or returns
- `metrics.risk.max_drawdown()` - Compute maximum drawdown

### Potentially Needed (Add to metrics library if missing)

- `metrics.risk.trailing_drawdown_threshold()` - Calculate trailing drawdown threshold from peak
- `metrics.risk.daily_drawdown()` - Calculate single-day drawdown (if different from regular drawdown)

**If any of these are missing, they MUST be added to the metrics library BEFORE implementing the simulator.**

