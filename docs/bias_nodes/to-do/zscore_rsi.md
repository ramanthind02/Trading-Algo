# Z-Score RSI Bias Node Specification

## Overview

The **Z-Score RSI** is a normalized version of the Relative Strength Index that measures how many standard deviations the current RSI value is from its own moving average. This creates an "adaptive" indicator that adjusts to market volatility, helping identify overextended moves even when price action is extremely volatile or unusually quiet.

**Key Benefit**: Provides volatility-adaptive overbought/oversold signals that adjust to current market conditions, unlike static RSI thresholds (30/70).

**Type**: Continuous feature (outputs z-score values, typically -3.0 to +3.0, requires binning)

## Formula

The Z-Score RSI calculation is a two-step process:

### Step 1: Calculate Standard RSI

```
RSI[t] = CalculateRSI(close_prices, rsi_length)
```

Use the standard RSI calculation (0-100 range).

### Step 2: Apply Z-Score Normalization

```
RSI_mean = mean(RSI[t-z_length+1] ... RSI[t])
RSI_std = std(RSI[t-z_length+1] ... RSI[t])

ZScoreRSI[t] = (RSI[t] - RSI_mean) / RSI_std
```

**Z-Score Formula**: `(Value - Mean) / Standard Deviation`

This measures how many standard deviations the current RSI is from its recent average.

## Interpretation

The Z-Score RSI uses statistical significance thresholds:

- **Above +2.0**: Overbought (RSI is 2+ standard deviations above its mean)
- **Below -2.0**: Oversold (RSI is 2+ standard deviations below its mean)
- **Near 0.0**: RSI is near its recent average (neutral)

**Typical Range**: -3.0 to +3.0 (though theoretically unbounded)

**Why It's Better**:
- **Volatility Adaptation**: During high volatility, classic RSI stays pinned at 10-20. Z-Score RSI requires extreme moves relative to current volatility.
- **Reduced Noise**: In low volatility, classic RSI may never reach 30/70. Z-Score RSI still triggers when moves are significant relative to quiet background.
- **Statistical Significance**: Provides mathematical basis for "overbought/oversold" rather than arbitrary thresholds.

## Implementation Strategy

### Recommended: Composition with RSI Node

Reuse the existing `RSI` node and track RSI values for z-score calculation:

```python
from nodes.rsi import RSI

class ZScoreRSI(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, z_length=20):
        # ... initialization ...
        # Create internal RSI node
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        # Circular buffer to store RSI values for z-score calculation
        self.rsi_buffer = np.zeros(z_length, dtype=np.float64)
        
    def _compute_candle(self, candle):
        # Get RSI value from internal node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Update RSI buffer
        # Calculate mean and std of RSI values
        # Calculate z-score
        z_score_rsi = (rsi - rsi_mean) / rsi_std
        
        return [z_score_rsi]
```

## Required Parameters

- `rsi_lookback` (int, default=14): Period for RSI calculation (typically 2-14)
- `z_length` (int, default=20): Period for calculating RSI mean and std (typically 20-100)

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = max(rsi_lookback, z_length)
```

This ensures we have:
1. Enough candles for RSI calculation (`rsi_lookback`)
2. Enough RSI values for z-score calculation (`z_length` RSI values)

**Minimum**: Need at least `max(rsi_lookback, z_length)` candles before valid Z-Score RSI output.

## State Management

You need to maintain:

1. **RSI Node** (via composition):
   - Internal `RSI` instance handles all RSI computation
   - No need to manage RSI buffers/state directly

2. **RSI Values Buffer**: Circular buffer to store RSI values
   - Size: `z_length`
   - Stores last `z_length` RSI values for mean/std calculation

3. **Buffer Index**: Track position in RSI values buffer

4. **Candle Count**: Track number of candles processed

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'zscorersi'` (or `'zscore_rsi'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'rsi_lookback': rsi_lookback, 'z_length': z_length}`
- [ ] Sets `self.front_bad = max(rsi_lookback, z_length)`
- [ ] Creates internal `RSI` node instance
- [ ] Initializes circular buffer for RSI values (size: `z_length`)
- [ ] Implements RSI value tracking
- [ ] Implements mean and std calculation of RSI values
- [ ] Implements z-score calculation: `(rsi - mean) / std`
- [ ] Handles warmup period (returns 0.0 - neutral z-score)
- [ ] Handles edge case: `std == 0` (return 0.0)
- [ ] Returns continuous value (z-score, typically -3.0 to +3.0)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode
from nodes.rsi import RSI  # Reuse existing RSI


class ZScoreRSI(BiasNode):
    """
    Z-Score RSI Bias Node - CONTINUOUS FEATURE
    
    Normalized version of RSI that measures how many standard deviations
    the current RSI is from its own moving average.
    
    Formula:
    1. Calculate standard RSI (0-100)
    2. Calculate mean and std of RSI over z_length period
    3. Z-Score = (RSI - mean) / std
    
    **Type**: Continuous feature (outputs z-score, requires binning)
    **Normalization**: Not needed (z-score is already normalized by definition)
    
    **Key Benefit**: Adapts to market volatility - identifies extreme moves
    relative to recent RSI distribution, not absolute thresholds.
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - z_length: Period for RSI mean/std calculation (default: 20)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        rsi_lookback: int = 14,
        z_length: int = 20
    ):
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.z_length = z_length
        
        # Standardization metadata
        self.module_name = 'zscorersi'
        self.output_features = ['signal']
        self.params = {
            'rsi_lookback': rsi_lookback,
            'z_length': z_length
        }
        
        # Warmup period: need max of RSI lookback and z_length
        self.front_bad = max(rsi_lookback, z_length)
        
        # Create internal RSI node for RSI calculation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        # Circular buffer to store RSI values for z-score calculation
        self.rsi_buffer = np.zeros(z_length, dtype=np.float64)
        self.rsi_buffer_idx = 0
        self.n_rsi_values = 0  # Count of RSI values stored
        
        # Track candle count
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Z-Score RSI for the given candle.
        
        Steps:
        1. Get RSI value from internal RSI node
        2. Store RSI in circular buffer
        3. Calculate mean and std of RSI values over z_length period
        4. Calculate z-score: (current_RSI - mean) / std
        5. Return z-score value
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Z-Score RSI value (typically -3.0 to +3.0)
        """
        self.n_prices += 1
        
        # Get RSI value from internal RSI node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Handle warmup period (RSI returns 50.0 during warmup)
        if self.n_prices < self.front_bad:
            self.output.append(0.0)  # Neutral z-score
            return [0.0]
        
        # Skip RSI values during RSI warmup (they're 50.0, not real values)
        # Only start tracking when RSI is valid
        if rsi == 50.0 and self.n_prices < self.rsi_lookback:
            self.output.append(0.0)
            return [0.0]
        
        # Update RSI buffer
        self.rsi_buffer[self.rsi_buffer_idx] = rsi
        self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.z_length
        if self.n_rsi_values < self.z_length:
            self.n_rsi_values += 1
        
        # Need at least z_length RSI values for z-score calculation
        if self.n_rsi_values < self.z_length:
            self.output.append(0.0)  # Neutral z-score during warmup
            return [0.0]
        
        # Calculate mean and standard deviation of RSI values
        rsi_mean = np.mean(self.rsi_buffer)
        rsi_std = np.std(self.rsi_buffer, ddof=1)  # Sample std dev (ddof=1)
        
        # Calculate z-score
        if rsi_std < 1e-10:  # Avoid division by zero
            z_score_rsi = 0.0  # Neutral if no variation
        else:
            z_score_rsi = (rsi - rsi_mean) / rsi_std
        
        self.output.append(z_score_rsi)
        return [z_score_rsi]
```

## Key Implementation Notes

1. **RSI Reuse**: The internal RSI node handles all RSI computation. You just call `self.rsi_node.add_candle(candle)[0]` to get the RSI value.

2. **RSI Value Tracking**: 
   - Store RSI values in a circular buffer of size `z_length`
   - Only start tracking after RSI warmup (when RSI != 50.0)

3. **Z-Score Calculation**:
   ```python
   rsi_mean = np.mean(self.rsi_buffer)
   rsi_std = np.std(self.rsi_buffer, ddof=1)  # Sample std dev
   z_score_rsi = (rsi - rsi_mean) / rsi_std
   ```
   Uses sample standard deviation (`ddof=1`) for proper statistical calculation.

4. **Edge Case Handling**:
   - If `rsi_std == 0` (all RSI values are the same) → return `0.0` (neutral)
   - During warmup → return `0.0` (neutral z-score)

5. **Output Range**: 
   - Theoretically unbounded (z-score can be any value)
   - Typically ranges from -3.0 to +3.0 in practice
   - Values beyond ±3.0 indicate extreme conditions

6. **No Additional Normalization**: Z-score is already normalized by definition. The output is the raw z-score value.

## Why Z-Score RSI is Better

### Volatility Adaptation Example

**High Volatility Scenario**:
- Classic RSI: Stays at 10-20 for days during crashes
- Z-Score RSI: Only triggers when RSI drops significantly below its recent average (e.g., if recent RSI mean is 15, current RSI of 5 would be -2.0 z-score)

**Low Volatility Scenario**:
- Classic RSI: May never reach 30/70 in quiet markets
- Z-Score RSI: Still triggers when RSI moves significantly relative to quiet background (e.g., if recent RSI mean is 50, current RSI of 60 would be +2.0 z-score)

## Trading Strategy Logic (Reference)

While this is a **continuous feature** (not rule-based), here's how it would typically be used:

**Mean Reversion Strategy**:
- **Long Entry**: When ZScoreRSI crosses below -2.0 (extreme oversold relative to recent RSI)
- **Long Exit**: When ZScoreRSI crosses back above 0 (returns to mean)
- **Short Entry**: When ZScoreRSI crosses above +2.0 (extreme overbought relative to recent RSI)
- **Short Exit**: When ZScoreRSI crosses back below 0

**Note**: These thresholds (+2.0, -2.0) would be learned by binning models, not hardcoded in the node.

## Edge Cases

1. **Zero Standard Deviation**:
   - If all RSI values in the z_length window are identical
   - `rsi_std == 0`
   - Return `0.0` (neutral z-score)

2. **Insufficient Data**:
   - During warmup, return `0.0` (neutral)
   - Need at least `z_length` valid RSI values

3. **RSI Warmup Period**:
   - RSI returns `50.0` during its warmup
   - Don't include these `50.0` values in z-score calculation
   - Wait until RSI is producing real values

## Performance Optimization

- **Circular Buffer**: Use NumPy array with modulo indexing for RSI values
- **Efficient Statistics**: Use `np.mean()` and `np.std()` on buffer
- **Minimize State**: Only store RSI values needed for z-score calculation

## Testing Considerations

When testing the implementation:

1. **Verify RSI Calculation**: Check that RSI values match standard RSI
2. **Verify Z-Score Calculation**:
   - When RSI is at mean → z-score should be ~0.0
   - When RSI is 1 std above mean → z-score should be ~+1.0
   - When RSI is 2 std above mean → z-score should be ~+2.0
3. **Verify Volatility Adaptation**: 
   - Test in high volatility (RSI pinned low) - z-score should still vary
   - Test in low volatility (RSI near 50) - z-score should still identify extremes
4. **Edge Cases**: Test with constant RSI values, zero std, etc.

## Expected Column Name

With parameters `rsi_lookback=14, z_length=20`:
```
zscorersi_signal_D_rsi_lookback_14_z_length_20
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reuse this via composition
- **Z-Score Example**: `nodes/archive/return_zscore.py` - Example of z-score calculation
- **Statistical Normalization**: Uses same z-score formula: `(value - mean) / std`

## Important Notes

1. **Continuous Feature**: This outputs z-score values (continuous), not discrete signals. Binning models will learn optimal thresholds.

2. **Volatility Adaptive**: Unlike static RSI thresholds (30/70), z-score adapts to current market volatility by comparing RSI to its recent distribution.

3. **Statistical Significance**: Z-scores provide a mathematical basis for identifying extremes. Values beyond ±2.0 are statistically significant (about 95% confidence interval for normal distribution).

4. **Output Range**: While z-scores are theoretically unbounded, in practice they typically range from -3.0 to +3.0. Values beyond ±3.0 indicate extremely rare/extreme conditions.

5. **No Additional Normalization**: Z-score is already normalized by definition. The output is the raw z-score value, which is comparable across different market conditions.
