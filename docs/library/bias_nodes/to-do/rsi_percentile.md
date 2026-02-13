# RSI Percentile Bias Node Specification

## Overview

The **RSI Percentile** is a non-parametric normalized version of the Relative Strength Index that measures where the current RSI value ranks within its historical distribution over a specified period. Unlike z-score normalization (which assumes normal distribution), percentile ranking is more robust to outliers and doesn't require distribution assumptions.

**Key Benefit**: Non-parametric approach that adapts to any RSI distribution shape. More robust than z-score for identifying extremes, making it particularly effective for mean reversion trading on stock indices.

**Type**: Continuous feature (outputs 0.0-100.0, requires binning)

**Normalization**: Not needed (output is already normalized to fixed 0-100 range)

## Formula

The RSI Percentile calculation is a two-step process:

### Step 1: Calculate Standard RSI

```
RSI[t] = CalculateRSI(close_prices, rsi_length)
```

Use the standard RSI calculation (0-100 range).

### Step 2: Calculate Percentile Rank

```
Count_below = Number of RSI values in period that are < RSI[t]
Count_equal = Number of RSI values in period that are == RSI[t]

RSI_Percentile[t] = (Count_below + 0.5 × Count_equal) / Total_Count × 100
```

**Percentile Formula**: `(Number of values below + 0.5 × Number of values equal) / Total × 100`

This measures what percentile the current RSI sits at within its recent distribution.

## Interpretation

The RSI Percentile uses percentile-based thresholds:

- **Above 80**: Overbought (RSI is in the top 20th percentile of its recent distribution)
- **Below 20**: Oversold (RSI is in the bottom 20th percentile of its recent distribution)
- **Near 50**: RSI is at the median of its recent distribution (neutral)

**Typical Range**: 0.0 to 100.0 (always bounded)

**Why It's Better**:
- **Non-Parametric**: Doesn't assume normal distribution (unlike z-score)
- **Robust to Outliers**: Percentile ranking is less affected by extreme values
- **Distribution Agnostic**: Works with any RSI distribution shape
- **Intuitive Scale**: 0-100 range represents percentile directly
- **Mean Reversion**: Excellent for mean reversion on indices

## Implementation Strategy

### Recommended: Composition with RSI Node

Reuse the existing `RSI` node and track RSI values for percentile calculation:

```python
from nodes.rsi import RSI

class RSIPercentile(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, percentile_period=20):
        # ... initialization ...
        # Create internal RSI node
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        # Circular buffer to store RSI values for percentile calculation
        self.rsi_buffer = np.zeros(percentile_period, dtype=np.float64)
        
    def _compute_candle(self, candle):
        # Get RSI value from internal node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Update RSI buffer
        # Calculate percentile rank
        count_below = np.sum(self.rsi_buffer < rsi)
        count_equal = np.sum(self.rsi_buffer == rsi)
        total = len(self.rsi_buffer)
        
        rsi_percentile = ((count_below + 0.5 * count_equal) / total) * 100.0
        
        return [rsi_percentile]
```

## Required Parameters

- `rsi_lookback` (int, default=14): Period for RSI calculation (typically 2-14)
- `percentile_period` (int, default=20): Period for calculating percentile rank (typically 20-100)

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = max(rsi_lookback, percentile_period)
```

This ensures we have:
1. Enough candles for RSI calculation (`rsi_lookback`)
2. Enough RSI values for percentile calculation (`percentile_period` RSI values)

**Minimum**: Need at least `max(rsi_lookback, percentile_period)` candles before valid RSI Percentile output.

## State Management

You need to maintain:

1. **RSI Node** (via composition):
   - Internal `RSI` instance handles all RSI computation
   - No need to manage RSI buffers/state directly

2. **RSI Values Buffer**: Circular buffer to store RSI values
   - Size: `percentile_period`
   - Stores last `percentile_period` RSI values for percentile calculation

3. **Buffer Index**: Track position in RSI values buffer

4. **Candle Count**: Track number of candles processed

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'rsipercentile'` (or `'rsi_percentile'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'rsi_lookback': rsi_lookback, 'percentile_period': percentile_period}`
- [ ] Sets `self.front_bad = max(rsi_lookback, percentile_period)`
- [ ] Creates internal `RSI` node instance
- [ ] Initializes circular buffer for RSI values (size: `percentile_period`)
- [ ] Implements RSI value tracking
- [ ] Implements percentile rank calculation
- [ ] Handles warmup period (returns 50.0 - median percentile)
- [ ] Returns continuous value (0.0-100.0, always bounded)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from nodes.rsi import RSI  # Reuse existing RSI


class RSIPercentile(BiasNode):
    """
    RSI Percentile Bias Node - CONTINUOUS FEATURE
    
    Non-parametric normalized version of RSI that measures where the current
    RSI value ranks within its historical distribution.
    
    Formula:
    1. Calculate standard RSI (0-100)
    2. Calculate percentile rank: (count_below + 0.5 × count_equal) / total × 100
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    **Normalization**: Not needed (output is already normalized to fixed 0-100 range)
    
    **Key Benefit**: Non-parametric approach that adapts to any RSI distribution
    shape. More robust than z-score for identifying extremes.
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - percentile_period: Period for percentile calculation (default: 20)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        rsi_lookback: int = 14,
        percentile_period: int = 20
    ):
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.percentile_period = percentile_period
        
        # Standardization metadata
        self.module_name = 'rsipercentile'
        self.output_features = ['signal']
        self.params = {
            'rsi_lookback': rsi_lookback,
            'percentile_period': percentile_period
        }
        
        # Warmup period: need max of RSI lookback and percentile_period
        self.front_bad = max(rsi_lookback, percentile_period)
        
        # Create internal RSI node for RSI calculation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        # Circular buffer to store RSI values for percentile calculation
        self.rsi_buffer = np.zeros(percentile_period, dtype=np.float64)
        self.rsi_buffer_idx = 0
        self.n_rsi_values = 0  # Count of RSI values stored
        
        # Track candle count
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI Percentile for the given candle.
        
        Steps:
        1. Get RSI value from internal RSI node
        2. Store RSI in circular buffer
        3. Calculate percentile rank: (count_below + 0.5 × count_equal) / total × 100
        4. Return RSI Percentile value (0.0-100.0)
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the RSI Percentile value (0.0-100.0)
        """
        self.n_prices += 1
        
        # Get RSI value from internal RSI node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Handle warmup period (RSI returns 50.0 during warmup)
        if self.n_prices < self.front_bad:
            self.output.append(50.0)  # Median percentile
            return [50.0]
        
        # Skip RSI values during RSI warmup (they're 50.0, not real values)
        # Only start tracking when RSI is valid
        if rsi == 50.0 and self.n_prices < self.rsi_lookback:
            self.output.append(50.0)
            return [50.0]
        
        # Update RSI buffer
        self.rsi_buffer[self.rsi_buffer_idx] = rsi
        self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.percentile_period
        if self.n_rsi_values < self.percentile_period:
            self.n_rsi_values += 1
        
        # Need at least percentile_period RSI values for percentile calculation
        if self.n_rsi_values < self.percentile_period:
            self.output.append(50.0)  # Median percentile during warmup
            return [50.0]
        
        # Calculate percentile rank
        # Count values below current RSI
        count_below = np.sum(self.rsi_buffer < rsi)
        # Count values equal to current RSI
        count_equal = np.sum(self.rsi_buffer == rsi)
        # Total count (should be percentile_period)
        total = self.n_rsi_values
        
        # Percentile formula: (count_below + 0.5 × count_equal) / total × 100
        # The 0.5 × count_equal handles ties (splits them evenly)
        rsi_percentile = ((count_below + 0.5 * count_equal) / total) * 100.0
        
        # Clip to ensure bounds (shouldn't be necessary, but safe)
        rsi_percentile = np.clip(rsi_percentile, 0.0, 100.0)
        
        # Update bias based on RSI Percentile value
        if rsi_percentile > 80:
            self.bias = Bias.BEARISH  # Overbought (top 20th percentile)
        elif rsi_percentile < 20:
            self.bias = Bias.BULLISH  # Oversold (bottom 20th percentile)
        else:
            self.bias = Bias.NEUTRAL
        
        self.output.append(rsi_percentile)
        return [rsi_percentile]
```

## Key Implementation Notes

1. **RSI Reuse**: The internal RSI node handles all RSI computation. You just call `self.rsi_node.add_candle(candle)[0]` to get the RSI value.

2. **RSI Value Tracking**: 
   - Store RSI values in a circular buffer of size `percentile_period`
   - Only start tracking after RSI warmup (when RSI != 50.0)

3. **Percentile Calculation**:
   ```python
   count_below = np.sum(self.rsi_buffer < rsi)
   count_equal = np.sum(self.rsi_buffer == rsi)
   total = self.n_rsi_values
   rsi_percentile = ((count_below + 0.5 * count_equal) / total) * 100.0
   ```
   Uses percentile ranking formula that handles ties by splitting them evenly.

4. **Edge Case Handling**:
   - During warmup → return `50.0` (median percentile)
   - If all values are equal → percentile will be 50.0 (middle)

5. **Output Range**: 
   - Always bounded between 0.0 and 100.0
   - 0.0 = RSI is at minimum (0th percentile)
   - 100.0 = RSI is at maximum (100th percentile)
   - 50.0 = RSI is at median (50th percentile)

6. **No Additional Normalization**: RSI Percentile is already normalized to 0-100 range. The output is the raw percentile value.

## Why RSI Percentile is Better Than Z-Score

### Non-Parametric Advantage

**Z-Score Assumes Normal Distribution**:
- If RSI distribution is skewed, z-score may misidentify extremes
- Sensitive to outliers (one extreme value can shift mean/std significantly)

**Percentile is Distribution Agnostic**:
- Works with any distribution shape (normal, skewed, bimodal, etc.)
- Robust to outliers (percentile rank is based on position, not distance from mean)
- More stable in non-normal market conditions

### Example: Skewed Distribution

**Scenario**: RSI values are skewed left (most values 20-40, few values 60-80)

- **Z-Score**: May not identify RSI=60 as extreme (if mean is low, std is small)
- **Percentile**: Correctly identifies RSI=60 as 90th percentile (clearly extreme)

## Trading Strategy Logic (Reference)

While this is a **continuous feature** (not rule-based), here's how it would typically be used:

**Mean Reversion Strategy**:
- **Long Entry**: When RSI Percentile crosses below 20 (RSI is in bottom 20th percentile)
- **Long Exit**: When RSI Percentile crosses back above 50 (returns to median)
- **Short Entry**: When RSI Percentile crosses above 80 (RSI is in top 20th percentile)
- **Short Exit**: When RSI Percentile crosses back below 50

**Note**: These thresholds (20, 50, 80) would be learned by binning models, not hardcoded in the node.

## Edge Cases

1. **All Values Equal**:
   - If all RSI values in the percentile_period window are identical
   - Percentile will be 50.0 (median) - correct behavior

2. **Insufficient Data**:
   - During warmup, return `50.0` (median percentile)
   - Need at least `percentile_period` valid RSI values

3. **RSI Warmup Period**:
   - RSI returns `50.0` during its warmup
   - Don't include these `50.0` values in percentile calculation
   - Wait until RSI is producing real values

## Performance Optimization

- **Circular Buffer**: Use NumPy array with modulo indexing for RSI values
- **Efficient Counting**: Use `np.sum()` with boolean masks for counting
- **Minimize State**: Only store RSI values needed for percentile calculation

## Testing Considerations

When testing the implementation:

1. **Verify RSI Calculation**: Check that RSI values match standard RSI
2. **Verify Percentile Calculation**:
   - When RSI is minimum → Percentile should be 0.0
   - When RSI is maximum → Percentile should be 100.0
   - When RSI is median → Percentile should be ~50.0
3. **Verify Robustness**: 
   - Test with skewed distributions
   - Test with outliers
   - Verify percentile is less affected than z-score
4. **Edge Cases**: Test with constant RSI values, insufficient data, etc.

## Expected Column Name

With parameters `rsi_lookback=14, percentile_period=20`:
```
rsipercentile_signal_D_rsi_lookback_14_percentile_period_20
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reuse this via composition
- **Similar Patterns**: 
  - `docs/bias_nodes/to-do/zscore_rsi.md` - Statistical normalization approach
  - `docs/bias_nodes/to-do/stoch_rsi.md` - Min/max normalization approach

## Important Notes

1. **Continuous Feature**: This outputs RSI Percentile values (continuous), not discrete signals. Binning models will learn optimal thresholds.

2. **Non-Parametric**: Unlike z-score which assumes normal distribution, percentile ranking works with any distribution shape.

3. **Robust to Outliers**: Percentile rank is based on position in sorted order, not distance from mean, making it less sensitive to extreme values.

4. **Mean Reversion**: Works exceptionally well for mean reversion strategies on stock indices, as it identifies when RSI is at extremes relative to its recent distribution without distribution assumptions.

5. **No Additional Normalization**: RSI Percentile is already normalized to 0-100 range. The output is the raw percentile value.
