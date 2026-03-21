# Stochastic RSI (StochRSI) Bias Node Specification

## Overview

The **Stochastic RSI (StochRSI)** is a normalized version of the Relative Strength Index that measures RSI's position relative to its own range over a specified period. By normalizing RSI to a 0-100 scale, StochRSI makes the indicator more sensitive to overbought and oversold conditions, making it particularly effective for mean reversion trading on stock indices.

**Key Benefit**: More sensitive than raw RSI, making it easier to identify extreme conditions. The 0-100 range provides intuitive thresholds for mean reversion strategies.

**Type**: Continuous feature (outputs 0.0-100.0, requires binning)

**Normalization**: Not needed (output is already normalized to fixed 0-100 range)

## Formula

The StochRSI calculation is a two-step process:

### Step 1: Calculate Standard RSI

```
RSI[t] = CalculateRSI(close_prices, rsi_length)
```

Use the standard RSI calculation (0-100 range).

### Step 2: Apply Stochastic Normalization

```
RSI_min = min(RSI[t-period+1] ... RSI[t])
RSI_max = max(RSI[t-period+1] ... RSI[t])

StochRSI[t] = (RSI[t] - RSI_min) / (RSI_max - RSI_min) × 100
```

**Stochastic Formula**: `(Current - Minimum) / (Maximum - Minimum) × 100`

This measures where the current RSI sits within its recent range, normalized to 0-100.

## Interpretation

The StochRSI uses the same interpretation as standard RSI, but with enhanced sensitivity:

- **Above 80**: Overbought (RSI is in the top 20% of its recent range)
- **Below 20**: Oversold (RSI is in the bottom 20% of its recent range)
- **Near 50**: RSI is near the middle of its recent range (neutral)

**Typical Range**: 0.0 to 100.0 (always bounded)

**Why It's Better**:
- **Enhanced Sensitivity**: More responsive to RSI extremes than raw RSI
- **Intuitive Scale**: 0-100 range makes thresholds easy to interpret
- **Relative Positioning**: Shows where RSI sits relative to its recent distribution
- **Mean Reversion**: Works exceptionally well for mean reversion on indices

## Implementation Strategy

### Recommended: Composition with RSI Node

Reuse the existing `RSI` node and track RSI values for stochastic calculation:

```python
from nodes.rsi import RSI

class StochRSI(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, stoch_period=14):
        # ... initialization ...
        # Create internal RSI node
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        # Circular buffer to store RSI values for stochastic calculation
        self.rsi_buffer = np.zeros(stoch_period, dtype=np.float64)
        
    def _compute_candle(self, candle):
        # Get RSI value from internal node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Update RSI buffer
        # Calculate min and max of RSI values
        rsi_min = np.min(self.rsi_buffer)
        rsi_max = np.max(self.rsi_buffer)
        
        # Calculate StochRSI
        if rsi_max - rsi_min > 1e-10:
            stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100.0
        else:
            stoch_rsi = 50.0  # Neutral if no variation
        
        return [stoch_rsi]
```

## Required Parameters

- `rsi_lookback` (int, default=14): Period for RSI calculation (typically 2-14)
- `stoch_period` (int, default=14): Period for calculating RSI min/max (typically 14-21)

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = max(rsi_lookback, stoch_period)
```

This ensures we have:
1. Enough candles for RSI calculation (`rsi_lookback`)
2. Enough RSI values for stochastic calculation (`stoch_period` RSI values)

**Minimum**: Need at least `max(rsi_lookback, stoch_period)` candles before valid StochRSI output.

## State Management

You need to maintain:

1. **RSI Node** (via composition):
   - Internal `RSI` instance handles all RSI computation
   - No need to manage RSI buffers/state directly

2. **RSI Values Buffer**: Circular buffer to store RSI values
   - Size: `stoch_period`
   - Stores last `stoch_period` RSI values for min/max calculation

3. **Buffer Index**: Track position in RSI values buffer

4. **Candle Count**: Track number of candles processed

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'stochrsi'` (or `'stoch_rsi'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'rsi_lookback': rsi_lookback, 'stoch_period': stoch_period}`
- [ ] Sets `self.front_bad = max(rsi_lookback, stoch_period)`
- [ ] Creates internal `RSI` node instance
- [ ] Initializes circular buffer for RSI values (size: `stoch_period`)
- [ ] Implements RSI value tracking
- [ ] Implements min and max calculation of RSI values
- [ ] Implements stochastic calculation: `(rsi - min) / (max - min) × 100`
- [ ] Handles warmup period (returns 50.0 - neutral value)
- [ ] Handles edge case: `max == min` (return 50.0)
- [ ] Returns continuous value (0.0-100.0, always bounded)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from nodes.rsi import RSI  # Reuse existing RSI


class StochRSI(BiasNode):
    """
    Stochastic RSI (StochRSI) Bias Node - CONTINUOUS FEATURE
    
    Normalized version of RSI that measures RSI's position relative to its
    own range over a specified period.
    
    Formula:
    1. Calculate standard RSI (0-100)
    2. Calculate min and max of RSI over stoch_period
    3. StochRSI = (RSI - min) / (max - min) × 100
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    **Normalization**: Not needed (output is already normalized to fixed 0-100 range)
    
    **Key Benefit**: More sensitive than raw RSI, making it easier to identify
    extreme conditions for mean reversion strategies.
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - stoch_period: Period for RSI min/max calculation (default: 14)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        rsi_lookback: int = 14,
        stoch_period: int = 14
    ):
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.stoch_period = stoch_period
        
        # Standardization metadata
        self.module_name = 'stochrsi'
        self.output_features = ['signal']
        self.params = {
            'rsi_lookback': rsi_lookback,
            'stoch_period': stoch_period
        }
        
        # Warmup period: need max of RSI lookback and stoch_period
        self.front_bad = max(rsi_lookback, stoch_period)
        
        # Create internal RSI node for RSI calculation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        # Circular buffer to store RSI values for stochastic calculation
        self.rsi_buffer = np.zeros(stoch_period, dtype=np.float64)
        self.rsi_buffer_idx = 0
        self.n_rsi_values = 0  # Count of RSI values stored
        
        # Track candle count
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Stochastic RSI for the given candle.
        
        Steps:
        1. Get RSI value from internal RSI node
        2. Store RSI in circular buffer
        3. Calculate min and max of RSI values over stoch_period
        4. Calculate StochRSI: (current_RSI - min) / (max - min) × 100
        5. Return StochRSI value (0.0-100.0)
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the StochRSI value (0.0-100.0)
        """
        self.n_prices += 1
        
        # Get RSI value from internal RSI node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Handle warmup period (RSI returns 50.0 during warmup)
        if self.n_prices < self.front_bad:
            self.output.append(50.0)  # Neutral value
            return [50.0]
        
        # Skip RSI values during RSI warmup (they're 50.0, not real values)
        # Only start tracking when RSI is valid
        if rsi == 50.0 and self.n_prices < self.rsi_lookback:
            self.output.append(50.0)
            return [50.0]
        
        # Update RSI buffer
        self.rsi_buffer[self.rsi_buffer_idx] = rsi
        self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.stoch_period
        if self.n_rsi_values < self.stoch_period:
            self.n_rsi_values += 1
        
        # Need at least stoch_period RSI values for stochastic calculation
        if self.n_rsi_values < self.stoch_period:
            self.output.append(50.0)  # Neutral value during warmup
            return [50.0]
        
        # Calculate min and max of RSI values
        rsi_min = np.min(self.rsi_buffer)
        rsi_max = np.max(self.rsi_buffer)
        rsi_range = rsi_max - rsi_min
        
        # Calculate StochRSI
        if rsi_range > 1e-10:  # Avoid division by zero
            stoch_rsi = ((rsi - rsi_min) / rsi_range) * 100.0
            # Clip to ensure bounds (shouldn't be necessary, but safe)
            stoch_rsi = np.clip(stoch_rsi, 0.0, 100.0)
        else:
            stoch_rsi = 50.0  # Neutral if no variation
        
        # Update bias based on StochRSI value
        if stoch_rsi > 80:
            self.bias = Bias.BEARISH  # Overbought
        elif stoch_rsi < 20:
            self.bias = Bias.BULLISH  # Oversold
        else:
            self.bias = Bias.NEUTRAL
        
        self.output.append(stoch_rsi)
        return [stoch_rsi]
```

## Key Implementation Notes

1. **RSI Reuse**: The internal RSI node handles all RSI computation. You just call `self.rsi_node.add_candle(candle)[0]` to get the RSI value.

2. **RSI Value Tracking**: 
   - Store RSI values in a circular buffer of size `stoch_period`
   - Only start tracking after RSI warmup (when RSI != 50.0)

3. **Stochastic Calculation**:
   ```python
   rsi_min = np.min(self.rsi_buffer)
   rsi_max = np.max(self.rsi_buffer)
   rsi_range = rsi_max - rsi_min
   stoch_rsi = ((rsi - rsi_min) / rsi_range) * 100.0
   ```
   Uses min/max normalization to map RSI to 0-100 range.

4. **Edge Case Handling**:
   - If `rsi_max == rsi_min` (all RSI values are the same) → return `50.0` (neutral)
   - During warmup → return `50.0` (neutral value)

5. **Output Range**: 
   - Always bounded between 0.0 and 100.0
   - 0.0 = RSI is at its minimum in the period
   - 100.0 = RSI is at its maximum in the period
   - 50.0 = RSI is at the middle of its range

6. **No Additional Normalization**: StochRSI is already normalized to 0-100 range. The output is the raw stochastic value.

## Why StochRSI is Better for Mean Reversion

### Enhanced Sensitivity Example

**Scenario 1: RSI Range 30-70**:
- Raw RSI at 70: Still in "normal" range
- StochRSI at 70: Would be 100.0 (at maximum of recent range) - clearly overbought

**Scenario 2: RSI Range 10-30**:
- Raw RSI at 30: Might seem "normal" but is actually at the top of its recent range
- StochRSI at 30: Would be 100.0 (at maximum) - clearly overbought relative to recent conditions

**Key Insight**: StochRSI adapts to the current RSI distribution, making it more sensitive to relative extremes than absolute RSI values.

## Trading Strategy Logic (Reference)

While this is a **continuous feature** (not rule-based), here's how it would typically be used:

**Mean Reversion Strategy**:
- **Long Entry**: When StochRSI crosses below 20 (RSI is in bottom 20% of recent range)
- **Long Exit**: When StochRSI crosses back above 50 (returns to middle of range)
- **Short Entry**: When StochRSI crosses above 80 (RSI is in top 20% of recent range)
- **Short Exit**: When StochRSI crosses back below 50

**Note**: These thresholds (20, 50, 80) would be learned by binning models, not hardcoded in the node.

## Edge Cases

1. **Zero Range**:
   - If all RSI values in the stoch_period window are identical
   - `rsi_max == rsi_min`
   - Return `50.0` (neutral value)

2. **Insufficient Data**:
   - During warmup, return `50.0` (neutral)
   - Need at least `stoch_period` valid RSI values

3. **RSI Warmup Period**:
   - RSI returns `50.0` during its warmup
   - Don't include these `50.0` values in stochastic calculation
   - Wait until RSI is producing real values

## Performance Optimization

- **Circular Buffer**: Use NumPy array with modulo indexing for RSI values
- **Efficient Min/Max**: Use `np.min()` and `np.max()` on buffer
- **Minimize State**: Only store RSI values needed for stochastic calculation

## Testing Considerations

When testing the implementation:

1. **Verify RSI Calculation**: Check that RSI values match standard RSI
2. **Verify Stochastic Calculation**:
   - When RSI is at min → StochRSI should be 0.0
   - When RSI is at max → StochRSI should be 100.0
   - When RSI is at middle → StochRSI should be ~50.0
3. **Verify Sensitivity**: 
   - StochRSI should be more sensitive than raw RSI
   - Should identify extremes relative to recent RSI distribution
4. **Edge Cases**: Test with constant RSI values, zero range, etc.

## Expected Column Name

With parameters `rsi_lookback=14, stoch_period=14`:
```
stochrsi_signal_D_rsi_lookback_14_stoch_period_14
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reuse this via composition
- **Similar Patterns**: `docs/bias_nodes/to-do/zscore_rsi.md` - Similar structure for RSI normalization

## Important Notes

1. **Continuous Feature**: This outputs StochRSI values (continuous), not discrete signals. Binning models will learn optimal thresholds.

2. **Enhanced Sensitivity**: Unlike raw RSI which uses fixed thresholds (30/70), StochRSI adapts to the current RSI distribution, making it more sensitive to relative extremes.

3. **Bounded Range**: StochRSI is always bounded between 0.0 and 100.0, making it easier to interpret than unbounded indicators.

4. **Mean Reversion**: Works exceptionally well for mean reversion strategies on stock indices, as it identifies when RSI is at extremes relative to its recent behavior.

5. **No Additional Normalization**: StochRSI is already normalized to 0-100 range. The output is the raw stochastic value.
