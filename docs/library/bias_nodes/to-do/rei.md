# DeMark REI (Range Expansion Index) Bias Node Specification

## Overview

The **DeMark REI (Range Expansion Index)** is a momentum oscillator developed by Tom DeMark that measures range expansion in price action. It uses High and Low prices (not Close) and identifies overbought/oversold conditions based on range expansion patterns.

**Key Benefit**: Provides a mean-reversion signal that works best on daily and higher timeframes, identifying exhaustion points in price ranges.

**Type**: Continuous feature (outputs -100.0 to +100.0, requires binning)

**Important**: The 5-bar offset is **mandatory** - this is where many implementations go wrong. Do not smooth REI again as it already embeds smoothing.

## Formula

The REI calculation is a three-step process:

### Step 1: Directional Conditions (5-bar offset)

For each candle at time `t`, compare with candle at `t-5`:

```
up_cond[t]   = (High[t] >= High[t-5]) AND (Low[t] >= Low[t-5])
down_cond[t] = (High[t] <= High[t-5]) AND (Low[t] <= Low[t-5])
```

**Critical**: The 5-bar offset is mandatory. This compares current High/Low with High/Low from 5 bars ago.

### Step 2: Compute Range Expansion Values

For each candle at time `t`, compute numerator and denominator:

```
if up_cond[t]:
    num[t] = (High[t] - High[t-2]) + (Low[t] - Low[t-2])
else if down_cond[t]:
    num[t] = -((High[t] - High[t-2]) + (Low[t] - Low[t-2]))
else:
    num[t] = 0

den[t] = abs(High[t] - High[t-2]) + abs(Low[t] - Low[t-2])
```

**Note**: Uses `t-2` offset for the range expansion calculation (2 bars back from current).

### Step 3: Rolling Sums (8-bar window)

For each candle at time `t`, sum over the last 8 bars:

```
NUM = sum(num[t-7 .. t])  # Sum of last 8 num values
DEN = sum(den[t-7 .. t])  # Sum of last 8 den values

if DEN == 0:
    REI[t] = 0
else:
    REI[t] = 100 * NUM / DEN
```

## Interpretation

The REI uses standard DeMark interpretation:
- **Above +60**: Overbought / upside exhaustion (potential mean reversion)
- **Below -60**: Oversold / downside exhaustion (potential mean reversion)
- **Near 0**: Neutral range expansion

**Best Practices**:
- Works best on daily and higher timeframes
- Best used with price flip / TD setup / trend context
- Signals are mean-reversion, not momentum

## Required Parameters

- `lookback` (int, default=8): Period for rolling sum calculation (8-bar window)
- `offset` (int, default=5): Offset for directional condition comparison (mandatory: 5 bars)
- `range_offset` (int, default=2): Offset for range expansion calculation (2 bars back)

**Note**: The `offset=5` is **mandatory** per DeMark's definition. Do not change this unless you're implementing a variant.

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = offset + range_offset + lookback - 1
front_bad = 5 + 2 + 8 - 1 = 14
```

This ensures we have:
1. Enough candles for the 5-bar offset comparison
2. Enough candles for the 2-bar range expansion calculation
3. Enough candles for the 8-bar rolling sum

**Minimum**: Need at least 14 candles before valid REI output.

## State Management

You need to maintain:

1. **High/Low History**: Circular buffers to store High and Low prices
   - Need to access `High[t]`, `High[t-2]`, `High[t-5]`
   - Need to access `Low[t]`, `Low[t-2]`, `Low[t-5]`
   - Buffer size: `max(offset, range_offset) + 1 = max(5, 2) + 1 = 6`

2. **Num/Den Buffers**: Circular buffers for numerator and denominator values
   - Size: `lookback = 8`
   - Store last 8 `num` and `den` values for rolling sum

3. **Buffer Indices**: Track positions in circular buffers

4. **Candle Count**: Track number of candles processed

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'rei'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'lookback': lookback, 'offset': offset, 'range_offset': range_offset}`
- [ ] Sets `self.front_bad = offset + range_offset + lookback - 1` (default: 14)
- [ ] Initializes circular buffers for High/Low prices
- [ ] Initializes circular buffers for num/den values
- [ ] Implements directional condition checks (5-bar offset)
- [ ] Implements range expansion calculation (2-bar offset)
- [ ] Implements rolling sum (8-bar window)
- [ ] Handles warmup period (returns 0.0 - neutral value)
- [ ] Handles edge case: `DEN == 0` (return 0.0)
- [ ] Returns continuous value (-100.0 to +100.0)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class REI(BiasNode):
    """
    DeMark REI (Range Expansion Index) Bias Node - CONTINUOUS FEATURE
    
    Measures range expansion in price action using High and Low prices.
    Identifies overbought/oversold conditions based on range expansion patterns.
    
    Formula:
    1. Check directional conditions: Compare High/Low[t] with High/Low[t-5]
    2. Compute range expansion: num and den using High/Low[t] vs High/Low[t-2]
    3. Rolling sum: Sum last 8 bars of num and den, then REI = 100 * NUM / DEN
    
    **Type**: Continuous feature (outputs -100.0 to +100.0, requires binning)
    **Normalization**: Not needed (fixed -100 to +100 range)
    
    **Critical**: The 5-bar offset is mandatory. Do not smooth REI again.
    
    Parameters:
    - lookback: Period for rolling sum (default: 8)
    - offset: Offset for directional condition (default: 5, mandatory)
    - range_offset: Offset for range expansion (default: 2)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        lookback: int = 8,
        offset: int = 5,
        range_offset: int = 2
    ):
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.offset = offset  # Mandatory: 5-bar offset for directional conditions
        self.range_offset = range_offset  # 2-bar offset for range expansion
        
        # Standardization metadata
        self.module_name = 'rei'
        self.output_features = ['signal']
        self.params = {
            'lookback': lookback,
            'offset': offset,
            'range_offset': range_offset
        }
        
        # Warmup period: need offset + range_offset + lookback - 1
        # For default: 5 + 2 + 8 - 1 = 14
        self.front_bad = offset + range_offset + lookback - 1
        
        # Circular buffers for High/Low prices
        # Need to access up to offset bars back (5 bars)
        buffer_size = max(offset, range_offset) + 1  # Need max(5, 2) + 1 = 6
        self.high_buffer = np.zeros(buffer_size, dtype=np.float64)
        self.low_buffer = np.zeros(buffer_size, dtype=np.float64)
        self.price_buffer_idx = 0
        self.n_prices = 0
        
        # Circular buffers for num and den values (for rolling sum)
        self.num_buffer = np.zeros(lookback, dtype=np.float64)
        self.den_buffer = np.zeros(lookback, dtype=np.float64)
        self.num_den_idx = 0
        self.n_num_den = 0  # Count of num/den values stored
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute REI for the given candle.
        
        Steps:
        1. Store High/Low in circular buffer
        2. Check directional conditions (5-bar offset)
        3. Calculate num and den (2-bar offset)
        4. Update num/den buffers
        5. Calculate rolling sum and REI (8-bar window)
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the REI value (-100.0 to +100.0)
        """
        curr_high = candle.high
        curr_low = candle.low
        
        # Update High/Low buffers
        self.high_buffer[self.price_buffer_idx] = curr_high
        self.low_buffer[self.price_buffer_idx] = curr_low
        self.price_buffer_idx = (self.price_buffer_idx + 1) % len(self.high_buffer)
        self.n_prices += 1
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            self.output.append(0.0)  # Neutral value
            return [0.0]
        
        # Step 1: Check directional conditions (5-bar offset)
        # Need to access High[t-5] and Low[t-5]
        idx_t_minus_5 = (self.price_buffer_idx - offset) % len(self.high_buffer)
        high_t_minus_5 = self.high_buffer[idx_t_minus_5]
        low_t_minus_5 = self.low_buffer[idx_t_minus_5]
        
        up_cond = (curr_high >= high_t_minus_5) and (curr_low >= low_t_minus_5)
        down_cond = (curr_high <= high_t_minus_5) and (curr_low <= low_t_minus_5)
        
        # Step 2: Compute range expansion values (2-bar offset)
        # Need to access High[t-2] and Low[t-2]
        idx_t_minus_2 = (self.price_buffer_idx - range_offset) % len(self.high_buffer)
        high_t_minus_2 = self.high_buffer[idx_t_minus_2]
        low_t_minus_2 = self.low_buffer[idx_t_minus_2]
        
        if up_cond:
            num = (curr_high - high_t_minus_2) + (curr_low - low_t_minus_2)
        elif down_cond:
            num = -((curr_high - high_t_minus_2) + (curr_low - low_t_minus_2))
        else:
            num = 0.0
        
        den = abs(curr_high - high_t_minus_2) + abs(curr_low - low_t_minus_2)
        
        # Update num/den buffers
        self.num_buffer[self.num_den_idx] = num
        self.den_buffer[self.num_den_idx] = den
        self.num_den_idx = (self.num_den_idx + 1) % self.lookback
        if self.n_num_den < self.lookback:
            self.n_num_den += 1
        
        # Need at least lookback num/den values for rolling sum
        if self.n_num_den < self.lookback:
            self.output.append(0.0)  # Neutral value during warmup
            return [0.0]
        
        # Step 3: Rolling sum (8-bar window)
        NUM = np.sum(self.num_buffer)
        DEN = np.sum(self.den_buffer)
        
        # Calculate REI
        if DEN < 1e-10:  # Avoid division by zero
            rei = 0.0
        else:
            rei = 100.0 * NUM / DEN
            # Clamp to -100 to +100 range (safety check)
            rei = np.clip(rei, -100.0, 100.0)
        
        self.output.append(rei)
        return [rei]
```

## Key Implementation Notes

1. **High/Low Buffers**: 
   - Use circular buffer of size `max(offset, range_offset) + 1 = 6`
   - Need to access `t`, `t-2`, and `t-5` positions
   - Handle wrap-around when accessing historical values

2. **Directional Conditions**:
   ```python
   up_cond = (High[t] >= High[t-5]) AND (Low[t] >= Low[t-5])
   down_cond = (High[t] <= High[t-5]) AND (Low[t] <= Low[t-5])
   ```
   **Critical**: The 5-bar offset is mandatory - this is where many implementations go wrong.

3. **Range Expansion Calculation**:
   ```python
   if up_cond:
       num = (High[t] - High[t-2]) + (Low[t] - Low[t-2])
   elif down_cond:
       num = -((High[t] - High[t-2]) + (Low[t] - Low[t-2]))
   else:
       num = 0.0
   
   den = abs(High[t] - High[t-2]) + abs(Low[t] - Low[t-2])
   ```
   Uses 2-bar offset for range expansion.

4. **Rolling Sum**:
   - Sum last 8 `num` values → `NUM`
   - Sum last 8 `den` values → `DEN`
   - `REI = 100 * NUM / DEN` (if DEN > 0)

5. **Edge Case Handling**:
   - If `DEN == 0` → return `0.0` (neutral)
   - During warmup → return `0.0` (neutral)

6. **Output**: Continuous value -100.0 to +100.0 (fixed range, no normalization needed)

7. **Do NOT Smooth**: REI already embeds smoothing in the 8-bar rolling sum. Do not apply additional smoothing.

## Circular Buffer Access Pattern

When accessing historical values from circular buffers:

```python
# Current index is self.price_buffer_idx
# To get value from N bars ago:
idx_n_bars_ago = (self.price_buffer_idx - N) % len(self.high_buffer)
value = self.high_buffer[idx_n_bars_ago]
```

**Example**:
- Current: `self.price_buffer_idx = 3`
- Get `t-5`: `idx = (3 - 5) % 6 = (-2) % 6 = 4`
- Get `t-2`: `idx = (3 - 2) % 6 = 1`

## Edge Cases

1. **Zero Denominator**:
   - If all `den` values in the 8-bar window are 0
   - Return `0.0` (neutral REI)

2. **Insufficient Data**:
   - During warmup, return `0.0` (neutral)
   - Need at least 14 candles for valid REI

3. **Constant Prices**:
   - If High/Low don't change, `den` will be 0
   - REI will be 0.0 (neutral)

## Performance Optimization

- **Circular Buffers**: Use NumPy arrays with modulo indexing
- **Efficient Sums**: Use `np.sum()` on buffer slices
- **Minimize State**: Only store High/Low and num/den buffers

## Testing Considerations

When testing the implementation:

1. **Verify 5-Bar Offset**: 
   - Check that directional conditions use `t-5` comparison
   - This is critical - many implementations get this wrong

2. **Verify Range Expansion**:
   - Check that num/den use `t-2` offset
   - Verify sign: positive for up_cond, negative for down_cond

3. **Verify Rolling Sum**:
   - Check that it sums last 8 bars correctly
   - Verify REI calculation: `100 * NUM / DEN`

4. **Verify Output Range**:
   - REI should be between -100 and +100
   - Test edge cases (all zeros, constant prices)

5. **Compare with Reference**:
   - Test against known REI values from DeMark's definition
   - Verify it matches expected behavior on sample data

## Expected Column Name

With default parameters `lookback=8, offset=5, range_offset=2`:
```
rei_signal_D_lookback_8_offset_5_range_offset_2
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **High/Low Example**: `nodes/williamsr.py` - Example of using High/Low prices
- **ATR Example**: `nodes/atr.py` - Example of using High/Low/Close for True Range
- **Candle Model**: `utils/models.py` - `Candle` class with `high` and `low` attributes

## Important Notes

1. **5-Bar Offset is Mandatory**: This is a critical part of DeMark's REI definition. Do not change this to a different offset.

2. **Uses High/Low, Not Close**: REI is unique in that it uses High and Low prices, not Close prices. This is intentional and part of the indicator design.

3. **No Additional Smoothing**: REI already includes smoothing via the 8-bar rolling sum. Do not apply additional moving averages or smoothing.

4. **Mean Reversion Signal**: REI identifies exhaustion points, not momentum. Values above +60 suggest upside exhaustion (potential reversal down), values below -60 suggest downside exhaustion (potential reversal up).

5. **Timeframe Considerations**: Works best on daily and higher timeframes. May be noisy on lower timeframes.
