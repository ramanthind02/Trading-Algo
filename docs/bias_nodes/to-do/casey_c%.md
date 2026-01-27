# Casey C% Bias Node Specification

## Overview

The **Casey C%** (Casey Close Change Percent Rank) is a momentum oscillator that ranks percent changes in closing prices over a lookback period, then smooths the result. Similar to RSI in that it oscillates between 0 and 100, but uses a different calculation method based on percent change ranking.

**Key Benefit**: Provides a momentum oscillator that shows overbought/oversold conditions similar to RSI, but with a different calculation approach that may capture different market dynamics.

**Type**: Continuous feature (outputs 0.0-100.0, requires binning)

## Formula

The Casey C% calculation is a multi-step process:

1. **Step 1: Calculate Percent Change**
   ```
   Change = (Close[t] - Close[t-1]) / Close[t-1]
   ```
   Percent change from previous close to current close.

2. **Step 2: Find Min/Max Over Lookback Period**
   ```
   LowestChange = min(Change[t-lookback+1] ... Change[t])
   HighestChange = max(Change[t-lookback+1] ... Change[t])
   ```
   Find the lowest and highest percent changes over the lookback window.

3. **Step 3: Rank Current Change (Normalize to 0-100)**
   ```
   Ranking = 100 * (Change[t] - LowestChange) / (HighestChange - LowestChange)
   ```
   If `HighestChange - LowestChange == 0`, return `50.0` (neutral).

4. **Step 4: Smooth the Ranking**
   ```
   CaseyC = mean(Ranking[t-smoothing+1] ... Ranking[t])
   ```
   Simple moving average of the ranked values over the smoothing period.

## Interpretation

The Casey C% uses similar interpretation to RSI:
- **Above 75**: Overbought (potential mean reversion opportunity)
- **Below 25**: Oversold (potential mean reversion opportunity)
- **50**: Neutral

## Implementation Strategy

### Recommended: Direct Implementation with Circular Buffers

This indicator can be implemented efficiently using circular buffers for:
- Percent changes (for ranking calculation)
- Ranked values (for smoothing)

```python
class CaseyC(BiasNode):
    def __init__(self, ticker, tf, lookback=7, smoothing=3):
        # ... initialization ...
        # Circular buffer for percent changes
        self.pct_change_buffer = np.zeros(lookback, dtype=np.float64)
        # Circular buffer for ranked values (for smoothing)
        self.ranked_buffer = np.zeros(smoothing, dtype=np.float64)
        
    def _compute_candle(self, candle):
        # Calculate percent change
        pct_change = (candle.close - self.prev_close) / self.prev_close
        
        # Update buffers and calculate ranking
        # ... implementation ...
```

## Required Parameters

- `lookback` (int, default=7): Period for finding min/max percent change (ranking window)
- `smoothing` (int, default=3): Period for smoothing the ranked values

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = lookback + smoothing
```

This ensures we have:
1. Enough candles for percent changes (need at least 2 for first pct_change)
2. Enough percent changes for ranking (need `lookback` changes)
3. Enough ranked values for smoothing (need `smoothing` ranked values)

## State Management

You need to maintain:

1. **Previous Close**: Store `self.prev_close` to calculate percent change
2. **Percent Change Buffer**: Circular buffer of size `lookback` for ranking calculation
3. **Ranked Values Buffer**: Circular buffer of size `smoothing` for smoothing
4. **Buffer Indices**: Track positions in circular buffers
5. **Candle Count**: Track number of candles processed

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'caseyc'` (or `'casey_c'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'lookback': lookback, 'smoothing': smoothing}`
- [ ] Sets `self.front_bad = lookback + smoothing`
- [ ] Initializes `self.prev_close = None`
- [ ] Initializes circular buffers for percent changes and ranked values
- [ ] Implements percent change calculation
- [ ] Implements ranking calculation (0-100 normalization)
- [ ] Implements smoothing (moving average of ranked values)
- [ ] Handles warmup period (returns 50.0 - neutral value)
- [ ] Handles edge case: `HighestChange - LowestChange == 0` (return 50.0)
- [ ] Returns continuous value (0.0-100.0)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class CaseyC(BiasNode):
    """
    Casey C% (Casey Close Change Percent Rank) Bias Node - CONTINUOUS FEATURE
    
    A momentum oscillator that ranks percent changes in closing prices over
    a lookback period, then smooths the result.
    
    Formula:
    1. Percent Change = (Close[t] - Close[t-1]) / Close[t-1]
    2. Ranking = 100 * (Change - min(Change)) / (max(Change) - min(Change))
    3. CaseyC = mean(Ranking) over smoothing period
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    **Normalization**: Not needed (same 0-100 range as RSI)
    
    Parameters:
    - lookback: Period for ranking calculation (default: 7)
    - smoothing: Period for smoothing ranked values (default: 3)
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        lookback: int = 7,
        smoothing: int = 3
    ):
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.smoothing = smoothing
        
        # Standardization metadata
        self.module_name = 'caseyc'
        self.output_features = ['signal']
        self.params = {
            'lookback': lookback,
            'smoothing': smoothing
        }
        
        # Warmup period: need lookback changes + smoothing ranked values
        # Plus 1 candle for first percent change calculation
        self.front_bad = lookback + smoothing + 1
        
        # Previous close for percent change calculation
        self.prev_close = None
        
        # Circular buffers for efficient storage
        self.pct_change_buffer = np.zeros(lookback, dtype=np.float64)
        self.pct_change_idx = 0
        self.n_pct_changes = 0  # Count of percent changes stored
        
        self.ranked_buffer = np.zeros(smoothing, dtype=np.float64)
        self.ranked_idx = 0
        self.n_ranked = 0  # Count of ranked values stored
        
        # Track candle count
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Casey C% for the given candle.
        
        Steps:
        1. Calculate percent change from previous close
        2. Update percent change buffer
        3. Calculate ranking (0-100) if enough data
        4. Update ranked buffer and smooth if enough data
        5. Return smoothed Casey C% value
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Casey C% value (0.0-100.0)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Handle warmup period
        if self.n_prices < 2:  # Need at least 2 candles for first percent change
            self.prev_close = curr_close
            self.output.append(50.0)  # Neutral value
            return [50.0]
        
        # Calculate percent change
        if self.prev_close is not None and self.prev_close > 1e-10:
            pct_change = (curr_close - self.prev_close) / self.prev_close
        else:
            # Handle edge case: prev_close is 0 or very small
            pct_change = 0.0
        
        # Update percent change buffer
        self.pct_change_buffer[self.pct_change_idx] = pct_change
        self.pct_change_idx = (self.pct_change_idx + 1) % self.lookback
        if self.n_pct_changes < self.lookback:
            self.n_pct_changes += 1
        
        # Update previous close for next iteration
        self.prev_close = curr_close
        
        # Need at least lookback percent changes for ranking
        if self.n_pct_changes < self.lookback:
            self.output.append(50.0)  # Neutral value during warmup
            return [50.0]
        
        # Calculate ranking: normalize current change to 0-100
        # Get the lookback window of percent changes
        if self.pct_change_idx == 0:
            # Buffer is full and contiguous
            recent_changes = self.pct_change_buffer
        else:
            # Buffer wraps around - need to reconstruct window
            recent_changes = np.concatenate([
                self.pct_change_buffer[self.pct_change_idx:],
                self.pct_change_buffer[:self.pct_change_idx]
            ])
        
        # Find min and max in the lookback window
        lowest_change = np.min(recent_changes)
        highest_change = np.max(recent_changes)
        
        # Current percent change (most recent)
        current_change = recent_changes[-1]
        
        # Calculate ranking (0-100)
        if highest_change - lowest_change < 1e-10:
            # No range - return neutral
            ranked = 50.0
        else:
            ranked = 100.0 * (current_change - lowest_change) / (highest_change - lowest_change)
            # Clamp to 0-100 range (safety check)
            ranked = np.clip(ranked, 0.0, 100.0)
        
        # Update ranked buffer for smoothing
        self.ranked_buffer[self.ranked_idx] = ranked
        self.ranked_idx = (self.ranked_idx + 1) % self.smoothing
        if self.n_ranked < self.smoothing:
            self.n_ranked += 1
        
        # Need at least smoothing ranked values for smoothing
        if self.n_ranked < self.smoothing:
            # Not enough for smoothing yet - return current ranked value
            casey_c = ranked
        else:
            # Smooth: calculate mean of ranked values
            casey_c = np.mean(self.ranked_buffer)
        
        self.output.append(casey_c)
        return [casey_c]
```

## Key Implementation Notes

1. **Percent Change Calculation**:
   ```python
   pct_change = (curr_close - prev_close) / prev_close
   ```
   Handle edge case where `prev_close` is 0 or very small (use 0.0 or skip).

2. **Circular Buffer for Percent Changes**:
   - Size: `lookback`
   - Stores percent changes for ranking calculation
   - Need to handle wrap-around when extracting the lookback window

3. **Ranking Calculation**:
   ```python
   if highest_change - lowest_change < 1e-10:
       ranked = 50.0  # Neutral if no range
   else:
       ranked = 100.0 * (current_change - lowest_change) / (highest_change - lowest_change)
   ```
   Normalizes current percent change to 0-100 based on min/max in lookback window.

4. **Smoothing**:
   - Use circular buffer of size `smoothing`
   - Calculate mean of ranked values in buffer
   - If not enough values yet, return current ranked value

5. **Warmup Handling**:
   - Need at least 2 candles for first percent change
   - Need `lookback` percent changes for ranking
   - Need `smoothing` ranked values for smoothing
   - Return `50.0` (neutral) during warmup

6. **Output**: Continuous value 0.0-100.0 (same range as RSI, no normalization needed)

## Edge Cases

1. **No Range in Percent Changes**:
   - If all percent changes in lookback window are the same
   - `HighestChange - LowestChange == 0`
   - Return `50.0` (neutral value)

2. **Division by Zero in Percent Change**:
   - If `prev_close` is 0 or very small
   - Use `0.0` for percent change or skip that candle

3. **Insufficient Data**:
   - During warmup, return `50.0` (neutral)
   - Before smoothing is ready, return current ranked value

## Performance Optimization

- **Circular Buffers**: Use NumPy arrays with modulo indexing instead of growing lists
- **Efficient Min/Max**: Use `np.min()` and `np.max()` on buffer slices
- **Avoid Repeated Calculations**: Cache intermediate values

## Testing Considerations

When testing the implementation:

1. **Verify Percent Change**: Check that percent changes are calculated correctly
2. **Verify Ranking**: 
   - When current change is minimum → should be close to 0.0
   - When current change is maximum → should be close to 100.0
   - When all changes are same → should be 50.0
3. **Verify Smoothing**: Check that smoothing reduces noise in the indicator
4. **Compare with RSI**: Both oscillate 0-100, but calculation differs
5. **Edge Cases**: Test with constant prices, zero prices, etc.

## Expected Column Name

With parameters `lookback=7, smoothing=3`:
```
caseyc_signal_D_lookback_7_smoothing_3
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Similar oscillator (0-100 range)
- **Archived Implementation**: `nodes/archive/casey_c.py` - Previous implementation with MA filter
- **Candle Model**: `utils/models.py` - `Candle` class definition

## Differences from Archived Implementation

The archived `casey_c.py` includes:
- 200-period MA trend filter
- Position tracking (long/flat)
- Multi-output (raw value + position)

This new specification focuses on:
- Pure Casey C% calculation (continuous feature)
- No trend filter (can be added separately if needed)
- Single output (Casey C% value 0-100)
- Simpler, more focused implementation
