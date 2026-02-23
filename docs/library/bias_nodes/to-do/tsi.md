# Trend Strength Index (TSI) Bias Node Specification

## Overview

The **Trend Strength Index (TSI)**, specifically the correlation-based variant discussed by Ali Casey (StatOasis), is an oscillator that measures the statistical relationship between price and time using the **Pearson Correlation Coefficient**. Unlike the more common "True Strength Index" (which uses multiple EMAs), this version determines how "cleanly" an asset is trending by correlating price movements with linear time progression.

**Type**: Continuous feature (outputs -100.0 to +100.0, requires binning)

**Normalization**: Not needed (output is already normalized to fixed range -100 to +100)

**Market**: Works best on indices (S&P 500, SPY) due to "buy the dip" characteristics

**Timeframe**: Daily (or any timeframe)

---

## Technical Specifications

### Parameters

| Parameter | Default Value | Optimized Range | Description |
|-----------|---------------|-----------------|-------------|
| `lookback` | 5 | 3 to 5 | The window used to calculate the correlation between price and time (bar index) |
| `price_source` | Close | N/A | The price data used for the correlation (typically close price) |

### Output Range

The TSI outputs a normalized correlation value:
- **Range**: `-100.0` to `+100.0`
- **Interpretation**:
  - **+100.0**: Perfect positive correlation (price rose perfectly linearly)
  - **0.0**: No correlation (random price movement)
  - **-100.0**: Perfect negative correlation (price fell perfectly linearly)

**Note**: The output is already normalized from the raw correlation range (-1.0 to +1.0) by multiplying by 100.

---

## Mathematical Foundation

### Pearson Correlation Coefficient

The TSI calculates the Pearson correlation coefficient between:
1. **Price Series**: Closing prices over the lookback period
2. **Time Series**: Linear bar indices (0, 1, 2, ..., lookback-1)

**Formula**:
```
Correlation = Σ((Price[i] - Price_mean) × (Time[i] - Time_mean)) / 
              (√(Σ(Price[i] - Price_mean)²) × √(Σ(Time[i] - Time_mean)²))
```

**Simplified using NumPy**:
```python
correlation = np.corrcoef(prices, time_indices)[0, 1]
```

### Normalization

The raw correlation coefficient ranges from `-1.0` to `+1.0`. To create a standard oscillator scale:

```
TSI = Correlation × 100
```

This maps the correlation to `-100.0` to `+100.0`.

---

## Strategy Logic (Mean Reversion)

While the TSI node outputs a **continuous value**, trading strategies typically use thresholds:

### Long Entry (Buy Signal)

Enter when `TSI` crosses below a threshold (e.g., `-80` or `-90`). This signifies a "clean" washout where the asset has dropped consistently and is likely due for a bounce.

**Example**: `TSI < -90` indicates strong downward trend consistency.

### Long Exit (Sell Signal)

Exit when `TSI` crosses back above a threshold (e.g., `+80`). This indicates reversion to strength.

**Example**: `TSI > +80` indicates strong upward trend consistency.

**Note**: These thresholds are applied in downstream binning models, not in the bias node itself. The node outputs the raw TSI value for flexibility.

---

## Implementation Strategy

### State Management

The node needs to maintain:

1. **Price Buffer**: Circular buffer for `lookback` closing prices
2. **Time Index Array**: Static array `[0, 1, 2, ..., lookback-1]` (doesn't change)

### Correlation Calculation

For each candle:
1. Update price buffer with current close
2. Extract prices from buffer (in chronological order)
3. Create time indices array `[0, 1, 2, ..., lookback-1]`
4. Calculate Pearson correlation between prices and time indices
5. Scale to -100 to +100 range

---

## Implementation Details

### Warmup Period

```
front_bad = lookback
```

We need at least `lookback` candles to calculate the correlation.

### Circular Buffer

Use a circular buffer for efficient storage:

```python
# Buffer for prices
self.price_buffer = np.zeros(lookback, dtype=np.float64)
self.price_buffer_idx = 0

# Static time indices (doesn't change)
self.time_indices = np.arange(lookback, dtype=np.float64)
```

### Core Computation Logic

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    self.n_prices += 1
    
    # Update price buffer
    self.price_buffer[self.price_buffer_idx] = curr_close
    self.price_buffer_idx = (self.price_buffer_idx + 1) % self.lookback
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        tsi = 0.0  # Neutral value during warmup
        self.output.append(tsi)
        return [tsi]
    
    # Extract prices in chronological order from circular buffer
    if self.price_buffer_idx == 0:
        # Buffer is full and in order
        prices = self.price_buffer
    else:
        # Need to reorder: [idx:end] + [0:idx]
        prices = np.concatenate([
            self.price_buffer[self.price_buffer_idx:],
            self.price_buffer[:self.price_buffer_idx]
        ])
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(prices, self.time_indices)[0, 1]
    
    # Handle NaN (can occur if prices are constant)
    if np.isnan(correlation):
        tsi = 0.0  # Neutral if no variation
    else:
        # Normalize to -100 to +100
        tsi = correlation * 100.0
        # Clip to ensure bounds (shouldn't be necessary, but safe)
        tsi = np.clip(tsi, -100.0, 100.0)
    
    self.output.append(tsi)
    return [tsi]
```

### Edge Cases

1. **Constant Prices**: If all prices in the buffer are the same, correlation is undefined (NaN). Return `0.0` (neutral).
2. **Insufficient Data**: During warmup, return `0.0` (neutral).
3. **Perfect Correlation**: If correlation is exactly `1.0` or `-1.0`, output `100.0` or `-100.0` respectively.

---

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode


class TSI(BiasNode):
    """
    Trend Strength Index (TSI) Bias Node - CONTINUOUS FEATURE
    
    A correlation-based oscillator that measures the statistical relationship
    between price and time using the Pearson Correlation Coefficient.
    
    Unlike the "True Strength Index" (which uses EMAs), this version determines
    how "cleanly" an asset is trending by correlating price movements with
    linear time progression.
    
    **Type**: Continuous feature (outputs -100.0 to +100.0, requires binning)
    
    **Normalization**: Not needed (output is already normalized to fixed range)
    
    **Mathematical Foundation**:
    - Correlation = Pearson correlation between Price and Time (bar index)
    - TSI = Correlation × 100 (normalizes -1 to +1 range to -100 to +100)
    
    **Interpretation**:
    - +100.0: Perfect positive correlation (price rose perfectly linearly)
    - 0.0: No correlation (random price movement)
    - -100.0: Perfect negative correlation (price fell perfectly linearly)
    
    Parameters:
    - lookback: Period for correlation calculation (default: 5)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 5
    ):
        """
        Initialize TSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Period for correlation calculation (default: 5)
        """
        super().__init__(ticker, tf)
        
        # Store parameters
        self.lookback = lookback
        
        # Standardization metadata
        self.module_name = 'tsi'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # Warmup period: need lookback candles
        self.front_bad = lookback
        
        # Circular buffer for prices
        self.price_buffer = np.zeros(lookback, dtype=np.float64)
        self.price_buffer_idx = 0
        
        # Static time indices array (0, 1, 2, ..., lookback-1)
        self.time_indices = np.arange(lookback, dtype=np.float64)
        
        # State tracking
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute TSI for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the TSI value (-100.0 to +100.0)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Update price buffer
        self.price_buffer[self.price_buffer_idx] = curr_close
        self.price_buffer_idx = (self.price_buffer_idx + 1) % self.lookback
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            tsi = 0.0  # Neutral value during warmup
            self.output.append(tsi)
            return [tsi]
        
        # Extract prices in chronological order from circular buffer
        if self.price_buffer_idx == 0:
            # Buffer is full and in order (wrapped around completely)
            prices = self.price_buffer.copy()
        else:
            # Need to reorder: [idx:end] + [0:idx] to get chronological order
            prices = np.concatenate([
                self.price_buffer[self.price_buffer_idx:],
                self.price_buffer[:self.price_buffer_idx]
            ])
        
        # Calculate Pearson correlation between prices and time indices
        correlation_matrix = np.corrcoef(prices, self.time_indices)
        correlation = correlation_matrix[0, 1]
        
        # Handle NaN (can occur if prices are constant - zero variance)
        if np.isnan(correlation):
            tsi = 0.0  # Neutral if no variation
        else:
            # Normalize to -100 to +100 range
            tsi = correlation * 100.0
            # Clip to ensure bounds (shouldn't be necessary, but safe)
            tsi = np.clip(tsi, -100.0, 100.0)
        
        # Update bias based on TSI value
        if tsi > 0:
            self.bias = Bias.BULLISH
        elif tsi < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        self.output.append(tsi)
        return [tsi]
```

---

## Key Strategic Insights

### 1. Signal Frequency

TSI (Lookback 3-5) generates roughly **60% more trades** than RSI2 while maintaining a similar win rate. This is because correlation identifies trend consistency more frequently than magnitude-based indicators.

### 2. Drawdown Management

TSI generally offers a higher **Return-to-Drawdown Ratio** because the correlation math identifies the *consistency* of a sell-off better than the *magnitude* of a sell-off (as RSI does).

### 3. Stability

3D optimization models show that TSI has a wider "plateau" of profitable parameters (specifically at Lookback 5), making it more robust against curve-fitting than RSI.

### 4. Indices vs. Individual Stocks

Like most mean-reversion systems, TSI thrives on indices (which have "buy the dip" characteristics) but can be dangerous on individual stocks that can "trend to zero" without a bounce.

---

## Implementation Considerations

### Pearson Correlation Calculation

Use NumPy's `corrcoef()` function for efficient calculation:

```python
correlation_matrix = np.corrcoef(prices, time_indices)
correlation = correlation_matrix[0, 1]  # Extract correlation coefficient
```

### Circular Buffer Ordering

When extracting prices from a circular buffer, ensure chronological order:

```python
if self.price_buffer_idx == 0:
    # Buffer is in order (wrapped around)
    prices = self.price_buffer.copy()
else:
    # Reorder: [idx:end] + [0:idx]
    prices = np.concatenate([
        self.price_buffer[self.price_buffer_idx:],
        self.price_buffer[:self.price_buffer_idx]
    ])
```

### Time Indices

The time indices array is static and doesn't need updating:

```python
self.time_indices = np.arange(lookback, dtype=np.float64)  # [0, 1, 2, ..., lookback-1]
```

---

## Edge Cases & Considerations

### 1. Constant Prices (Zero Variance)

If all prices in the buffer are identical, the correlation calculation will return `NaN` (division by zero in variance calculation). Handle this by returning `0.0` (neutral value).

### 2. Insufficient Data

During warmup (`n_prices < front_bad`), return `0.0` (neutral value).

### 3. Perfect Correlation

If correlation is exactly `1.0` or `-1.0`, the output will be `100.0` or `-100.0` respectively. This is valid and indicates perfect linear trend.

### 4. Single Price Point

If `lookback = 1`, correlation is undefined. Ensure `lookback >= 2` in parameter validation.

### 5. Floating-Point Precision

Correlation values should naturally be in `[-1.0, +1.0]` range, but clip the final output to `[-100.0, +100.0]` for safety.

---

## Performance Notes

1. **Circular Buffers**: Use NumPy arrays with modulo indexing for O(1) updates
2. **Correlation Calculation**: `np.corrcoef()` is O(n) where n is the lookback period (typically 3-5, very fast)
3. **Buffer Reordering**: Only needed when `buffer_idx != 0`, and uses `np.concatenate()` which is O(n)
4. **Avoid Reallocation**: Pre-allocate all buffers in `__init__`
5. **Static Time Array**: Time indices array is created once and reused

---

## Testing Considerations

### Unit Tests

1. **Warmup Period**: Verify neutral values (`0.0`) during warmup
2. **Perfect Positive Correlation**: Test with prices that increase linearly (should output `100.0`)
3. **Perfect Negative Correlation**: Test with prices that decrease linearly (should output `-100.0`)
4. **No Correlation**: Test with random prices (should output values near `0.0`)
5. **Constant Prices**: Test with all same prices (should output `0.0`, not NaN)
6. **Edge Cases**: Test with `lookback = 2`, `lookback = 3`, etc.

### Integration Tests

1. **Feature Extraction**: Verify column naming matches expected format
2. **Output Range**: Verify output is always in `[-100.0, +100.0]` range
3. **Performance**: Measure computation time per candle (should be < 1ms)
4. **Binning Compatibility**: Ensure output can be binned by downstream models

### Backtest Validation

1. **Signal Frequency**: Verify TSI generates more signals than RSI2
2. **Win Rate**: Verify win rate is similar to RSI2 on indices
3. **Parameter Sensitivity**: Test different lookback values (3, 4, 5)
4. **Market Conditions**: Verify strategy performs well on mean-reverting markets (indices)

---

## Checklist

Before implementing, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'tsi'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = lookback`
- [ ] Uses circular buffer for price storage
- [ ] Creates static time indices array
- [ ] Implements Pearson correlation calculation using `np.corrcoef()`
- [ ] Handles buffer reordering for chronological price extraction
- [ ] Normalizes correlation to -100 to +100 range
- [ ] Returns neutral value (`0.0`) during warmup
- [ ] Handles edge cases (constant prices → NaN → 0.0)
- [ ] Clips output to `[-100.0, +100.0]` range
- [ ] Updates `self.bias` based on TSI value
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Includes comprehensive docstring (notes continuous feature type and normalization)
- [ ] Uses efficient data structures (NumPy arrays, circular buffers)

---

## References

- **Base Spec Guide**: `docs/bias_nodes/base_bias_node_specs.md`
- **Continuous Feature Examples**: 
  - `nodes/rsi.py` (RSI indicator - similar oscillator)
  - `nodes/ewmac.py` (momentum indicator)
- **Correlation Examples**: 
  - `nodes/archive/return_autocorr.py` (autocorrelation calculation)
- **BiasNode Base Class**: `nodes/__init__.py`
- **Strategy Source**: Ali Casey (StatOasis) - Trend Strength Index

---

## Notes

- **Platform Support**: Most platforms (TradingView, MetaTrader) require custom code, as the "True Strength Index" found in standard menus is not the correct version (that one uses EMAs, not correlation).
- **Indices Only**: Like most mean-reversion systems, this thrives on indices but can be dangerous on individual stocks that can "trend to zero" without a bounce.
