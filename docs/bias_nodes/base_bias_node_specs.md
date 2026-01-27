# Bias Node Specification Guide

This document provides a comprehensive guide for creating new bias nodes in the Trading-Algo system. Use this specification to ensure consistency, performance, and proper integration with the feature extraction pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Base Class Structure](#base-class-structure)
3. [Required Implementation](#required-implementation)
4. [Standardization Requirements](#standardization-requirements)
5. [Rule-Based vs Continuous Features](#rule-based-vs-continuous-features)
6. [Template: RSI Example](#template-rsi-example)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

---

## Overview

A **Bias Node** is a stateful component that processes candle data and produces trading signals or features. All bias nodes inherit from the `BiasNode` abstract base class and implement the `_compute_candle()` method.

### Key Characteristics

- **Stateful**: Maintains internal state across multiple candles
- **Streaming**: Processes one candle at a time via `add_candle()`
- **Cached**: Automatic caching prevents duplicate computations
- **Standardized**: Uses consistent naming for feature columns
- **Singleton**: Can use `get_instance()` for shared instances

---

## Base Class Structure

### Inheritance

```python
from nodes import BiasNode
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from typing import List

class YourNode(BiasNode):
    # Implementation here
```

### Base Class Attributes (Inherited)

The `BiasNode` base class provides these attributes:

- `self.name`: Class name (automatically set)
- `self.ticker`: Ticker symbol (from `__init__`)
- `self.tf`: TimeFrame enum (from `__init__`)
- `self.bias`: Current bias state (Bias enum, defaults to NEUTRAL)
- `self.output`: List of historical outputs (appended automatically)
- `self.module_name`: Module name for column naming (set in `__init__`)
- `self.output_features`: List of feature names (set in `__init__`)
- `self.params`: Dictionary of parameters (set in `__init__`)
- `self._cache`: Internal caching dictionary (managed by base class)

### Base Class Methods (Inherited)

- `add_candle(candle: Candle) -> List`: Wrapper that handles caching and calls `_compute_candle()`
- `get_column_names() -> List[str]`: Returns standardized column names
- `ensure_standardized_columns() -> None`: Sets `self.columns` to standardized names

---

## Required Implementation

### 1. Constructor (`__init__`)

**Required Steps:**

1. Call `super().__init__(ticker, tf)`
2. Store any parameters as instance variables
3. Set standardization metadata:
   - `self.module_name`: Lowercase module name (e.g., `'rsi'`, `'ewmac'`)
   - `self.output_features`: List of feature names (e.g., `['signal']` or `['atr', 'atrPct']`)
   - `self.params`: Dictionary of all parameters (used for column naming)
4. Set `self.front_bad`: Number of candles needed before valid output
5. Initialize any computation state (buffers, accumulators, etc.)
6. Call `self.ensure_standardized_columns()` at the end

**Example:**

```python
def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14):
    super().__init__(ticker, tf)
    
    self.lookback = lookback
    # Standardized naming metadata
    self.module_name = 'rsi'
    self.output_features = ['signal']
    self.params = {'lookback': lookback}
    
    # Number of candles needed before we can compute valid output
    self.front_bad = lookback
    
    # Initialize computation state
    self.upsum = 1e-60
    self.dnsum = 1e-60
    # ... more state initialization ...
    
    # Define standardized columns
    self.ensure_standardized_columns()
```

### 2. Core Computation (`_compute_candle`)

**Signature:**

```python
def _compute_candle(self, candle: Candle) -> List:
    """
    Compute output for the given candle.
    
    Parameters:
    - candle: The candle to process
    
    Returns:
    - List containing output values (one per output_feature)
    """
```

**Required Steps:**

1. Extract needed data from `candle` (e.g., `candle.close`, `candle.high`, etc.)
2. Update internal state (buffers, accumulators, etc.)
3. Handle warmup period: return neutral/default values if `n_prices < front_bad`
4. Perform computation
5. Append result to `self.output` (for historical tracking)
6. Return `List` with one value per `output_feature`

**Example:**

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    
    # Update state
    self.n_prices += 1
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        self.output.append(50.0)  # Neutral value
        return [50.0]
    
    # Perform computation
    rsi = self._calculate_rsi(curr_close)
    
    # Append and return
    self.output.append(rsi)
    return [rsi]
```

---

## Standardization Requirements

### Module Name

- **Format**: Lowercase, no underscores (e.g., `'rsi'`, `'ewmac'`, `'donchianchannel'`)
- **Purpose**: Used in feature column naming
- **Example**: `self.module_name = 'rsi'`

### Output Features

- **Format**: List of strings, typically `['signal']` for single-output nodes
- **Purpose**: Defines what features this node produces
- **Multi-output example**: `self.output_features = ['atr', 'atrPct']`
- **Single-output example**: `self.output_features = ['signal']`

### Parameters Dictionary

- **Format**: Dictionary with parameter names as keys
- **Purpose**: Used in column naming to distinguish parameterized variants
- **Example**: `self.params = {'lookback': 14, 'smoothing': 2}`
- **Note**: All constructor parameters (except `ticker` and `tf`) should be included

### Column Naming

The system automatically generates column names using:

```
{module_name}_{feature}_{timeframe}_{param1}_{value1}_{param2}_{value2}...
```

**Example**: `rsi_signal_D_lookback_14`

**Generated by**: `self.ensure_standardized_columns()` which calls `get_column_names()`

---

## Rule-Based vs Continuous Features

Bias nodes can produce two types of outputs: **rule-based** (discrete signals) or **continuous** (values that require binning). Understanding this distinction is crucial for proper implementation.

### Rule-Based Bias Nodes

**Definition**: Output discrete position signals directly.

**Output Values**:
- `1`: Long position (bullish signal)
- `0`: Neutral/no position
- `-1`: Short position (bearish signal)

**Characteristics**:
- **No binning required**: Output is already a trading signal
- **Deterministic**: Clear rules determine the output value
- **Immediate use**: Can be used directly in trading logic
- **Example use cases**: 
  - Breakout strategies (Donchian Channel)
  - Crossover strategies (MA crossovers)
  - Threshold-based signals (price above/below level)

**Example Implementation:**

```python
def _compute_candle(self, candle: Candle) -> List:
    # Rule-based logic
    if candle.close > self.upper_band:
        signal = 1  # Long signal
    elif candle.close < self.lower_band:
        signal = -1  # Short signal
    else:
        signal = 0  # Neutral
    
    self.output.append(signal)
    return [signal]
```

**When to Use**:
- You have clear entry/exit rules
- The signal is binary or ternary (long/neutral/short)
- No need for nuanced strength measurement

### Continuous Features

**Definition**: Output continuous numerical values that represent indicator strength or magnitude.

**Output Values**:
- Any floating-point number (e.g., `0.0` to `100.0` for RSI, `-50.0` to `50.0` for momentum)
- Represents the **strength** or **magnitude** of a signal, not the position itself

**Characteristics**:
- **Requires binning**: Must be discretized into bins before use in trading
- **Gradient information**: Preserves information about signal strength
- **Flexible**: Allows downstream models to learn optimal thresholds
- **Example use cases**:
  - Technical indicators (RSI, MACD, momentum)
  - Volatility measures (ATR, standard deviation)
  - Statistical measures (z-scores, percentiles)

**Example Implementation:**

```python
def _compute_candle(self, candle: Candle) -> List:
    # Continuous value calculation
    rsi = self._calculate_rsi(candle.close)  # Returns 0.0-100.0
    
    self.output.append(rsi)
    return [rsi]  # Continuous value
```

**Binning Process**: Continuous features are later processed by binning models (e.g., `QuantileBinningModel`, `UniformBinningModel`) that:
1. Divide the continuous range into bins
2. Select optimal bins based on performance metrics
3. Convert bins to binary signals (1 if in selected bin, 0 otherwise)

**Normalization Requirements**:

Continuous features fall into two categories based on their value ranges:

1. **Fixed-Range Features** (No normalization needed):
   - Have the same range across all assets
   - Examples:
     - **RSI**: Always 0.0-100.0 regardless of asset price
     - **Percent-based indicators**: Always 0-100% or -100% to +100%
     - **Normalized oscillators**: Already scaled to fixed ranges
   - **Implementation**: Output raw values directly
   ```python
   # RSI example - no normalization needed
   rsi = self._calculate_rsi(candle.close)  # Always 0.0-100.0
   return [rsi]
   ```

2. **Dynamic-Range Features** (Normalization required):
   - Range depends on asset price or volatility
   - Examples:
     - **Moving average differences**: MA_diff = MA_fast - MA_slow
       - For ES (futures ~$4000): might range -50 to +50
       - For RTY (futures ~$2000): might range -25 to +25
       - Different assets have different scales!
     - **Price-based momentum**: Raw price changes vary by asset price
     - **Volatility measures**: ATR values differ significantly between assets
   - **Implementation**: Normalize by price or volatility
   ```python
   # MA difference example - normalization needed
   ma_diff = self.ma_fast - self.ma_slow  # Raw difference (price-dependent)
   ma_diff_pct = (ma_diff / candle.close) * 100  # Normalize by price
   return [ma_diff_pct]  # Now comparable across assets
   ```

**Normalization Methods**:

- **Price normalization**: Divide by current price (or average price)
  ```python
  normalized = (raw_value / candle.close) * 100  # Percentage
  ```
- **Volatility normalization**: Divide by ATR or standard deviation
  ```python
  normalized = raw_value / self.atr_value  # ATR-scaled
  ```
- **Z-score normalization**: (value - mean) / std (requires historical data)

**Key Principle**: If your feature's range varies significantly between assets (e.g., high-priced vs low-priced stocks), you **must** normalize it. Otherwise, binning models will struggle to find meaningful patterns across different assets.

**When to Use**:
- You want to preserve signal strength information
- The indicator has a meaningful continuous scale
- You want downstream models to learn optimal thresholds
- The feature benefits from statistical analysis

### Comparison Table

| Aspect | Rule-Based | Continuous |
|--------|-----------|------------|
| **Output Format** | `1`, `0`, `-1` | Any float (e.g., `0.0-100.0`) |
| **Binning Required** | No | Yes |
| **Information Content** | Discrete position | Signal strength + direction |
| **Downstream Processing** | Direct use | Binning → Signal |
| **Flexibility** | Fixed rules | Learnable thresholds |
| **Example Nodes** | `DonchianChannel`, `BuyHold` | `RSI`, `EWMAC`, `Momentum` |

### Decision Guide

**Choose Rule-Based if**:
- ✅ You have explicit trading rules (e.g., "buy when price breaks above 20-day high")
- ✅ The signal is inherently binary/ternary
- ✅ You want deterministic, interpretable signals
- ✅ No need to measure signal strength

**Choose Continuous if**:
- ✅ You want to preserve gradient information
- ✅ The indicator has meaningful magnitude (e.g., RSI strength, momentum size)
- ✅ You want models to learn optimal thresholds
- ✅ The feature benefits from statistical analysis

### Hybrid Approach

Some nodes can output both types:
- **Primary output**: Continuous value (e.g., RSI value)
- **Secondary output**: Rule-based signal (e.g., RSI > 70 = overbought)

However, typically you should choose one approach based on your primary use case.

---

## Template: RSI Example

Here's a complete example using the RSI node as a template:

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class RSI(BiasNode):
    """
    RSI (Relative Strength Index) Bias Node - CONTINUOUS FEATURE
    
    Computes the standard RSI indicator using exponential moving average
    of up and down price movements.
    
    RSI = 100 * (average_gain) / (average_gain + average_loss)
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    
    Parameters:
    - lookback: Period for RSI calculation (default: 14)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14):
        """
        Initialize RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: RSI period (default: 14)
        """
        # 1. Call super().__init__()
        super().__init__(ticker, tf)
        
        # 2. Store parameters
        self.lookback = lookback
        
        # 3. Set standardization metadata
        self.module_name = 'rsi'
        self.output_features = ['signal']
        self.params = {'lookback': lookback}
        
        # 4. Set warmup period
        self.front_bad = lookback
        
        # 5. Initialize computation state
        self.upsum = 1e-60  # Small value to avoid division by zero
        self.dnsum = 1e-60
        
        # Circular buffer for efficient storage
        self.buffer_size = lookback + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        
        # 6. Ensure standardized columns
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the RSI value
        """
        # Extract data
        curr_close = candle.close
        
        # Update state
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            self.prev_close = curr_close
            self.output.append(50.0)  # Neutral RSI value
            return [50.0]
        
        # Initialize on first valid computation
        if self.n_prices == self.front_bad:
            # Extract initialization data from buffer
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[:self.lookback]
            else:
                init_prices = np.concatenate([
                    self.close_buffer[self.buffer_idx:],
                    self.close_buffer[:self.buffer_idx]
                ])
            self.upsum, self.dnsum = self._compute_rsi_initial(init_prices)
        
        # Update RSI
        self.upsum, self.dnsum, rsi = self._update_rsi(
            self.prev_close,
            curr_close,
            self.upsum,
            self.dnsum,
            self.lookback
        )
        
        # Update state for next iteration
        self.prev_close = curr_close
        
        # Append and return
        self.output.append(rsi)
        return [rsi]
    
    def _compute_rsi_initial(self, prices: np.ndarray) -> tuple:
        """Initialize RSI from price array."""
        # Implementation here
        pass
    
    def _update_rsi(self, prev_close: float, curr_close: float, 
                    upsum: float, dnsum: float, lookback: int) -> tuple:
        """Update RSI incrementally."""
        # Implementation here
        pass
```

---

## Best Practices

### 1. Performance Optimization

- **Use circular buffers** instead of growing lists for historical data
- **Minimize state**: Only store what's needed for next computation
- **Avoid repeated calculations**: Cache intermediate values
- **Use NumPy arrays** for numerical operations

**Example (Circular Buffer):**

```python
# Instead of: self.prices.append(candle.close)  # O(n) append
# Use:
self.buffer[self.buffer_idx] = candle.close
self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
```

### 2. Warmup Period Handling

- Always return a **neutral/default value** during warmup
- Choose values that won't skew downstream analysis:
  - **Rule-based nodes**: Use `0.0` (neutral signal)
  - **Continuous oscillators (0-100)**: Use middle value (e.g., `50.0` for RSI)
  - **Continuous returns**: Use `0.0`
  - **Continuous ratios**: Use `1.0`
  - **Continuous momentum**: Use `0.0` (no momentum)

**Example:**

```python
if self.n_prices < self.front_bad:
    self.output.append(50.0)  # Neutral RSI
    return [50.0]
```

### 3. State Management

- Initialize all state variables in `__init__`
- Update state **before** computing output
- Keep state minimal and focused on computation needs

### 4. Error Handling

- Handle edge cases (e.g., division by zero, empty buffers)
- Use safe defaults (e.g., `1e-60` instead of `0.0` for denominators)
- Don't raise exceptions in `_compute_candle()` - return default values instead

### 5. Documentation

- Include docstring with formula/algorithm description
- Document all parameters in `__init__` docstring
- Explain any non-obvious state variables

---

## Common Patterns

### Pattern 1: Simple Moving Average (Fixed-Range)

```python
def __init__(self, ticker: Ticker, tf: TimeFrame, period: int = 20):
    super().__init__(ticker, tf)
    self.period = period
    self.module_name = 'sma'
    self.output_features = ['signal']
    self.params = {'period': period}
    self.front_bad = period
    
    # Circular buffer for prices
    self.buffer = np.zeros(period, dtype=np.float64)
    self.buffer_idx = 0
    self.n_prices = 0
    
    self.ensure_standardized_columns()

def _compute_candle(self, candle: Candle) -> List:
    self.buffer[self.buffer_idx] = candle.close
    self.buffer_idx = (self.buffer_idx + 1) % self.period
    self.n_prices += 1
    
    if self.n_prices < self.front_bad:
        self.output.append(0.0)
        return [0.0]
    
    sma = np.mean(self.buffer)
    self.output.append(sma)
    return [sma]  # Note: SMA values are price-dependent
```

**Note**: If using SMA as a continuous feature across multiple assets, consider normalizing by price:
```python
sma_pct = ((sma - candle.close) / candle.close) * 100  # Deviation from price as %
return [sma_pct]
```

### Pattern 2: Moving Average Difference (Dynamic-Range with Normalization)

```python
class MADiff(BiasNode):
    """
    Moving Average Difference - Continuous feature with normalization.
    
    Computes the difference between two moving averages and normalizes by price
    to make it comparable across assets with different price levels.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, fast_period: int = 10, slow_period: int = 20):
        super().__init__(ticker, tf)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.module_name = 'ma_diff'
        self.output_features = ['signal']
        self.params = {'fast_period': fast_period, 'slow_period': slow_period}
        self.front_bad = slow_period  # Need slow_period candles
        
        # Buffers for both MAs
        self.fast_buffer = np.zeros(fast_period, dtype=np.float64)
        self.slow_buffer = np.zeros(slow_period, dtype=np.float64)
        self.fast_idx = 0
        self.slow_idx = 0
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        # Update buffers
        self.fast_buffer[self.fast_idx] = candle.close
        self.slow_buffer[self.slow_idx] = candle.close
        self.fast_idx = (self.fast_idx + 1) % self.fast_period
        self.slow_idx = (self.slow_idx + 1) % self.slow_period
        self.n_prices += 1
        
        if self.n_prices < self.front_bad:
            self.output.append(0.0)
            return [0.0]
        
        # Calculate MAs
        ma_fast = np.mean(self.fast_buffer)
        ma_slow = np.mean(self.slow_buffer)
        
        # Raw difference (price-dependent, varies by asset)
        raw_diff = ma_fast - ma_slow
        
        # NORMALIZE by price to make comparable across assets
        # This converts to percentage deviation
        ma_diff_pct = (raw_diff / candle.close) * 100
        
        self.output.append(ma_diff_pct)
        return [ma_diff_pct]  # Normalized value (percentage)
```

**Key Point**: The raw difference `ma_fast - ma_slow` would be ~$50 for ES futures but ~$25 for RTY futures. Normalizing by price makes both comparable as percentages.

### Pattern 3: Multi-Output Node

```python
def __init__(self, ticker: Ticker, tf: TimeFrame):
    super().__init__(ticker, tf)
    self.module_name = 'atr'
    self.output_features = ['atr', 'atrPct']  # Multiple outputs
    self.params = {}
    self.front_bad = 14
    # ... state initialization ...
    self.ensure_standardized_columns()

def _compute_candle(self, candle: Candle) -> List:
    # ... computation ...
    atr = self._calculate_atr(candle)
    atr_pct = (atr / candle.close) * 100
    
    self.output.append((atr, atr_pct))  # Store tuple
    return [atr, atr_pct]  # Return list with both values
```

### Pattern 4: Exponential Moving Average (Stateful - Continuous)

```python
def __init__(self, ticker: Ticker, tf: TimeFrame, span: int = 20):
    super().__init__(ticker, tf)
    self.span = span
    self.module_name = 'ema'
    self.output_features = ['signal']
    self.params = {'span': span}
    self.front_bad = span
    
    self.alpha = 2.0 / (span + 1.0)  # EMA smoothing factor
    self.ema = None  # Will be initialized on first valid candle
    self.n_prices = 0
    
    self.ensure_standardized_columns()

def _compute_candle(self, candle: Candle) -> List:
    self.n_prices += 1
    
    if self.n_prices < self.front_bad:
        self.output.append(0.0)
        return [0.0]
    
    # Initialize EMA on first valid candle
    if self.ema is None:
        self.ema = candle.close
    else:
        # Update EMA: EMA = alpha * price + (1 - alpha) * EMA_prev
        self.ema = self.alpha * candle.close + (1 - self.alpha) * self.ema
    
    self.output.append(self.ema)
    return [self.ema]  # Continuous value (price level)
```

### Pattern 5: Rule-Based Breakout Strategy

```python
class BreakoutStrategy(BiasNode):
    """
    Rule-based bias node that outputs discrete signals.
    
    Outputs:
    - 1: Price breaks above upper band (long signal)
    - -1: Price breaks below lower band (short signal)
    - 0: Price within bands (neutral)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, period: int = 20):
        super().__init__(ticker, tf)
        self.period = period
        self.module_name = 'breakout'
        self.output_features = ['signal']
        self.params = {'period': period}
        self.front_bad = period
        
        # Circular buffer for high/low prices
        self.high_buffer = np.zeros(period, dtype=np.float64)
        self.low_buffer = np.zeros(period, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        # Update buffers
        self.high_buffer[self.buffer_idx] = candle.high
        self.low_buffer[self.buffer_idx] = candle.low
        self.buffer_idx = (self.buffer_idx + 1) % self.period
        self.n_prices += 1
        
        # Handle warmup period - return neutral signal
        if self.n_prices < self.front_bad:
            self.output.append(0)  # Neutral signal
            return [0]
        
        # Calculate bands
        upper_band = np.max(self.high_buffer)
        lower_band = np.min(self.low_buffer)
        
        # Rule-based logic: discrete signals
        if candle.close > upper_band:
            signal = 1  # Long signal
        elif candle.close < lower_band:
            signal = -1  # Short signal
        else:
            signal = 0  # Neutral
        
        self.output.append(signal)
        return [signal]  # Discrete: 1, 0, or -1
```

---

## Checklist for New Bias Nodes

Before submitting a new bias node, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] **Decided on output type**: Rule-based (`1`, `0`, `-1`) or Continuous (float values)
- [ ] Sets `self.module_name` (lowercase, no underscores)
- [ ] Sets `self.output_features` (list of feature names)
- [ ] Sets `self.params` (dictionary of all parameters)
- [ ] Sets `self.front_bad` (warmup period)
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Implements `_compute_candle(candle: Candle) -> List`
- [ ] Handles warmup period appropriately:
  - Rule-based: Returns `0` (neutral signal)
  - Continuous: Returns appropriate neutral value (e.g., `50.0` for 0-100 oscillators, `0.0` for returns)
- [ ] Appends result to `self.output`
- [ ] Returns list with correct number of values (one per `output_feature`)
- [ ] **For continuous features**: Determined if normalization is needed:
  - Fixed-range (e.g., RSI 0-100): No normalization needed
  - Dynamic-range (e.g., MA differences, price-based): Normalize by price or volatility
- [ ] Includes comprehensive docstrings (note if rule-based or continuous, and normalization approach)
- [ ] Uses efficient data structures (circular buffers, NumPy arrays)
- [ ] Handles edge cases gracefully (no exceptions in `_compute_candle`)

---

## Additional Resources

- **Base Class**: `nodes/__init__.py` - `BiasNode` abstract base class
- **Example Nodes**: 
  - `nodes/rsi.py` - RSI indicator (complete example)
  - `nodes/ewmac.py` - EWMAC momentum
  - `nodes/atr.py` - Average True Range
- **Column Naming**: `utils/helpers.py` - `build_feature_column_name()` function
- **Candle Model**: `utils/models.py` - `Candle` class definition

---

## Questions?

If you need help implementing a bias node:

1. Review the RSI example above
2. Check existing nodes in `nodes/` directory
3. Ensure all checklist items are completed
4. Test with sample candle data to verify output format
