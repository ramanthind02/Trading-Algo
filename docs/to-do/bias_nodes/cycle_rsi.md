# Cyclical RSI (CycRSI) Bias Node Specification

## Overview

The **Cyclical RSI (CycRSI)** is a digital signal processing (DSP) approach to the Relative Strength Index that uses cycle analysis to generate dynamic overbought and oversold levels. Unlike classic RSI which uses static thresholds (e.g., 70/30), the Cyclical RSI adapts to the current market rhythm by measuring "Cycle Momentum" rather than simple relative strength.

**Type**: Continuous feature (outputs normalized cycle signal, typically -1.0 to +1.0 or 0.0 to 100.0, requires binning)

**Normalization**: Not needed (output is already normalized to a fixed range)

**Based on**: Work by Lars von Thienen (Decoding the Hidden Market Rhythm)

---

## Technical Specifications

### Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `dom_cycle` | 20 | The Dominant Cycle length. Determines the primary lookback for cycle analysis. |
| `vibration` | 10 | A sensitivity filter. Lower values increase the frequency of signals; higher values smooth the indicator. |
| `leveling` | 10 | Controls the width/tightness of the dynamic bands. Increasing this value forces more frequent crosses (signals). |

### Output Range

The Cyclical RSI outputs a normalized cycle signal. The exact range depends on implementation:
- **Option 1**: `-1.0` to `+1.0` (centered oscillator)
- **Option 2**: `0.0` to `100.0` (similar to classic RSI)

For consistency with other oscillators in the system, **Option 2 (0.0-100.0)** is recommended.

---

## Algorithm Description

### Step 1: Pre-Filtering (Centered Oscillator)

The price data is passed through a bandpass filter or centering oscillator to:
- Remove the trend (D.C. component)
- Filter high-frequency noise
- Extract only the cyclical components

**Common Implementation Approaches**:
1. **Centered Moving Average**: `CycleSignal = Price - SMA(Price, DomCycle)`
2. **Bandpass Filter**: More sophisticated DSP filter tuned to `DomCycle` frequency
3. **Hilbert Transform**: Advanced cycle detection method

### Step 2: Cycle Oscillation Calculation

Calculate the oscillation within a normalized range:

```
CycleSignal = CenteredOscillator(Price, DomCycle, Vibration)
```

The `Vibration` parameter acts as a damping/smoothing factor on the cycle signal.

### Step 3: Dynamic Thresholds (Dynamic Bands)

Instead of fixed horizontal lines, the bands adapt to cycle amplitude:

```
UpperBand = Highest(CycleSignal, Leveling) * 0.9
LowerBand = Lowest(CycleSignal, Leveling) * 0.9
```

The `0.9` multiplier (or similar) adjusts the sensitivity. The `Leveling` period determines how quickly the bands adapt.

### Step 4: Normalization (Optional)

If using a centered oscillator (-1.0 to +1.0), normalize to 0-100 range:

```
CycRSI = 50.0 + (CycleSignal * 50.0)  # Maps [-1, +1] to [0, 100]
```

---

## Implementation Strategy

### Recommended Approach: Centered Moving Average

For simplicity and performance, use a centered moving average approach:

1. **Detrend Price**: `DetrendedPrice = Price - SMA(Price, DomCycle)`
2. **Smooth with Vibration**: Apply additional smoothing if needed
3. **Normalize to 0-100**: Scale the detrended signal to RSI-like range
4. **Track Dynamic Bands**: Maintain rolling highest/lowest over `Leveling` period

### State Management

The node needs to maintain:

1. **Price Buffer**: Circular buffer for `DomCycle` prices (for SMA calculation)
2. **Cycle Signal Buffer**: Circular buffer for `Leveling` cycle signals (for dynamic bands)
3. **Smoothing State**: If using exponential smoothing for `Vibration`

---

## Implementation Details

### Warmup Period

```
front_bad = max(DomCycle, Leveling) + Vibration
```

We need:
- `DomCycle` candles to calculate the centered oscillator
- `Leveling` candles to establish dynamic bands
- Additional `Vibration` candles for smoothing (if applicable)

### Circular Buffers

Use circular buffers for efficient storage:

```python
# Price buffer for SMA calculation
self.price_buffer = np.zeros(dom_cycle, dtype=np.float64)
self.price_buffer_idx = 0

# Cycle signal buffer for dynamic bands
self.cycle_buffer = np.zeros(leveling, dtype=np.float64)
self.cycle_buffer_idx = 0
```

### Core Computation Logic

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    self.n_prices += 1
    
    # Update price buffer
    self.price_buffer[self.price_buffer_idx] = curr_close
    self.price_buffer_idx = (self.price_buffer_idx + 1) % self.dom_cycle
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        self.output.append(50.0)  # Neutral value (middle of 0-100 range)
        return [50.0]
    
    # Step 1: Calculate centered oscillator (detrend)
    sma = np.mean(self.price_buffer)
    cycle_signal = curr_close - sma  # Detrended price
    
    # Step 2: Apply vibration smoothing (if needed)
    # For simplicity, we can use a simple EMA or skip if vibration is low
    if self.vibration > 1:
        # Smooth the cycle signal
        cycle_signal = self._smooth_cycle(cycle_signal)
    
    # Step 3: Update cycle buffer for dynamic bands
    self.cycle_buffer[self.cycle_buffer_idx] = cycle_signal
    self.cycle_buffer_idx = (self.cycle_buffer_idx + 1) % self.leveling
    
    # Step 4: Normalize to 0-100 range
    # First, normalize cycle_signal to [-1, +1] range using recent volatility
    # Then scale to [0, 100]
    cyc_rsi = self._normalize_cycle_signal(cycle_signal)
    
    self.output.append(cyc_rsi)
    return [cyc_rsi]
```

### Normalization Method

The cycle signal needs to be normalized. Two approaches:

**Approach 1: Fixed Range Normalization**
```python
# Assume cycle_signal ranges roughly from -max_dev to +max_dev
# Normalize to [-1, +1] then scale to [0, 100]
max_dev = np.std(self.cycle_buffer) * 2.0  # Use 2 std devs as range
if max_dev > 1e-10:
    normalized = np.clip(cycle_signal / max_dev, -1.0, 1.0)
    cyc_rsi = 50.0 + (normalized * 50.0)  # Map to [0, 100]
else:
    cyc_rsi = 50.0  # Neutral if no variation
```

**Approach 2: Percentile-Based Normalization**
```python
# Use rolling min/max over Leveling period
cycle_min = np.min(self.cycle_buffer)
cycle_max = np.max(self.cycle_buffer)
range_size = cycle_max - cycle_min

if range_size > 1e-10:
    normalized = (cycle_signal - cycle_min) / range_size  # [0, 1]
    cyc_rsi = normalized * 100.0  # [0, 100]
else:
    cyc_rsi = 50.0  # Neutral if no variation
```

**Recommendation**: Use Approach 2 (percentile-based) as it adapts to current cycle amplitude.

---

## Parameter Behavior Guide

### To Get MORE Signals:

- **Decrease `dom_cycle`**: Makes the indicator look for shorter, more frequent cycles
- **Decrease `vibration`**: Reduces damping, allowing the indicator to reach thresholds more easily
- **Increase `leveling`**: Tightens the dynamic bands toward the center line, causing more frequent breaches

### To Get FEWER (More Stable) Signals:

- **Increase `dom_cycle`**: Focuses on long-term market rhythms
- **Increase `vibration`**: Filters out minor price fluctuations
- **Decrease `leveling`**: Widens the bands, requiring a more extreme cyclical move to trigger a trade

---

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class CyclicalRSI(BiasNode):
    """
    Cyclical RSI (CycRSI) Bias Node - CONTINUOUS FEATURE
    
    A digital signal processing approach to RSI that uses cycle analysis
    to generate dynamic overbought/oversold levels that adapt to market rhythm.
    
    Based on work by Lars von Thienen (Decoding the Hidden Market Rhythm).
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    
    **Normalization**: Not needed (output is already normalized to fixed range)
    
    Parameters:
    - dom_cycle: Dominant cycle length (default: 20)
    - vibration: Sensitivity filter (default: 10)
    - leveling: Dynamic band width control (default: 10)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        dom_cycle: int = 20,
        vibration: int = 10,
        leveling: int = 10
    ):
        """
        Initialize Cyclical RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - dom_cycle: Dominant cycle length (default: 20)
        - vibration: Sensitivity filter (default: 10)
        - leveling: Dynamic band width control (default: 10)
        """
        super().__init__(ticker, tf)
        
        # Store parameters
        self.dom_cycle = dom_cycle
        self.vibration = vibration
        self.leveling = leveling
        
        # Standardization metadata
        self.module_name = 'cyclicalrsi'
        self.output_features = ['signal']
        self.params = {
            'domCycle': dom_cycle,
            'vibration': vibration,
            'leveling': leveling
        }
        
        # Warmup period: need max(dom_cycle, leveling) + vibration
        self.front_bad = max(dom_cycle, leveling) + vibration
        
        # Circular buffers
        self.price_buffer = np.zeros(dom_cycle, dtype=np.float64)
        self.price_buffer_idx = 0
        
        self.cycle_buffer = np.zeros(leveling, dtype=np.float64)
        self.cycle_buffer_idx = 0
        
        # State tracking
        self.n_prices = 0
        
        # Smoothing state (if using EMA for vibration)
        if vibration > 1:
            self.alpha = 2.0 / (vibration + 1.0)
            self.smoothed_cycle = None
        else:
            self.alpha = None
            self.smoothed_cycle = None
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Cyclical RSI for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the CycRSI value (0.0-100.0)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Update price buffer
        self.price_buffer[self.price_buffer_idx] = curr_close
        self.price_buffer_idx = (self.price_buffer_idx + 1) % self.dom_cycle
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            self.output.append(50.0)  # Neutral value
            return [50.0]
        
        # Step 1: Calculate centered oscillator (detrend)
        sma = np.mean(self.price_buffer)
        cycle_signal = curr_close - sma  # Detrended price
        
        # Step 2: Apply vibration smoothing (optional)
        if self.vibration > 1 and self.alpha is not None:
            if self.smoothed_cycle is None:
                self.smoothed_cycle = cycle_signal
            else:
                # EMA smoothing
                self.smoothed_cycle = (
                    self.alpha * cycle_signal + 
                    (1.0 - self.alpha) * self.smoothed_cycle
                )
            cycle_signal = self.smoothed_cycle
        
        # Step 3: Update cycle buffer for dynamic bands
        self.cycle_buffer[self.cycle_buffer_idx] = cycle_signal
        self.cycle_buffer_idx = (self.cycle_buffer_idx + 1) % self.leveling
        
        # Step 4: Normalize to 0-100 range using percentile method
        cycle_min = np.min(self.cycle_buffer)
        cycle_max = np.max(self.cycle_buffer)
        range_size = cycle_max - cycle_min
        
        if range_size > 1e-10:
            # Normalize to [0, 1] then scale to [0, 100]
            normalized = (cycle_signal - cycle_min) / range_size
            cyc_rsi = normalized * 100.0
            cyc_rsi = np.clip(cyc_rsi, 0.0, 100.0)  # Ensure bounds
        else:
            cyc_rsi = 50.0  # Neutral if no variation
        
        self.output.append(cyc_rsi)
        return [cyc_rsi]
```

---

## Edge Cases & Considerations

### 1. Insufficient Data

- During warmup, return `50.0` (neutral value)
- If `cycle_buffer` has no variation (all same values), return `50.0`

### 2. Zero Range

- If `cycle_max - cycle_min` is near zero, return neutral value `50.0`

### 3. Extreme Values

- Clip normalized values to `[0.0, 100.0]` to ensure valid range

### 4. Parameter Validation

- Ensure `dom_cycle >= 2`
- Ensure `vibration >= 1`
- Ensure `leveling >= 2`

---

## Performance Notes

1. **Circular Buffers**: Use NumPy arrays with modulo indexing for O(1) updates
2. **SMA Calculation**: Use `np.mean()` on circular buffer (O(n) but n is small)
3. **Min/Max Calculation**: Use `np.min()` and `np.max()` on cycle buffer (O(n) but n is small)
4. **Avoid Reallocation**: Pre-allocate all buffers in `__init__`

---

## Testing Considerations

### Unit Tests

1. **Warmup Period**: Verify neutral values (`50.0`) during warmup
2. **Normalization**: Verify output is always in `[0.0, 100.0]` range
3. **Parameter Effects**: Test that increasing `dom_cycle` smooths the output
4. **Edge Cases**: Test with constant prices (should return `50.0`)

### Integration Tests

1. **Feature Extraction**: Verify column naming matches expected format
2. **Binning Compatibility**: Ensure output can be binned by downstream models
3. **Performance**: Measure computation time per candle (should be < 1ms)

---

## Checklist

Before implementing, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'cyclicalrsi'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = max(dom_cycle, leveling) + vibration`
- [ ] Uses circular buffers for price and cycle signal storage
- [ ] Implements centered oscillator (detrending)
- [ ] Implements vibration smoothing (optional)
- [ ] Implements normalization to 0-100 range
- [ ] Returns neutral value (`50.0`) during warmup
- [ ] Handles edge cases (zero range, constant prices)
- [ ] Clips output to `[0.0, 100.0]` range
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Includes comprehensive docstring (notes continuous feature type)
- [ ] Uses efficient data structures (NumPy arrays, circular buffers)

---

## Strategic Advantage

The primary advantage of Cyclical RSI over Classic RSI is **Portfolio Diversification**. Because it uses cycle-based math rather than simple gains/losses math, its equity curve is often uncorrelated with standard RSI strategies. Traders can combine both to smooth out overall portfolio drawdowns.

---

## References

- **Base Spec Guide**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Template**: `nodes/rsi.py` (for reference on continuous feature implementation)
- **BiasNode Base Class**: `nodes/__init__.py`

---

## Questions?

If implementing this node:

1. Review the RSI example in `nodes/rsi.py` for continuous feature patterns
2. Consider using composition with existing SMA/EMA nodes if available
3. Test normalization method with various market conditions
4. Verify output range is always `[0.0, 100.0]` for consistency with other oscillators
