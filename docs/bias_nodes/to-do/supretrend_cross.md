# SuperTrend Crossover Strategy Bias Node Specification

## Overview

The **SuperTrend Crossover Strategy** is a rule-based trend-following system that uses two SuperTrend indicators (Fast and Slow) to identify trend alignment and generate trading signals. The SuperTrend indicator is based on the Average True Range (ATR) and acts as an adaptive trailing stop.

**Type**: Rule-based feature (outputs discrete signals: `1` for long, `0` for neutral, `-1` for short)

**Market**: Works best on trending markets, especially US Indices (Nasdaq-100, S&P 500)

**Timeframe**: Daily or intraday (60-minute charts increase trade frequency significantly)

**Strategy Logic**: Enter when short-term momentum (Fast ST) aligns with long-term trend (Slow ST)

---

## Technical Specifications

### Parameters

| Parameter | Default Value | Optimized Range | Description |
|-----------|---------------|-----------------|-------------|
| `fast_atr_period` | 10 | 5 to 15 | ATR period for Fast SuperTrend |
| `fast_multiplier` | 2.0 | 1.5 to 3.0 | Multiplier for Fast SuperTrend bands |
| `slow_atr_period` | 30 | 20 to 40 | ATR period for Slow SuperTrend |
| `slow_multiplier` | 4.0 | 3.0 to 5.0 | Multiplier for Slow SuperTrend bands |
| `use_sma_filter` | False | N/A | Optional: Only trade when price > 200-day SMA |
| `sma_period` | 200 | N/A | Period for SMA filter (if enabled) |

### Output Values

- `1`: Long position (entry signal triggered)
- `0`: Neutral/no position (cash)
- `-1`: Short position (if strategy is extended to short trades)

**Note**: The classic Dual SuperTrend strategy is **long-only**. It outputs `1` when in a long position and `0` when in cash.

---

## SuperTrend Indicator Calculation

### Step 1: Calculate ATR and Median Price

```
Current_ATR = ATR(ATR_Period)
Median_Price = (High + Low) / 2
```

### Step 2: Calculate Basic Bands

```
Basic_Upper_Band = Median_Price + (Multiplier × Current_ATR)
Basic_Lower_Band = Median_Price - (Multiplier × Current_ATR)
```

### Step 3: Final Band Logic (Trailing Effect)

The key feature of SuperTrend is that bands can only move in one direction:

**Lower Band (for Uptrend)**:
```
IF (Basic_Lower_Band > Lower_Band[prev]) OR (Close[prev] < Lower_Band[prev])
    Lower_Band = Basic_Lower_Band
ELSE
    Lower_Band = Lower_Band[prev]  // Stay flat
```

**Upper Band (for Downtrend)**:
```
IF (Basic_Upper_Band < Upper_Band[prev]) OR (Close[prev] > Upper_Band[prev])
    Upper_Band = Basic_Upper_Band
ELSE
    Upper_Band = Upper_Band[prev]  // Stay flat
```

### Step 4: Trend Direction Assignment

```
IF (Close > Upper_Band[prev])
    Trend = Uptrend (Plot Lower_Band, Green)
ELSE IF (Close < Lower_Band[prev])
    Trend = Downtrend (Plot Upper_Band, Red)
ELSE
    Trend = Trend[prev]  // Maintain previous trend
```

---

## Strategy Logic

### Long Entry (Buy Signal)

Enter when **both** conditions are met:

1. **Fast SuperTrend** switches from Downtrend (Red) to Uptrend (Green)
2. **Slow SuperTrend** is already in Uptrend (Green)
3. **(Optional)** Price is above 200-day SMA (if `use_sma_filter = True`)

**Pseudo-code**:
```
IF (FastTrend == Uptrend) AND (SlowTrend == Uptrend) AND
   (NOT use_sma_filter OR Close > SMA(200))
   THEN Buy (output = 1)
```

### Long Exit (Sell Signal)

Exit when:

1. **Fast SuperTrend** switches from Uptrend (Green) to Downtrend (Red)

**Pseudo-code**:
```
IF (FastTrend == Downtrend)
   THEN Sell (output = 0)
```

---

## Implementation Strategy

### State Management

The node needs to maintain:

1. **Position State**: Track whether currently in a long position (`self.position = 1`) or neutral (`self.position = 0`)
2. **Fast SuperTrend State**:
   - ATR calculation state (circular buffer for true ranges)
   - Previous lower band value
   - Previous upper band value
   - Current trend direction (1 = uptrend, -1 = downtrend)
3. **Slow SuperTrend State**: Same as Fast SuperTrend
4. **SMA State** (if filter enabled): Circular buffer for 200-day SMA

### Position Logic Flow

```
Current State: position = 0 (neutral/cash)
├─ Calculate Fast SuperTrend
├─ Calculate Slow SuperTrend
├─ Check Entry Conditions:
│  ├─ FastTrend == Uptrend? → Yes
│  ├─ SlowTrend == Uptrend? → Yes
│  └─ (Optional) Close > SMA(200)? → Yes
│     └─ ENTER LONG: position = 1, output = 1
│
Current State: position = 1 (long)
├─ Calculate Fast SuperTrend
├─ Check Exit Condition:
│  └─ FastTrend == Downtrend?
│     └─ YES → EXIT: position = 0, output = 0
│     └─ NO → Stay in position: output = 1
```

---

## Implementation Details

### Warmup Period

```
front_bad = max(fast_atr_period, slow_atr_period, sma_period if enabled)
```

We need at least the maximum of all lookback periods.

### Circular Buffers

Use circular buffers for efficient storage:

```python
# Fast SuperTrend ATR
self.fast_true_ranges = np.zeros(fast_atr_period, dtype=np.float64)
self.fast_tr_idx = 0

# Slow SuperTrend ATR
self.slow_true_ranges = np.zeros(slow_atr_period, dtype=np.float64)
self.slow_tr_idx = 0

# SMA (if filter enabled)
if use_sma_filter:
    self.sma_buffer = np.zeros(sma_period, dtype=np.float64)
    self.sma_idx = 0
```

### SuperTrend Calculation Helper

```python
def _calculate_supertrend(
    self,
    candle: Candle,
    atr_period: int,
    multiplier: float,
    true_ranges: np.ndarray,
    tr_idx: int,
    prev_lower_band: float,
    prev_upper_band: float,
    prev_trend: int
) -> tuple:
    """
    Calculate SuperTrend for a single indicator.
    
    Returns:
    - (lower_band, upper_band, trend, new_tr_idx)
    - trend: 1 = uptrend, -1 = downtrend
    """
    # Calculate True Range
    if self.prev_close is not None:
        hl = candle.high - candle.low
        hc = abs(candle.high - self.prev_close)
        lc = abs(candle.low - self.prev_close)
        true_range = max(hl, hc, lc)
    else:
        true_range = candle.high - candle.low
    
    # Update ATR buffer
    true_ranges[tr_idx] = true_range
    tr_idx = (tr_idx + 1) % atr_period
    
    # Calculate ATR (use available data during warmup)
    n_filled = min(self.n_prices, atr_period)
    if n_filled < atr_period:
        atr = np.mean(true_ranges[:n_filled]) if n_filled > 0 else 0.0
    else:
        atr = np.mean(true_ranges)
    
    # Calculate median price
    median_price = (candle.high + candle.low) / 2.0
    
    # Calculate basic bands
    basic_upper_band = median_price + (multiplier * atr)
    basic_lower_band = median_price - (multiplier * atr)
    
    # Initialize bands on first calculation
    if prev_lower_band is None:
        lower_band = basic_lower_band
        upper_band = basic_upper_band
        trend = 1 if candle.close > upper_band else -1
        return lower_band, upper_band, trend, tr_idx
    
    # Trailing band logic
    # Lower Band (for uptrend): can only move up or stay flat
    if (basic_lower_band > prev_lower_band) or (self.prev_close < prev_lower_band):
        lower_band = basic_lower_band
    else:
        lower_band = prev_lower_band  # Stay flat
    
    # Upper Band (for downtrend): can only move down or stay flat
    if (basic_upper_band < prev_upper_band) or (self.prev_close > prev_upper_band):
        upper_band = basic_upper_band
    else:
        upper_band = prev_upper_band  # Stay flat
    
    # Determine trend direction
    if candle.close > prev_upper_band:
        trend = 1  # Uptrend
    elif candle.close < prev_lower_band:
        trend = -1  # Downtrend
    else:
        trend = prev_trend  # Maintain previous trend
    
    return lower_band, upper_band, trend, tr_idx
```

### Core Computation Logic

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    self.n_prices += 1
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        signal = 0  # Neutral during warmup
        self.prev_close = curr_close
        self.output.append(signal)
        return [signal]
    
    # Calculate Fast SuperTrend
    (self.fast_lower_band, self.fast_upper_band, self.fast_trend, self.fast_tr_idx) = \
        self._calculate_supertrend(
            candle,
            self.fast_atr_period,
            self.fast_multiplier,
            self.fast_true_ranges,
            self.fast_tr_idx,
            self.fast_lower_band,
            self.fast_upper_band,
            self.fast_trend
        )
    
    # Calculate Slow SuperTrend
    (self.slow_lower_band, self.slow_upper_band, self.slow_trend, self.slow_tr_idx) = \
        self._calculate_supertrend(
            candle,
            self.slow_atr_period,
            self.slow_multiplier,
            self.slow_true_ranges,
            self.slow_tr_idx,
            self.slow_lower_band,
            self.slow_upper_band,
            self.slow_trend
        )
    
    # Calculate SMA (if filter enabled)
    if self.use_sma_filter:
        self.sma_buffer[self.sma_idx] = curr_close
        self.sma_idx = (self.sma_idx + 1) % self.sma_period
        n_sma_filled = min(self.n_prices, self.sma_period)
        if n_sma_filled >= self.sma_period:
            sma_200 = np.mean(self.sma_buffer)
        else:
            sma_200 = np.mean(self.sma_buffer[:n_sma_filled]) if n_sma_filled > 0 else curr_close
    else:
        sma_200 = None
    
    # Check if currently in a position
    if self.position == 0:  # Neutral/Cash
        # Check entry conditions
        fast_uptrend = (self.fast_trend == 1)
        slow_uptrend = (self.slow_trend == 1)
        sma_filter_ok = (not self.use_sma_filter) or (curr_close > sma_200)
        
        # Check if Fast ST just switched to uptrend (crossover)
        fast_switched_up = (self.fast_trend == 1) and (self.prev_fast_trend == -1)
        
        if fast_switched_up and slow_uptrend and sma_filter_ok:
            # ENTER LONG
            self.position = 1
            self.bias = Bias.BULLISH
            signal = 1
        else:
            # Stay neutral
            signal = 0
            self.bias = Bias.NEUTRAL
    
    else:  # Currently long (position == 1)
        # Check exit condition: Fast ST switches to downtrend
        fast_switched_down = (self.fast_trend == -1) and (self.prev_fast_trend == 1)
        
        if fast_switched_down:
            # EXIT
            self.position = 0
            self.bias = Bias.NEUTRAL
            signal = 0
        else:
            # Stay in position
            signal = 1
            self.bias = Bias.BULLISH
    
    # Store previous trend values for crossover detection
    self.prev_fast_trend = self.fast_trend
    self.prev_slow_trend = self.slow_trend
    self.prev_close = curr_close
    
    self.output.append(signal)
    return [signal]
```

---

## Example Implementation Structure

```python
from typing import List, Optional, Tuple
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode


class SuperTrendCross(BiasNode):
    """
    SuperTrend Crossover Strategy Bias Node - RULE-BASED FEATURE
    
    A trend-following strategy using two SuperTrend indicators (Fast and Slow)
    to identify trend alignment and generate trading signals.
    
    **Type**: Rule-based feature (outputs 1 for long, 0 for neutral)
    
    **Strategy Logic**:
    - Entry: Fast ST switches to uptrend AND Slow ST is already in uptrend
    - Exit: Fast ST switches to downtrend
    - Optional: 200-day SMA filter
    
    **SuperTrend Calculation**:
    - Based on ATR and median price (H+L)/2
    - Bands can only move in one direction (trailing stop behavior)
    - Trend determined by price position relative to bands
    
    Parameters:
    - fast_atr_period: ATR period for Fast SuperTrend (default: 10)
    - fast_multiplier: Multiplier for Fast SuperTrend (default: 2.0)
    - slow_atr_period: ATR period for Slow SuperTrend (default: 30)
    - slow_multiplier: Multiplier for Slow SuperTrend (default: 4.0)
    - use_sma_filter: Enable 200-day SMA filter (default: False)
    - sma_period: Period for SMA filter (default: 200)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        fast_atr_period: int = 10,
        fast_multiplier: float = 2.0,
        slow_atr_period: int = 30,
        slow_multiplier: float = 4.0,
        use_sma_filter: bool = False,
        sma_period: int = 200
    ):
        """
        Initialize SuperTrend Crossover Strategy node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - fast_atr_period: ATR period for Fast SuperTrend (default: 10)
        - fast_multiplier: Multiplier for Fast SuperTrend (default: 2.0)
        - slow_atr_period: ATR period for Slow SuperTrend (default: 30)
        - slow_multiplier: Multiplier for Slow SuperTrend (default: 4.0)
        - use_sma_filter: Enable 200-day SMA filter (default: False)
        - sma_period: Period for SMA filter (default: 200)
        """
        super().__init__(ticker, tf)
        
        # Store parameters
        self.fast_atr_period = fast_atr_period
        self.fast_multiplier = fast_multiplier
        self.slow_atr_period = slow_atr_period
        self.slow_multiplier = slow_multiplier
        self.use_sma_filter = use_sma_filter
        self.sma_period = sma_period
        
        # Standardization metadata
        self.module_name = 'supertrendcross'
        self.output_features = ['signal']
        self.params = {
            'fastAtrPeriod': fast_atr_period,
            'fastMultiplier': fast_multiplier,
            'slowAtrPeriod': slow_atr_period,
            'slowMultiplier': slow_multiplier,
            'useSmaFilter': use_sma_filter,
            'smaPeriod': sma_period if use_sma_filter else None
        }
        
        # Warmup period: need max of all lookback periods
        self.front_bad = max(fast_atr_period, slow_atr_period, sma_period if use_sma_filter else 0)
        
        # Circular buffers for ATR calculation
        self.fast_true_ranges = np.zeros(fast_atr_period, dtype=np.float64)
        self.fast_tr_idx = 0
        
        self.slow_true_ranges = np.zeros(slow_atr_period, dtype=np.float64)
        self.slow_tr_idx = 0
        
        # SMA buffer (if filter enabled)
        if use_sma_filter:
            self.sma_buffer = np.zeros(sma_period, dtype=np.float64)
            self.sma_idx = 0
        else:
            self.sma_buffer = None
            self.sma_idx = None
        
        # Fast SuperTrend state
        self.fast_lower_band: Optional[float] = None
        self.fast_upper_band: Optional[float] = None
        self.fast_trend: int = 0  # 1 = uptrend, -1 = downtrend, 0 = uninitialized
        self.prev_fast_trend: int = 0
        
        # Slow SuperTrend state
        self.slow_lower_band: Optional[float] = None
        self.slow_upper_band: Optional[float] = None
        self.slow_trend: int = 0
        self.prev_slow_trend: int = 0
        
        # Position state (0 = neutral, 1 = long)
        self.position = 0
        
        # Previous close (for ATR and trend detection)
        self.prev_close: Optional[float] = None
        
        # State tracking
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _calculate_supertrend(
        self,
        candle: Candle,
        atr_period: int,
        multiplier: float,
        true_ranges: np.ndarray,
        tr_idx: int,
        prev_lower_band: Optional[float],
        prev_upper_band: Optional[float],
        prev_trend: int
    ) -> Tuple[float, float, int, int]:
        """
        Calculate SuperTrend for a single indicator.
        
        Parameters:
        - candle: Current candle
        - atr_period: ATR period
        - multiplier: Band multiplier
        - true_ranges: Circular buffer for true ranges
        - tr_idx: Current index in true ranges buffer
        - prev_lower_band: Previous lower band value
        - prev_upper_band: Previous upper band value
        - prev_trend: Previous trend (1 = uptrend, -1 = downtrend)
        
        Returns:
        - (lower_band, upper_band, trend, new_tr_idx)
        - trend: 1 = uptrend, -1 = downtrend
        """
        # Calculate True Range
        if self.prev_close is not None:
            hl = candle.high - candle.low
            hc = abs(candle.high - self.prev_close)
            lc = abs(candle.low - self.prev_close)
            true_range = max(hl, hc, lc)
        else:
            true_range = candle.high - candle.low
        
        # Update ATR buffer
        true_ranges[tr_idx] = true_range
        new_tr_idx = (tr_idx + 1) % atr_period
        
        # Calculate ATR (use available data during warmup)
        n_filled = min(self.n_prices, atr_period)
        if n_filled < atr_period:
            atr = np.mean(true_ranges[:n_filled]) if n_filled > 0 else 0.0
        else:
            atr = np.mean(true_ranges)
        
        # Calculate median price
        median_price = (candle.high + candle.low) / 2.0
        
        # Calculate basic bands
        basic_upper_band = median_price + (multiplier * atr)
        basic_lower_band = median_price - (multiplier * atr)
        
        # Initialize bands on first calculation
        if prev_lower_band is None:
            lower_band = basic_lower_band
            upper_band = basic_upper_band
            trend = 1 if candle.close > upper_band else -1
            return lower_band, upper_band, trend, new_tr_idx
        
        # Trailing band logic
        # Lower Band (for uptrend): can only move up or stay flat
        if (basic_lower_band > prev_lower_band) or (self.prev_close is not None and self.prev_close < prev_lower_band):
            lower_band = basic_lower_band
        else:
            lower_band = prev_lower_band  # Stay flat
        
        # Upper Band (for downtrend): can only move down or stay flat
        if (basic_upper_band < prev_upper_band) or (self.prev_close is not None and self.prev_close > prev_upper_band):
            upper_band = basic_upper_band
        else:
            upper_band = prev_upper_band  # Stay flat
        
        # Determine trend direction
        if candle.close > prev_upper_band:
            trend = 1  # Uptrend
        elif candle.close < prev_lower_band:
            trend = -1  # Downtrend
        else:
            trend = prev_trend  # Maintain previous trend
        
        return lower_band, upper_band, trend, new_tr_idx
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute SuperTrend Crossover signal for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal (1 for long, 0 for neutral)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            signal = 0  # Neutral during warmup
            self.prev_close = curr_close
            self.output.append(signal)
            return [signal]
        
        # Calculate Fast SuperTrend
        (self.fast_lower_band, self.fast_upper_band, self.fast_trend, self.fast_tr_idx) = \
            self._calculate_supertrend(
                candle,
                self.fast_atr_period,
                self.fast_multiplier,
                self.fast_true_ranges,
                self.fast_tr_idx,
                self.fast_lower_band,
                self.fast_upper_band,
                self.fast_trend
            )
        
        # Calculate Slow SuperTrend
        (self.slow_lower_band, self.slow_upper_band, self.slow_trend, self.slow_tr_idx) = \
            self._calculate_supertrend(
                candle,
                self.slow_atr_period,
                self.slow_multiplier,
                self.slow_true_ranges,
                self.slow_tr_idx,
                self.slow_lower_band,
                self.slow_upper_band,
                self.slow_trend
            )
        
        # Calculate SMA (if filter enabled)
        sma_value = None
        if self.use_sma_filter:
            self.sma_buffer[self.sma_idx] = curr_close
            self.sma_idx = (self.sma_idx + 1) % self.sma_period
            n_sma_filled = min(self.n_prices, self.sma_period)
            if n_sma_filled >= self.sma_period:
                sma_value = np.mean(self.sma_buffer)
            else:
                sma_value = np.mean(self.sma_buffer[:n_sma_filled]) if n_sma_filled > 0 else curr_close
        
        # Check if currently in a position
        if self.position == 0:  # Neutral/Cash
            # Check entry conditions
            # Fast ST just switched to uptrend (crossover detection)
            fast_switched_up = (self.fast_trend == 1) and (self.prev_fast_trend == -1)
            slow_uptrend = (self.slow_trend == 1)
            sma_filter_ok = (not self.use_sma_filter) or (sma_value is not None and curr_close > sma_value)
            
            if fast_switched_up and slow_uptrend and sma_filter_ok:
                # ENTER LONG
                self.position = 1
                self.bias = Bias.BULLISH
                signal = 1
            else:
                # Stay neutral
                signal = 0
                self.bias = Bias.NEUTRAL
        
        else:  # Currently long (position == 1)
            # Check exit condition: Fast ST switches to downtrend
            fast_switched_down = (self.fast_trend == -1) and (self.prev_fast_trend == 1)
            
            if fast_switched_down:
                # EXIT
                self.position = 0
                self.bias = Bias.NEUTRAL
                signal = 0
            else:
                # Stay in position
                signal = 1
                self.bias = Bias.BULLISH
        
        # Store previous trend values for crossover detection
        self.prev_fast_trend = self.fast_trend
        self.prev_slow_trend = self.slow_trend
        self.prev_close = curr_close
        
        self.output.append(signal)
        return [signal]
```

---

## Key Strategic Insights

### 1. Timeframe Sensitivity

Switching from Daily to 60-minute charts significantly increases trade frequency (from ~100 to 500+ trades) while maintaining a high win rate. This makes the strategy suitable for both swing trading and day trading.

### 2. Market Regime

On US Indices, SuperTrend acts as a trend-following system. It performs best in trending markets and stays in cash (or flat) during choppy, directionless periods. This helps preserve capital during unfavorable conditions.

### 3. Trailing Stop Behavior

The "flat" behavior of the bands makes the SuperTrend an excellent trailing stop. Because the band only moves in the direction of the trade (up for longs), it protects profits during sudden reversals while allowing the trade to continue in favorable conditions.

### 4. Dual SuperTrend Advantage

Using two SuperTrend indicators (Fast and Slow) ensures you only enter when short-term momentum aligns with long-term trend direction, reducing false signals and improving win rate.

---

## Implementation Considerations

### Trailing Band Logic

The key complexity is the trailing band logic. Bands can only move in one direction:
- **Lower Band** (for uptrend): Can only move **up** or stay **flat**
- **Upper Band** (for downtrend): Can only move **down** or stay **flat**

This is implemented by comparing the new basic band value with the previous band value and the previous close price.

### Crossover Detection

To detect when Fast ST "switches" to uptrend, compare `fast_trend` with `prev_fast_trend`:
- Entry: `fast_trend == 1` AND `prev_fast_trend == -1` (switched from down to up)
- Exit: `fast_trend == -1` AND `prev_fast_trend == 1` (switched from up to down)

### ATR Calculation

Reuse the ATR calculation logic from existing nodes. Use circular buffers for efficiency and handle warmup period by using available data.

---

## Edge Cases & Considerations

### 1. Initialization

On the first valid calculation, initialize bands to basic bands and determine initial trend based on price position.

### 2. Insufficient Data

During warmup, return `0` (neutral). Ensure all buffers are populated before calculating SuperTrend.

### 3. Constant Prices

If prices are constant, ATR will be zero, and bands will be at median price. Handle this gracefully.

### 4. Trend Persistence

If price is between bands, maintain the previous trend. This prevents whipsaw signals.

### 5. SMA Filter

If SMA filter is enabled but not enough data is available, skip the filter check (treat as if filter passed).

---

## Performance Notes

1. **Circular Buffers**: Use NumPy arrays with modulo indexing for O(1) updates
2. **ATR Calculation**: Reuse efficient ATR logic (can use Cython-optimized version if available)
3. **Band Calculation**: Simple arithmetic operations (O(1))
4. **Avoid Reallocation**: Pre-allocate all buffers in `__init__`
5. **State Management**: Keep position and trend state minimal (integers)

---

## Testing Considerations

### Unit Tests

1. **Warmup Period**: Verify neutral signals (`0`) during warmup
2. **Entry Logic**: Test that entry occurs when Fast ST switches up AND Slow ST is up
3. **Exit Logic**: Test that exit occurs when Fast ST switches down
4. **Trailing Bands**: Test that bands only move in one direction
5. **SMA Filter**: Test entry with and without SMA filter
6. **Edge Cases**: Test with constant prices, insufficient data

### Integration Tests

1. **Feature Extraction**: Verify column naming matches expected format
2. **Signal Output**: Verify output is always `0` or `1` (no invalid values)
3. **Performance**: Measure computation time per candle (should be < 1ms)

### Backtest Validation

1. **Win Rate**: Verify win rate is high on trending markets
2. **Trade Frequency**: Verify trade frequency increases on shorter timeframes
3. **Drawdown**: Verify strategy stays in cash during choppy markets
4. **Parameter Sensitivity**: Test different ATR periods and multipliers

---

## Checklist

Before implementing, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'supertrendcross'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = max(fast_atr_period, slow_atr_period, sma_period if enabled)`
- [ ] Uses circular buffers for ATR calculation (Fast and Slow)
- [ ] Implements SuperTrend calculation with trailing band logic
- [ ] Tracks position state (`self.position = 0 or 1`)
- [ ] Tracks trend direction for Fast and Slow SuperTrend
- [ ] Implements crossover detection (compares current trend with previous trend)
- [ ] Implements entry logic: Fast ST switches up AND Slow ST is up
- [ ] Implements exit logic: Fast ST switches down
- [ ] Implements optional SMA filter
- [ ] Returns neutral value (`0`) during warmup
- [ ] Handles edge cases (initialization, constant prices, insufficient data)
- [ ] Updates `self.bias` based on position state
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Includes comprehensive docstring (notes rule-based feature type)
- [ ] Uses efficient data structures (NumPy arrays, circular buffers)

---

## References

- **Base Spec Guide**: `docs/bias_nodes/base_bias_node_specs.md`
- **Rule-Based Examples**: 
  - `nodes/donchian_channel.py` (position state management)
  - `nodes/double_7.py` (similar trend-following strategy)
- **ATR Reference**: `nodes/atr.py` (ATR calculation)
- **BiasNode Base Class**: `nodes/__init__.py`
