# Double 7s Strategy Bias Node Specification

## Overview

The **Double 7s Strategy**, developed by Larry Connors, is a simple but powerful mean-reversion system designed for equities and ETFs. It seeks to buy assets that are in a long-term uptrend but have experienced a short-term "pullback" or sell-off.

**Type**: Rule-based feature (outputs discrete signals: `1` for long, `0` for neutral, `-1` for short)

**Market**: Equities, ETFs (works best on SPY, QQQ, and high-liquidity stocks)

**Timeframe**: Daily (or any timeframe, but optimized for daily)

**Win Rate**: Typically > 70% in backtests

---

## Technical Specifications

### Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `trend_period` | 200 | Period for the trend filter (200-day SMA). The asset must be in a long-term uptrend to take long signals. |
| `short_period` | 7 | Period for identifying entry (7-day low) and exit (7-day high) points. |

### Output Values

- `1`: Long position (entry signal triggered)
- `0`: Neutral/no position (cash)
- `-1`: Short position (if strategy is extended to short trades, though classic version is long-only)

**Note**: The classic Double 7s strategy is **long-only**. It outputs `1` when in a long position and `0` when in cash. For simplicity, we'll implement the long-only version.

---

## Strategy Logic

### Long Entry (Buy Signal)

A position is opened when **both** conditions are met:

1. **Trend Filter**: The current closing price is **above** the 200-day Simple Moving Average (SMA)
2. **Pullback**: The current closing price is the **lowest close** of the last 7 trading days

**Pseudo-code**:
```
IF (Close > SMA(Close, 200)) AND (Close == Lowest(Close, 7))
   THEN Buy (output = 1)
```

### Long Exit (Sell Signal)

The position is closed when the short-term weakness has reversed:

1. **Reversion**: The current closing price is the **highest close** of the last 7 trading days

**Pseudo-code**:
```
IF (Close == Highest(Close, 7))
   THEN Sell (output = 0)
```

### Additional Exit Condition

The position is also closed if the trend filter is violated:

```
IF (Close < SMA(Close, 200))
   THEN Sell (output = 0)  // Exit due to trend violation
```

---

## Implementation Strategy

### State Management

The node needs to maintain:

1. **Position State**: Track whether currently in a long position (`self.position = 1`) or neutral (`self.position = 0`)
2. **Price Buffer for SMA**: Circular buffer for `trend_period` (200) prices
3. **Price Buffer for Min/Max**: Circular buffer for `short_period` (7) prices

### Position Logic Flow

```
Current State: position = 0 (neutral/cash)
├─ Check Entry Conditions:
│  ├─ Close > SMA(200)? → Yes
│  └─ Close == Lowest(Close, 7)? → Yes
│     └─ ENTER LONG: position = 1, output = 1
│
Current State: position = 1 (long)
├─ Check Exit Conditions:
│  ├─ Close == Highest(Close, 7)? → Yes
│  │  └─ EXIT: position = 0, output = 0
│  └─ Close < SMA(200)? → Yes
│     └─ EXIT (trend violation): position = 0, output = 0
```

---

## Implementation Details

### Warmup Period

```
front_bad = max(trend_period, short_period) = max(200, 7) = 200
```

We need at least `trend_period` candles to calculate the SMA.

### Circular Buffers

Use circular buffers for efficient storage:

```python
# Buffer for 200-day SMA calculation
self.trend_buffer = np.zeros(trend_period, dtype=np.float64)
self.trend_buffer_idx = 0

# Buffer for 7-day min/max calculation
self.short_buffer = np.zeros(short_period, dtype=np.float64)
self.short_buffer_idx = 0
```

### Core Computation Logic

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    self.n_prices += 1
    
    # Update buffers
    self.trend_buffer[self.trend_buffer_idx] = curr_close
    self.trend_buffer_idx = (self.trend_buffer_idx + 1) % self.trend_period
    
    self.short_buffer[self.short_buffer_idx] = curr_close
    self.short_buffer_idx = (self.short_buffer_idx + 1) % self.short_period
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        signal = 0  # Neutral during warmup
        self.output.append(signal)
        return [signal]
    
    # Calculate SMA(200)
    sma_200 = np.mean(self.trend_buffer)
    
    # Calculate 7-day min and max
    min_7 = np.min(self.short_buffer)
    max_7 = np.max(self.short_buffer)
    
    # Check if currently in a position
    if self.position == 0:  # Neutral/Cash
        # Check entry conditions
        if curr_close > sma_200 and curr_close == min_7:
            # ENTER LONG
            self.position = 1
            self.bias = Bias.BULLISH
            signal = 1
        else:
            # Stay neutral
            signal = 0
            self.bias = Bias.NEUTRAL
    
    else:  # Currently long (position == 1)
        # Check exit conditions
        if curr_close == max_7:
            # EXIT: Reversion to mean
            self.position = 0
            self.bias = Bias.NEUTRAL
            signal = 0
        elif curr_close < sma_200:
            # EXIT: Trend violation
            self.position = 0
            self.bias = Bias.NEUTRAL
            signal = 0
        else:
            # Stay in position
            signal = 1
            self.bias = Bias.BULLISH
    
    self.output.append(signal)
    return [signal]
```

### Edge Cases

1. **Multiple 7-day lows/highs**: If multiple candles have the same value as the min/max, use the **most recent** candle (current candle) for comparison
2. **SMA exactly equal to close**: Use strict comparison (`>` and `<`) to avoid floating-point issues
3. **Insufficient data**: Return `0` (neutral) during warmup period

---

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode


class Double7(BiasNode):
    """
    Double 7s Strategy Bias Node - RULE-BASED FEATURE
    
    A mean-reversion strategy developed by Larry Connors that buys assets
    in a long-term uptrend (above 200-day SMA) when they hit a 7-day low,
    and exits when they reach a 7-day high.
    
    **Type**: Rule-based feature (outputs 1 for long, 0 for neutral)
    
    **Strategy Logic**:
    - Entry: Close > SMA(200) AND Close == Lowest(Close, 7)
    - Exit: Close == Highest(Close, 7) OR Close < SMA(200)
    
    Parameters:
    - trend_period: Period for trend filter SMA (default: 200)
    - short_period: Period for entry/exit detection (default: 7)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        trend_period: int = 200,
        short_period: int = 7
    ):
        """
        Initialize Double 7s Strategy node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - trend_period: Period for trend filter SMA (default: 200)
        - short_period: Period for entry/exit detection (default: 7)
        """
        super().__init__(ticker, tf)
        
        # Store parameters
        self.trend_period = trend_period
        self.short_period = short_period
        
        # Standardization metadata
        self.module_name = 'double7'
        self.output_features = ['signal']
        self.params = {
            'trendPeriod': trend_period,
            'shortPeriod': short_period
        }
        
        # Warmup period: need trend_period candles for SMA
        self.front_bad = trend_period
        
        # Circular buffers
        self.trend_buffer = np.zeros(trend_period, dtype=np.float64)
        self.trend_buffer_idx = 0
        
        self.short_buffer = np.zeros(short_period, dtype=np.float64)
        self.short_buffer_idx = 0
        
        # Position state (0 = neutral, 1 = long)
        self.position = 0
        
        # State tracking
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Double 7s signal for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal (1 for long, 0 for neutral)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Update buffers
        self.trend_buffer[self.trend_buffer_idx] = curr_close
        self.trend_buffer_idx = (self.trend_buffer_idx + 1) % self.trend_period
        
        self.short_buffer[self.short_buffer_idx] = curr_close
        self.short_buffer_idx = (self.short_buffer_idx + 1) % self.short_period
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            signal = 0  # Neutral during warmup
            self.output.append(signal)
            return [signal]
        
        # Calculate SMA(200)
        sma_200 = np.mean(self.trend_buffer)
        
        # Calculate 7-day min and max
        min_7 = np.min(self.short_buffer)
        max_7 = np.max(self.short_buffer)
        
        # Check if currently in a position
        if self.position == 0:  # Neutral/Cash
            # Check entry conditions
            # Note: Use np.isclose() for floating-point comparison, or check if current is min
            is_7_day_low = np.isclose(curr_close, min_7, rtol=1e-9, atol=1e-9)
            is_above_sma = curr_close > sma_200
            
            if is_above_sma and is_7_day_low:
                # ENTER LONG
                self.position = 1
                self.bias = Bias.BULLISH
                signal = 1
            else:
                # Stay neutral
                signal = 0
                self.bias = Bias.NEUTRAL
        
        else:  # Currently long (position == 1)
            # Check exit conditions
            is_7_day_high = np.isclose(curr_close, max_7, rtol=1e-9, atol=1e-9)
            is_below_sma = curr_close < sma_200
            
            if is_7_day_high:
                # EXIT: Reversion to mean
                self.position = 0
                self.bias = Bias.NEUTRAL
                signal = 0
            elif is_below_sma:
                # EXIT: Trend violation
                self.position = 0
                self.bias = Bias.NEUTRAL
                signal = 0
            else:
                # Stay in position
                signal = 1
                self.bias = Bias.BULLISH
        
        self.output.append(signal)
        return [signal]
```

---

## Key Performance Insights

Based on technical backtests:

1. **Low Volatility**: Because the strategy exits on a 7-day high, it captures quick "bursts" of recovery rather than trying to ride long trends.

2. **Low Drawdown**: Compared to "Buy and Hold" (which can see 50%+ drawdowns), this strategy historically maintains significantly lower drawdowns (e.g., ~12.7% on SPY since 1993).

3. **Portfolio Diversification**: You can create a "Portfolio of Double 7s" by applying the same logic to different assets (SPY, QQQ, DIA) or varying the lookback period (using 6-day or 8-day windows) to smooth out the equity curve.

4. **Cash Position**: The strategy often sits in cash during extended bear markets (whenever price is below the 200 SMA), which preserves capital during crashes like 2001 and 2008.

---

## Implementation Considerations

### No Stop Loss

Classic Double 7s does not use a fixed stop loss. The strategy relies on the statistical probability that an asset in a long-term uptrend will eventually recover from a 7-day low.

### Execution Timing

Orders are traditionally executed "Market on Close" (MOC) right before the daily bell, or at the open of the following day. The bias node outputs the signal at the close, and execution can happen at the next bar's open.

### Floating-Point Comparison

When checking if `Close == Lowest(Close, 7)`, use `np.isclose()` for floating-point comparison to avoid precision issues:

```python
is_7_day_low = np.isclose(curr_close, min_7, rtol=1e-9, atol=1e-9)
```

Alternatively, you can check if the current close is the minimum by comparing indices:

```python
# Get index of minimum value in buffer
min_idx = np.argmin(self.short_buffer)
# Check if current index matches minimum index
is_7_day_low = (self.short_buffer_idx - 1) % self.short_period == min_idx
```

However, the `np.isclose()` approach is simpler and handles edge cases better.

---

## Edge Cases & Considerations

### 1. Multiple Equal Min/Max Values

If multiple candles in the 7-day window have the same value as the min/max, the strategy should trigger on the **current candle** if it matches the min/max. This is handled correctly by checking if `curr_close == min_7` or `curr_close == max_7`.

### 2. Trend Filter Violation During Position

If price drops below SMA(200) while in a position, exit immediately. This preserves capital during trend reversals.

### 3. Insufficient Data

- During warmup (`n_prices < front_bad`), return `0` (neutral)
- If buffers aren't fully populated yet, return `0`

### 4. Parameter Validation

- Ensure `trend_period >= 2`
- Ensure `short_period >= 2`
- Ensure `trend_period >= short_period` (logically, trend should be longer than short-term)

---

## Performance Notes

1. **Circular Buffers**: Use NumPy arrays with modulo indexing for O(1) updates
2. **SMA Calculation**: Use `np.mean()` on circular buffer (O(n) but n is fixed at 200)
3. **Min/Max Calculation**: Use `np.min()` and `np.max()` on circular buffer (O(n) but n is fixed at 7)
4. **Avoid Reallocation**: Pre-allocate all buffers in `__init__`
5. **State Management**: Keep position state minimal (single integer: 0 or 1)

---

## Testing Considerations

### Unit Tests

1. **Warmup Period**: Verify neutral signals (`0`) during warmup
2. **Entry Logic**: Test that entry occurs when both conditions are met
3. **Exit Logic**: Test that exit occurs on 7-day high
4. **Trend Violation**: Test that position exits when price drops below SMA(200)
5. **Position Persistence**: Test that position stays `1` between entry and exit
6. **Edge Cases**: Test with constant prices, multiple equal min/max values

### Integration Tests

1. **Feature Extraction**: Verify column naming matches expected format
2. **Signal Output**: Verify output is always `0` or `1` (no invalid values)
3. **Performance**: Measure computation time per candle (should be < 1ms)

### Backtest Validation

1. **Win Rate**: Verify win rate is > 70% on SPY/QQQ
2. **Drawdown**: Verify max drawdown is significantly lower than buy-and-hold
3. **Cash Periods**: Verify strategy sits in cash during bear markets (below SMA)

---

## Checklist

Before implementing, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'double7'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = trend_period`
- [ ] Uses circular buffers for trend and short-term price storage
- [ ] Tracks position state (`self.position = 0 or 1`)
- [ ] Implements entry logic: `Close > SMA(200) AND Close == Lowest(Close, 7)`
- [ ] Implements exit logic: `Close == Highest(Close, 7) OR Close < SMA(200)`
- [ ] Returns neutral value (`0`) during warmup
- [ ] Handles edge cases (floating-point comparison, multiple equal min/max)
- [ ] Updates `self.bias` based on position state
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Includes comprehensive docstring (notes rule-based feature type)
- [ ] Uses efficient data structures (NumPy arrays, circular buffers)

---

## References

- **Base Spec Guide**: `docs/bias_nodes/base_bias_node_specs.md`
- **Rule-Based Examples**: 
  - `nodes/donchian_channel.py` (position state management)
  - `nodes/buy_hold.py` (simple rule-based signal)
- **BiasNode Base Class**: `nodes/__init__.py`
- **Strategy Source**: Larry Connors - "Double 7s Strategy"

---


