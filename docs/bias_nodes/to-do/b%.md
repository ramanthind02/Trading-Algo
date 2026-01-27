# %B (Percent B) Strategy Bias Node Specification

## Overview

The **%B Strategy** is a mean-reversion system using %B (Percent B), a derivative of Bollinger Bands. While standard Bollinger Bands indicate where price is, %B quantifies price position relative to the bands as a percentage (or decimal), allowing for more precise entries and sophisticated optimizations.

**Type**: Rule-based feature (outputs discrete signals: `1` for long, `0` for neutral)

**Market**: Equities, ETFs (works best on S&P 500 / SPY, major indices)

**Timeframe**: Daily (or any timeframe, but optimized for daily)

**Win Rate**: Typically > 70% on S&P 500

---

## Mathematical Foundation: %B

The %B indicator calculates where the current price sits between the Upper and Lower Bollinger Bands:

```
%B = (Close - Lower Band) / (Upper Band - Lower Band)
```

**Interpretation**:
- **%B > 1.0**: Price is above the Upper Band (overbought)
- **%B = 0.5**: Price is at the Middle Band (SMA)
- **%B < 0.0**: Price is below the Lower Band (oversold)

---

## Technical Specifications

### Parameters

| Parameter | Default Value | Optimized Range | Description |
|-----------|---------------|-----------------|-------------|
| `bb_length` | 20 | 5 to 20 | Period for the Simple Moving Average (SMA) used in Bollinger Bands calculation |
| `bb_std_dev` | 2.0 | 1.0 to 2.5 | Standard deviation multiplier for Bollinger Bands width |
| `entry_threshold` | 0.0 | -0.3 to +0.3 | %B threshold for entry condition (e.g., -0.2 means buying deeper into oversold territory) |
| `lookback` | 4 | 3 to 10 | Number of bars to look back for CountIf condition |
| `occurrence` | 2 | 2 to 5 | Number of times %B must be below threshold within lookback period |

### Output Values

- `1`: Long position (entry signal triggered)
- `0`: Neutral/no position (cash)

**Note**: The classic %B strategy is **long-only** (mean-reversion). It outputs `1` when in a long position and `0` when in cash.

---

## Strategy Logic

### Bollinger Bands Calculation

```
SMA = Simple Moving Average(Close, bb_length)
StdDev = Standard Deviation(Close, bb_length)
Upper Band = SMA + (bb_std_dev × StdDev)
Lower Band = SMA - (bb_std_dev × StdDev)
```

### %B Calculation

```
%B = (Close - Lower Band) / (Upper Band - Lower Band)
```

**Edge Case**: If `Upper Band == Lower Band` (zero volatility), return `0.5` (middle of range).

### Long Entry (Buy Signal)

The strategy uses a "sustained pressure" logic. Instead of buying the first time price touches the band, it uses a CountIf function to ensure the asset is sufficiently oversold over a short period.

**Entry Condition**: Enter if %B has been below the threshold (e.g., -0.2) at least `occurrence` times within the last `lookback` bars.

**Pseudo-code**:
```
IF (Count(%B < entry_threshold, lookback) >= occurrence)
   THEN Buy (output = 1)
```

**Example**: If `entry_threshold = -0.2`, `lookback = 4`, and `occurrence = 2`, then enter if %B was below -0.2 at least 2 times in the last 4 bars.

### Long Exit (Sell Signal)

The strategy uses a fast mean-reversion exit to capture quick bounces.

**Exit Condition**: Exit when the current closing price is higher than the previous bar's high.

**Pseudo-code**:
```
IF (Close > High[1])
   THEN Sell (output = 0)
```

---

## Implementation Strategy

### State Management

The node needs to maintain:

1. **Position State**: Track whether currently in a long position (`self.position = 1`) or neutral (`self.position = 0`)
2. **Price Buffer for SMA/StdDev**: Circular buffer for `bb_length` prices
3. **%B History Buffer**: Circular buffer for `lookback` %B values (for CountIf logic)
4. **Previous Bar's High**: Store previous candle's high for exit condition

### Position Logic Flow

```
Current State: position = 0 (neutral/cash)
├─ Calculate %B
├─ Update %B history buffer
├─ Check Entry Condition:
│  └─ Count(%B < threshold, lookback) >= occurrence?
│     └─ YES → ENTER LONG: position = 1, output = 1
│     └─ NO → Stay neutral: output = 0
│
Current State: position = 1 (long)
├─ Check Exit Condition:
│  └─ Close > High[1]?
│     └─ YES → EXIT: position = 0, output = 0
│     └─ NO → Stay in position: output = 1
```

---

## Implementation Details

### Warmup Period

```
front_bad = bb_length + 1
```

We need:
- `bb_length` candles to calculate SMA and standard deviation
- 1 additional candle to have a previous high for exit condition

### Circular Buffers

Use circular buffers for efficient storage:

```python
# Buffer for SMA and StdDev calculation
self.price_buffer = np.zeros(bb_length, dtype=np.float64)
self.price_buffer_idx = 0

# Buffer for %B history (for CountIf logic)
self.percent_b_buffer = np.zeros(lookback, dtype=np.float64)
self.percent_b_buffer_idx = 0
```

### Core Computation Logic

```python
def _compute_candle(self, candle: Candle) -> List:
    curr_close = candle.close
    self.n_prices += 1
    
    # Update price buffer
    self.price_buffer[self.price_buffer_idx] = curr_close
    self.price_buffer_idx = (self.price_buffer_idx + 1) % self.bb_length
    
    # Handle warmup period
    if self.n_prices < self.front_bad:
        signal = 0  # Neutral during warmup
        self.output.append(signal)
        return [signal]
    
    # Calculate SMA and Standard Deviation
    sma = np.mean(self.price_buffer)
    std_dev = np.std(self.price_buffer, ddof=1)  # Sample std dev
    
    # Calculate Bollinger Bands
    upper_band = sma + (self.bb_std_dev * std_dev)
    lower_band = sma - (self.bb_std_dev * std_dev)
    
    # Calculate %B
    band_width = upper_band - lower_band
    if band_width > 1e-10:  # Avoid division by zero
        percent_b = (curr_close - lower_band) / band_width
    else:
        percent_b = 0.5  # Neutral if no volatility
    
    # Update %B history buffer
    self.percent_b_buffer[self.percent_b_buffer_idx] = percent_b
    self.percent_b_buffer_idx = (self.percent_b_buffer_idx + 1) % self.lookback
    
    # Check if currently in a position
    if self.position == 0:  # Neutral/Cash
        # Check entry condition: Count(%B < threshold, lookback) >= occurrence
        count_below_threshold = np.sum(
            self.percent_b_buffer < self.entry_threshold
        )
        
        if count_below_threshold >= self.occurrence:
            # ENTER LONG
            self.position = 1
            self.bias = Bias.BULLISH
            signal = 1
        else:
            # Stay neutral
            signal = 0
            self.bias = Bias.NEUTRAL
    
    else:  # Currently long (position == 1)
        # Check exit condition: Close > High[1]
        if self.prev_high is not None and curr_close > self.prev_high:
            # EXIT: Mean reversion bounce
            self.position = 0
            self.bias = Bias.NEUTRAL
            signal = 0
        else:
            # Stay in position
            signal = 1
            self.bias = Bias.BULLISH
    
    # Store current high for next iteration's exit check
    self.prev_high = candle.high
    
    self.output.append(signal)
    return [signal]
```

### Edge Cases

1. **Zero Band Width**: If `Upper Band == Lower Band` (no volatility), set `%B = 0.5` (middle)
2. **Insufficient %B History**: During initial `lookback` period, use available data for CountIf
3. **No Previous High**: On first valid candle, `prev_high` is `None`, so exit condition is skipped

---

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode


class PercentB(BiasNode):
    """
    %B (Percent B) Strategy Bias Node - RULE-BASED FEATURE
    
    A mean-reversion strategy using %B, a derivative of Bollinger Bands.
    %B quantifies price position relative to the bands as a percentage,
    allowing for more precise entries than standard Bollinger Band strategies.
    
    **Type**: Rule-based feature (outputs 1 for long, 0 for neutral)
    
    **Strategy Logic**:
    - Entry: Count(%B < threshold, lookback) >= occurrence
    - Exit: Close > High[1] (mean-reversion bounce)
    
    **Mathematical Foundation**:
    - %B = (Close - Lower Band) / (Upper Band - Lower Band)
    - Upper Band = SMA + (bb_std_dev × StdDev)
    - Lower Band = SMA - (bb_std_dev × StdDev)
    
    Parameters:
    - bb_length: Period for SMA (default: 20)
    - bb_std_dev: Standard deviation multiplier (default: 2.0)
    - entry_threshold: %B threshold for entry (default: 0.0)
    - lookback: Bars to look back for CountIf (default: 4)
    - occurrence: Times %B must be below threshold (default: 2)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        bb_length: int = 20,
        bb_std_dev: float = 2.0,
        entry_threshold: float = 0.0,
        lookback: int = 4,
        occurrence: int = 2
    ):
        """
        Initialize %B Strategy node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - bb_length: Period for SMA (default: 20)
        - bb_std_dev: Standard deviation multiplier (default: 2.0)
        - entry_threshold: %B threshold for entry (default: 0.0)
        - lookback: Bars to look back for CountIf (default: 4)
        - occurrence: Times %B must be below threshold (default: 2)
        """
        super().__init__(ticker, tf)
        
        # Store parameters
        self.bb_length = bb_length
        self.bb_std_dev = bb_std_dev
        self.entry_threshold = entry_threshold
        self.lookback = lookback
        self.occurrence = occurrence
        
        # Standardization metadata
        self.module_name = 'percentb'
        self.output_features = ['signal']
        self.params = {
            'bbLength': bb_length,
            'bbStdDev': bb_std_dev,
            'entryThreshold': entry_threshold,
            'lookback': lookback,
            'occurrence': occurrence
        }
        
        # Warmup period: need bb_length candles + 1 for prev_high
        self.front_bad = bb_length + 1
        
        # Circular buffers
        self.price_buffer = np.zeros(bb_length, dtype=np.float64)
        self.price_buffer_idx = 0
        
        self.percent_b_buffer = np.zeros(lookback, dtype=np.float64)
        self.percent_b_buffer_idx = 0
        
        # Position state (0 = neutral, 1 = long)
        self.position = 0
        
        # Previous bar's high (for exit condition)
        self.prev_high = None
        
        # State tracking
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute %B Strategy signal for the given candle.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal (1 for long, 0 for neutral)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Update price buffer
        self.price_buffer[self.price_buffer_idx] = curr_close
        self.price_buffer_idx = (self.price_buffer_idx + 1) % self.bb_length
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            signal = 0  # Neutral during warmup
            self.prev_high = candle.high  # Store high for future use
            self.output.append(signal)
            return [signal]
        
        # Calculate SMA and Standard Deviation
        sma = np.mean(self.price_buffer)
        std_dev = np.std(self.price_buffer, ddof=1)  # Sample std dev
        
        # Calculate Bollinger Bands
        upper_band = sma + (self.bb_std_dev * std_dev)
        lower_band = sma - (self.bb_std_dev * std_dev)
        
        # Calculate %B
        band_width = upper_band - lower_band
        if band_width > 1e-10:  # Avoid division by zero
            percent_b = (curr_close - lower_band) / band_width
        else:
            percent_b = 0.5  # Neutral if no volatility
        
        # Update %B history buffer
        self.percent_b_buffer[self.percent_b_buffer_idx] = percent_b
        self.percent_b_buffer_idx = (self.percent_b_buffer_idx + 1) % self.lookback
        
        # Check if currently in a position
        if self.position == 0:  # Neutral/Cash
            # Check entry condition: Count(%B < threshold, lookback) >= occurrence
            # Count how many values in buffer are below threshold
            count_below_threshold = np.sum(
                self.percent_b_buffer < self.entry_threshold
            )
            
            if count_below_threshold >= self.occurrence:
                # ENTER LONG
                self.position = 1
                self.bias = Bias.BULLISH
                signal = 1
            else:
                # Stay neutral
                signal = 0
                self.bias = Bias.NEUTRAL
        
        else:  # Currently long (position == 1)
            # Check exit condition: Close > High[1]
            if self.prev_high is not None and curr_close > self.prev_high:
                # EXIT: Mean reversion bounce
                self.position = 0
                self.bias = Bias.NEUTRAL
                signal = 0
            else:
                # Stay in position
                signal = 1
                self.bias = Bias.BULLISH
        
        # Store current high for next iteration's exit check
        self.prev_high = candle.high
        
        self.output.append(signal)
        return [signal]
```

---

## Key Strategic Insights

### 1. Precision via %B

By using a threshold like `-0.2` instead of just "below the band" (`0.0`), you ensure you are buying deeper into the washout, which often improves the win rate on indices.

### 2. Increased Frequency

Reducing the Moving Average length (from 20 to 5) and the Standard Deviation (from 2.0 to 1.0) creates a tighter channel, significantly increasing the number of trade opportunities while maintaining robustness.

### 3. Low Correlation

This strategy can be duplicated into a portfolio by applying different filters (e.g., only trading when volume is high, or when the 200 SMA is trending up). Because the entries are based on short-term volatility "pockets," these variations often have low correlation with each other.

### 4. Statistical Reliability

On the S&P 500, this strategy historically yields a high win rate (70%+) because it exploits the "buy the dip" characteristic inherent in US equity indices.

---

## Implementation Considerations

### CountIf Logic

The CountIf function counts how many times %B was below the threshold within the lookback period. This is implemented efficiently using NumPy:

```python
count_below_threshold = np.sum(self.percent_b_buffer < self.entry_threshold)
```

### Standard Deviation Calculation

Use sample standard deviation (`ddof=1`) for consistency with typical Bollinger Bands implementations:

```python
std_dev = np.std(self.price_buffer, ddof=1)  # Sample std dev
```

### Previous High Tracking

Store the previous candle's high at the end of each computation:

```python
self.prev_high = candle.high  # Store for next iteration
```

This allows the exit condition to check `Close > High[1]` on the next candle.

---

## Edge Cases & Considerations

### 1. Zero Band Width (No Volatility)

If `Upper Band == Lower Band` (zero volatility), set `%B = 0.5` (middle of range). This prevents division by zero and represents a neutral state.

### 2. Insufficient %B History

During the initial `lookback` period, the %B buffer may not be fully populated. Use available data for the CountIf calculation:

```python
# Count only on available data (may be less than lookback initially)
count_below_threshold = np.sum(
    self.percent_b_buffer < self.entry_threshold
)
```

### 3. No Previous High

On the first valid candle (after warmup), `prev_high` is `None`, so the exit condition is skipped. This is handled by checking `if self.prev_high is not None`.

### 4. Parameter Validation

- Ensure `bb_length >= 2`
- Ensure `bb_std_dev > 0`
- Ensure `lookback >= 1`
- Ensure `occurrence >= 1` and `occurrence <= lookback`

---

## Performance Notes

1. **Circular Buffers**: Use NumPy arrays with modulo indexing for O(1) updates
2. **SMA Calculation**: Use `np.mean()` on circular buffer (O(n) but n is fixed)
3. **StdDev Calculation**: Use `np.std()` on circular buffer (O(n) but n is fixed)
4. **CountIf Calculation**: Use `np.sum()` with boolean mask (O(n) but n is small, typically 4)
5. **Avoid Reallocation**: Pre-allocate all buffers in `__init__`
6. **State Management**: Keep position state minimal (single integer: 0 or 1)

---

## Testing Considerations

### Unit Tests

1. **Warmup Period**: Verify neutral signals (`0`) during warmup
2. **Entry Logic**: Test that entry occurs when CountIf condition is met
3. **Exit Logic**: Test that exit occurs when `Close > High[1]`
4. **Position Persistence**: Test that position stays `1` between entry and exit
5. **Zero Volatility**: Test that %B = 0.5 when bands have zero width
6. **Edge Cases**: Test with constant prices, insufficient history

### Integration Tests

1. **Feature Extraction**: Verify column naming matches expected format
2. **Signal Output**: Verify output is always `0` or `1` (no invalid values)
3. **Performance**: Measure computation time per candle (should be < 1ms)

### Backtest Validation

1. **Win Rate**: Verify win rate is > 70% on SPY/S&P 500
2. **Parameter Sensitivity**: Test different parameter combinations
3. **Market Conditions**: Verify strategy performs well in mean-reverting markets

---

## Checklist

Before implementing, ensure:

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'percentb'`
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = bb_length + 1`
- [ ] Uses circular buffers for price and %B history storage
- [ ] Tracks position state (`self.position = 0 or 1`)
- [ ] Implements Bollinger Bands calculation (SMA, StdDev, Upper/Lower bands)
- [ ] Implements %B calculation with zero-width handling
- [ ] Implements entry logic: `Count(%B < threshold, lookback) >= occurrence`
- [ ] Implements exit logic: `Close > High[1]`
- [ ] Tracks previous bar's high for exit condition
- [ ] Returns neutral value (`0`) during warmup
- [ ] Handles edge cases (zero volatility, insufficient history)
- [ ] Updates `self.bias` based on position state
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Includes comprehensive docstring (notes rule-based feature type)
- [ ] Uses efficient data structures (NumPy arrays, circular buffers)

---

## References

- **Base Spec Guide**: `docs/bias_nodes/base_bias_node_specs.md`
- **Rule-Based Examples**: 
  - `nodes/donchian_channel.py` (position state management)
  - `nodes/double_7.py` (similar mean-reversion strategy)
- **Bollinger Bands Reference**: `nodes/archive/bollinger_bands.py` (for calculation reference)
- **BiasNode Base Class**: `nodes/__init__.py`
