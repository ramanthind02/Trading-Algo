# Adaptive RSI Bias Node Specification

## Overview

The **Adaptive RSI** is a dynamic version of the Relative Strength Index that adjusts its lookback period based on market volatility or cycle length. Unlike fixed-period RSI, Adaptive RSI uses shorter periods in volatile markets (for faster response) and longer periods in quiet markets (for stability), making it more responsive to changing market conditions.

**Key Benefit**: Adapts to changing market volatility, providing faster signals in volatile markets and more stable signals in quiet markets. This makes it particularly effective for mean reversion trading on stock indices across different market regimes.

**Type**: Continuous feature (outputs 0.0-100.0, requires binning)

**Normalization**: Not needed (output is already normalized to fixed 0-100 range, same as standard RSI)

## Formula

The Adaptive RSI calculation involves dynamically adjusting the RSI lookback period:

### Step 1: Calculate Volatility Measure

```
Volatility = ATR(volatility_period) / Close
```

Or use standard deviation of returns:
```
Volatility = StdDev(Returns, volatility_period)
```

### Step 2: Map Volatility to RSI Period

```
Normalized_Volatility = (Volatility - Min_Volatility) / (Max_Volatility - Min_Volatility)
Adaptive_Period = Min_Period + (Normalized_Volatility × (Max_Period - Min_Period))
```

Or use inverse relationship (higher volatility → shorter period):
```
Adaptive_Period = Max_Period - (Normalized_Volatility × (Max_Period - Min_Period))
```

### Step 3: Calculate RSI with Adaptive Period

```
RSI = CalculateRSI(close_prices, Adaptive_Period)
```

**Note**: The adaptive period changes each bar, so RSI calculation needs to handle variable periods.

## Interpretation

The Adaptive RSI uses the same interpretation as standard RSI:

- **Above 70**: Overbought (potential mean reversion opportunity)
- **Below 30**: Oversold (potential mean reversion opportunity)
- **Near 50**: Neutral

**Typical Range**: 0.0 to 100.0 (same as standard RSI)

**Why It's Better**:
- **Volatility Adaptation**: Shorter periods in volatile markets (faster response), longer periods in quiet markets (stability)
- **Regime Awareness**: Automatically adjusts to different market conditions
- **Reduced Lag**: Responds faster to changing market dynamics
- **Mean Reversion**: More effective for mean reversion as it adapts to current volatility

## Implementation Strategy

### Recommended: Volatility-Based Adaptation

Use ATR-based volatility to adjust RSI period:

```python
from nodes.rsi import RSI
from nodes.atr import ATRNode

class AdaptiveRSI(BiasNode):
    def __init__(self, ticker, tf, min_period=2, max_period=14, 
                 volatility_period=20, volatility_method='atr'):
        # ... initialization ...
        # Create ATR node for volatility calculation
        self.atr_node = ATRNode(ticker, tf, period=volatility_period)
        # Track volatility history for normalization
        self.volatility_buffer = np.zeros(volatility_period, dtype=np.float64)
        
    def _compute_candle(self, candle):
        # Calculate ATR-based volatility
        atr, atr_pct = self.atr_node.add_candle(candle)
        volatility = atr_pct  # Use ATR as percentage
        
        # Update volatility buffer
        # Normalize volatility to [0, 1] range
        normalized_vol = self._normalize_volatility(volatility)
        
        # Calculate adaptive period (inverse: high vol → short period)
        adaptive_period = int(self.max_period - (normalized_vol * (self.max_period - self.min_period)))
        adaptive_period = max(self.min_period, min(self.max_period, adaptive_period))
        
        # Calculate RSI with adaptive period
        # Note: Need to recalculate RSI with new period each time
        rsi = self._calculate_adaptive_rsi(candle, adaptive_period)
        
        return [rsi]
```

### Alternative: Cycle-Based Adaptation

Use cycle length detection to adjust RSI period (more complex):

```python
# Detect dominant cycle length
cycle_length = self._detect_cycle_length()
adaptive_period = cycle_length // 2  # Use half cycle length
```

## Required Parameters

- `min_period` (int, default=2): Minimum RSI lookback period (for high volatility)
- `max_period` (int, default=14): Maximum RSI lookback period (for low volatility)
- `volatility_period` (int, default=20): Period for volatility calculation
- `volatility_method` (str, default='atr'): Method for volatility ('atr' or 'stddev')

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = max(max_period, volatility_period) + 1
```

This ensures we have:
1. Enough candles for maximum RSI period calculation
2. Enough candles for volatility calculation
3. Additional buffer for volatility normalization

**Minimum**: Need at least `max(max_period, volatility_period)` candles before valid Adaptive RSI output.

## State Management

You need to maintain:

1. **Volatility Calculation**:
   - ATR node (or standard deviation calculation)
   - Volatility history buffer for normalization

2. **RSI Calculation**:
   - Since period changes, need to recalculate RSI each time
   - Can use incremental RSI calculation or full recalculation
   - Store price history for variable-period RSI calculation

3. **Price History Buffer**: Circular buffer for prices (size: `max_period + 1`)

4. **Volatility Buffer**: Circular buffer for volatility values (size: `volatility_period`)

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)` in constructor
- [ ] Sets `self.module_name = 'adaptiversi'` (or `'adaptive_rsi'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params` with all parameters
- [ ] Sets `self.front_bad = max(max_period, volatility_period) + 1`
- [ ] Creates volatility calculation (ATR node or stddev)
- [ ] Initializes price history buffer (size: `max_period + 1`)
- [ ] Initializes volatility buffer for normalization
- [ ] Implements volatility normalization
- [ ] Implements adaptive period calculation
- [ ] Implements variable-period RSI calculation
- [ ] Handles warmup period (returns 50.0 - neutral RSI)
- [ ] Returns continuous value (0.0-100.0, same as RSI)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from nodes.atr import ATRNode  # For volatility calculation


class AdaptiveRSI(BiasNode):
    """
    Adaptive RSI Bias Node - CONTINUOUS FEATURE
    
    Dynamic version of RSI that adjusts its lookback period based on market
    volatility. Uses shorter periods in volatile markets (faster response) and
    longer periods in quiet markets (stability).
    
    Formula:
    1. Calculate volatility (ATR or stddev)
    2. Normalize volatility to [0, 1] range
    3. Map to adaptive period: high vol → short period, low vol → long period
    4. Calculate RSI with adaptive period
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    **Normalization**: Not needed (same 0-100 range as standard RSI)
    
    **Key Benefit**: Adapts to changing market volatility, providing faster
    signals in volatile markets and more stable signals in quiet markets.
    
    Parameters:
    - min_period: Minimum RSI period (for high volatility, default: 2)
    - max_period: Maximum RSI period (for low volatility, default: 14)
    - volatility_period: Period for volatility calculation (default: 20)
    - volatility_method: Method for volatility ('atr' or 'stddev', default: 'atr')
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        min_period: int = 2,
        max_period: int = 14,
        volatility_period: int = 20,
        volatility_method: str = 'atr'
    ):
        super().__init__(ticker, tf)
        
        self.min_period = min_period
        self.max_period = max_period
        self.volatility_period = volatility_period
        self.volatility_method = volatility_method
        
        # Standardization metadata
        self.module_name = 'adaptiversi'
        self.output_features = ['signal']
        self.params = {
            'minPeriod': min_period,
            'maxPeriod': max_period,
            'volatilityPeriod': volatility_period,
            'volatilityMethod': volatility_method
        }
        
        # Warmup period: need max of max_period and volatility_period
        self.front_bad = max(max_period, volatility_period) + 1
        
        # Create ATR node for volatility calculation (if using ATR method)
        if volatility_method == 'atr':
            self.atr_node = ATRNode(ticker, tf, period=volatility_period)
        else:
            self.atr_node = None
        
        # Price history buffer for variable-period RSI calculation
        self.price_buffer = np.zeros(max_period + 1, dtype=np.float64)
        self.price_buffer_idx = 0
        
        # Volatility buffer for normalization
        self.volatility_buffer = np.zeros(volatility_period, dtype=np.float64)
        self.volatility_buffer_idx = 0
        self.n_volatility_values = 0
        
        # RSI state (for incremental calculation)
        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.prev_close = None
        
        # Track candle count
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _calculate_volatility(self, candle: Candle) -> float:
        """
        Calculate volatility measure.
        
        Returns:
        - Volatility value (ATR percentage or stddev)
        """
        if self.volatility_method == 'atr':
            # Use ATR as percentage
            atr, atr_pct = self.atr_node.add_candle(candle)
            return atr_pct
        else:
            # Use standard deviation of returns
            # This would require return calculation
            # For simplicity, using ATR is recommended
            return 0.0
    
    def _normalize_volatility(self, volatility: float) -> float:
        """
        Normalize volatility to [0, 1] range using recent volatility distribution.
        
        Parameters:
        - volatility: Current volatility value
        
        Returns:
        - Normalized volatility in [0, 1] range
        """
        if self.n_volatility_values < self.volatility_period:
            # Not enough data yet, use current value as baseline
            return 0.5  # Neutral
        
        vol_min = np.min(self.volatility_buffer)
        vol_max = np.max(self.volatility_buffer)
        vol_range = vol_max - vol_min
        
        if vol_range > 1e-10:
            normalized = (volatility - vol_min) / vol_range
            return np.clip(normalized, 0.0, 1.0)
        else:
            return 0.5  # Neutral if no variation
    
    def _calculate_adaptive_rsi(self, candle: Candle, period: int) -> float:
        """
        Calculate RSI with specified period.
        
        Parameters:
        - candle: Current candle
        - period: RSI lookback period
        
        Returns:
        - RSI value (0.0-100.0)
        """
        curr_close = candle.close
        
        # Update price buffer
        self.price_buffer[self.price_buffer_idx] = curr_close
        self.price_buffer_idx = (self.price_buffer_idx + 1) % (self.max_period + 1)
        
        # Need at least period candles
        if self.n_prices < period:
            return 50.0  # Neutral during warmup
        
        # Extract prices for RSI calculation
        # Get last period+1 prices (need period+1 for RSI calculation)
        if self.price_buffer_idx == 0:
            prices = self.price_buffer[-(period+1):]
        else:
            # Handle circular buffer wrapping
            if self.price_buffer_idx > period:
                prices = self.price_buffer[self.price_buffer_idx - period - 1:self.price_buffer_idx]
            else:
                # Need to wrap around
                prices = np.concatenate([
                    self.price_buffer[self.price_buffer_idx - period - 1:],
                    self.price_buffer[:self.price_buffer_idx]
                ])
        
        # Calculate RSI from price array
        # Use standard RSI calculation
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        # Calculate average gain and loss
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Avoid division by zero
        if avg_loss < 1e-10:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Adaptive RSI for the given candle.
        
        Steps:
        1. Calculate volatility (ATR or stddev)
        2. Normalize volatility to [0, 1] range
        3. Calculate adaptive period (inverse: high vol → short period)
        4. Calculate RSI with adaptive period
        5. Return Adaptive RSI value (0.0-100.0)
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Adaptive RSI value (0.0-100.0)
        """
        self.n_prices += 1
        
        # Handle warmup period
        if self.n_prices < self.front_bad:
            self.prev_close = candle.close
            self.output.append(50.0)  # Neutral RSI
            return [50.0]
        
        # Calculate volatility
        volatility = self._calculate_volatility(candle)
        
        # Update volatility buffer
        self.volatility_buffer[self.volatility_buffer_idx] = volatility
        self.volatility_buffer_idx = (self.volatility_buffer_idx + 1) % self.volatility_period
        if self.n_volatility_values < self.volatility_period:
            self.n_volatility_values += 1
        
        # Normalize volatility
        normalized_vol = self._normalize_volatility(volatility)
        
        # Calculate adaptive period
        # Inverse relationship: high volatility → short period (faster response)
        # low volatility → long period (stability)
        adaptive_period = int(
            self.max_period - (normalized_vol * (self.max_period - self.min_period))
        )
        adaptive_period = max(self.min_period, min(self.max_period, adaptive_period))
        
        # Calculate RSI with adaptive period
        adaptive_rsi = self._calculate_adaptive_rsi(candle, adaptive_period)
        
        # Update state
        self.prev_close = candle.close
        
        # Update bias based on Adaptive RSI value
        if adaptive_rsi > 70:
            self.bias = Bias.BEARISH  # Overbought
        elif adaptive_rsi < 30:
            self.bias = Bias.BULLISH  # Oversold
        else:
            self.bias = Bias.NEUTRAL
        
        self.output.append(adaptive_rsi)
        return [adaptive_rsi]
```

## Key Implementation Notes

1. **Volatility Calculation**: 
   - Use ATR node for simplicity (recommended)
   - Or calculate standard deviation of returns (more complex)

2. **Volatility Normalization**: 
   - Normalize volatility to [0, 1] using recent volatility distribution
   - Use min/max of volatility buffer for normalization

3. **Adaptive Period Calculation**:
   - Inverse relationship: high volatility → short period (faster response)
   - Low volatility → long period (stability)
   - Clip to [min_period, max_period] range

4. **Variable-Period RSI Calculation**:
   - Since period changes each bar, need to recalculate RSI each time
   - Use price history buffer to extract prices for current period
   - Can use standard RSI calculation on extracted prices

5. **Edge Case Handling**:
   - During warmup → return `50.0` (neutral RSI)
   - If volatility range is zero → use neutral period (middle of range)
   - If insufficient prices for period → return `50.0`

6. **Output Range**: 
   - Same as standard RSI: 0.0 to 100.0
   - Interpretation is the same as standard RSI

## Why Adaptive RSI is Better

### Volatility Adaptation Example

**High Volatility Market**:
- Standard RSI(14): May lag behind rapid price movements
- Adaptive RSI: Uses shorter period (e.g., 2-5), responds faster to volatility spikes

**Low Volatility Market**:
- Standard RSI(14): May be too sensitive, generating false signals
- Adaptive RSI: Uses longer period (e.g., 10-14), more stable, filters noise

**Key Insight**: Adaptive RSI automatically adjusts to market conditions, providing optimal responsiveness without manual parameter tuning.

## Trading Strategy Logic (Reference)

While this is a **continuous feature** (not rule-based), here's how it would typically be used:

**Mean Reversion Strategy**:
- **Long Entry**: When Adaptive RSI crosses below 30 (oversold)
- **Long Exit**: When Adaptive RSI crosses back above 50 (returns to neutral)
- **Short Entry**: When Adaptive RSI crosses above 70 (overbought)
- **Short Exit**: When Adaptive RSI crosses back below 50

**Note**: These thresholds (30, 50, 70) would be learned by binning models, not hardcoded in the node.

## Edge Cases

1. **Insufficient Data**:
   - During warmup, return `50.0` (neutral RSI)
   - Need at least `max_period` candles for RSI calculation
   - Need at least `volatility_period` candles for volatility normalization

2. **Zero Volatility Range**:
   - If all volatility values are identical
   - Use neutral period (middle of min/max range)

3. **Period Bounds**:
   - Clip adaptive period to [min_period, max_period]
   - Ensure period is at least 2 (minimum for RSI calculation)

## Performance Optimization

- **Circular Buffers**: Use NumPy arrays with modulo indexing
- **Efficient RSI Calculation**: Recalculate RSI each time (period changes)
- **Volatility Caching**: ATR node handles its own caching
- **Minimize State**: Only store necessary price and volatility history

## Testing Considerations

When testing the implementation:

1. **Verify Volatility Calculation**: Check that volatility values are reasonable
2. **Verify Period Adaptation**: 
   - High volatility should produce shorter periods
   - Low volatility should produce longer periods
3. **Verify RSI Calculation**: 
   - Adaptive RSI should match standard RSI when period is constant
   - Should respond faster in volatile markets
4. **Verify Stability**: 
   - Should be stable in quiet markets
   - Should be responsive in volatile markets

## Expected Column Name

With parameters `min_period=2, max_period=14, volatility_period=20, volatility_method='atr'`:
```
adaptiversi_signal_D_minPeriod_2_maxPeriod_14_volatilityPeriod_20_volatilityMethod_atr
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reference for RSI calculation
- **ATR Implementation**: `nodes/atr.py` - For volatility calculation
- **Similar Patterns**: Other adaptive indicators that adjust parameters based on market conditions

## Important Notes

1. **Continuous Feature**: This outputs Adaptive RSI values (continuous), not discrete signals. Binning models will learn optimal thresholds.

2. **Volatility Adaptation**: Unlike fixed-period RSI, Adaptive RSI adjusts its period based on market volatility, providing optimal responsiveness.

3. **Regime Awareness**: Automatically adapts to different market conditions without manual parameter tuning.

4. **Mean Reversion**: More effective for mean reversion as it adapts to current volatility, providing faster signals in volatile markets and more stable signals in quiet markets.

5. **Implementation Complexity**: More complex than standard RSI due to variable period calculation. Consider performance implications of recalculating RSI each bar.
