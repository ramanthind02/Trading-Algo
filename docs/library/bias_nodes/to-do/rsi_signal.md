# RSI Signal Bias Node Specification

## Overview

The **RSI Signal** node is a rule-based bias node that converts continuous RSI values into discrete trading signals (`1`, `0`, `-1`) based on entry/exit thresholds and trading direction.

**Key Benefit**: Provides deterministic, interpretable trading signals directly from RSI without requiring binning.

**Type**: Rule-based (outputs `1`, `0`, `-1` - no binning required)

## Trading Logic

The node maintains position state and generates signals based on RSI thresholds:

### For Long-Only Strategy (`direction='long'`):
- **Entry**: When RSI crosses below `entry_level` → output `1` (long signal)
- **Exit**: When RSI crosses above `exit_level` → output `0` (neutral/flat)
- **Maintain**: While in position, continue outputting `1` until exit condition

### For Short-Only Strategy (`direction='short'`):
- **Entry**: When RSI crosses above `entry_level` → output `-1` (short signal)
- **Exit**: When RSI crosses below `exit_level` → output `0` (neutral/flat)
- **Maintain**: While in position, continue outputting `-1` until exit condition

### For Long-Short Strategy (`direction='long-short'`):
- **Enter Long**: When RSI crosses below `entry_level` → output `1`
- **Exit Long / Enter Short**: When RSI crosses above `exit_level` → output `-1`
- **Exit Short / Enter Long**: When RSI crosses below `entry_level` → output `1`
- **Always in market**: Alternates between `1` and `-1` based on RSI levels

## Formula

```
1. Calculate RSI using existing RSI node
2. Apply rule-based logic based on direction:
   - long: RSI < entry_level → 1, RSI > exit_level → 0
   - short: RSI > entry_level → -1, RSI < exit_level → 0
   - long-short: RSI < entry_level → 1, RSI > exit_level → -1
```

## Implementation Strategy

### Recommended: Composition with RSI Node

Reuse the existing `RSI` node by creating an internal instance:

```python
from nodes.rsi import RSI

class RSISignal(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, entry_level=30, exit_level=70, direction='long'):
        # ... initialization ...
        # Create internal RSI node
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        # Track current position state
        self.position = 0  # 0=flat, 1=long, -1=short
        
    def _compute_candle(self, candle):
        # Get RSI value from internal node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Apply rule-based logic based on direction
        signal = self._apply_rsi_rules(rsi)
        
        return [signal]
```

## Required Parameters

- `rsi_lookback` (int, default=14): Period for RSI calculation
- `entry_level` (float, default=30.0): RSI level to enter position
- `exit_level` (float, default=70.0): RSI level to exit position
- `direction` (str, default='long'): Trading direction - `'long'`, `'short'`, or `'long-short'`

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = rsi_lookback
```

The RSI node handles its own warmup, so we just need to wait for RSI to be valid.

## State Management

You need to maintain:

1. **Position State**:
   - `self.position`: Current position (`0`=flat, `1`=long, `-1`=short)
   - Updated based on RSI crossing thresholds

2. **RSI Node** (via composition):
   - Internal `RSI` instance handles all RSI state
   - No need to manage RSI buffers/state directly

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'rsi_signal'` (or `'rsisignal'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'rsi_lookback': rsi_lookback, 'entry_level': entry_level, 'exit_level': exit_level, 'direction': direction}`
- [ ] Sets `self.front_bad = rsi_lookback`
- [ ] Creates internal `RSI` node instance
- [ ] Initializes `self.position = 0` (flat/neutral)
- [ ] Implements rule-based logic for each direction
- [ ] Handles warmup period (returns `0` - neutral signal)
- [ ] Returns discrete signal (`1`, `0`, or `-1`)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
from utils.models import Candle
from utils.enums import Ticker, TimeFrame, Bias
from nodes import BiasNode
from nodes.rsi import RSI  # Reuse existing RSI


class RSISignal(BiasNode):
    """
    RSI Signal Bias Node - RULE-BASED
    
    Converts continuous RSI values into discrete trading signals based on
    entry/exit thresholds and trading direction.
    
    **Type**: Rule-based (outputs 1, 0, -1 - no binning required)
    
    Trading Logic:
    - Long: Enter when RSI < entry_level, exit when RSI > exit_level
    - Short: Enter when RSI > entry_level, exit when RSI < exit_level
    - Long-Short: Always in market, switch between long/short based on thresholds
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - entry_level: RSI level to enter position (default: 30.0)
    - exit_level: RSI level to exit position (default: 70.0)
    - direction: Trading direction - 'long', 'short', or 'long-short' (default: 'long')
    """
    
    def __init__(
        self, 
        ticker: Ticker, 
        tf: TimeFrame,
        rsi_lookback: int = 14,
        entry_level: float = 30.0,
        exit_level: float = 70.0,
        direction: str = 'long'
    ):
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.entry_level = entry_level
        self.exit_level = exit_level
        
        # Validate direction
        if direction not in ['long', 'short', 'long-short']:
            raise ValueError(f"direction must be 'long', 'short', or 'long-short', got '{direction}'")
        self.direction = direction
        
        # Standardization metadata
        self.module_name = 'rsi_signal'
        self.output_features = ['signal']
        self.params = {
            'rsi_lookback': rsi_lookback,
            'entry_level': entry_level,
            'exit_level': exit_level,
            'direction': direction
        }
        
        # Warmup period (same as RSI)
        self.front_bad = rsi_lookback
        
        # Create internal RSI node for RSI calculation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        # Position state: 0=flat, 1=long, -1=short
        self.position = 0
        
        # Track candle count for warmup
        self.n_prices = 0
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI signal for the given candle.
        
        Returns discrete signal:
        - 1: Long position
        - 0: Neutral/flat
        - -1: Short position
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the signal value (1, 0, or -1)
        """
        self.n_prices += 1
        
        # Get RSI value from internal RSI node
        rsi = self.rsi_node.add_candle(candle)[0]
        
        # Handle warmup period (RSI returns 50.0 during warmup)
        if self.n_prices < self.front_bad:
            self.position = 0
            self.output.append(0)
            return [0]
        
        # Apply rule-based logic based on direction
        if self.direction == 'long':
            signal = self._apply_long_logic(rsi)
        elif self.direction == 'short':
            signal = self._apply_short_logic(rsi)
        else:  # long-short
            signal = self._apply_long_short_logic(rsi)
        
        # Update position state
        self.position = signal
        
        # Update bias for diagnostics
        if signal == 1:
            self.bias = Bias.BULLISH
        elif signal == -1:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        self.output.append(signal)
        return [signal]
    
    def _apply_long_logic(self, rsi: float) -> int:
        """
        Apply long-only trading logic.
        
        Entry: RSI < entry_level
        Exit: RSI > exit_level
        """
        if self.position == 0:
            # Currently flat - check for entry
            if rsi < self.entry_level:
                return 1  # Enter long
            else:
                return 0  # Stay flat
        else:
            # Currently long - check for exit
            if rsi > self.exit_level:
                return 0  # Exit long
            else:
                return 1  # Maintain long
    
    def _apply_short_logic(self, rsi: float) -> int:
        """
        Apply short-only trading logic.
        
        Entry: RSI > entry_level
        Exit: RSI < exit_level
        """
        if self.position == 0:
            # Currently flat - check for entry
            if rsi > self.entry_level:
                return -1  # Enter short
            else:
                return 0  # Stay flat
        else:
            # Currently short - check for exit
            if rsi < self.exit_level:
                return 0  # Exit short
            else:
                return -1  # Maintain short
    
    def _apply_long_short_logic(self, rsi: float) -> int:
        """
        Apply long-short trading logic (always in market).
        
        Enter long: RSI < entry_level
        Enter short: RSI > exit_level
        """
        if rsi < self.entry_level:
            return 1  # Long position
        elif rsi > self.exit_level:
            return -1  # Short position
        else:
            # Between thresholds - maintain current position
            return self.position if self.position != 0 else 0
```

## Key Implementation Notes

1. **RSI Reuse**: The internal RSI node handles all RSI computation. You just call `self.rsi_node.add_candle(candle)[0]` to get the RSI value.

2. **Position State**: Maintain `self.position` to track current position state. This is crucial for entry/exit logic.

3. **Threshold Logic**: 
   - Use `<` for entry (oversold/overbought conditions)
   - Use `>` for exit (reversal conditions)
   - For long-short, maintain position when between thresholds

4. **Warmup Handling**: 
   - During warmup, RSI returns `50.0` (neutral)
   - Return `0` (flat/neutral signal) during warmup

5. **Direction Parameter**: 
   - `'long'`: Only long positions (1 or 0)
   - `'short'`: Only short positions (-1 or 0)
   - `'long-short'`: Always in market (1 or -1, never 0 after first entry)

6. **Output**: Always returns discrete values: `1` (long), `0` (flat), or `-1` (short)

## Common Use Cases

### Mean Reversion (Long-Only)
```python
# Enter when oversold, exit when overbought
RSISignal(ticker, tf, entry_level=30, exit_level=70, direction='long')
```

### Momentum (Short-Only)
```python
# Enter short when overbought, exit when oversold
RSISignal(ticker, tf, entry_level=70, exit_level=30, direction='short')
```

### Always-In-Market (Long-Short)
```python
# Switch between long and short based on RSI extremes
RSISignal(ticker, tf, entry_level=30, exit_level=70, direction='long-short')
```

## Testing Considerations

When testing the implementation:

1. **Verify Entry Logic**: 
   - Long: RSI drops below entry_level → should output `1`
   - Short: RSI rises above entry_level → should output `-1`

2. **Verify Exit Logic**:
   - Long: RSI rises above exit_level → should output `0`
   - Short: RSI drops below exit_level → should output `0`

3. **Verify Position Maintenance**:
   - While in position and between thresholds, maintain current position

4. **Verify Long-Short Logic**:
   - Should alternate between `1` and `-1` based on RSI extremes
   - Never output `0` after first entry (always in market)

5. **Compare with Standard RSI**:
   - Signals should align with RSI overbought/oversold levels
   - Entry/exit should occur at specified thresholds

## Expected Column Name

With parameters `rsi_lookback=14, entry_level=30, exit_level=70, direction='long'`:
```
rsi_signal_signal_D_rsi_lookback_14_entry_level_30.0_exit_level_70.0_direction_long
```

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reuse this via composition
- **Rule-Based Example**: `nodes/donchian_channel.py` - Example of rule-based node with position state
- **Archived RSI Signal**: `nodes/archive/rsi_signal.py` - Previous implementation (long-only with MA filter)

## Differences from Archived Implementation

The archived `rsi_signal.py` only supports:
- Long-only positions
- Includes 200-period MA trend filter
- Binary output (0 or 1)

This new specification supports:
- Long, short, or long-short strategies
- No trend filter (pure RSI-based)
- Ternary output (1, 0, -1)
- Configurable entry/exit levels
