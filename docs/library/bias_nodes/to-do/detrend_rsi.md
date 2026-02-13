# Detrended RSI (dRSI) Bias Node Specification

## Overview

The **Detrended Relative Strength Index (dRSI)** is a variation of the standard RSI indicator that removes long-term trend bias before calculating RSI. This makes it more effective for mean reversion trading systems.

**Key Benefit**: By removing the trend first, dRSI avoids long-term bias and works better for mean reversion systems.

## Formula

The dRSI calculation is a two-step process:

1. **Step 1: Calculate Detrended Price**
   ```
   Detrended Price = Price - SMA(50)
   ```
   Where `SMA(50)` is a 50-period Simple Moving Average of the close price.

2. **Step 2: Calculate dRSI**
   ```
   dRSI = RSI(Detrended Price)
   ```
   Apply the standard RSI calculation to the detrended price series instead of raw prices.

## Interpretation

The dRSI uses the same interpretation as standard RSI:
- **Above 70**: Overbought (potential mean reversion opportunity)
- **Below 30**: Oversold (potential mean reversion opportunity)

## Implementation Strategy

### Option 1: Composition (Recommended)

Reuse the existing `RSI` node by creating an internal instance and feeding it detrended prices:

```python
from nodes.rsi import RSI

class DetrendedRSI(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, sma_period=50):
        # ... initialization ...
        # Create internal RSI node for detrended prices
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
    def _compute_candle(self, candle):
        # Calculate SMA(50) of close prices
        sma_50 = self._calculate_sma(candle.close)
        
        # Detrend: subtract SMA from current price
        detrended_price = candle.close - sma_50
        
        # Create a "detrended candle" to feed to RSI node
        detrended_candle = Candle(
            datetime=candle.datetime,
            ticker=candle.ticker,
            open=candle.open - sma_50,  # Detrend all OHLC for consistency
            high=candle.high - sma_50,
            low=candle.low - sma_50,
            close=detrended_price,  # Main value
            volume=candle.volume
        )
        
        # Feed detrended candle to RSI node
        drsi = self.rsi_node.add_candle(detrended_candle)[0]
        
        return [drsi]
```

### Option 2: Direct RSI Calculation

Reuse RSI calculation logic directly using `utils.rsi_helpers`:

```python
from utils.rsi_helpers import compute_rsi_initial, update_rsi

class DetrendedRSI(BiasNode):
    def __init__(self, ticker, tf, rsi_lookback=14, sma_period=50):
        # ... initialization ...
        # RSI state (same as RSI node)
        self.upsum = 1e-60
        self.dnsum = 1e-60
        # ... RSI buffers ...
        
    def _compute_candle(self, candle):
        # Calculate SMA(50)
        sma_50 = self._calculate_sma(candle.close)
        
        # Detrend price
        detrended_close = candle.close - sma_50
        
        # Calculate RSI on detrended price (reuse RSI logic)
        # ... use update_rsi() with detrended_close ...
```

**Recommendation**: Use Option 1 (composition) as it's cleaner, reuses existing tested code, and is easier to maintain.

## Required Parameters

- `rsi_lookback` (int, default=14): Period for RSI calculation
- `sma_period` (int, default=50): Period for SMA used in detrending

## Warmup Period

The warmup period (`front_bad`) should be:
```
front_bad = max(sma_period, rsi_lookback)
```

This ensures we have enough data for both:
1. SMA(50) calculation (needs 50 candles)
2. RSI calculation (needs `rsi_lookback` candles)

## State Management

You'll need to maintain:

1. **SMA State**:
   - Circular buffer for close prices (size: `sma_period`)
   - Buffer index for circular access
   - Running sum or calculate from buffer

2. **RSI State** (if using Option 2):
   - Same as standard RSI node (upsum, dnsum, buffers, etc.)
   - OR delegate to internal RSI node (Option 1)

## Implementation Checklist

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `self.module_name = 'detrendedrsi'` (or `'detrend_rsi'`)
- [ ] Sets `self.output_features = ['signal']`
- [ ] Sets `self.params = {'rsi_lookback': rsi_lookback, 'sma_period': sma_period}`
- [ ] Sets `self.front_bad = max(sma_period, rsi_lookback)`
- [ ] Implements SMA(50) calculation with circular buffer
- [ ] Implements detrending: `detrended_price = close - sma_50`
- [ ] Reuses RSI calculation (via composition or direct helpers)
- [ ] Handles warmup period (returns 50.0 for RSI neutral value)
- [ ] Returns continuous value (0.0-100.0, same as RSI)
- [ ] Calls `self.ensure_standardized_columns()`

## Example Implementation Structure

```python
from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode
from nodes.rsi import RSI  # Reuse existing RSI


class DetrendedRSI(BiasNode):
    """
    Detrended RSI (dRSI) Bias Node - CONTINUOUS FEATURE
    
    Computes RSI on detrended prices to remove long-term trend bias.
    This makes dRSI more effective for mean reversion systems.
    
    Formula:
    1. Detrended Price = Price - SMA(50)
    2. dRSI = RSI(Detrended Price)
    
    **Type**: Continuous feature (outputs 0.0-100.0, requires binning)
    **Normalization**: Not needed (same 0-100 range as RSI)
    
    Parameters:
    - rsi_lookback: Period for RSI calculation (default: 14)
    - sma_period: Period for SMA detrending (default: 50)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, 
                 rsi_lookback: int = 14, sma_period: int = 50):
        super().__init__(ticker, tf)
        
        self.rsi_lookback = rsi_lookback
        self.sma_period = sma_period
        
        # Standardization metadata
        self.module_name = 'detrendedrsi'
        self.output_features = ['signal']
        self.params = {
            'rsi_lookback': rsi_lookback,
            'sma_period': sma_period
        }
        
        # Warmup period: need max of both calculations
        self.front_bad = max(sma_period, rsi_lookback)
        
        # SMA state (circular buffer for close prices)
        self.sma_buffer = np.zeros(sma_period, dtype=np.float64)
        self.sma_buffer_idx = 0
        self.sma_n_prices = 0
        
        # Create internal RSI node for detrended prices
        # This reuses the existing RSI implementation
        self.rsi_node = RSI(ticker, tf, lookback=rsi_lookback)
        
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute dRSI for the given candle.
        
        Steps:
        1. Calculate SMA(50) of close prices
        2. Detrend: subtract SMA from current close
        3. Feed detrended price to RSI node
        4. Return RSI value (0.0-100.0)
        """
        curr_close = candle.close
        
        # Update SMA buffer
        self.sma_buffer[self.sma_buffer_idx] = curr_close
        self.sma_buffer_idx = (self.sma_buffer_idx + 1) % self.sma_period
        self.sma_n_prices += 1
        
        # Handle warmup period
        if self.sma_n_prices < self.sma_period:
            # Not enough data for SMA yet
            self.output.append(50.0)  # Neutral RSI value
            return [50.0]
        
        # Calculate SMA(50)
        sma_50 = np.mean(self.sma_buffer)
        
        # Detrend: subtract SMA from current close
        detrended_close = curr_close - sma_50
        
        # Create detrended candle for RSI calculation
        # Note: We detrend all OHLC for consistency, but RSI only uses close
        detrended_candle = Candle(
            datetime=candle.datetime,
            ticker=candle.ticker,
            open=candle.open - sma_50,
            high=candle.high - sma_50,
            low=candle.low - sma_50,
            close=detrended_close,
            volume=candle.volume
        )
        
        # Feed detrended candle to RSI node
        drsi = self.rsi_node.add_candle(detrended_candle)[0]
        
        self.output.append(drsi)
        return [drsi]
```

## Key Implementation Notes

1. **SMA Calculation**: Use a circular buffer for efficient SMA calculation:
   ```python
   sma = np.mean(self.sma_buffer)  # O(1) after buffer is filled
   ```

2. **Detrending**: Subtract SMA from the current close price:
   ```python
   detrended_close = candle.close - sma_50
   ```

3. **RSI Reuse**: The internal RSI node handles all RSI state management, initialization, and computation. You just feed it detrended prices.

4. **Warmup Handling**: 
   - During SMA warmup: return 50.0 (neutral RSI)
   - RSI node handles its own warmup internally

5. **Output**: Same as RSI - continuous value 0.0-100.0, no normalization needed

## Testing Considerations

When testing the implementation:

1. **Verify detrending**: Check that detrended prices oscillate around zero
2. **Verify RSI range**: dRSI should still be 0.0-100.0
3. **Compare with standard RSI**: dRSI should show less long-term bias
4. **Mean reversion**: dRSI should be more effective for mean reversion strategies

## References

- **Base Specification**: `docs/bias_nodes/base_bias_node_specs.md`
- **RSI Implementation**: `nodes/rsi.py` - Reuse this via composition
- **RSI Helpers**: `utils/rsi_helpers.py` - If implementing RSI directly
- **Candle Model**: `utils/models.py` - For creating detrended candle objects

## Expected Column Name

With parameters `rsi_lookback=14, sma_period=50`:
```
detrendedrsi_signal_D_rsi_lookback_14_sma_period_50
```
