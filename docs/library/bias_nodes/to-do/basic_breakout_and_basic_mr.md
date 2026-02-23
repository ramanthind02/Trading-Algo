# BasicBreakout and BasicMR Bias Node Specification

This document specifies two rule-based bias nodes derived from the StatOasis Market Edge Analysis strategy: **BasicBreakout** (prior-bar high/low breakout) and **BasicMR** (mean reversion = inverse of breakout). BasicMR is implemented by wrapping BasicBreakout and negating its output; no duplicated logic.

**Architecture note:** Both nodes output a **signed integer in `[-max_positions, max_positions]`** (default max 5). The value represents stacked position size: e.g. 3 = three long units, -2 = two short units. On warmup or when mode clamps, output is 0. Stop losses, take profit, limit orders, and stop orders are not managed inside the node; the bias output is the position directive for downstream systems.

**EA alignment:** The StatOasis EA holds position on inside bars and adds one order per bar in the same direction up to `MaxOrders` (5). The node mirrors this by (1) **holding** the last non-zero output on inside bars, and (2) **stacking** same-direction breakouts up to `max_positions`.

---

## PositionMode Enum

Both nodes use a **position mode** parameter to allow long-only, short-only, or long-short behaviour. Define an enum (e.g. in `utils/enums.py`) and use it for the parameter (per workspace rules: no booleans for domain-level APIs).

**Suggested definition in `utils/enums.py`:**

```python
class PositionMode(Enum):
    LONG_SHORT = "long_short"   # Default: output in [-max_positions, max_positions]
    LONG_ONLY = "long_only"     # Clamp negative to 0 (output 0..max_positions)
    SHORT_ONLY = "short_only"   # Clamp positive to 0 (output -max_positions..0)
```

- **LONG_SHORT (default):** Output the stacked signal in [-max_positions, max_positions].
- **LONG_ONLY:** If stacked signal is negative, output 0; otherwise output as-is (0..max_positions).
- **SHORT_ONLY:** If stacked signal is positive, output 0; otherwise output as-is (-max_positions..0).

Include `mode` and `max_positions` in `params` for column naming (e.g. `basicbreakout_signal_D_mode_long_short_max_5`).

---

# Section 1: BasicBreakout

## Overview

**BasicBreakout** (Market Edge / prior bar breakout) signals when price closes beyond the prior bar's high or low.

- **Type:** Rule-based, **stateful** (outputs stacked level in [-max_positions, max_positions], or 0).
- **Logic:** Close above previous bar high → long (stack or start); close below previous bar low → short (stack or start); **inside bar → hold** last non-zero output.
- **Stacking:** Same-direction breakout increments magnitude (cap at max_positions); opposite breakout resets to 1 in the new direction.
- **Source:** Translated from StatOasis Market Edge Analysis MQL5 EA (prior bar high/low, hold, MaxOrders=5).

**No SL/TP/limit/stop inside the node;** architecture uses bias output only.

---

## Technical Specifications (BasicBreakout)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | PositionMode | LONG_SHORT | LONG_SHORT, LONG_ONLY, or SHORT_ONLY. LONG_ONLY clamps negative→0; SHORT_ONLY clamps positive→0. |
| `max_positions` | int | 5 | Maximum stacked level (1..max_positions long or -1..-max_positions short). Matches EA MaxOrders. |

Optional future extension: `lookback: int = 1` to use the bar N bars ago as "prior" (canonical is 1).

### Output Values

- **Integer in [-max_positions, max_positions].**
- **Positive (1..max_positions):** Long; magnitude = stacked long count (same-direction breakout adds one, cap at max_positions).
- **Negative (-1..-max_positions):** Short; magnitude = stacked short count.
- **0:** Warmup bar(s), or mode clamp (LONG_ONLY when would be short; SHORT_ONLY when would be long).
- **Inside bar:** Output = previous output (hold); no reset.

### Column Naming

Include `mode` and `max_positions` in `params`. Example: `basicbreakout_signal_D_mode_long_short_max_5`.

---

## Formula / Rules (BasicBreakout)

Let `prev_high` and `prev_low` be the high and low of the **previous** bar. On the **current** bar:

**Raw bar signal (strict inequalities):**
```
IF close > prev_high  → raw = 1
ELSE IF close < prev_low → raw = -1
ELSE → raw = 0   (inside bar)
```

**Stateful stacked signal:**
```
last_signal is state from previous bar (init 0).

IF raw == 0:
    stacked = last_signal   (hold)
ELIF raw == 1:
    IF last_signal <= 0:  stacked = 1
    ELSE:                 stacked = min(max_positions, last_signal + 1)
ELSE:  (raw == -1)
    IF last_signal >= 0:  stacked = -1
    ELSE:                 stacked = -min(max_positions, -last_signal + 1)

Apply mode:
  IF mode == LONG_ONLY  AND stacked < 0 → signal = 0
  ELIF mode == SHORT_ONLY AND stacked > 0 → signal = 0
  ELSE → signal = stacked

last_signal = signal   (for next bar)
```

Use **strict** inequalities so that `close == prev_high` or `close == prev_low` is inside bar (hold).

---

## State and Streaming (BasicBreakout)

- **State:** `prev_high`, `prev_low` (from the last processed candle); `n_prices` (candle count); **`last_signal`** (int, last output for hold/stack).
- **Warmup:** Need at least 2 candles (current + prior). `front_bad = 2`. During warmup output 0 and do not update `last_signal`; **do** update `prev_high` and `prev_low` from the current candle so the next bar has the correct prior bar.
- **Streaming:** For candle `t`, use candle `t`'s close and candle `t-1`'s high/low. After computing the signal, set `prev_high = candle.high`, `prev_low = candle.low`, `last_signal = signal` for the next call.

No circular buffer needed (only one prior bar).

---

## Implementation Details (BasicBreakout)

- **module_name:** `basicbreakout`
- **output_features:** `['signal']`
- **params:** `{'mode': mode, 'max_positions': max_positions}`
- **front_bad:** 2

### Pseudocode for `_compute_candle`

```
n_prices += 1
IF n_prices < front_bad:
    output.append(0)
    prev_high = candle.high
    prev_low  = candle.low
    return [0]

raw = 1  IF close > prev_high  ELSE -1 IF close < prev_low  ELSE 0

IF raw == 0:
    stacked = last_signal
ELIF raw == 1:
    stacked = 1 if last_signal <= 0 else min(max_positions, last_signal + 1)
ELSE:
    stacked = -1 if last_signal >= 0 else -min(max_positions, -last_signal + 1)

IF mode == LONG_ONLY  AND stacked < 0: signal = 0
ELIF mode == SHORT_ONLY AND stacked > 0: signal = 0
ELSE: signal = stacked

last_signal = signal
prev_high = candle.high
prev_low  = candle.low
output.append(signal)
return [signal]
```

---

## Example Class Skeleton (BasicBreakout)

```python
from typing import List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, PositionMode
from nodes import BiasNode


class BasicBreakout(BiasNode):
    """
    Prior-bar high/low breakout bias node (Market Edge). Rule-based, stateful.
    Holds on inside bar; stacks same-direction breakouts up to max_positions (default 5).
    Output: integer in [-max_positions, max_positions]; 0 for warmup or mode clamp.
    Position mode: LONG_SHORT (default), LONG_ONLY, or SHORT_ONLY.
    """

    def __init__(self, ticker: Ticker, tf: TimeFrame, mode: PositionMode = PositionMode.LONG_SHORT, max_positions: int = 5):
        super().__init__(ticker, tf)
        self.mode = mode
        self.max_positions = max_positions
        self.module_name = 'basicbreakout'
        self.output_features = ['signal']
        self.params = {'mode': mode, 'max_positions': max_positions}
        self.front_bad = 2

        self.prev_high: float = 0.0
        self.prev_low: float = 0.0
        self.n_prices: int = 0
        self.last_signal: int = 0

        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        self.n_prices += 1
        if self.n_prices < self.front_bad:
            self.output.append(0)
            self.prev_high = candle.high
            self.prev_low = candle.low
            return [0]

        if candle.close > self.prev_high:
            raw = 1
        elif candle.close < self.prev_low:
            raw = -1
        else:
            raw = 0

        if raw == 0:
            stacked = self.last_signal
        elif raw == 1:
            stacked = 1 if self.last_signal <= 0 else min(self.max_positions, self.last_signal + 1)
        else:
            stacked = -1 if self.last_signal >= 0 else -min(self.max_positions, -self.last_signal + 1)

        if self.mode == PositionMode.LONG_ONLY and stacked < 0:
            signal = 0
        elif self.mode == PositionMode.SHORT_ONLY and stacked > 0:
            signal = 0
        else:
            signal = stacked

        self.last_signal = signal
        self.prev_high = candle.high
        self.prev_low = candle.low
        self.output.append(signal)
        return [signal]
```

---

## Edge Cases (BasicBreakout)

- **First bar:** No prior bar → warmup, return 0; `last_signal` remains 0.
- **close == prev_high or close == prev_low:** Strict inequalities → inside bar → hold `last_signal`.
- **Repeated same-direction breakouts:** Stack until max_positions (e.g. 5).
- **Opposite breakout:** Reset to 1 in new direction (EA closes opposite side then opens one).
- No division or normalization; no Cython required.

---

## Checklist (BasicBreakout)

- [ ] Inherits from `BiasNode`
- [ ] Calls `super().__init__(ticker, tf)`
- [ ] Sets `module_name = 'basicbreakout'`, `output_features = ['signal']`, `params = {'mode': mode, 'max_positions': max_positions}`
- [ ] Sets `front_bad = 2`; state includes `last_signal`
- [ ] Implements `_compute_candle(candle: Candle) -> List` with hold (inside bar) and stack (same direction, cap max_positions)
- [ ] Handles warmup (return 0)
- [ ] Applies position mode after stacked signal (LONG_ONLY: negative→0; SHORT_ONLY: positive→0)
- [ ] Appends result to `self.output` and returns list of length 1
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`
- [ ] Docstrings note rule-based, stateful, hold + stack; output in [-max_positions, max_positions]; no SL/TP/orders in node
- [ ] PositionMode enum defined in `utils/enums.py` and used for `mode` parameter

---

# Section 2: BasicMR (Mean Reversion)

## Overview

**BasicMR** is the **inverse** of BasicBreakout. Use case: fade prior-bar breakouts (mean reversion).

- When BasicBreakout would output `k` (1..max_positions or -1..-max_positions or 0), BasicMR outputs `-k`.
- Same **hold** and **stack** semantics apply to the negated signal (MR holds inverse level on inside bar; stacks in the opposite direction up to max_positions).

The same **position mode** and **max_positions** apply to the **negated** stacked signal: LONG_ONLY / SHORT_ONLY / LONG_SHORT (e.g. MR with LONG_ONLY outputs 0..max_positions; negative MR output is clamped to 0).

---

## Technical Specifications (BasicMR)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | PositionMode | LONG_SHORT | Same as BasicBreakout; applied to the negated stacked signal. |
| `max_positions` | int | 5 | Passed to inner BasicBreakout; output range is [-max_positions, max_positions] before mode. |

### Output Values

- Negated BasicBreakout stacked value, then mode-filtered: integer in [-max_positions, max_positions] (LONG_ONLY clamps negative→0; SHORT_ONLY clamps positive→0).

### Column Naming

Example: `basicmr_signal_D_mode_long_short_max_5`.

---

## Implementation (BasicMR as Wrapper)

BasicMR **does not duplicate** breakout logic. It holds an instance of BasicBreakout and negates its output.

- **module_name:** `basicmr`
- **output_features:** `['signal']`
- **params:** `{'mode': mode, 'max_positions': max_positions}`
- **front_bad:** 2 (same as inner node)

### Constructor

- Import BasicBreakout.
- `self._breakout = BasicBreakout(ticker, tf, mode=mode, max_positions=max_positions)`
- `self.mode = mode`; `self.max_positions = max_positions`
- Set `module_name`, `output_features`, `params`, `front_bad` as above.
- Call `self.ensure_standardized_columns()`.

### `_compute_candle`

- Call `raw = self._breakout._compute_candle(candle)[0]` (do **not** call `add_candle` on the inner node; only `_compute_candle` so inner state updates without double appends/cache).
- `negated = -int(raw)`  (preserves stacked magnitude: e.g. 3 → -3, -2 → 2).
- Apply mode: if `mode == LONG_ONLY` and `negated < 0` then `signal = 0`; elif `mode == SHORT_ONLY` and `negated > 0` then `signal = 0`; else `signal = negated`.
- Append `signal` to `self.output`, return `[signal]`.

---

## Example Class Skeleton (BasicMR)

```python
from typing import List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame, PositionMode
from nodes import BiasNode
from nodes.basic_breakout import BasicBreakout


class BasicMR(BiasNode):
    """
    Mean-reversion bias node: inverse of BasicBreakout (fade prior-bar breakouts).
    Wraps BasicBreakout and negates stacked output. Same hold/stack semantics; output in [-max_positions, max_positions] (or mode-clamped). Supports same position mode and max_positions.
    """

    def __init__(self, ticker: Ticker, tf: TimeFrame, mode: PositionMode = PositionMode.LONG_SHORT, max_positions: int = 5):
        super().__init__(ticker, tf)
        self.mode = mode
        self.max_positions = max_positions
        self._breakout = BasicBreakout(ticker, tf, mode=mode, max_positions=max_positions)

        self.module_name = 'basicmr'
        self.output_features = ['signal']
        self.params = {'mode': mode, 'max_positions': max_positions}
        self.front_bad = 2

        self.ensure_standardized_columns()

    def _compute_candle(self, candle: Candle) -> List:
        raw = self._breakout._compute_candle(candle)[0]
        negated = -int(raw)

        if self.mode == PositionMode.LONG_ONLY and negated < 0:
            signal = 0
        elif self.mode == PositionMode.SHORT_ONLY and negated > 0:
            signal = 0
        else:
            signal = negated

        self.output.append(signal)
        return [signal]
```

---

## Checklist (BasicMR)

- [ ] Inherits from `BiasNode`
- [ ] Imports and holds `BasicBreakout(ticker, tf, mode=mode, max_positions=max_positions)`; does not call `add_candle` on it
- [ ] Sets `module_name = 'basicmr'`, `output_features = ['signal']`, `params = {'mode': mode, 'max_positions': max_positions}`, `front_bad = 2`
- [ ] In `_compute_candle`: calls `_breakout._compute_candle(candle)[0]`, negates (stacked value), applies mode (LONG_ONLY: negative→0; SHORT_ONLY: positive→0), appends and returns
- [ ] Docstring states that BasicMR wraps BasicBreakout and negates stacked output; same hold/stack; supports same position mode and max_positions
- [ ] Calls `self.ensure_standardized_columns()` at end of `__init__`

---

## References

- **Base spec:** [docs/bias_nodes/base_bias_node_specs.md](../base_bias_node_specs.md)
- **Rule-based example:** `nodes/donchian_channel.py`
- **Breakout pattern:** [docs/bias_nodes/base_bias_node_specs.md](../base_bias_node_specs.md) Pattern 5 (Rule-Based Breakout Strategy)
- **Spec style:** [docs/bias_nodes/to-do/supretrend_cross.md](supretrend_cross.md)
