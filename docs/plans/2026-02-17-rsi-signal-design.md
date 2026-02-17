# RSI Signal Bias Node Redesign

Date: 2026-02-17

## Summary
Update `nodes/rsi_signal.py` to implement rule-based RSI signals with explicit strategy modes (`long`, `short`, `long-short`) and fixed-exit support. Keep RSI calculation and cache behavior intact, but replace signal logic with a state machine based on threshold crossings and bar-count exits.

## Context / References
- `docs/library/bias_nodes/base_bias_node_specs.md`
- `docs/library/bias_nodes/to-do/rsi_signal.md`
- `nodes/rsi_signal.py`
- `docs/api/nodes.md`

## Goals
- Support `strategy_mode` values: `long`, `short`, `long-short`.
- Enter on RSI threshold **crosses** (oversold/overbought), not raw level checks.
- Hold until opposite threshold cross or fixed bar exit (default 5) fires.
- In `long-short`, fixed exit forces flat (0), then waits for next cross.

## Non-Goals
- Changing RSI math or moving to composition with `nodes.rsi.RSI`.
- Changing cache mechanics beyond parameter metadata.
- Adding trend filters or additional indicators.

## Proposed API
RSISignal constructor parameters will be updated to:
- `rsi_period: int = 14`
- `oversold: float = 30.0`
- `overbought: float = 70.0`
- `strategy_mode: str = "long"` (`long`, `short`, `long-short`)
- `exit_policy: str = "threshold_or_bars"` (`threshold`, `threshold_or_bars`)
- `exit_bars: int = 5`

## State & Data Flow
- Continue using existing RSI computation state (`close_buffer`, `upsum`, `dnsum`, `prev_close`).
- Add:
  - `position: int` in `{ -1, 0, 1 }`.
  - `bars_in_position: int` (increments when in position, resets on entry/exit).
  - `prev_rsi: Optional[float]` for cross detection.

Per candle:
1. Compute RSI (existing logic).
2. Warmup: return `0`, reset `position` and `bars_in_position`.
3. Detect crosses with `prev_rsi`:
   - Cross below oversold: `prev_rsi > oversold` and `rsi <= oversold`.
   - Cross above overbought: `prev_rsi < overbought` and `rsi >= overbought`.
4. Update position per `strategy_mode`:
   - `long`: enter on oversold cross; exit on overbought cross or fixed-exit.
   - `short`: enter on overbought cross; exit on oversold cross or fixed-exit.
   - `long-short`: enter long on oversold cross, enter short on overbought cross; fixed-exit goes flat.
5. Increment `bars_in_position` while `position != 0`.

Fixed exit:
- Enabled when `exit_policy == "threshold_or_bars"`.
- If `bars_in_position >= exit_bars`, exit to `0` regardless of RSI.
- Resets on each new entry.

## Validation / Errors
- Validate `strategy_mode` and `exit_policy` values; raise `ValueError` on invalid strings.
- Validate `exit_bars >= 1`.

## Docs & Tests
- Update `docs/api/nodes.md` to include `RSISignal` parameters and behavior.
- Update unit tests in `tests/test_new_bias_nodes.py` (or create focused unit tests) covering:
  - Long/short/long-short cross-based entries.
  - Fixed-exit after `exit_bars` and reset on re-entry.
  - Flat behavior after fixed exit in long-short mode.

## Risks
- Downstream expectations for `rsisignal` column naming and param keys. Ensure standardized metadata is updated and consistent with existing naming conventions.
