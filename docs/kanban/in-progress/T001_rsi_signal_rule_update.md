# T001 — RSI Signal Rule Update

## Goal
Update the RSI signal bias node to use cross-based entries, explicit strategy modes, and fixed-exit support while preserving existing RSI computation and cache behavior.

## Context / References
- `docs/library/bias_nodes/base_bias_node_specs.md`
- `docs/library/bias_nodes/to-do/rsi_signal.md`
- `nodes/rsi_signal.py`
- `docs/api/nodes.md`

## Scope
In scope:
- Update `RSISignal` constructor signature and rule logic.
- Add deterministic unit tests for cross-based entries and fixed exits.
- Update `docs/api/nodes.md` with the new RSISignal API and behavior.

Out of scope:
- Changing RSI calculation math or switching to composition with `nodes.rsi.RSI`.
- Adding trend filters or additional indicators.

## Interfaces (must match)
- Modify: `nodes/rsi_signal.py` — update `RSISignal.__init__` params and rule logic.
- Modify: `tests/test_new_bias_nodes.py` — add unit tests for new modes and fixed exits.
- Modify: `docs/api/nodes.md` — document RSISignal API/behavior.

## Data Contracts
- Output remains discrete `-1`, `0`, `1` signals.
- Standardized column naming uses `module_name`, `output_features`, and params.

## Dependencies
- `nodes/rsi_signal.py`
- `tests/test_new_bias_nodes.py`
- `docs/api/nodes.md`

## Invariants / Constraints
- Deterministic: same candles and params ⇒ same output.
- No lookahead: signal changes depend on current/previous RSI only.
- Warmup returns `0` until `rsi_period` candles.

## Acceptance tests
1. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_mode_fixed_exit -v`
2. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_short_mode_fixed_exit -v`
3. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_short_mode_fixed_exit -v`

## Definition of done
- [ ] Tests added under `tests/test_new_bias_nodes.py`
- [ ] Docs updated under `docs/api/nodes.md`
- [ ] `pytest tests/test_new_bias_nodes.py::TestRSISignal -v` passes

## Notes
- Fixed exit defaults to 5 bars; for long-short mode it exits to flat.
