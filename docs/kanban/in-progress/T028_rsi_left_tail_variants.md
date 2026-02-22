# T028 — RSI Left-Tail Mean-Reversion Variants

## Goal
Add standalone RSI-derived bias nodes engineered for short-term (2-10 period) left-tail mean-reversion behavior on stock indices.

## Context / References
- `docs/library/bias_nodes/base_bias_node_specs.md`
- `docs/plans/2026-02-21-rsi-left-tail-mean-reversion-design.md`
- `nodes/mean_reversion/rsi/rsi.py`
- `utils/functime.py`

## Scope
In scope:
- Add new RSI-derived continuous nodes for lag, tail pressure, streak persistence, and rebound velocity.
- Register canonical node module mappings for dynamic node construction.
- Add unit tests for deterministic behavior and warmup/range contracts.
- Update `docs/api/nodes.md` with new public node entrypoints.

Out of scope:
- Rule-based signal conversion or execution policy changes.
- Cross-layer integration tests and cache population workflows.
- Refactoring existing RSI variant implementations.

## Interfaces (must match)
- Add: `nodes/mean_reversion/rsi/lagged_rsi.py` — `class LaggedRSI(BiasNode)`
- Add: `nodes/mean_reversion/rsi/rsi_left_tail_pressure.py` — `class RSILeftTailPressure(BiasNode)`
- Add: `nodes/mean_reversion/rsi/rsi_left_tail_streak.py` — `class RSILeftTailStreak(BiasNode)`
- Add: `nodes/mean_reversion/rsi/rsi_rebound_velocity.py` — `class RSIReboundVelocity(BiasNode)`
- Add wrappers in `nodes/` for canonical imports.
- Modify: `nodes/_taxonomy.py` to register module keys for new nodes.
- Modify: `docs/api/nodes.md` to document public observable behavior.

## Data Contracts
- Input: `Candle` stream ordered by datetime.
- Output: One continuous scalar feature per node (`signal`) per candle.
- Alignment: strictly causal (current and prior bars only), no lookahead.

## Dependencies
- `utils.fast_nodes` (RSI kernels)
- `nodes.BiasNode` base API
- `numpy` and standard library deques

## Invariants / Constraints
- Deterministic: same candle stream and params => same outputs.
- No lookahead: outputs depend only on current/prior bars.
- Warmup outputs are explicit neutral values per node.
- Feature ranges are bounded where applicable (RSI-like 0-100; streak ratio 0-1).

## Acceptance tests
1. `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py -v`
2. `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_nodes_are_deterministic -v`
3. `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_behavior_on_down_then_rebound_path -v`

### Integration Test Data Contract (required when integration tests are in scope)
- Not in scope for this task.

## Definition of done
- [ ] New node files added under `nodes/mean_reversion/rsi/`
- [ ] Canonical wrappers and taxonomy entries added
- [ ] Unit tests added and passing
- [ ] `docs/api/nodes.md` updated for new entrypoints

## Notes
- Keep parameters and column metadata compatible with existing parser conventions (snake_case module name, camelCase param keys).
- Use short RSI periods by default (2-10 range emphasis) where suitable.
