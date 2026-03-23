# Bias Node API

## Module

- [`nodes/__init__.py`](../../../nodes/__init__.py)

## Public Surface

- `LookbackContribution`
- `LookbackWindow` alias
- `BiasNode.lookback_param_names`
- `BiasNode.hardcoded_lookbacks`
- `BiasNode.cold_rebuild_buffer_ratio`
- `BiasNode.lookback_contributions()`
- `BiasNode.max_lookback()`
- `BiasNode.cold_rebuild_candle_count()`

## Types

### `LookbackContribution`

Machine-readable warmup contribution for a bias node.

Fields:

- `label: str`
- `bars: int`

Notes:

- `LookbackWindow` is a compatibility alias for the same dataclass
- contributions are bar counts, not timedeltas

## `BiasNode` Warmup Metadata

Class variables:

- `lookback_param_names: frozenset[str]`
  - names inside `self.params` that should be interpreted as rolling-window lookbacks
- `hardcoded_lookbacks: tuple[tuple[str, int], ...]`
  - fixed bar-count windows that are not represented directly in `self.params`
- `cold_rebuild_buffer_ratio: float`
  - extra fractional buffer appended on top of `max_lookback()` for stateless rebuilds

Methods:

- `lookback_contributions() -> tuple[LookbackContribution, ...]`
  - returns all declared warmup windows for the node instance
  - includes contributions from `lookback_param_names`, `hardcoded_lookbacks`, `_extra_lookback_contributions()`, and `front_bad` when present
- `max_lookback() -> int`
  - returns the largest single bar-count contribution needed to warm the node from scratch
- `cold_rebuild_candle_count() -> int`
  - returns the full stateless rebuild window
  - current default behavior is `max_lookback + ceil(max_lookback * cold_rebuild_buffer_ratio)` with a minimum of `1`

## Authoring Rules

- Bias nodes must register every rolling-window constructor param that drives warmup in `lookback_param_names`
- Fixed or derived windows must be declared in `hardcoded_lookbacks` or `_extra_lookback_contributions()`
- `front_bad` remains part of the contract, but callers should size warmup from `max_lookback()` / `cold_rebuild_candle_count()`, not from `front_bad` alone

## Operational Notes

- Cache rebuilds use the stateless cold-rebuild invariant: instantiate a fresh node, stream `cold_rebuild_candle_count()` candles ending at the requested boundary, then trim the artifact back to the requested coverage
- This metadata exists so cache orchestration, multi-timeframe prediction, and future live refresh paths can size warmup windows without hardcoding node-specific rules

## Related

- [Creating Bias Nodes](../../library/bias_nodes/creating_nodes.md) - authoring guide and examples
- [Central Cache API](../cache/central_cache.md) - cache lifecycle and refresh contract
