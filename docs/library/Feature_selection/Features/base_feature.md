# Base Feature

> [!summary]
> A base feature is one production-ready bias-node contract plus one feature column used by ensembles.

## Current contract

| Type | Output | Production status |
|---|---|---|
| Native signed-signal bias node | `-1`, `0`, `+1` | Canonical |
| Continuous bias node | Float | Research only |

## Runtime

- Input: OHLCV candles
- Output: signed signal column `-1/0/+1`
- No production-time bin fitting or wrapper translation step

## Naming

- Column names should reflect the native node recipe.
- Changing the production recipe should create a new feature identity.

## Related

- [[bias_nodes/creating_nodes]]
- [[Feature_selection/pipeline]]
- [[Vault/architecture]]
