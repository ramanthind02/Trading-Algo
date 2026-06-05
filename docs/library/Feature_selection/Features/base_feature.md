# Base Feature

> [!summary]
> A base feature is one production-ready bias-node contract plus one feature column used by ensembles.

## Current contract

| Type | Output | Production status |
|---|---|---|
| Native signed-signal bias node | `-1`, `0`, `+1` | Canonical |
| Continuous bias node | Float | Research only |

`research.feature.config.ResearchConfig` enforces this in `__post_init__`: it accepts
only `FeatureType.SIGNED_SIGNAL`. Continuous-node binning / EDA research has moved to the
standalone `research.feature.binning` package (run `python -m research.feature binning`).

## Runtime

- Input: OHLCV candles
- Output: signed signal column `-1/0/+1`
- No production-time bin fitting or wrapper translation step

The legacy runtime base-model classes (`BinningModelBase`, `ContinuousBinningModel`,
`RuleBasedModel`) have been **deleted** — the old `feature_selection/base_models/base_model.py`
stub module no longer exists (only `features/models/feature_base_model.py` survives). There is
no production-time bin/threshold fitting path anymore.

## Naming

- Column names should reflect the native node recipe.
- Changing the production recipe should create a new feature identity.

## Related

- [[bias_nodes/creating_nodes]]
- [[Feature_selection/pipeline]]
- [[Vault/architecture]]

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
