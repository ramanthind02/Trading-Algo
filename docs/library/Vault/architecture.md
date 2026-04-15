# Vault Architecture

> [!summary]
> The vault is the persistence layer for selected feature definitions, ensemble membership, and immutable portfolio snapshots.

## Ownership boundary

The vault owns:

- working ensemble directories
- feature control files
- portfolio snapshots

The central cache owns:

- candles
- derived runtime artifacts
- live materialized predictions

## Feature control files

A feature control file stores one persisted production feature for one ensemble.

Current invariant:

- one file
- one native signed-signal bias-node spec
- one saved base-model row

The important fields are:

- `feature_name`
- `bias_node_spec`
- `tickers`
- `base_models`
- timestamps

Each base-model row stores:

- `model_id`
- `model_name`
- `model_type: "signed_signal"`
- `strategy`
- `bias_node_spec`

## Identity

Stable identity comes from:

- timeframe
- ensemble directory
- feature name
- deterministic `model_id`

`model_id` is derived from the signed-signal bias-node spec.

## Write path

1. Choose a working ensemble.
2. Save a native signed-signal feature.
3. Load or fit portfolios against those saved definitions.
4. Snapshot the final portfolio when needed.

## Snapshot path

Portfolio snapshots live under:

`vault/portfolio_snapshots/<portfolio_id>/`

They store frozen copies of the referenced ensemble files so deployment does not depend on the mutable working vault.

## Related

- [[Vault/user_guide]]
- [[Vault/vault]]
- [[Vault/portfolio_snapshots_and_predictions]]
- [[Cache/architecture]]
