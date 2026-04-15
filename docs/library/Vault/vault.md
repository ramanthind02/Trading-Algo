# Vault

> [!summary]
> The vault stores persisted feature definitions, working ensemble membership, and frozen portfolio snapshots.

## What lives here

- ensemble directories under `vault/<TF>/<ensemble_name>/`
- `features/*.json` control files
- immutable portfolio snapshots under `vault/portfolio_snapshots/<portfolio_id>/`

The vault does not own candles or mutable runtime cache artifacts.

## Current feature contract

- `model_type` is `signed_signal`
- `bias_node_spec` points directly to a native discrete bias node
- one feature file currently corresponds to one saved base-model entry

## Typical workflow

1. Create an ensemble directory.
2. Save a native signed-signal feature.
3. Load ensembles from the vault for research or portfolio construction.
4. Save a `GlobalPortfolio` snapshot when you need a frozen deployable artifact.

## Model IDs

Model IDs are generated from the signed-signal bias-node spec so semantic duplicates collide deterministically.

## Related

- [[Vault/architecture]]
- [[Vault/user_guide]]
- [[Vault/portfolio_snapshots_and_predictions]]
- [[Cache/architecture]]
