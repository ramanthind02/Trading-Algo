# Vault Architecture

> [!summary]
> The vault is the persisted model-definition layer of the system. It stores selected feature control files, fitted base-model state, working ensemble structure, and immutable portfolio snapshots. It does not own mutable runtime candles or fresh/stale derived artifacts; those belong to the central cache.

Related: [[Vault/user_guide]], [[Vault/vault]], [[Vault/portfolio_snapshots_and_predictions]], [[Cache/architecture]], [[Deployment/live_cache_refresh]].

## Purpose

The vault exists to answer:

- which features and base-model variants have been selected?
- what fitted base-model state should survive runs?
- which ensemble directories are currently active?
- which frozen portfolio snapshot is the deployable source of truth?

This makes the vault the persistence boundary between research selection and runtime inference.

## Ownership boundary

The vault owns persisted model intent:

- working ensemble directories under `vault/<TF>/<ensemble_name>/`
- feature control files under `features/*.json`
- fitted base-model params stored in those control files
- current working-vault membership
- immutable portfolio snapshots under `vault/portfolio_snapshots/<portfolio_id>/`

The vault does **not** own:

- runtime candles
- fresh/stale artifact lifecycle state
- materialized live forecasts
- runtime revision lineage

Those belong to the central cache.

## Canonical implementation

Primary code lives in:

- `ensemble/vault_manager.py`
- `ensemble/portfolio_vault.py`
- `ensemble/portfolio.py` for the public snapshot entrypoints

Important entrypoints:

- `initialize_vault(...)`
- `create_ensemble_directory(...)`
- `load_feature_base_models(...)`
- `remove_base_model_variant(...)`
- `load_ensemble_from_vault(...)`
- `ensure_vault_cache_coverage(...)`
- `GlobalPortfolio.save_to_vault(...)`
- `load_global_portfolio_snapshot(...)`

## Directory model

The working vault is organized as:

```text
vault/
├── D/
│   └── <ensemble_name>/
│       ├── ensemble_config.json
│       └── features/
│           └── *.json
├── W/
├── M/
└── portfolio_snapshots/
    └── <portfolio_id>/
```

The key hierarchy is:

1. timeframe
2. ensemble directory
3. feature control file
4. base-model entries inside that feature file

This is why the vault is a model-definition store rather than a raw file dump.

## Working-vault ensemble structure

Each working ensemble directory contains:

- `ensemble_config.json`
- `features/*.json`

The ensemble directory encodes:

- the trading timeframe
- the ensemble name / direction identity
- the selected feature set for that ensemble

The feature files then define the fitted base-model variants attached to each feature.

## Feature control file schema

A feature control file is the canonical persisted record for one feature in one ensemble.

It typically stores:

- `feature_name`
- `created_at`
- `updated_at`
- `bias_node_spec`
- `tickers`
- `base_models`

Each `base_models` entry stores one variant, including:

- `model_id`
- `model_name`
- `bias_node_params`
- `binning_model_type`
- `strategy`
- `binning_model_params`
- `requires_fit`
- `is_fitted`
- `fitted_params`

Current working-vault invariant:

- the file still uses a `base_models` list shape
- but the working-vault code currently enforces **one base-model entry per feature file**
- older docs that describe “all variants in one feature file” are stale for the current implementation

That means the practical identity model today is:

- one feature file
- one saved base-model variant
- one persisted fitted-state payload for that variant

## Identity rules

The vault relies on stable identities.

### Ensemble identity

The working ensemble is identified by:

- timeframe
- ensemble directory name

### Feature identity

The feature is identified by:

- the feature JSON path
- `feature_name`

### Base-model identity

A base-model variant is identified by:

- `model_id`

This matters because downstream materialization and cleanup logic use:

- timeframe
- ensemble name
- feature name
- model id

as the active base-model identity key.

`model_id` is deterministic from normalized model type, key hyperparameters, and per-model bias params. That is why save/load/materialization can treat it as a stable identity rather than a user-chosen label.

## Write path: base models

The base-model save flow is:

1. create or choose a working ensemble directory
2. fit a base model in research
3. save it into the appropriate feature control file
4. persist fitted params and metadata

The vault is the long-lived result of that workflow. It is not rebuilt every time candles change.

Important invariant:

- live inference uses the already fitted vault state
- it does not refit and rewrite the vault on each bar

## Load path: ensembles

`load_ensemble_from_vault(...)` reconstructs an ensemble from the working vault by:

- reading the ensemble config
- reading feature control files
- reconstructing saved base-model variants
- restoring fitted params where present
- attaching vault metadata used later by materialization and cleanup

Implementation detail worth knowing:

- ensemble loading is a translation layer
- the working-vault feature files are read and converted into the unified control-file shape expected by the ensemble loader
- this is why the working-vault schema and the ensemble control-file schema are related but not identical

That vault metadata is important because later systems need to know:

- which working ensemble directory a loaded ensemble came from
- what its timeframe and ensemble name are
- what its base-model identities are

This is how runtime materialization can map predictions back to stable working-vault identities.

## Monitoring store

The vault also has a monitoring-adjacent persistence role.

Saved fitted models can seed monitoring data under the working vault so decay analysis stays aligned with the saved feature/model identity. That monitoring store is adjacent to the working vault identity model, not part of the runtime cache materialization tree.

## Working vault vs portfolio snapshots

These are different layers and should not be conflated.

### Working vault

The working vault is mutable and researcher-managed.

It answers:

- which features/models are active right now?
- which ensemble dirs should be refreshed?
- which base-model materializations should still exist?

### Portfolio snapshots

Portfolio snapshots are immutable.

They are written under:

```text
vault/portfolio_snapshots/<portfolio_id>/
```

Each snapshot stores:

- `snapshot.json`
- frozen copies of referenced `ensemble_config.json`
- frozen copies of referenced `features/*.json`

The snapshot hash is derived from semantic portfolio state, not wall-clock write time, so identical saved state intentionally reuses the same `portfolio_id`.

The purpose of snapshots is different from the working vault:

- the working vault is today's mutable selection surface
- the snapshot is a frozen deployable or replayable portfolio artifact

## Why snapshots include copied ensemble files

Snapshots copy the referenced live ensemble files so that reloads do not depend on the current working vault.

This protects against:

- later edits to working-vault feature files
- later additions/removals of model variants
- drift between deployment and current researcher state

That is what makes `portfolio_id` a stable deployment identity.

## How the vault interacts with the cache

The vault does not perform inference by itself. It works together with the cache.

The vault drives the cache by declaring model intent:

- `ensure_vault_cache_coverage(...)` reads the working-vault feature files
- active ensemble tickers come from the working vault
- active base-model identities come from the working vault
- live refresh manifests point to working-vault ensemble dirs

The cache serves the vault by providing runtime inputs:

- current candles
- fresh bias artifacts
- EWSD volatility lineage
- materialized prediction outputs

This relationship is directional:

- vault = persisted chosen state
- cache = current runtime state

## Research lifecycle

In research, the vault is a write target.

Typical sequence:

1. fit or validate features/base models
2. save selected fitted models into the working vault
3. load ensembles back from the working vault
4. fit and evaluate portfolios
5. save a portfolio snapshot if needed

Research changes the vault intentionally.

Legacy note:

- the vault manager still carries migration helpers for older fragmented or `members`-based schemas
- empty legacy `members` keys can be migrated away
- non-empty legacy `members` payloads are rejected

That is another reason the architecture doc should treat the current working-vault invariant as authoritative rather than older prose.

## Live lifecycle

In live operation, the vault is mostly a read surface.

Typical sequence:

1. keep LIVE candles current in the cache
2. rebuild stale cache artifacts as needed
3. load deployed portfolios from immutable snapshots
4. scan the working vault for currently active base-model identities
5. materialize live outputs
6. prune stale base-model materialization files that no longer match the working vault

Live does not normally rewrite the working vault.

## Cleanup semantics

The working vault controls what is active.

This is why base-model materialization cleanup scans the working vault rather than portfolio snapshots.

Behavior:

- inactive base-model materialization parquet files can be deleted
- historical portfolio materializations are retained
- portfolio snapshots are retained

This preserves deployment/audit history while preventing bloated base-model runtime output trees.

## What does not belong in the vault

Do not use the vault for:

- runtime candles
- stale/fresh cache artifacts
- transient runtime forecast rows
- revision counters
- latest-only deployment status

Those are cache concerns, not vault concerns.

## Design invariants

Keep these rules in mind:

- the working vault is mutable
- portfolio snapshots are immutable
- fitted base-model state belongs in the vault
- runtime inference outputs belong in the cache
- working-vault membership determines active base-model identities
- deployment should point to `portfolio_id`, not to the mutable working vault

## Common mistakes

- Treating the working vault as if it were an immutable deployment artifact
- Expecting live cache refresh to refit and rewrite vault state
- Storing runtime outputs in the vault instead of the cache
- Deleting historical portfolio snapshots during base-model cleanup
- Forgetting that the working vault and the deployed snapshot can intentionally diverge over time

## Related

- [[Vault/user_guide]] — practical day-to-day usage
- [[Vault/vault]] — quick reference for layout and APIs
- [[Vault/portfolio_snapshots_and_predictions]] — snapshot/materialization contract
- [[Cache/architecture]] — runtime cache architecture
