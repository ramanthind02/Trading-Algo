# Vault Architecture

> [!summary]
> The vault is the persistence layer for selected feature definitions and working ensemble membership. Portfolio *predictions* are persisted separately, in the central cache (not the vault) — see [[Vault/portfolio_snapshots_and_predictions]].

## Ownership boundary

The vault owns:

- working ensemble directories (nested or flat; see below)
- feature control files

The central cache owns:

- candles
- derived runtime artifacts (bias-node outputs, EWSD)
- materialized portfolio and base-model predictions

> [!note] No immutable portfolio snapshot today
> Earlier drafts of this doc described a `GlobalPortfolio.save_to_vault(...)` /
> `load_global_portfolio_snapshot(...)` pair writing immutable snapshots under
> `<vault_root>/portfolio_snapshots/`. That layer is **not** in the current code. A
> deployed portfolio is re-assembled from its working-vault ensemble directories
> (`build_global_portfolio_from_ensemble_dirs`), and its predictions are materialized
> into the central cache keyed by a caller-supplied `portfolio_id`. See
> [[Vault/portfolio_snapshots_and_predictions]].

### Ensemble directory layout

**Preferred (nested):** `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_leaf>/`

Group folder names must be members of `VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES` in `ensemble/vault/constants.py`. Each feature JSON carries a `weight_hierarchy_group` field that should match its folder when using nested storage.

**Legacy (flat):** `<vault_root>/<TF>/<ensemble_leaf>/` — still discovered by `ensemble.vault.discovery` and portfolio research config; new work should prefer nested paths for consistency with the weight layer.

See [[Vault/vault]] for the current list of group folder names.

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
- `weight_hierarchy_group`
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

`model_id` is derived from the signed-signal bias-node spec (`generate_model_id` in `ensemble/vault/manager.py`), so semantic duplicates collide deterministically.

## Write path

1. Choose a working ensemble directory (`create_ensemble_directory`).
2. Save a native signed-signal feature (`BaseModel.save_to_vault(...)` → `add_feature_to_ensemble(...)`).
3. Load ensembles from the vault and fit/predict portfolios against those saved definitions.
4. Materialize portfolio predictions into the central cache when needed.

## Prediction materialization path

Portfolio and base-model predictions are written by
`materialize_global_portfolio_predictions(...)` under the central cache, **not** the
vault:

```text
.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet
.cache/trading_algo/central_cache/materialized/<scope>/base_models/<TF>/<ensemble_name>/<identity_hash>.parquet
```

This keeps mutable runtime prediction state out of the working vault. See
[[Vault/portfolio_snapshots_and_predictions]] and [[Cache/architecture]].

## Related

- [[Vault/user_guide]]
- [[Vault/vault]]
- [[Vault/portfolio_snapshots_and_predictions]]
- [[Cache/architecture]]

> _Verified against current code via CodeGraph on 2026-06-07._
