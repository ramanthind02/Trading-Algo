# Vault

> [!summary]
> The vault stores persisted feature definitions, working ensemble membership, and frozen portfolio snapshots.

## Prop vs personal roots

The repo supports **two default vault roots** (same on-disk layout under each):

| Profile | Default directory (under repo root) | Environment variable |
|--------|----------------------------------------|-------------------------|
| **Prop** (firm) | `vault/` | `TRADING_ALGO_VAULT_PROP`, or legacy `TRADING_ALGO_VAULT_ROOT` if `VAULT_PROP` is unset |
| **Personal** | `vault_personal/` | `TRADING_ALGO_VAULT_PERSONAL` |

- Any API that takes an explicit `vault_root` path still wins over profile defaults.
- Repo-relative paths (for example in `PortfolioResearchConfig.ensemble_dirs`, `VaultSaveConfig.existing_ensemble_dir`, and hierarchy helpers) must use the correct **top-level folder** (`vault/...` vs `vault_personal/...`).
- Feature research: set `VaultSaveConfig.vault_profile` to `"personal"` when `vault_root` is omitted, or run `python -m feature_research.save_feature_to_vault --vault-profile personal`.
- Portfolio discovery scans each existing root returned by `default_vault_discovery_dirnames()` (prop and personal defaults). If the same ensemble **leaf** name exists in both trees, the first root in that discovery order keeps the entry.
- Feature-research portfolio admission (`PortfolioSourceConfig`) discovers from **one** vault via `vault_profile` (`prop` → `vault/`, `personal` → `vault_personal/`) or an explicit `vault_root`. `portfolio_research.config.load_config()` defaults to the prop vault only.

## What lives here

- **Working ensembles** under the timeframe folders `<vault_root>/D/`, `<vault_root>/W/`, and `<vault_root>/M/` (for example `vault/D/` for the prop tree).
- **Nested layout (current):** `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_leaf>/`
  - `<ensemble_leaf>` is the usual `{ensemble_name}_{direction}` directory (for example `buy_hold_long`).
  - `<weight_hierarchy_group>` is one of the manual global-weight-layer buckets (folder names must match `weight_hierarchy_group` in each feature JSON):
    - `mean_reversion_indices`
    - `buy_hold`
    - `es_tlt`
    - `seasonal`
    - `momentum`
- **Legacy flat layout (still supported):** `<vault_root>/<TF>/<ensemble_leaf>/` — discovery and cache preflight resolve both shapes.
- `features/*.json` control files (each tagged with `weight_hierarchy_group` for the weight hierarchy).
- Immutable portfolio snapshots under `<vault_root>/portfolio_snapshots/<portfolio_id>/` (prop and personal trees each have their own `portfolio_snapshots/` if you use both).

The vault does not own candles or mutable runtime cache artifacts.

## Current feature contract

- `model_type` is `signed_signal`
- `bias_node_spec` points directly to a native discrete bias node
- `weight_hierarchy_group` identifies the manual group used by the global `WeightLayer` hierarchy (and should match the on-disk group folder when using nested layout)
- one feature file currently corresponds to one saved base-model entry

## Typical workflow

1. Create an ensemble directory (optionally under a `weight_hierarchy_group` folder).
2. Save a native signed-signal feature.
3. Load ensembles from the vault for research or portfolio construction.
4. Save a `GlobalPortfolio` snapshot when you need a frozen deployable artifact.

## Model IDs

Model IDs are generated from the signed-signal bias-node spec so semantic duplicates collide deterministically.

## Related

- [[Vault/architecture]]
- [[Vault/user_guide]]
- [[Vault/portfolio_snapshots_and_predictions]]
- [[Ensemble/weight_layer]]
- [[Cache/architecture]]
