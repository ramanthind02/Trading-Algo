# Vault

> [!summary]
> The vault stores persisted feature definitions and working ensemble membership. Portfolio *predictions* are materialized separately into the central cache, not the vault (see [[Vault/portfolio_snapshots_and_predictions]]).

## Prop vs personal roots

The repo supports **three default vault roots** (same on-disk layout under each):

| Profile | Default directory (under repo root) | Environment variable |
|--------|----------------------------------------|-------------------------|
| **Prop** (futures firm) | `vault/` | `TRADING_ALGO_VAULT_PROP`, or legacy `TRADING_ALGO_VAULT_ROOT` if `VAULT_PROP` is unset |
| **Personal** | `vault_personal/` | `TRADING_ALGO_VAULT_PERSONAL` |
| **CFD Prop** | `vault_cfd_prop/` | `TRADING_ALGO_VAULT_CFD_PROP` |

- Any API that takes an explicit `vault_root` path still wins over profile defaults.
- Repo-relative paths (for example in `PortfolioResearchConfig.ensemble_dirs`, `VaultSaveConfig.existing_ensemble_dir`, and hierarchy helpers) must use the correct **top-level folder** (`vault/...` vs `vault_personal/...` vs `vault_cfd_prop/...`).
- Feature research: set `VaultSaveConfig.vault_profile` to `"personal"` or `"cfd_prop"` when `vault_root` is omitted, or run `python -m research.feature.save_feature_to_vault --vault-profile <profile>`.
- Portfolio discovery scans each existing root returned by `default_vault_discovery_dirnames()` (prop, personal, and cfd_prop defaults). If the same ensemble **leaf** name exists in multiple trees, the first root in that discovery order keeps the entry.
- Feature-research portfolio admission (`PortfolioSourceConfig`) discovers from **one** vault via `vault_profile` (`prop`/`futures_prop` → `vault/`, `personal` → `vault_personal/`, `cfd_prop` → `vault_cfd_prop/`) or an explicit `vault_root`. `research.portfolio.config.load_config()` defaults to the prop vault only.

## What lives here

- **Working ensembles** under the timeframe folders `<vault_root>/D/`, `<vault_root>/W/`, and `<vault_root>/M/` (for example `vault/D/` for the prop tree).
- **Nested layout (current):** `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_leaf>/`
  - `<ensemble_leaf>` is the usual `{ensemble_name}_{direction}` directory (for example `buy_hold_long`).
  - `<weight_hierarchy_group>` is one of the manual global-weight-layer buckets. Folder names must be members of `VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES` in `ensemble/vault/constants.py` and must match `weight_hierarchy_group` in each feature JSON. The current built-in registry is:
    - `mean_reversion_indices`
    - `buy_hold`
    - `es_tlt`
    - `seasonal`
    - `momentum`
    - `trend_following`
    - `momentum_gc`
    - `crude_oil_mr`
    - `gc_breakout`
    - `cl_breakout`
    - `breakout`
    - `silver_mr`
    - `silver_trend`
  - This built-in set is a `frozenset`; user-defined sleeves are additive and persisted to `ensemble/vault/custom_sleeves.json` via `add_custom_sleeve(name)`. Call `valid_weight_hierarchy_groups()` at runtime to get the full set (built-ins + custom). Treat both as the source of truth, not this list.
- **Legacy flat layout (still supported):** `<vault_root>/<TF>/<ensemble_leaf>/` — discovery and cache preflight resolve both shapes.
- `features/*.json` control files (each tagged with `weight_hierarchy_group` for the weight hierarchy).

The vault does not own candles, mutable runtime cache artifacts, or materialized portfolio predictions. Portfolio/base-model predictions are written to the central cache (`.cache/trading_algo/central_cache/materialized/<scope>/`) — see [[Vault/portfolio_snapshots_and_predictions]].

## Current feature contract

- `model_type` is `signed_signal`
- `bias_node_spec` points directly to a native discrete bias node
- `weight_hierarchy_group` identifies the manual group used by the global `WeightLayer` hierarchy (and should match the on-disk group folder when using nested layout)
- one feature file currently corresponds to one saved base-model entry

## Typical workflow

1. Create an ensemble directory (optionally under a `weight_hierarchy_group` folder).
2. Save a native signed-signal feature.
3. Load ensembles from the vault for research or portfolio construction.
4. Materialize portfolio predictions into the central cache when you need a persisted, queryable prediction set (see [[Vault/portfolio_snapshots_and_predictions]]).

## Model IDs

Model IDs are generated from the signed-signal bias-node spec so semantic duplicates collide deterministically.

## Related

- [[Vault/architecture]]
- [[Vault/user_guide]]
- [[Vault/portfolio_snapshots_and_predictions]]
- [[Ensemble/weight_layer]]
- [[Cache/architecture]]

> _Verified against current code via CodeGraph on 2026-06-07._
