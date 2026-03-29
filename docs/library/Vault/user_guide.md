# Vault User Guide

> [!tip]
> Use this page when you want the practical “how do I use the vault?” version without reading the deeper architecture page first.

Related: [[Vault/architecture]], [[Vault/vault]], [[Vault/portfolio_snapshot_usage]], [[Vault/portfolio_snapshots_and_predictions]], [[Deployment/live_cache_refresh]].

## The short version

Use the vault for persisted model state:

- selected feature files
- fitted base-model variants
- working ensemble structure
- frozen portfolio snapshots

Do **not** use the vault for:

- candles
- fresh/stale runtime artifacts
- current live forecasts

Those belong in the central cache.

## What the vault is for

You usually touch the vault in four situations:

1. creating or organizing working ensembles
2. saving fitted base models
3. loading ensembles back for research or portfolio construction
4. saving a deployable `GlobalPortfolio` snapshot

## What the vault looks like

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

Quick mental model:

- `D/W/M` folders = working ensembles
- `features/*.json` = saved feature + base-model definition files
- `portfolio_snapshots/` = frozen deployable portfolios

Current working-vault rule:

- one feature file currently corresponds to one saved base-model variant
- older descriptions that imply many active variants per feature file are stale for the current implementation

## Common tasks

### 1. Initialize the vault

```python
from ensemble.vault_manager import initialize_vault

initialize_vault("vault")
```

Use this once when setting up a new vault root.

### 2. Create an ensemble directory

```python
from ensemble.vault_manager import create_ensemble_directory
from utils.core.enums import Direction, TimeFrame

ensemble_dir = create_ensemble_directory(
    vault_root="vault",
    timeframe=TimeFrame.D,
    ensemble_name="commodity_breakout",
    direction=Direction.LONG,
)
```

This creates the working location where feature control files will live.

### 3. Save a fitted base model

Typical workflow:

1. fit the base model in research
2. save it into the correct working ensemble

```python
model_id = base_model.save_to_vault(ensemble_dir)
```

What gets persisted:

- feature identity
- bias-node spec
- binning-model params
- fitted params
- model id and metadata

### 4. Load base models back from the vault

```python
from ensemble.vault_manager import load_feature_base_models

models = load_feature_base_models(
    ensemble_dir="vault/D/commodity_breakout_long",
    feature_column="rsi_signal_D_lookback_14",
    fitted_only=True,
)
```

Use this when you want one feature’s saved base-model variants.

In practice today, that usually means one saved variant per feature file in the working vault.

### 5. Load a full ensemble from the vault

```python
from ensemble.vault_manager import load_ensemble_from_vault

ensemble = load_ensemble_from_vault(
    "vault/M/buy_hold_long",
    refit=False,
    target_volatility=0.15,
)
```

Use this when you want the full working-vault ensemble reconstructed for research, testing, or portfolio use.

### 6. Remove a base-model variant

```python
from ensemble.vault_manager import remove_base_model_variant

remove_base_model_variant(
    ensemble_dir="vault/D/commodity_breakout_long",
    feature_column="rsi_signal_D_lookback_14",
    model_id="quantile_binning_3",
)
```

Use this when a variant should no longer be part of the working vault.

Important downstream effect:

- once removed, future base-model materialization cleanup can prune runtime parquet files for that inactive identity

### 7. Save a portfolio snapshot

```python
portfolio_id = global_portfolio.save_to_vault(
    fit_start=train_query.start,
    fit_end=train_query.end,
    vault_root="vault",
)
```

Use this when a fitted `GlobalPortfolio` should become:

- reproducible
- reloadable
- deployable
- auditable

### 8. Reload a portfolio snapshot

```python
from ensemble.portfolio import load_global_portfolio_snapshot

portfolio = load_global_portfolio_snapshot(
    portfolio_id=portfolio_id,
    vault_root="vault",
)
```

This reloads from the frozen snapshot files, not from the current working vault.

## How to think about working vault vs snapshot

### Working vault

Use the working vault for:

- what is active right now
- what should be part of the next refresh/preflight
- what researchers are adding/removing/updating

### Snapshot

Use a snapshot for:

- what is deployed
- what should reload exactly later
- what should stay stable even if the working vault changes

If you only remember one distinction, remember this one.

## Typical workflows

### Research workflow

1. Update/prepare candles in the cache.
2. Refresh cache coverage for the working-vault ensembles.
3. Fit base models and save them into the vault.
4. Load ensembles from the vault.
5. Fit/evaluate the portfolio.
6. Save a snapshot if the portfolio should become a stable artifact.

### Deployment workflow

1. Choose the fitted `GlobalPortfolio`.
2. Save a snapshot and capture `portfolio_id`.
3. Put that `portfolio_id` into `deployment/config/live_cache_refresh.json`.
4. Keep LIVE candles current.
5. Let runtime materialize live outputs automatically.

## What the vault controls indirectly

Even though the vault is not the runtime cache, it still drives runtime behavior.

The working vault influences:

- which ensemble dirs are refreshed by `ensure_vault_cache_coverage(...)`
- which tickers/features/models are considered active
- which base-model materialization files should still exist

This is why editing the working vault has downstream runtime consequences.

## What not to do

- Do not store candles in the vault.
- Do not treat the working vault as an immutable deployment artifact.
- Do not expect live candle updates to rewrite vault files automatically.
- Do not confuse a materialized base-model parquet file with the fitted base model itself.
- Do not point deployment directly at the mutable working vault when a saved `portfolio_id` exists.

## Easy checklist

When using the vault day to day:

1. Keep the working vault accurate.
2. Save fitted state only when it should persist.
3. Remove variants that are no longer part of the active working set.
4. Save snapshots for deployment or reproducibility.
5. Use `portfolio_id` as the stable deployment handle.

## If something looks wrong

### “My live forecasts changed after I edited the vault”

That can be expected if:

- the working vault changed the active ensemble set
- live refresh used the updated working-vault identities for bias refresh or cleanup

Check whether you meant to change:

- the mutable working vault
- or the deployed `portfolio_id`

### “I removed a model from the vault but still see runtime files”

Run the relevant materialization flow or live refresh cycle again. Cleanup happens during refresh/materialization, not at the moment you edit the JSON file.

### “Reloaded portfolio behavior does not match the current working vault”

That can be correct. A snapshot reload uses frozen copied files, not the current working-vault files.

## Related

- [[Vault/architecture]] — deeper explanation of the vault’s role
- [[Vault/vault]] — quick reference page
- [[Vault/portfolio_snapshot_usage]] — snapshot workflow examples
- [[Cache/user_guide]] — runtime cache usage
