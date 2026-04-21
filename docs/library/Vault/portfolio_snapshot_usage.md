# Portfolio Snapshot Usage

> [!summary]
> Use this flow when you want to persist a fitted `GlobalPortfolio`, materialize its predictions for research or live workflows, and later reload the exact same fitted portfolio from the vault.

Related: [[Vault/architecture]], [[Vault/user_guide]], [[Vault/portfolio_snapshots_and_predictions]], [[Vault/vault]], [[Cache/architecture]], [[Ensemble/portfolio]].

Paths below use the prop tree (`vault/...`) as examples; pass `vault_root=` / use `vault_personal/...` for the personal vault ([[Vault/vault]]).

## What gets persisted

When you call `GlobalPortfolio.save_to_vault(...)`:

- a new immutable snapshot is written under `<vault_root>/portfolio_snapshots/<portfolio_id>/`
- `snapshot.json` stores the fitted portfolio state and fit window
- frozen copies of the referenced ensemble files are copied into the snapshot folder

This means snapshot reloads do not depend on the mutable working vault contents.

## Typical research workflow

```python
from ensemble.portfolio import (
    GlobalPortfolio,
    PortfolioCacheQuery,
    PortfolioWorld,
    TFPortfolio,
    load_global_portfolio_snapshot,
    materialize_global_portfolio_predictions,
)
from ensemble.vault_manager import load_ensemble_from_vault
from utils.cache.central_cache_models import ArtifactScope
from utils.core.enums import TimeFrame

ensemble = load_ensemble_from_vault(
    "vault/M/buy_hold/buy_hold_long",
    refit=True,
    target_volatility=0.15,
)
ensemble.use_cache = True
ensemble.retry_on_cache_miss = False
for model in ensemble.base_models.values():
    model.use_cache = True

tf_portfolio = TFPortfolio(
    ensembles=[ensemble],
    trading_timeframe=TimeFrame.M,
    target_volatility=0.15,
    max_position_pct=3.5,
    use_cache=True,
)
global_portfolio = GlobalPortfolio(
    tf_portfolios=[tf_portfolio],
    max_position_pct=3.5,
)

train_query = PortfolioCacheQuery(...)
val_query = PortfolioCacheQuery(...)
test_query = PortfolioCacheQuery(...)
instrument_returns = ...

global_portfolio.fit_from_cache(train_query, instrument_returns)

portfolio_id = global_portfolio.save_to_vault(
    fit_start=train_query.start,
    fit_end=train_query.end,
    vault_root="vault",
)

materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=train_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.TRAIN,
    research_run_id="wf_2026_03",
    scope=ArtifactScope.LIVE,
)
materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=val_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.VAL,
    research_run_id="wf_2026_03",
    scope=ArtifactScope.LIVE,
)
materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=test_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.TEST,
    research_run_id="wf_2026_03",
    scope=ArtifactScope.LIVE,
)

reloaded = load_global_portfolio_snapshot(
    portfolio_id=portfolio_id,
    vault_root="vault",
)
positions = reloaded.predict_from_cache(test_query)
```

## Live workflow

For live inference, the pattern is the same:

1. Fit or select the deployed `GlobalPortfolio`.
2. Save one immutable snapshot and keep the returned `portfolio_id`.
3. Put that `portfolio_id` into `deployment/config/live_cache_refresh.json` if you want LIVE candle writes to keep it materialized automatically.
4. Materialize live predictions with `world=PortfolioWorld.LIVE`.
5. Use `portfolio_id` as the stable join key for monitoring, dashboards, and audits.

Example:

```python
materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=live_query,
    portfolio_id=portfolio_id,
    world=PortfolioWorld.LIVE,
    scope=ArtifactScope.LIVE,
)
```

## Directory quick reference

Snapshot files:

```text
<vault_root>/portfolio_snapshots/<portfolio_id>/snapshot.json
<vault_root>/portfolio_snapshots/<portfolio_id>/ensembles/<TF>/<ensemble_name>/ensemble_config.json
<vault_root>/portfolio_snapshots/<portfolio_id>/ensembles/<TF>/<ensemble_name>/features/*.json
```

Materialized outputs:

```text
.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet
.cache/trading_algo/central_cache/materialized/<scope>/base_models/<timeframe>/<ensemble_name>/<feature_name>__<model_id>.parquet
```

## Cleanup behavior

Use `prune_inactive_base_model_materializations(...)` when you want the base-model materialization tree to match the current working vault.

```python
from ensemble.portfolio import prune_inactive_base_model_materializations
from utils.cache.central_cache_models import ArtifactScope

cleanup = prune_inactive_base_model_materializations(
    vault_root="vault",
    scope=ArtifactScope.LIVE,
)
```

Rules:

- active base-model identities are derived from the current working vault
- stale base-model parquet files are deleted
- historical portfolio-level parquet files are kept
- vault snapshots are not deleted by this cleanup

## Operational rules

- `fit()` and `fit_from_cache()` do not write snapshots automatically
- snapshot ids are hash-derived from semantic portfolio state, not wall-clock write time
- use `PortfolioWorld.VAL` for validation materialization
- pass `research_run_id` when you want to group rows from the same research run
- rerunning materialization for the same row key updates rows instead of duplicating them

## Common mistakes

- Do not expect snapshot reloads to read the current working vault. They read the frozen copies inside the snapshot folder.
- Do not use raw strings like `"validation"` for `world`; use `PortfolioWorld.VAL` or the exact stored value `val`.
- Do not expect cleanup to delete historical portfolio parquet files. Cleanup only targets stale base-model materializations.
