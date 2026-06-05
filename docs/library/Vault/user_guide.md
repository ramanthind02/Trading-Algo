# Vault User Guide

**Vault roots:** default prop firm tree is `vault/`; personal trading defaults to `vault_personal/` at the repo root. Env vars and `vault_profile` / explicit paths are documented in [[Vault/vault]]. Examples below use the prop tree unless noted.

## Short version

Use the vault for persisted feature and ensemble definitions:

- saved signed-signal features
- working ensemble structure

Do not use it for candles, live cache artifacts, current forecasts, or materialized
portfolio predictions (those live in the central cache — see
[[Vault/portfolio_snapshots_and_predictions]]).

## Common tasks

### Initialize the vault

```python
from ensemble.vault_manager import initialize_vault

initialize_vault("vault")  # or None for prop default; use vault_personal / resolve_vault_personal() for personal
```

### Create an ensemble directory

```python
from ensemble.vault_manager import create_ensemble_directory
from lib.core.enums import Direction, TimeFrame

# Nested under a manual weight-hierarchy group (recommended for new ensembles)
ensemble_dir = create_ensemble_directory(
    timeframe=TimeFrame.D,
    ensemble_name="commodity_breakout",
    direction=Direction.LONG,
    weight_hierarchy_group="momentum",
)
# → vault/D/momentum/commodity_breakout_long

# Omit weight_hierarchy_group for the legacy flat path vault/D/<ensemble_leaf>/
```

See [[Vault/vault]] for the list of group folder names.

### Save a feature

Only native signed-signal bias nodes should be saved to the vault.

```python
model_id = base_model.save_to_vault(ensemble_dir)
```

### Load a feature back

```python
from ensemble.vault_manager import load_feature_base_models

models = load_feature_base_models(
    ensemble_dir="vault/D/mean_reversion_indices/mr_indices_long",
    feature_name="rsi_signal_D",
    fitted_only=True,
)
```

### Materialize portfolio predictions

There is no `GlobalPortfolio.save_to_vault(...)`. To persist predictions, fit the portfolio
and materialize its outputs into the central cache under a `portfolio_id` you choose:

```python
from ensemble.portfolio import PortfolioWorld, materialize_global_portfolio_predictions
from lib.cache import ArtifactScope

materialize_global_portfolio_predictions(
    portfolio=global_portfolio,
    query=train_query,
    portfolio_id="my_portfolio_2024",
    world=PortfolioWorld.TRAIN,
    scope=ArtifactScope.LIVE,
)
```

See [[Vault/portfolio_snapshot_usage]] for the full workflow.

## Research workflow

1. Research the idea in `research.feature` using the canonical three-phase model: `exploration -> validation -> portfolio_addition`.
2. Treat `docs/SaaS/robustness_tests/` as the workflow source of truth; local module names are still migrating toward that structure.
3. If the idea is continuous, convert it into a native signed-signal node before production.
4. Save the signed-signal feature to the vault only after it has survived the individual research phases.
5. Build and evaluate portfolios.
6. Materialize portfolio predictions into the central cache when you need a persisted, queryable prediction set.

## Related

- [[Vault/architecture]]
- [[Vault/vault]]
- [[Ensemble/weight_layer]] - Manual `hierarchy_equal` groups match vault folder names
- [[Vault/portfolio_snapshot_usage]]
- [[Deployment/live_cache_refresh]]

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
