# Vault User Guide

**Vault roots:** default prop firm tree is `vault/`; personal trading defaults to `vault_personal/` at the repo root. Env vars and `vault_profile` / explicit paths are documented in [[Vault/vault]]. Examples below use the prop tree unless noted.

## Short version

Use the vault for persisted feature and portfolio state:

- saved signed-signal features
- working ensemble structure
- frozen portfolio snapshots

Do not use it for candles, live cache artifacts, or current forecasts.

## Common tasks

### Initialize the vault

```python
from ensemble.vault_manager import initialize_vault

initialize_vault("vault")  # or None for prop default; use vault_personal / resolve_vault_personal() for personal
```

### Create an ensemble directory

```python
from ensemble.vault_manager import create_ensemble_directory
from utils.core.enums import Direction, TimeFrame

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

### Save a portfolio snapshot

```python
portfolio_id = global_portfolio.save_to_vault(
    fit_start=train_query.start,
    fit_end=train_query.end,
    vault_root="vault",
)
```

## Research workflow

1. Research the idea in `feature_research` using the canonical three-phase model: `exploration -> validation -> portfolio_addition`.
2. Treat `docs/SaaS/robustness_tests/` as the workflow source of truth; local module names are still migrating toward that structure.
3. If the idea is continuous, convert it into a native signed-signal node before production.
4. Save the signed-signal feature to the vault only after it has survived the individual research phases.
5. Build and evaluate portfolios.
6. Save a snapshot when you need a frozen deployment artifact.

## Related

- [[Vault/architecture]]
- [[Vault/vault]]
- [[Ensemble/weight_layer]] - Manual `hierarchy_equal` groups match vault folder names
- [[Vault/portfolio_snapshot_usage]]
- [[Deployment/live_cache_refresh]]
