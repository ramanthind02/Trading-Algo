# Vault User Guide

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

initialize_vault("vault")
```

### Create an ensemble directory

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

### Save a feature

Only native signed-signal bias nodes should be saved to the vault.

```python
model_id = base_model.save_to_vault(ensemble_dir)
```

### Load a feature back

```python
from ensemble.vault_manager import load_feature_base_models

models = load_feature_base_models(
    ensemble_dir="vault/D/commodity_breakout_long",
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

1. Research the idea in `feature_research`.
2. If it is continuous, convert the idea into a native signed-signal node before production.
3. Save the signed-signal feature to the vault.
4. Build and evaluate portfolios.
5. Save a snapshot when you need a frozen deployment artifact.

## Related

- [[Vault/architecture]]
- [[Vault/vault]]
- [[Vault/portfolio_snapshot_usage]]
- [[Deployment/live_cache_refresh]]
