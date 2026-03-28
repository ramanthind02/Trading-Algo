# Vault

> [!summary] What Is the Vault?
> Centralized, validated storage for fitted [[base_model|base models]] and feature control files.
> Organized by **timeframe → ensemble → feature**. In the current working-vault implementation, each feature file effectively stores one saved base-model variant plus its fitted state.

Start here:

- [[Vault/architecture]] — in-depth explanation of vault ownership, working-vault invariants, and portfolio snapshots
- [[Vault/user_guide]] — shorter task-oriented usage guide

---

## Directory Structure

```
vault/
├── D/                                   # TimeFrame.D
│   ├── commodity_breakout_long/         # {strategy_name}_{direction}
│   │   └── features/
│   │       ├── rsi_signal_D.json
│   │       └── momentum_signal_D.json
│   └── universal_momentum_short/
│       └── features/
└── W/
    └── momentum_weekly_long/
        └── features/
```

- Timeframe nesting: `D`, `W`, `M`
- Ensemble naming: `{strategy_name}_{direction}` (e.g. `commodity_breakout_long`)
- One JSON per feature — current working-vault code effectively enforces one saved base-model variant per file

---

## Model ID Auto-Generation

IDs are derived from binning type + hyperparameters + per-model bias params — no manual naming needed.

| Binning model | Generated ID |
|---|---|
| `rule_based + lookback=2` | `rule_based_lookback_2` |
| `continuous_binning(n_bins=5) + lookback=14` | `continuous_binning_5_lookback_14` |

Duplicate IDs (same hyperparams) raise `ValueError`.

---

## Quick Workflows

### Initialize Vault
```python
from ensemble.vault_manager import initialize_vault
initialize_vault('vault')
```

### Create Ensemble Directory
```python
from ensemble.vault_manager import create_ensemble_directory
from utils.core.enums import TimeFrame, Direction

ensemble_dir = create_ensemble_directory(
    vault_root='vault', timeframe=TimeFrame.D,
    ensemble_name='commodity_breakout', direction=Direction.LONG
)
```

### Fit and Save a Base Model
```python
from feature_selection.base_models import BaseModel, QuantileBinningModel

base_model = BaseModel(
    bias_node_spec={'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
    binning_model=QuantileBinningModel(n_bins=3, strategy='long'),
    ticker=Ticker.ES
)
base_model.fit(candles_df, target_data)
model_id = base_model.save_to_vault(ensemble_dir)  # -> 'quantile_binning_3'
```

### Load Models
```python
from ensemble.vault_manager import load_feature_base_models

models = load_feature_base_models(
    ensemble_dir='vault/D/commodity_breakout_long',
    feature_column='rsi_signal_D_lookback_14',
    fitted_only=True
)
base_model = models['quantile_binning_3']
```

### List / Remove
```python
from ensemble.vault_manager import list_ensembles, list_features, remove_base_model_variant

list_ensembles('vault')                        # DataFrame: name, timeframe, direction, n_features
list_features('vault/D/commodity_breakout_long')  # DataFrame: feature, n_base_models, n_fitted

remove_base_model_variant(
    ensemble_dir='vault/D/commodity_breakout_long',
    feature_column='rsi_signal_D_lookback_14',
    model_id='quantile_binning_3'
)
```

---

## Key Rules

- **Strategy must match direction** — long models → `*_long/` ensembles. Mismatch raises `ValueError`.
- **Feature naming**: `{module}_{feature}_{timeframe}_{param_key}_{param_value}` (e.g. `rsi_signal_D_lookback_14`). Timeframe in name must match ensemble timeframe.
- **Model ID uniqueness** — auto-generated; duplicate hyperparams rejected.
- Call `fit()` before `save_to_vault()` — feature column is set during fit.

---

## Control File Schema (JSON)

Each feature JSON contains:
- `feature_name` — canonical feature key (for example `rsi_signal_D`)
- `bias_node_spec` — top-level shared spec (`module_name`, `timeframes`) with no params
- `tickers` — training ticker universe for the ensemble feature
- `base_models` — list shape retained by schema, with exactly one base-model entry in the current working-vault implementation
- `created_at`, `updated_at` — metadata timestamps

Each base model entry contains:
- `model_id`, `model_name`
- `bias_node_params` (params moved from top-level spec into per-model entries)
- `binning_model_type`, `strategy`, `binning_model_params`
- `requires_fit`, `is_fitted`, `fitted_params`

### `is_fit` Flag + Selection Metadata

```python
metadata = {
    'is_fit': True,
    'version': '2.0.0',
    'selection_method': 'walkforward_stability',   # or 'permutation_test', 'manual'
    'selection_hyperparams': {'min_stability_score': 0.7, 'n_folds': 5, 'metric': 'sharpe'}
}
```

## Portfolio snapshots

`GlobalPortfolio.save_to_vault(fit_start, fit_end, vault_root="vault")` writes a frozen portfolio snapshot under `vault/portfolio_snapshots/<portfolio_id>/`.

The snapshot contains:

- `snapshot.json` with the portfolio's semantic state and fit window
- frozen copies of the referenced vault ensemble files under `ensembles/<TF>/<ensemble_name>/`

`load_global_portfolio_snapshot(portfolio_id, vault_root="vault")` reloads the portfolio from those frozen copies instead of reading the mutable working vault.

See [[Vault/portfolio_snapshots_and_predictions]] for the snapshot and materialization layout.

---

## See Also

- [[Vault/architecture]] — in-depth vault architecture and lifecycle
- [[Vault/user_guide]] — practical vault usage guide
- [[Cache/architecture]] — runtime cache design and invalidation behavior
- [[Cache/user_guide]] — practical cache usage guide
- [[base_model]] — BaseModel composition and binning strategy
- [[portfolio]] — how vault ensembles are loaded into Portfolio
- [[monitoring]] — strategy decay monitoring store (signal/target vectors, CUSUM, rolling Sharpe)

---

## Portfolio Auto-Load

`Portfolio` supports vault auto-load:

```python
Portfolio(ensembles=None, vault_root="vault", ensemble_names=None)
```

- `ensembles=None`: load all ensemble directories under `vault/{D,W,M}/*`
- `ensemble_names=[...]`: filter by full directory names (for example `mean-reversion_indices_long`)
- Explicit `ensembles=` (including `[]`) bypasses auto-load entirely
