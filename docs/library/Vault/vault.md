# Vault

> [!summary] What Is the Vault?
> Centralized, validated storage for fitted [[base_model|base models]] and feature control files.
> Organized by **timeframe → ensemble → feature**. Each feature has one canonical JSON control file holding all model variants.

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
- One JSON per feature — all model variants (fitted + unfitted) coexist in that file

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
- `base_models` — list of model variants
- `created_at`, `updated_at` — metadata timestamps

Each base model entry contains:
- `model_id`, `model_name`
- `bias_node_params` (params moved from top-level spec into per-model entries)
- `binning_model_type`, `strategy`, `binning_model_params`
- `requires_fit`, `is_fitted`, `fitted_params`
- optional `members` list (may be empty or omitted)

Each member entry contains:
- `member_name`
- `binning_model_type`, `binning_model_params`
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

---

## See Also

- [[base_model]] — BaseModel composition and binning strategy
- [[portfolio]] — how vault ensembles are loaded into Portfolio

---

## Portfolio Auto-Load

`Portfolio` supports vault auto-load:

```python
Portfolio(ensembles=None, vault_root="vault", ensemble_names=None)
```

- `ensembles=None`: load all ensemble directories under `vault/{D,W,M}/*`
- `ensemble_names=[...]`: filter by full directory names (for example `mean-reversion_indices_long`)
- Explicit `ensembles=` (including `[]`) bypasses auto-load entirely
