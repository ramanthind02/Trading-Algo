# Vault System User Guide

## Overview

The Vault is a centralized storage system for validated trading features and base models. It provides a clean, organized way to manage hundreds of features across multiple ensembles with automatic model ID generation and comprehensive validation.

## Quick Start

### 1. Initialize Vault

```python
from ensemble.vault_manager import initialize_vault

initialize_vault('vault')
```

### 2. Create an Ensemble

```python
from utils.enums import TimeFrame, Direction
from ensemble.vault_manager import create_ensemble_directory

ensemble_dir = create_ensemble_directory(
    vault_root='vault',
    timeframe=TimeFrame.D,
    ensemble_name='commodity_breakout',
    direction=Direction.LONG
)
```

### 3. Create and Save a Base Model

```python
from feature_selection.base_models import BaseModel, QuantileBinningModel
from utils.enums import Ticker, TimeFrame

# Create binning model
binning_model = QuantileBinningModel(n_bins=3, selection_metric='sortino', strategy='long')

# Define bias node specification
bias_spec = {
    'module_name': 'rsi',
    'timeframes': [TimeFrame.D],
    'params': {'lookback': 14}
}

# Create base model
base_model = BaseModel(
    bias_node_spec=bias_spec,
    binning_model=binning_model,
    ticker=Ticker.ES
)

# Fit the model
base_model.fit(candles_df, target_data)

# Save to vault (model ID auto-generated)
model_id = base_model.save_to_vault(ensemble_dir)
print(f"Saved model: {model_id}")  # 'quantile_binning_3'
```

## Core Concepts

### Directory Structure

```
vault/
├── D/                               # Daily timeframe
│   ├── commodity_breakout_long/     # Ensemble: {name}_{direction}
│   │   └── features/                 # Feature control files
│   │       ├── rsi_signal_D_lookback_14.json
│   │       └── momentum_signal_D_lookback_20.json
│   └── universal_momentum_short/
│       └── features/
│           └── rsi_signal_D_lookback_14.json
└── W/                               # Weekly timeframe
    └── momentum_weekly_long/
        └── features/
            └── momentum_signal_W_lookback_10.json
```

**Key Points:**
- **Timeframe-based nesting**: Ensembles organized by timeframe (D, W, M)
- **Ensemble naming**: `{strategy_name}_{direction}` (e.g., `commodity_breakout_long`)
- **Feature files**: One JSON file per feature containing all base model variants

### Model ID Auto-Generation

Model IDs are automatically generated from binning model type and hyperparameters:

- `ContinuousBinningModel(n_bins=3)` → `continuous_binning_3`
- `ContinuousBinningModel(n_bins=5)` → `continuous_binning_5`

No manual naming required - the system ensures uniqueness.

## Common Workflows

### Adding Multiple Model Variants

```python
# Add 3-bin variant
binning_3 = QuantileBinningModel(n_bins=3, strategy='long')
base_3 = BaseModel(bias_node_spec=bias_spec, binning_model=binning_3, ticker=Ticker.ES)
base_3.fit(candles_df, target_data)
model_id_3 = base_3.save_to_vault(ensemble_dir)  # 'quantile_binning_3'

# Add 5-bin variant
binning_5 = QuantileBinningModel(n_bins=5, strategy='long')
base_5 = BaseModel(bias_node_spec=bias_spec, binning_model=binning_5, ticker=Ticker.ES)
base_5.fit(candles_df, target_data)
model_id_5 = base_5.save_to_vault(ensemble_dir)  # 'quantile_binning_5'
```

### Loading Models from Vault

```python
from ensemble.vault_manager import load_feature_base_models

# Load all variants for a feature
models = load_feature_base_models(
    ensemble_dir='vault/D/commodity_breakout_long',
    feature_column='rsi_signal_D_lookback_14'
)

# Access specific model
base_model = models['quantile_binning_3']

# Check if fitted
if base_model.binning_model.is_fitted_:
    # Ready for prediction
    predictions = base_model.predict(candles_df, strategy='long')
else:
    # Fit it first
    base_model.fit(candles_df, target_data)
    base_model.update_fitted_params_in_vault(
        ensemble_dir='vault/D/commodity_breakout_long',
        model_id='quantile_binning_3',
        train_start='2020-01-01',
        train_end='2024-12-31'
    )
```

### Production Deployment

```python
from ensemble.vault_manager import get_all_base_model_names, load_feature_base_models

ensemble_dir = 'vault/D/commodity_breakout_long'

# Get all model names
model_names = get_all_base_model_names(ensemble_dir)
# ['rsi_signal_D_lookback_14::quantile_binning_3', ...]

# Load all fitted models
fitted_models = {}
for model_name in model_names:
    feature_column, model_id = model_name.split('::')
    models = load_feature_base_models(ensemble_dir, feature_column, fitted_only=True)
    if model_id in models:
        fitted_models[model_name] = models[model_id]

# Generate predictions
for model_name, base_model in fitted_models.items():
    # Stream candles
    for candle in candle_stream:
        base_model.add_candle(candle, TimeFrame.D)
    
    # Predict
    predictions = base_model.predict(candles_df, strategy='long')
```

## Vault Management Functions

### Listing Ensembles

```python
from ensemble.vault_manager import list_ensembles

df = list_ensembles('vault')
# Returns DataFrame with: ensemble_name, timeframe, direction, n_features, path
```

### Listing Features

```python
from ensemble.vault_manager import list_features

df = list_features('vault/D/commodity_breakout_long')
# Returns DataFrame with: feature_column, n_base_models, n_fitted, created_at, updated_at
```

### Removing Model Variants

```python
from ensemble.vault_manager import remove_base_model_variant

remove_base_model_variant(
    ensemble_dir='vault/D/commodity_breakout_long',
    feature_column='rsi_signal_D_lookback_14',
    model_id='quantile_binning_3'
)
```

### Validation

```python
from ensemble.vault_manager import validate_ensemble_directory

# Validate single ensemble
validate_ensemble_directory('vault/D/commodity_breakout_long')

# Validate entire vault
from ensemble.vault_manager import validate_vault
validate_vault('vault')
```

## Important Rules

### Strategy Matching

**Models must match ensemble direction:**
- Long models (`strategy='long'`) → Long ensembles (`*_long/`)
- Short models (`strategy='short'`) → Short ensembles (`*_short/`)

The system validates this automatically when saving - mismatches raise `ValueError`.

### Feature Column Naming

Feature columns must follow the standardized format:
- `{module}_{feature}_{timeframe}_{param_key}_{param_value}`
- Example: `rsi_signal_D_lookback_14`

The timeframe in the feature column must match the ensemble timeframe.

### Model ID Uniqueness

Model IDs are auto-generated and must be unique within each feature. If you try to save a duplicate model ID, the system raises `ValueError`.

## Architecture Notes

### BaseModel Composition

The `BaseModel` class:
- **Owns bias nodes**: Instantiates and manages bias nodes from `bias_node_spec`
- **Owns binning model**: Contains a `BinningModel` instance via composition
- **Handles workflow**: Streams candles, extracts features, delegates binning

This separation allows:
- Independent testing of binning logic
- Reusable binning models across features
- Clear separation of concerns

### Control File Structure

Each feature has a JSON control file containing:
- `bias_node_spec`: Everything needed to reconstruct bias nodes
- `base_models`: List of all model variants (fitted and unfitted)
- Metadata: `created_at`, `updated_at`, `feature_column`

Fitted and unfitted models coexist in the same file - no duplication.

### Multi-Member Schema (Required)

As of v2.0.0, all base model configurations must include a `members` array:

```python
base_model_config = {
    'name': 'my_feature_model',
    'model_type': 'continuous_binning',
    'feature_column': 'rsi_signal_D_lookback_14',
    'strategy': 'long',
    'constructor_params': {'n_bins': 3},
    'members': [
        {'member_id': 'member_1', 'bin_index': 0},
        {'member_id': 'member_2', 'bin_index': 1},
        {'member_id': 'member_3', 'bin_index': 2},
    ]
}
```

Each member represents a bin in the model with:
- `member_id`: Unique identifier for the member
- `bin_index`: Which bin this member represents

The `members` array is required and must be non-empty. Legacy schemas without `members` are rejected.

### Selection Method and Hyperparameters

When saving a fitted ensemble, you can persist the selection method used:

```python
metadata = {
    'is_fit': True,
    'version': '2.0.0',
    'selection_method': 'walkforward_stability',
    'selection_hyperparams': {
        'min_stability_score': 0.7,
        'n_folds': 5,
        'metric': 'sharpe'
    }
}
```

Supported selection methods:
- `walkforward_stability`: Walk-forward stability-based selection
- `permutation_test`: Permutation testing-based selection
- `manual`: Manual researcher selection

Selection hyperparameters vary by method and are stored in the control file for reproducibility.

## Troubleshooting

### "Strategy mismatch" Error

**Problem**: Trying to save a long model to a short ensemble (or vice versa).

**Solution**: Ensure `base_model.binning_model.strategy` matches the ensemble direction.

### "Model ID already exists" Error

**Problem**: Trying to save a model with the same hyperparameters as an existing model.

**Solution**: This is expected - model IDs are based on hyperparameters. If you want a different model, change the hyperparameters.

### "Feature column not set" Error

**Problem**: Calling `save_to_vault()` before `fit()`.

**Solution**: Call `fit()` first to set the `feature_column` from bias node outputs.

## Next Steps

- See `docs/to-do/vault_specs.md` for detailed specifications
- Check `tests/test_vault_system.py` for usage examples
- Review `ensemble/vault_manager.py` for API documentation
