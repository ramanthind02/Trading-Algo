# Vault - Base Model Feature Storage

This vault stores validated trading features and their associated base models.

## Directory Structure

```
vault/
├── D/                    # Daily timeframe ensembles
│   └── {ensemble}_{direction}/
│       └── features/
│           └── {feature_column}.json
├── W/                    # Weekly timeframe ensembles
└── M/                    # Monthly timeframe ensembles
```

## Feature Control Files

Each feature has its own control file (`features/{feature_column}.json`) containing:
- Bias node specification (for feature reconstruction)
- All base model variants for that feature
- Fitted and unfitted model configurations

## Usage

See `docs/to-do/vault_specs.md` for complete documentation.

### Saving from feature research

To save a research model to the vault after you are satisfied with results, set `vault_save` in `feature_research.config.load_config()` (ensemble name, direction, and optional `params_to_save`), then run:

```bash
python -m feature_research.save_to_vault
```
