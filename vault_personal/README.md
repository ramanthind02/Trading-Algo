# Vault - Working Model Storage

This vault stores validated trading features and their saved base-model state.

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
- A `base_models` list shape that currently holds one saved base-model variant in the working-vault implementation
- Fitted or unfitted model configuration for that saved variant

## Usage

See:

- `docs/library/Vault/architecture.md`
- `docs/library/Vault/user_guide.md`

### Saving from feature research

To save a research model to the vault after you are satisfied with results, set `vault_save` (`VaultSaveConfig`: direction, ensemble target, optional tickers override, `dry_run`; the feature always uses `eval_bias_spec` from the same config) in `feature_research.config.load_config()`, then run:

```bash
python -m feature_research.save_feature_to_vault
```
