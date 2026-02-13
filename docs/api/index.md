# API Documentation

This folder documents public API surfaces and shared data contracts for the repository.

## How to use
- Start with `_inventory.md` for package boundaries, entrypoints, and cross-module surfaces.
- Use module pages for signatures, behavior, constraints, and minimal examples.

## Index
- [_Inventory](./_inventory.md) - Repository map, script entrypoints, and cross-package symbol surface

## Modules
- [utils.md](./utils.md) - Shared enums, models, helpers, cache, logging, and fast compute interfaces
- [nodes.md](./nodes.md) - Bias-node contracts and major indicator entrypoints
- [base_models.md](./base_models.md) - Base model and binning APIs used by selection and ensemble layers
- [feature_selection.md](./feature_selection.md) - Feature selector, walkforward, and integration entrypoints
- [feature_validator_api.md](./feature_validator_api.md) - Stage-based validation API and report dataclasses
- [ensemble.md](./ensemble.md) - Diversified ensemble, weighting, portfolio, and vault/control interfaces
- [metrics.md](./metrics.md) - Performance/risk/equity metrics and reporting helpers
- [deployment.md](./deployment.md) - Forecast server, connectors, notifier, and training pipeline APIs
- [data_pipeline.md](./data_pipeline.md) - Data cleaning, feature extraction, EDA, and research entrypoints
- [testing_tools.md](./testing_tools.md) - Permutation, robustness, prop-firm simulation, and plotting APIs
