# Trading-Algo Library Index

> [!note]
> Pipeline (conceptual):
> `Candles (OHLCV)` -> `Bias nodes` -> `DiversifiedEnsemble` -> `WeightLayer` -> `Portfolio` -> `PositionSizer`

Production uses native signed-signal bias nodes that emit `-1/0/+1`. Continuous nodes are for research unless they are later reimplemented as native discrete nodes.

## Core docs

- [[workflow]] - Daily research loop and agent presets
- [[cursor_sub_bridge]] - Use ChatGPT Pro / Claude Max in Cursor via Sub Bridge

## Cache

- [[Cache/architecture]] - Central-cache design and lifecycle
- [[Cache/user_guide]] - Practical cache usage

## Bias nodes

- [[bias_nodes/index]] - Hub for node authoring and composition
- [[bias_nodes/creating_nodes]] - How to implement a node
- [[bias_nodes/bias_node_arch]] - Bias-node architecture notes

## Feature selection

- [[Feature_selection/pipeline]] - Current `feature_research` phase model (`exploration -> validation -> portfolio_addition`)
- [[SaaS/robustness_tests/index]] - Canonical robustness workflow that the local `feature_research` docs now mirror
- [[Feature_selection/Features/base_feature]] - Base feature contract
- [[Feature_selection/Features/feature_model]] - Feature model overview
- [[Feature_selection/Features/rule_based]] - Native discrete feature notes
- [[Feature_selection/exploration]] - Exploration artifacts and hand-off to parameter lock
- [[Feature_selection/permutation_testing]] - Exploration-phase permutation subset
- [[Feature_selection/parameter_sensitivity]] - Parameter sensitivity and plateau-selection guide
- [[Feature_selection/validation]] - Validation stage reference
- [[Feature_selection/portfolio_addition]] - Portfolio-addition stage reference
- [[SaaS/robustness_tests/portfolio_holdout]] - Downstream portfolio holdout source of truth
- [[SaaS/robustness_tests/monitoring]] - Downstream monitoring source of truth

## Data

- [[Data/norgate]] - Norgate data integration and OHLCV sourcing

## Ensemble

- [[Ensemble/base_model]] - Base-model concepts
- [[Ensemble/weight_layer]] - Weighting layer
- [[Ensemble/portfolio]] - Portfolio layer, multi-TF orchestration, and the portfolio-addition gate (older docs/code may still say `inclusion`)

## Vault

Default prop tree: `vault/<D|W|M>/<group>/<ensemble>/` (manual weight-hierarchy groups); personal: `vault_personal/...` — see [[Vault/vault]]. Legacy flat `<vault_root>/<TF>/<ensemble>/` remains supported.

- [[Vault/architecture]] - Vault ownership and invariants
- [[Vault/user_guide]] - Saving and loading features and ensembles
- [[Vault/vault]] - Quick reference
- [[Vault/monitoring]] - Monitoring store
- [[Vault/portfolio_snapshots_and_predictions]] - Snapshot and prediction materialization
- [[Vault/portfolio_snapshot_usage]] - Snapshot workflows

## Deployment

- [[Deployment/production]] - Forecast server and production training
- [[Deployment/live_cache_refresh]] - Live cache refresh
- [[Deployment/cython]] - Cython build notes

## Research reading order

1. [[workflow]]
2. [[bias_nodes/index]] -> [[bias_nodes/creating_nodes]]
3. [[Feature_selection/pipeline]]
4. [[SaaS/robustness_tests/index]]
5. [[Feature_selection/exploration]]
6. [[Feature_selection/validation]]
7. [[Vault/user_guide]]

## Naming convention

`{module}_{feature}_{timeframe}_{param}_{value}`

Example: `rsi_signal_D_lookback_14`
