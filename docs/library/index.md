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

- [[Feature_selection/pipeline]] - End-to-end research flow
- [[Feature_selection/Features/base_feature]] - Base feature contract
- [[Feature_selection/Features/feature_model]] - Feature model overview
- [[Feature_selection/Features/rule_based]] - Native discrete feature notes
- [[Feature_selection/Phase_1_IS/eda]] - In-sample EDA
- [[Feature_selection/Phase_1_IS/permutation_testing]] - In-sample permutation
- [[Feature_selection/Phase_2_WF/walkforward]] - Walk-forward reference
- [[Feature_selection/Phase_3_OOS/oos_validation]] - OOS validation

## Data

- [[Data/norgate]] - Norgate data integration and OHLCV sourcing

## Ensemble

- [[Ensemble/base_model]] - Base-model concepts
- [[Ensemble/weight_layer]] - Weighting layer
- [[Ensemble/portfolio]] - Portfolio layer

## Vault

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
4. [[Feature_selection/Phase_1_IS/permutation_testing]]
5. [[Feature_selection/Phase_2_WF/walkforward]]
6. [[Vault/user_guide]]

## Naming convention

`{module}_{feature}_{timeframe}_{param}_{value}`

Example: `rsi_signal_D_lookback_14`
