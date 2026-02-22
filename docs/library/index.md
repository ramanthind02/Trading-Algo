# Trading-Algo Library Index

> [!note] Pipeline
> `Candles (OHLCV)` → `Bias Nodes` → `Base Models` → `DiversifiedEnsemble` → `WeightLayer` → `Portfolio` → `PositionSizer`

## Core Docs

| Doc | Purpose |
|-----|---------|
| [[workflow]] | Daily research loop, agent presets, RAG/memory commands |

## Pipeline Components

### Bias Nodes
- [[bias_nodes/creating_nodes]] — How to implement a new technical indicator node
- [[bias_nodes/norgate]] — Norgate data integration and OHLCV sourcing

### Feature Selection
- [[Feature_selection/pipeline]] — End-to-end feature selection pipeline overview

**Feature Types**
- [[Feature_selection/Features/base_feature]] — Base feature contracts and naming conventions
- [[Feature_selection/Features/continuous_binning]] — Quantile / decision-tree binning strategies
- [[Feature_selection/Features/rule_based]] — Rule-based binary signal models

**Phase 1 — EDA**
- [[Feature_selection/Phase_1_EDA/eda]] — Exploratory data analysis for features

**Phase 2 — IS Screening**
- [[Feature_selection/Phase_2_IS_Screening/permutation_testing]] — Permutation tests (T013–T016)
- [[Feature_selection/Phase_2_IS_Screening/candle_permutation]] — Candle-level permutation for signal validation
- [[Feature_selection/Phase_2_IS_Screening/kfold]] — K-fold cross-validation setup
- [[Feature_selection/Phase_2_IS_Screening/cpcv]] — Combinatorial purged cross-validation

**Phase 3–4 — Walkforward**
- [[Feature_selection/Phase_3_4_Walkforward/walkforward]] — Walk-forward validation and stability
- [[Feature_selection/Phase_3_4_Walkforward/param_stability]] — Parameter sensitivity and stability analysis

### Ensemble
- [[Ensemble/base_model]] — `BinningModelBase` ABC, `get_fitted_vector()`
- [[Ensemble/weight_layer]] — Inverse-correlation weights + FDM formula
- [[Ensemble/portfolio]] — Instrument weights + IDM formula

### Vault
- [[Vault/vault]] — Validated feature storage, JSON control files, `is_fit` flag

### Deployment
- [[Deployment/production]] — REST forecast server, production training pipeline
- [[Deployment/cython]] — Compiling Cython extensions for performance

## Research Reading Order

> [!tip] New Researcher Start Here
> 1. [[workflow]] — understand tooling and daily loop
> 2. [[Feature_selection/pipeline]] — understand data flow
> 3. [[bias_nodes/creating_nodes]] — implement your first node
> 4. [[Ensemble/base_model]] → [[Ensemble/weight_layer]] → [[Ensemble/portfolio]]
> 5. [[Feature_selection/Phase_3_4_Walkforward/param_stability]] + [[Feature_selection/Phase_2_IS_Screening/permutation_testing]]
> 6. [[Vault/vault]] — understand how validated features are stored

## Key Naming Convention

`{module}_{feature}_{timeframe}_{param}_{value}` — e.g. `rsi_signal_D_lookback_14`

## Key Formulas

- Forecast: `F_i = (τ / (σ × √h_i)) × X_i`
- FDM / IDM: `min(√(1 / (mean_corr + 0.01)), cap)`
- Position: `contracts = (position_fraction × capital) / (price × multiplier × fx_rate)`
