# Repository Inventory

> **Generated:** 2026-02-13
> **Scope:** First-party top-level packages and `scripts/` entrypoints (excluding `venv`, `.git`, `tests`, caches, and data/artifacts directories)

## Major Packages / Modules

| Package | Path | Description | Doc Target |
|---------|------|-------------|------------|
| **utils** | `utils/` | Shared enums, typed candle models, helpers, cache tooling, logging, and fast numeric helpers | [utils.md](utils.md) |
| **nodes** | `nodes/` | Stateful bias-node indicators (RSI, ATR, EWMAC, momentum, seasonal, etc.) | [nodes.md](nodes.md) |
| **feature_selection.base_models** | `feature_selection/base_models/` | Base model and binning contracts used by ensemble and research layers | [base_models.md](base_models.md) |
| **feature_selection** | `feature_selection/` | In-sample/out-of-sample selectors and walkforward/permutation workflows | [feature_selection.md](feature_selection.md) |
| **feature_selection.validators** | `feature_selection/validators/` | Stage-based validator (`EDA -> permutation -> stability`) and report contracts | [feature_validator_api.md](feature_validator_api.md) |
| **ensemble** | `ensemble/` | Diversified ensemble construction, weight layer, portfolio, and vault/control helpers | [ensemble.md](ensemble.md) |
| **execution** | `execution/` | Position sizing from forecast fractions to contracts/notional | [ensemble.md](ensemble.md) |
| **metrics** | `metrics/` | Performance/risk/equity metrics and reporting/plotting utilities | [metrics.md](metrics.md) |
| **deployment** | `deployment/` | Forecast server, data connectors, notifier, and training pipeline surfaces | [deployment.md](deployment.md) |
| **data_cleaning** | `data_cleaning/` | Raw text/parquet ingest and D/W/M OHLC aggregation | [data_pipeline.md](data_pipeline.md) |
| **feature_extraction** | `feature_extraction/` | Feature extraction plus forward-return target alignment | [data_pipeline.md](data_pipeline.md) |
| **eda** | `eda/` | Feature exploration and parameter sensitivity tooling | [data_pipeline.md](data_pipeline.md) |
| **research** | `research/` | Research orchestration helpers for node and model evaluation | [data_pipeline.md](data_pipeline.md) |
| **utils.evaluation.permutation_test** | `utils/evaluation/permutation_test/` | Feature/bar permutation engines and walkforward-safe strategies | [testing_tools.md](testing_tools.md) |
| **utils.simulation.prop_firm_simulator** | `utils/simulation/prop_firm_simulator/` | Prop-firm challenge simulation and statistics contracts | [testing_tools.md](testing_tools.md) |
| **utils.evaluation.robustness_test** | `utils/evaluation/robustness_test/` | Monte Carlo/bootstrap/block-bootstrap robustness analysis | [testing_tools.md](testing_tools.md) |
| **plotting** | `plotting/` | Visualization helpers for robustness and prop-firm simulations | [testing_tools.md](testing_tools.md) |

## Key Entrypoints (`scripts/`)

| Script | Purpose |
|--------|---------|
| `scripts/tws_live_forecast.py` | Live/paper forecast loop via Interactive Brokers TWS with optional notifications |
| `scripts/run_manual_forecast.py` | Manual forecast execution and JSON output for selected tickers/timeframes |
| `scripts/benchmark_portfolio_backtest.py` | End-to-end portfolio backtest benchmark (config from portfolio_research) |
| `scripts/demo_forecast_pipeline.py` | Demonstrates full forecast stack from base model to position sizing |
| `scripts/demo_ib_data_fetch.py` | Interactive Brokers data fetch and streaming demonstrations |
| `scripts/demo_prop_firm_simulator.py` | Demonstrates prop-firm simulation workflow and outputs |
| `scripts/demo_robustness_test.py` | Demonstrates robustness resampling and comparison workflow |

## Cross-Module Dependency Surface

Key symbols imported across package boundaries:

| Symbol | Defined In | Used By |
|--------|-----------|---------|
| `Ticker` | `utils.core.enums` | `data_cleaning`, `deployment`, `ensemble`, `feature_extraction`, `feature_selection`, `nodes`, `research` |
| `TimeFrame` | `utils.core.enums` | `data_cleaning`, `deployment`, `ensemble`, `feature_extraction`, `feature_selection`, `nodes`, `research` |
| `helpers` | `utils.core.helpers` | `deployment`, `eda`, `ensemble`, `feature_extraction`, `feature_selection`, `nodes`, `research` |
| `Candle` | `utils.core.models` | `deployment`, `ensemble`, `feature_extraction`, `feature_selection`, `nodes` |
| `CacheMissError` | `utils.cache.bias_node_cache` | `feature_extraction`, `feature_selection`, `nodes` |
| `BiasNodeCache` | `utils.cache.bias_node_cache` | `feature_extraction`, `nodes` |
| `PermutationEngine` | `utils.evaluation.permutation_test.permutation_engine` | `eda`, `feature_selection` |
| `FeaturePermutationStrategy` | `utils.evaluation.permutation_test.permutation_engine` | `eda`, `feature_selection` |
| `Portfolio` | `ensemble.portfolio` | `deployment`, `feature_selection` |
| `PortfolioTester` | `ensemble.portfolio_tester` | `feature_selection` |
| `PositionSizer` | `execution.position_sizer` | `ensemble` |
| `SortinoRatio` | `metrics.performance` | `deployment`, `eda` |
| `generate_tearsheet` | `metrics.plotting.graphing.quantstats_reports` | `ensemble`, `feature_selection`, `research` |
| `BaseModel` / `ContinuousBinningModel` / `RuleBasedModel` / `TwoBinBinningModel` | `feature_selection.base_models` | `ensemble`, `research`, `deployment` |
| `OSFeatureSelector` | `feature_selection.os_feature_selector` | `deployment` |
| `extract_features_for_bias_node` | `feature_extraction.feature_extractor` | `research` |
| `get_logger` | `utils.core.logger` | `deployment`, `feature_extraction` |

## Doc Targets (10 groups)

1. `utils.md` - Core utility layer (types, helpers, cache, logging, fast modules)
2. `nodes.md` - Bias-node APIs and contracts
3. `base_models.md` - Base-model and binning contracts
4. `feature_selection.md` - Selector and walkforward APIs
5. `feature_validator_api.md` - Stage-based validator and report contracts
6. `ensemble.md` - Ensemble/weight/portfolio plus execution interface
7. `metrics.md` - Metrics, risk, equity, and report/plot helpers
8. `deployment.md` - Production serving/connectors/notifier/training APIs
9. `data_pipeline.md` - Data cleaning, extraction, EDA, and research entrypoints
10. `testing_tools.md` - Permutation, robustness, prop-firm simulation, and plotting helpers
