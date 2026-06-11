# Trading-Algo Library Index

> [!note]
> Pipeline (conceptual):
> `Candles (OHLCV)` -> `Bias nodes` -> `DiversifiedEnsemble` -> `WeightLayer` -> `Portfolio` -> `PositionSizer`

Production uses native signed-signal bias nodes that emit `-1/0/+1`. Continuous nodes are for research unless they are later reimplemented as native discrete nodes.

## Cache

- [[Cache/architecture]] - Central-cache design and lifecycle
- [[Cache/user_guide]] - Practical cache usage

## Bias nodes

- [[bias_nodes/index]] - Hub for node authoring and composition
- [[bias_nodes/creating_nodes]] - How to implement a node
- [[bias_nodes/bias_node_arch]] - Bias-node architecture notes
- [[bias_nodes/composed_nodes]] - Composite / gate nodes

## Feature research and robustness

The local `research.feature` phase model (`exploration -> validation -> portfolio_addition`)
now mirrors the canonical robustness workflow under `docs/SaaS/robustness_tests/`. There is no
longer a `docs/library/Feature_selection/` folder — use the robustness docs as the source of truth.

- [[SaaS/robustness_tests/index]] - Canonical robustness workflow (entry point)
- [[SaaS/robustness_tests/in_sample]] - In-sample exploration and parameter lock
- [[SaaS/robustness_tests/parameter_selection]] - Parameter selection
- [[SaaS/robustness_tests/parameter_sensitivity]] - Parameter sensitivity / plateau selection
- [[SaaS/robustness_tests/validation]] - Validation stage
- [[SaaS/robustness_tests/portfolio_addition]] - Portfolio-addition gate
- [[SaaS/robustness_tests/portfolio_holdout]] - Portfolio holdout source of truth
- [[SaaS/robustness_tests/monitoring]] - Monitoring source of truth
- [[Portfolio_research/holdout]] - Library-local holdout notes (`portfolio_fit_mode`)

## Strategy research

> The **frontend app** (`frontend/`, FastAPI + React/Vite) is now the primary way to build, run,
> and inspect a `StrategySpec`. The manual `research` agent skill (the "agent harness" below) is
> **superseded** but retained; agent and UI share the same spec JSON in `research/specs/`.

- [[Strategy_research/strategy_engineering]] - Front of the funnel: turn a strategy brief into a parsimonious design. Category taxonomy (mean-reversion / breakout / trend / seasonal / flow / pairs → node folders + vault sleeves), parsimony principles (few params, simplest construct, fix what needn't be optimized), study what already works
- [[Strategy_research/strategy_spec]] - `StrategySpec` source of truth: the flat, self-validating object an agent writes to define / evaluate / vault a strategy (windows, signal grid, vol scaling, execution, the two strategy types)
- [[Strategy_research/execution_architecture]] - Two engines: Engine A (level signal + market/passive-limit, no stops) and Engine B (deferred bracket/scalping state machine on partitioned instruments). Why we don't build the "node with intrabar stops" middle
- [[Strategy_research/pnl_lanes_and_validation]] - The three P&L lanes (vectorized / Nautilus realistic / final-validation), the lookahead-free causality contract (`backtest_conventions`, the `as_of` bound, Nautilus as oracle), and `scripts/validate_candidate.py` — the gold-standard pre-promotion check that runs the real live strategy with the signal generated on-the-fly
- [[Strategy_research/research_report]] - The report contract: pipeline emits raw data, agent judges holistically (rationale, plateau, overfit, IS→val persistence, degradation, diversification), no auto-gates. Replaces the buggy dashboard with an agent-written memo
- [[Strategy_research/agent_harness]] - The Claude Code wiring (manual, no scheduling): a `research` skill orchestrating three subagents (designer/executor/analyst), hooks as structural guardrails, deterministic scripts the agents call, the per-run workspace, and the two human checkpoints
- [[Strategy_research/README]] - **Start here** — overview & implementation handoff: goal, principles, architecture in brief, what exists, the build backlog in order

## Data

- [[Data/README]] - Data platform hub (`data_platform/` providers and catalogs)
- [[Data/futures_research_data]] - Norgate continuous back-adjusted futures and ratio adjustment
- [[Data/mt5_data_scraper]] - MT5 OHLCV / tick scraper
- [[Data/mt5_broker_config]] - MT5 broker config (`configs/mt5_brokers.yaml`)
- [[Data/mt5_timezones]] - MT5 broker EET/EEST timestamp handling
- [[Data/feed_comparison_and_adjustment]] - Norgate futures vs Darwinex CFD feed comparison
- [[Data/feed_and_execution_decision]] - Feed and execution decision
- [[Data/darwinex_universe]] - Darwinex CFD universe
- [[Data/hybrid_tick_backtest]] - Hybrid bars + ticks backtest

## Ensemble

- [[Ensemble/base_model]] - Base-model concepts (node-backed signed-signal adapter)
- [[Ensemble/weight_layer]] - Cross-stream weighting layer (9 methods + FDM)
- [[Ensemble/portfolio]] - Portfolio layer and the portfolio-addition gate (older docs/code may still say `inclusion`)
- [[Ensemble/multi_timeframe]] - Multi-timeframe orchestration

## Vault

Default prop tree: `vault/<D|W|M>/<group>/<ensemble>/` (manual weight-hierarchy groups); personal: `vault_personal/...`; CFD prop: `vault_cfd_prop/...` — see [[Vault/vault]]. Legacy flat `<vault_root>/<TF>/<ensemble>/` remains supported.

- [[Vault/architecture]] - Vault ownership and invariants
- [[Vault/user_guide]] - Saving and loading features and ensembles
- [[Vault/vault]] - Quick reference
- [[Vault/monitoring]] - Monitoring store
- [[Vault/portfolio_snapshots_and_predictions]] - Snapshot and prediction materialization
- [[Vault/portfolio_snapshot_usage]] - Snapshot workflows

## Deployment

- [[Deployment/production]] - Forecast server and production training
- [[Deployment/live_cache_refresh]] - Live cache refresh
- [[Deployment/live_multi_timeframe]] - Live multi-timeframe prediction from cache
- [[Deployment/cython]] - Cython build notes

## Live forecast

> Two live paths: (1) **IB/TWS** legacy daily-rebalance driver `scripts/enigma_live_forecast.py`
> (futures-prop + personal profiles); (2) **MT5/CFD** Nautilus vault runtime
> `deployment/live/run_vault_sandbox.py` (cfd_prop) — see [[Deployment/production]] and
> `deployment/live/README.md`. IB market-data ingestion (`data_platform/providers/ib/`) was removed.

- [[live_forecast/live_forecast_script]] - IB live forecast driver (`scripts/enigma_live_forecast.py` and prop/personal variants)
- [[live_forecast/prop_vs_personal_workflows]] - Prop vs personal workflow split
- [[live_forecast/testing_plan]] - Live testing plan

## Prop firms and ops

- [[prop_firms/prop_firm_and_trade_copier_recommendations]] - Prop firm and trade-copier recommendations
- [[Testing/prop_firms]] - Prop-firm report runners and tests
- [[remote_ssh_setup]] - Remote SSH setup

## Research reading order

1. [[bias_nodes/index]] -> [[bias_nodes/creating_nodes]]
2. [[SaaS/robustness_tests/index]]
3. [[SaaS/robustness_tests/in_sample]]
4. [[SaaS/robustness_tests/validation]]
5. [[Vault/user_guide]]

## Naming convention

`{module}_{feature}_{timeframe}_{param}_{value}`

Example: `rsi_signal_D_lookback_14`

> _Verified against current code via CodeGraph on 2026-06-07._
