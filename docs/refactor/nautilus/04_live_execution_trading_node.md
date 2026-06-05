# WP-4 — Live Execution → Nautilus TradingNode (IB + vendored MT5)

> **Depends on:** WP-2 (instruments/catalog). Shares the rebalance `Strategy` with
> WP-3 (research-to-live parity — one strategy, used in both backtest and live).
> **Invariant:** Deployment keeps working throughout (I3). Live cutover happens
> **only after a paper-trading gate**. Forecast generation is NOT changed here.

## Objective

Replace the homegrown order-placement / reconciliation layer with a Nautilus
`TradingNode` driving:

- the **Interactive Brokers adapter** (futures), and
- a **vendored fork of `aulekator/mt5-connect`** (CFD/MT5),

so both live venues run through one event-driven runtime. The same
`TargetRebalanceStrategy` built in WP-3 executes the daily `position_fraction` target
live — giving true research↔live parity.

## What stays vs. what moves

**Stays (do not change):**

- Forecast generation: `deployment/forecast_server.py`,
  `deployment/forecast_prediction_runtime.py`, `deployment/forecast_live_inputs.py`,
  the `DiversifiedEnsemble`/MLManager/volatility path, and the alpha core. These still
  produce `position_fraction` / forecasts.
- Telegram notifications (`deployment/telegram_notifier.py`) — keep (optionally wrap as
  a Nautilus `Actor` later; not required).
- Prop-firm reporting/optimization (`prop_firms/`) — unrelated to execution.

**Moves (homegrown → Nautilus), guarded by the paper gate:**

- `execution/ib_trade_executor.py` → Nautilus IB `ExecutionClient` under `TradingNode`.
- `execution/mt5_trade_executor.py` → vendored MT5 `ExecutionClient`.
- `execution/{rebalancer,mt5_rebalancer}.py` → `TargetRebalanceStrategy` (shared w/ WP-3).
- `execution/{order_safety,mt5_order_safety}.py` → Nautilus `RiskEngine` pre-trade checks
  **plus** any bespoke rules kept as a custom risk component (see Risk note).
- `execution/{run_execution,run_mt5_execution,run_mt5_weekend_close}.py` → `TradingNode`
  run scripts / config.
- `scripts/enigma_*_forecast.py` live-order portions → thin launchers that build the node.

**Keep until explicitly validated, then re-home (don't cull blindly):**

- `execution/approval_flow.py`, `execution/audit_log.py` — bespoke governance. Re-home as
  Nautilus `Actor`s or wrappers around the node; preserve behaviour.
- `execution/position_sizer.py` — sizing math is shared with WP-3; keep as the single
  source of contract sizing for both backtest and live.

## Current-state map (confirm with codegraph)

- Forecast→order flow today: `forecast_server` → `position_fraction` →
  `rebalancer`/`mt5_rebalancer` (`plan_symbol_actions`, partial-close logic, `_quantize`)
  → `ib_trade_executor` / `mt5_trade_executor` → venue, with `order_safety` /
  `mt5_order_safety` pre-trade checks and `audit_log` / `approval_flow` governance.
- Data connector: `deployment/mt5_data_connector.py` (`ForecastMT5DataConnector`).
- Models: `execution/models.py`, `execution/mt5_models.py`.

## Target design

```
deployment/nautilus_mt5/              # NEW — vendored fork of aulekator/mt5-connect
  ...                                 #   audited, pinned, MIT license preserved
deployment/live/                      # NEW
  node.py                             # build TradingNode (IB + MT5 exec/data clients)
  strategy.py                         # re-export/parametrize WP-3 TargetRebalanceStrategy
  risk.py                             # Nautilus RiskEngine config + bespoke rules port +
                                      #   TradingState (HALTED/REDUCING) for prop limits
  config_ib.py / config_mt5.py        # venue + account + adapter configs
  run_live.py                         # entrypoint replacing run_execution/run_mt5_execution
```

Design points:

- **One strategy, two venues.** `TargetRebalanceStrategy` consumes the daily
  `position_fraction` (from the unchanged forecast pipeline) and works orders per the
  same `ExecutionPolicy`/`ExecutionWindowPolicy` as WP-3. This is the DRY payoff:
  identical execution logic in research and production.
- **Reconciliation:** rely on Nautilus `ExecutionEngine` reconciliation + the adapter's
  account reconciliation (mt5connect advertises startup + continuous reconciliation).
- **Sizing:** `execution/position_sizer.py` is the shared contract-sizing authority.
- **Vendored MT5 adapter:** fork into `deployment/nautilus_mt5/`, pin a commit, audit the
  order lifecycle + reconnection paths, add tests. Do **not** depend on the PyPI dev
  package for funded accounts.

## Nautilus capabilities to leverage here (see [06_nautilus_feature_leverage.md](06_nautilus_feature_leverage.md))

- **`RiskEngine`** pre-trade checks map `order_safety`/`mt5_order_safety` (no weakening);
  add **`TradingState` HALTED/REDUCING** to enforce prop-firm daily/overall drawdown
  limits (auto exposure-reduce / halt near a limit). High-value, previously unplanned.
- **Bracket / `OTO`/`OCO`/`OUO`** orders for native SL/TP; same `ExecutionPolicy` +
  `ExecAlgorithm` (TWAP/custom) as WP-3 — identical execution logic research↔live.
- **Execution reconciliation** (`OrderStatusReport`/`FillReport`/`OrderWithFills`/
  `PositionStatusReport`, overfill + duplicate-fill handling) replaces bespoke
  reconciliation; pair with mt5connect's account reconciliation.
- **Actors** re-home deployment pieces: `telegram_notifier` (subscribe `on_order_filled`
  for fill/status pushes), a periodic `ReportingActor`, volatility refresh, weekend-close
  scheduler (via `Clock` timers).
- **Real-time `Portfolio`** (live PnL/exposure/margin) feeds the `TradingState` guardrails
  and Telegram; **`ReportProvider`** emits live execution diagnostics. Return/robustness
  metrics still come from `quantfoundry_core` (invariant I4).
- Daily `position_fraction` reaches the live strategy as the same **registered `CustomData`
  signal** used in WP-3 (forecast pipeline → publish → strategy `on_data`).
- **Crash-only restart** with externalized (Redis) state for the funded node (hardening).

## Live-simulation / validation ladder (mandatory before live cutover)

Climb these in order. The same `TargetRebalanceStrategy` used in WP-3 runs at every
rung — only the data source and fill authority change. NDX (MT5 feed already available)
is the natural first instrument.

1. **Nautilus sandbox (live data, local virtual fills).** Use the vendored MT5 adapter's
   live `DataClient` to stream real-time quotes/ticks into the `TradingNode`, with
   Nautilus's built-in `SandboxExecutionClient` / `SimulatedExchange` doing the fills
   (orders never leave the machine). Validates the **live runtime end to end** — data
   flow, the strategy reacting in real time, the `FillModel`, reconnection — at zero
   broker risk. This is the cheapest realistic test; do it first.
2. **MT5 demo account (live data, broker-side virtual fills).** Orders now route to the
   broker's demo server and fill against *their* engine. This is what validates the
   adapter's real **order lifecycle + reconciliation + partial fills + weekend-close**
   (`run_mt5_weekend_close` equivalent) + error handling — things sandbox cannot, because
   fills are no longer local. IB paper account is the equivalent rung for the IB venue.
   (FTMO demo/trial fits here.)
3. **Shadow mode (run across rungs 1–2).** Run the node in parallel with the existing
   executor on the same forecasts; log intended orders from both; assert they agree
   (same side, qty, instrument, within tolerance) over an agreed window.
4. **Governance preserved:** approval flow + audit log + Telegram must fire identically
   at every rung.
5. Only after 1–4 pass for the agreed soak duration is the legacy executor retired (WP-5).

> Sandbox is also a useful **continuous staging environment** post-migration: keep a
> sandbox node running on live data to catch adapter/data regressions before they reach
> funded accounts.

## Acceptance criteria (gate)

- Sandbox node runs on live MT5 data (NDX) with local virtual fills; strategy reacts in
  real time; reconnection verified.
- Node builds and connects to IB paper + MT5 demo; instruments load from the WP-2 catalog.
- Shadow-mode order agreement vs legacy executor over the validation window (documented
  diff = none / within tolerance).
- Weekend-close, partial-close, and flatten behaviours reproduced.
- Governance (approval/audit/Telegram) verified equivalent.
- Runbook written: how to start/stop the node, recover from a crash (Nautilus crash-only
  semantics), and roll back to the legacy executor.

## Risks / notes

- **mt5connect maturity** (≈6 commits, hobby, Windows-only): biggest risk. Vendoring +
  audit + paper soak is mandatory. Windows-only is fine (FTMO VPS is Windows).
- **Do not weaken safety.** Map every existing `order_safety`/`approval_flow` rule to a
  Nautilus `RiskEngine` check or a kept bespoke component. A missing safety check is a
  release blocker.
- **One node per process** (Nautilus global-singleton constraint). If IB and MT5 must run
  separately, use two processes.
- **Cutover is reversible** until WP-5 deletes the legacy path; keep the rollback runbook.

## Subagent instructions

- `codegraph_explore "forecast_server rebalancer mt5_rebalancer ib_trade_executor mt5_trade_executor order_safety position_sizer plan_symbol_actions"` to map the full live path before editing.
- Build the node + shadow mode first; do not touch funded accounts. Do not delete any
  legacy executor file — that is WP-5, gated on the paper soak.
- Hand back: the node, vendored+audited MT5 adapter, shadow-mode agreement report,
  paper-soak results, and the runbook.
