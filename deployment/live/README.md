# `deployment/live/` — Nautilus + vault live trading runtime (MT5)

A NautilusTrader runtime that trades the vault `GlobalPortfolio` on an MT5 venue.
It supersedes the legacy daily-rebalance chain (`scripts/enigma_live_forecast.py`
→ `execution/run_mt5_execution.py`, which drives the raw `MetaTrader5` API) by
running the same vault math *inside* a Nautilus `TradingNode`. The legacy path is
untouched and still production.

Current execution tier: **sandbox** — real broker quotes stream in, orders fill
LOCALLY via Nautilus's `SandboxExecutionClient` (NETTING) and **never reach the
broker**. Swapping to a real MT5 exec client is a config/factory change in
`runtime/node_builder.py`.

## Design — one Darwinex signal, per-broker execution

Signals come from a **single source of truth — Darwinex** — for every CFD broker.
Each broker still runs in its **own process** (MT5 is one-terminal-per-process)
with its own log (`logs/live_nautilus/{broker}_*.log`), but every process binds
the central cache to the **same shared signal namespace**
(`data/broker_cache/darwinex/`). Only **execution** (symbols, quotes, orders,
creds, magic) is per-broker.

```
EVERY BROKER PROCESS (e.g. ftmo):
  refresh_signal_daily: scraped Darwinex series (data/mt5_data via cfd_candles)
        → shared darwinex store (D + M candles) → bias artifacts (EWSD + nodes)
                                      │            [NO MT5 connection; no splice]
                                      ▼
  VaultForecastEngine(cache_root=darwinex) ─► target position_fraction/ticker  (identical for all brokers)
                                      │
                                      ▼
  VaultRebalanceStrategy ── daily decision @ 16:05 ET
     • warmup gate (cache coverage)
     • fraction→signed lots (reuses compute_target_signed_lots)
     • delta vs Nautilus net_position (pure plan_rebalance)
     • subscribe ticks ONLY in the execution window → market order → unsubscribe
        (live quotes are the EXECUTION broker's, used for sizing/fills only)
```

The Darwinex signal cache is rebuilt from the parquet files the daily Darwinex
scraper writes under `data/mt5_data` — no broker `copy_rates`, no ratio-splice
(Darwinex is do-not-disturb / `_NOT_ARMABLE`). Cross-broker price differences
therefore affect execution sizing/fills only, never the signal; a difference
large enough to flip a signal is a broker problem to investigate, not noise to
model. This also makes live signals identical to research/validation, which
already read Darwinex. Signal freshness depends on the scraper having run.

## Trading hours (different per broker)

The signal is the Darwinex daily close (index/energy close 23:00 EET — complete by
the 16:05-ET decision). Execution is on FTMO, whose index/energy CFDs close ~23:15
EET (a ~10-min window after the decision) while metals/FX run wider. Each decision
filters targets through `exec_hours.tradeable(exec_broker, …)`: a symbol is submitted
only when the **execution** broker's market is open **and** its quote is fresh;
closed/stale symbols stay pending and are retried on later polls within the session
(bounded by that day's session). So we never fire into a closed FTMO book, and the
basket is robust to holidays, the nightly rollover dead-zone, and odd restart times.

## Crash-restart robustness

The pipeline is built to survive a server crash + restart with no double-trade:

* **Durable decision marker** (`runtime/decision_state.py`): the once-per-day latch +
  executed-symbol set persist to `data/broker_cache/<broker>/decision_state.json`. On
  restart the strategy **resumes** the session — already-filled symbols are excluded,
  unfinished ones still execute.
* **Reconcile-before-first-decision** (`startup_grace_secs`): the first decision is
  withheld until account/position reconciliation populates, so a restart never sizes
  against a phantom-flat book. Combined with the netting-delta planner, a re-decide is
  a no-op once positions are known.
* **Atomic cache writes** (`lib/core/atomic_io.py`): candle + bias parquet writes are
  temp-then-`os.replace`; a corrupt post-crash file is quarantined and recomputed.
* **Exec-loop escalation**: on MT5 reconnect exhaustion the exec client alerts
  (Telegram) and degrades instead of running blind.
* **Warmup headroom**: `prediction_daily_max_bars` (750) > `warmup_min_bars` (500) — equal
  values caused a 1-bar off-by-one that flagged tickers "not ready".

## Sandbox simulation (dress rehearsal)

`scripts/run_sandbox_sim.py` (→ `sandbox_sim.py`) runs the **exact** live strategy in a
Nautilus `BacktestEngine` over a recent window — Darwinex signal cache, FTMO venue with
modeled spread/swap, the 16:05-ET clock, the market-hours gate, and the durable
decision-state — so it behaves identically to live (Nautilus backtest/live invariance).
Use it to validate the full workflow before arming. Example:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_sandbox_sim --days 25
```

## Components

| File | Role |
|---|---|
| `forecast_engine.py` | `VaultForecastEngine` — loads the vault, emits latest daily targets from cache (`cache_root` binds the shared Darwinex signal store, the same for every broker); `warmup_status()` gates trading until enough daily history. |
| `broker_data.py` | Shared Darwinex signal pipeline: `bind_signal_cache`, `refresh_signal_daily` (loads the scraped Darwinex series via `cfd_candles.load_cfd_candles_raw` + bias population into the shared `darwinex` store). `broker_cache_root` / `bind_broker_cache` remain as the generic namespace helpers. |
| `runtime/broker_clock.py` | US-Eastern decision clock (16:05 ET, `is_past_decision` = latch-free weekday/time gate) + broker-EET conversions + rollover dead-zone. |
| `runtime/exec_hours.py` | `tradeable(exec_broker, ticker, now, quote_age)` = `brokers.is_market_open` **AND** a fresh quote. Orders go out only when the EXECUTION broker's market is open (FTMO index/energy close 23:15 EET vs Darwinex 23:00); closed/stale symbols defer. |
| `runtime/decision_state.py` | Durable per-broker `decision_state.json` (`last_session_date` + executed tickers, atomic write). Reloaded on `on_start` so a crash+restart **resumes** an unfinished session instead of re-firing it (no double-trade). |
| `runtime/sizing.py` | `target_signed_lots` (reuses the tested sizer), `net_rebalance`, and the pure `plan_rebalance` decision core. |
| `runtime/tick_controller.py` | `DemandTickController` — demand-driven quote subscribe/unsubscribe per execution window. |
| `vault_strategy.py` | `VaultRebalanceStrategy` — imperative shell. Decision flow: startup/reconcile grace → build session plan once/day (latch via persisted `decision_state`) → **pump** open symbols, defer closed/stale → execute. `_build_engine`/`_evaluate_targets` are test seams. |
| `sandbox_sim.py` | Full live-trading SIMULATION (markets closed): real strategy in a BacktestEngine, Darwinex signal cache + FTMO venue with modeled spread (`SimCostModel`) + overnight swap (`_FinancingActor`). Run via `scripts/run_sandbox_sim.py`. |
| `credentials.py` | `BrokerCredentials.from_env(broker, tier)` (per-broker `.env` resolver) + `ExecTier` enum. `darwinex` is not armable (only the live scraper terminal exists). |
| `monitoring/heartbeat.py` | `PortfolioHeartbeat` actor — periodic equity/positions log via the Nautilus logger. |
| `monitoring/live_state.py` | **Node-mediated dashboard feed** (functional core): publishes `snapshot.json` (account/positions/targets/warmup/risk baseline) + consumes `command.json` (flatten) under `data/broker_cache/<broker>/live_state/`. Risk-baseline math, halt persistence (`halt.json`), bounded equity history, atomic IO. No Nautilus/MT5 imports. Read by `frontend/api/live.py`. |
| `runtime/node_builder.py` | `build_node(..., exec_tier)` — MT5 live data + (sandbox NETTING **or** real MT5 exec) on venue `MT5`; `build_sandbox_node` is the sandbox wrapper. |
| `run_vault_sandbox.py` | Operator entrypoint (`--dry-build` / `--arm`, `--exec-tier`, `--live`), target-broker safety guard. |
| `../../configs/live_nautilus_ftmo.json` | Runtime knobs (vault_root, tickers, warmup, decision time, sandbox balance, …). |

## Run

Execution tiers (`--exec-tier`, default `sandbox`): `sandbox` (local simulated
fills, zero broker risk) · `demo` (REAL orders to the broker demo account) ·
`live` (REAL orders to a funded account). `demo`/`live` additionally require the
explicit `--live` flag.

```powershell
# Validate wiring only (no terminal/connection) — both tiers:
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --dry-build
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --exec-tier demo --dry-build

# Armed SANDBOX run (paper; no broker orders). Refreshes the shared Darwinex signal cache, then runs:
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --arm --minutes 10

# Armed DEMO run (REAL orders to the FTMO demo account) — requires --live:
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --exec-tier demo --arm --live
```

`--arm` needs a SEPARATE FTMO MT5 terminal running (never the live Darwinex one;
`terminal_path` + creds resolve from the SAME `--broker`). `darwinex` is refused
as an execution broker (only its live scraper terminal exists). Each broker is a
separate invocation/process with its own `logs/live_nautilus/{broker}_*.log`, but
all read the **shared** `data/broker_cache/darwinex/` signal cache.

> **Not yet proven live:** FTMO demo is a *hedging* account; the netting-delta →
> per-ticket-close path is designed-correct in the adapter but unvalidated until
> the first open-market demo order round-trip. No prop risk-limit gates are built
> (the firm enforces those).

## Live monitor dashboard (node-mediated)

A read-only **Live monitor** page in the research frontend (`frontend/`, nav
`/live`) shows live positions, account/equity, prop-firm health gauges, an equity
curve, target-vs-actual, and a **flatten-all kill switch** — without ever opening
its own MT5 connection.

**Why node-mediated.** MT5's Python API binds one terminal per OS process, and an
out-of-process "flatten" would race the node's next rollover rebalance. So the
running node is the *single* MT5 owner: `VaultRebalanceStrategy` publishes
`snapshot.json` every ~5s and watches `command.json`; the FastAPI backend
(`frontend/api/live.py`, routes `/api/live/*`) only **reads** the snapshot and
**writes** a command file. No terminal contention, no double-trade race. All of
this is wrapped so a monitoring failure can never break trading, and it is disabled
(no files, no timer) unless the node is built via `node_builder` (tests /
`sandbox_sim` stay file-free).

**Kill switch.** `POST /api/live/flatten` (confirm-token + demo-only + node-online
gated) writes a `flatten` command. The node closes every magic-tagged broker
**ticket** per-ticket (`TRADE_ACTION_DEAL` + `position=ticket` — hedging-safe, like
`manual_trade.py`), reports the *real* synchronous result (executed / partial /
error), and **halts** durably (`halt.json`) so it never re-opens toward target —
even across a crash+restart. To resume: delete
`data/broker_cache/<broker>/live_state/halt.json` and restart the node.

Run the dashboard from the repo root (the live node runs separately, as above):

```powershell
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --port 5057   # serves built UI + /api/live/*
```

**Limitations (by design / deferred):**

- **Demo-only.** Flatten is refused on a funded (`exec_tier=live`) account at both
  the API and the node — a real-money kill switch is a deliberate, audited future
  change.
- **Intraday equity is live only during rollover windows.** Between windows the
  demand-tick model holds no quote subscription, so position marks freeze; the
  snapshot flags `marks_fresh=false` and the node does **not** record a frozen
  equity sample or anchor the daily gauge off it. A continuous low-frequency
  monitoring subscription (always-live equity) is a future enhancement.
- **The automated rollover EXIT still uses the netting-delta path** (the H8
  hedging limitation below); only the operator kill switch uses the per-ticket
  close. Wiring per-ticket close into the automated path is a separate, demo-
  verified change.
- **Prop-firm gauges are advisory.** Daily/total limits are modelled as a dollar
  amount = the firm's % of the initial balance; confirm the drawdown basis
  (static-from-initial vs trailing) and the daily reset timezone (server EET vs the
  firm's CE(S)T) against your program rules before trusting headroom near a limit.

## Tests

```powershell
# Offline (unit + integration; integration drives the real cache + a BacktestEngine):
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\deployment\live_nautilus tests\integration\live_nautilus -q

# Gated live smoke (needs the FTMO terminal): builds + runs the sandbox node, asserts data flow.
$env:MT5_LIVE_TESTS=1; .\.venv\Scripts\python.exe -m pytest tests\live_mt5\test_vault_sandbox_node_live.py -q
```

## Notes / future tiers

- `vault_root` defaults to `vault` (the fitted prop vault on disk); the
  `vault_cfd_prop` profile resolves to a directory not materialised here.
- Targets are filtered to broker-resolvable canonicals (`brokers.resolve`), so an
  instrument the broker doesn't offer (e.g. `TLT`) is skipped automatically.
- **OMS:** sandbox uses NETTING (clean target-position deltas). The real FTMO
  account is hedging — the demo/live tier reuses the per-ticket close path in
  `execution/mt5_rebalancer.plan_symbol_actions`.
- Not wired here: real-order tiers, the rollover swap-avoidance overlay
  (`execution/rollover_overlay.py`), Telegram/journal, Darwinex/FundedNext nodes.

> _Verified against current code via CodeGraph on 2026-06-07._
