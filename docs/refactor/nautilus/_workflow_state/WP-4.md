# WP-4 — Live Execution → Nautilus TradingNode · Discovery Manifest

**Mode:** READ-ONLY discovery. No code edited. This file is the only artifact written.
**Gate type:** NO research-parity gate. Gated by a **paper-trade / sandbox ladder** (invariant
**I3**: deployment keeps working; live execution is *migrated, not broken*; cut over **only after
the paper gate**). Nothing here may be silently cut over.
**Depends on:** WP-2 (instruments/catalog — COMPLETE, default still legacy). Shares the rebalance
`Strategy` with WP-3 (`TargetRebalanceStrategy`, Unit-2 still BLOCKED on the WP-2 precision fork).
**Decision log (00 §4):** Vendor/fork `aulekator/mt5-connect` (≈6 commits, hobby, Windows-only) →
must be vendored, audited, pinned, paper-validated before funded accounts. Nautilus ships an IB
adapter; **no MT5 adapter**.

---

## 0. Drift check — execution/ + deployment/ UNMOVED (verified via codegraph_files)

WP-8 ("consolidate flat top-level") did **NOT** move `execution/` or `deployment/`. Confirmed by
`codegraph_files`:

- **`execution/`** (15 modules, top-level): `__init__.py`, `approval_flow.py`, `audit_log.py`,
  `ib_trade_executor.py`, `models.py`, `mt5_models.py`, `mt5_order_safety.py`, `mt5_rebalancer.py`,
  `mt5_trade_executor.py`, `order_safety.py`, `position_sizer.py`, `rebalancer.py`,
  `run_execution.py`, `run_mt5_execution.py`, `run_mt5_weekend_close.py`.
- **`deployment/`** (6 modules, top-level): `_bootstrap.py`, `forecast_live_inputs.py`,
  `forecast_prediction_runtime.py`, `forecast_server.py`, `mt5_data_connector.py`,
  `telegram_notifier.py`.
- **`scripts/enigma_*.py`** (6) and **`deployment/ops/*.bat`** (4) are also unmoved — the scripts/ reorg
  was **deliberately HELD** because `deployment/ops/*.bat` cron entries call `scripts\enigma_*` by path.

No drift between the 04 doc's module list and the live index, with two corrections (see §6):
the doc says `execution/run_mt5_execution.py` orchestrates MT5; the live entrypoint function is
`run_cfd_prop_execution` (futures-prop IB path is `run_auto_execution`). Both live where the doc
expects.

---

## 1. Current live execution surface (file:line + order lifecycle)

Two independent broker paths exist today. They are wired by **profile** inside
`scripts/enigma_live_forecast.py:1634` (Step-12 dispatch):

- `profile in {futures_prop, personal}` → **IB path** (`execution.run_execution.run_auto_execution`)
- `profile == cfd_prop` → **MT5 path** (`execution.run_mt5_execution.run_cfd_prop_execution`)

### 1a. IB path (ETF/futures via Interactive Brokers — `ibapi`)

| Module | Role | Key entrypoint (file:line) |
|---|---|---|
| `execution/rebalancer.py` | **Pure** delta math: target_shares + current + prices → `List[OrderIntent]`. Dead-band suppression (`min_rebalance_shares`, `min_rebalance_notional_usd`). No I/O. | `compute_order_intents` `:24` |
| `execution/ib_trade_executor.py` | Thin `ibapi` EClient/EWrapper wrapper. Connect on dedicated `client_id`; fetch managedAccounts + positions; place MKT DAY fractional-share orders; block to terminal state. | `IBTradeClient` `:49`; `connect_and_start` `:78`; `submit_order` `:191`; `wait_for_terminal` `:203`; `cancel_all_open` `:220` |
| `execution/order_safety.py` | **Pure** pre-flight gates (lock-file, live-port-requires-flag, account-match, market-hours, max-orders). | `run_all_preflight_checks` `:99`; `write_lock_file` `:108` |
| `execution/run_execution.py` | **Orchestrator** for IB. Builds targets, connects, computes intents, runs preflight, requests Telegram approval, places orders sequentially, writes lock + closing positions, summarizes. | `run_auto_execution` `:62`; `_execute` `:113`; `_place_orders` `:281` |
| `execution/position_sizer.py` | Shared contract-sizing (`position_fraction → contracts`). `Position` dataclass `:72`; `PositionSizer` `:106`. **Keep — WP-3 shares it.** | — |
| `execution/audit_log.py` | Append-only JSONL audit (one file/run) + `positions_latest.json` for next-run reconciliation. | `AuditLog` `:26`; `log_order_placed` `:57`; `log_order_result` `:75`; `write_closing_positions` `:85`; `read_closing_positions` `:101` |
| `execution/approval_flow.py` | Telegram inline-button approval, fail-closed on outage. | `request_approval` `:124` (single); `request_batch_approval` `:278` (multi-account); `ApprovalDecision` `:46`; `BatchApprovalOutcome` `:76`; `new_run_id` `:100` |
| `deployment/telegram_notifier.py` | Telegram sender (forecast pushes, approval keyboards, health probe). | `TelegramNotifier` `:54`; `send_message` `:276`; `send_with_inline_keyboard(_rows)` `:364/:403`; `probe_health` `:515` |

**IB order lifecycle (place / wait / cancel):**
- **Place:** `submit_order` (`ib_trade_executor.py:191`) → `_next_id` → `_build_etf_contract`
  (STK/SMART/USD) + `_build_mkt_order` (MKT DAY, `totalQuantity=Decimal`, fractional) →
  `placeOrder(order_id, contract, order)`. Returns the IB order id.
- **Wait/terminal:** `wait_for_terminal` (`:203`) polls `_order_states` every 0.25s until status ∈
  `{Filled, Cancelled, ApiCancelled, Inactive}` or an error code ≥200 (≠202) → REJECTED; on timeout
  it **cancels** (`cancelOrder(order_id, "")`) and returns `TIMED_OUT`.
- **Cancel:** `cancelOrder` inside `wait_for_terminal`; bulk `cancel_all_open` (`:220`).
- **Sequencing:** `_place_orders` (`run_execution.py:281`) places intents one at a time; on
  REJECTED/ERROR/TIMED_OUT it **aborts the remaining batch** (`break`).
- No native bracket/SL/TP, no OCO — plain MKT DAY only.

### 1b. MT5 path (CFD prop via MetaTrader5 `mt5` python pkg)

| Module | Role | Key entrypoint (file:line) |
|---|---|---|
| `execution/mt5_models.py` | Domain types: `OrderSide` `:31`, `MT5Position` `:124` (`signed_volume`), `MT5Tick`/`MT5SymbolInfo`/`MT5AccountInfo`, `RebalanceAction`/`RebalanceActionKind` `:146/:153`, `AccountPlan` `:174`, `AccountExecutionReport` `:208`, `OrderResult`, `BatchApprovalOutcome` helpers. | — |
| `execution/mt5_rebalancer.py` | **Pure** sizing + ticket-level planning. `position_fraction → signed lots`; per-symbol action plan; **partial-close (oldest-first), direction-flip (close then open), dead-band, lot-step quantize**. | `SizingConfig` `:52`; `compute_target_signed_lots` `:85`; `_quantize` `:77`; `plan_symbol_actions` `:187`; `plan_account_actions` `:350` |
| `execution/mt5_trade_executor.py` | `mt5` terminal wrapper: initialize/login, fetch account/symbol/tick/positions, `order_calc_margin`, send market deals with retry. | `MT5TradeExecutor` ; `initialize` `:124`; `connect_and_verify` (ctxmgr) `:161`; `fetch_account_info` `:187`; `fetch_all_positions` `:248`; `_market_order` `:288`; `place_market_order` `:330`; `close_ticket` `:373` |
| `execution/mt5_order_safety.py` | **Pure** per-account gates: login/server match, trade_allowed/expert, symbol tradeable, trade-mode compat (LONG/SHORT/CLOSE_ONLY), **margin budget**, **daily-loss floor (prop DLL)**. | `run_preflight` `:125`; `check_daily_loss_floor` `:101`; `check_margin_budget` `:82`; `check_account_login_matches` `:26` |
| `execution/run_mt5_execution.py` | **Orchestrator** (multi-account). Per-account: connect → fetch state → size → plan → preflight; **one batch approval** for all accounts; execute; summarize. | `AccountConfig` `:72`; `_account_from_config` `:100`; `_build_account_plan` `:194`; `run_cfd_prop_execution` (top entry); `_execute_account` `:479` |
| `execution/run_mt5_weekend_close.py` | Weekend flatten: build CLOSE_TICKET-only plan, batch-approve, execute. | `run_cfd_prop_weekend_close`; `_execute_close_plan` `:271` |

**MT5 order lifecycle (place / modify / cancel):**
- **Place (open):** `place_market_order` (`mt5_trade_executor.py:330`) → `_market_order` (`:288`)
  builds `TRADE_ACTION_DEAL` request (`type=ORDER_TYPE_BUY/SELL`, `type_filling=ORDER_FILLING_IOC`,
  `type_time=ORDER_TIME_GTC`, `deviation=50`, `magic`) → `mt5.order_send`. Retries on
  `RETRYABLE_RETCODES` up to `max_retries=3`; success = `retcode == TRADE_RETCODE_DONE`.
- **Close (THE critical mechanic):** `close_ticket` (`:373`) sends an **opposite-side IOC deal with
  `position=<ticket>`** so MT5 treats it as a partial/full close of that *specific* ticket rather
  than opening a hedge (required for hedging-mode accounts). Partial closes are emitted oldest-first
  by `_build_partial_close_actions` (`mt5_rebalancer.py:148`).
- **"Modify":** there is no modify — exposure changes are expressed as additional OPEN_NEW (increase)
  or partial CLOSE_TICKET (decrease); direction flips close all then open fresh
  (`plan_symbol_actions:247`).
- **Magic-number isolation:** `_our_positions` filters by `magic` — only our tickets are managed;
  other strategies/manual trades are ignored.
- **Connect/disconnect:** per plan-build AND per execute, the orchestrator opens a fresh
  `MT5TradeExecutor.connect_and_verify(...)` context (shut down after plan-build, reconnect to
  execute) — `run_mt5_execution.py:257` and `:531`.

---

## 2. The deployment forecast→order path (I3 — must keep working)

Trace (cron → forecast → position_fraction → executor → broker):

```
deployment\ops\run_prop_forecast.bat:24      → python scripts\enigma_prop_forecast.py --port 7497
deployment\ops\run_personal_forecast.bat:27  → python scripts\enigma_personal_forecast.py --port 7497
(cfd_prop has no .bat in deployment/ops/; scripts\enigma_cfd_prop_forecast.py is the entry)
        │  (each enigma_*_forecast.py is a THIN wrapper: injects --profile and calls
        │   scripts.enigma_live_forecast.main — see enigma_cfd_prop_forecast.py:40-48)
        ▼
scripts\enigma_live_forecast.py : main() :1221
  build_portfolio(config) → ensure_cache_ready → data fetch
     (IB via IBDataClient  |  MT5 via scripts.mt5_data_fetch.sync_mt5_dailies_into_central_cache
      |  use_cached_only)   :1394-1449
  portfolio.fit_from_cache(...) :1465
  positions_df = portfolio.predict_from_cache(query) :1483   ← FROZEN alpha core output
     cols: ['ticker','datetime','forecast_score','position_fraction']
  Step 9 sizing (profile-specific):
     futures_prop → calculate_futures_contracts(positions_df, prices, capital) :1542
     cfd_prop     → NO global $ sizing; per-account lots computed at execute time :1554-...
     personal     → ETF shares
  format_telegram_message + notifier.send_message :1620   ← signal still stands regardless
        ▼  Step-12 dispatch :1634
  cfd_prop → execution.run_mt5_execution.run_cfd_prop_execution(args, config, forecasts_df=shares_df) :1647
  else     → execution.run_execution.run_auto_execution(args, config, shares_df, capital, profile) :1653
```

Deployment supporting modules (forecast server lane — **STAYS, do not change** per 04 doc):
`deployment/forecast_server.py` (46 symbols), `deployment/forecast_prediction_runtime.py`,
`deployment/forecast_live_inputs.py`, `deployment/mt5_data_connector.py`
(`ForecastMT5DataConnector`). These produce forecasts/`position_fraction`; the alpha core
(`nodes/`→`feature_selection/`→`ensemble/`→`position_fraction`) is FROZEN (00 §3).

**What must NOT break (I3):**
1. The `enigma_*_forecast.py` → `enigma_live_forecast.main` → `predict_from_cache` →
   `position_fraction` chain (signal generation), regardless of executor.
2. The Telegram **signal** post (`format_telegram_message` / `send_message`) — it must keep firing
   even if execution is disabled/fails (`run_auto_execution` returns silently when `--execute` is
   off and never raises on safety violations; `:73-80`).
3. The Step-12 dispatch contract: `run_auto_execution(args, config, shares_df, capital, profile)`
   and `run_cfd_prop_execution(args, config, forecasts_df)` signatures.
4. The execute gating flags (`--execute`, `--dry-run-execute`, `--approve-via-telegram`, `--live`,
   `--allow-rerun`) and their semantics (`enigma_live_forecast.py:1242-1268`).
5. `deployment/ops/*.bat` calling `scripts\enigma_*` **by path** — the scripts/ reorg is HELD for exactly
   this. Moving/renaming these scripts breaks the scheduled tasks.

---

## 3. TradingNode migration target (per 04 doc)

**Target layout (NEW, additive):** `deployment/nautilus_mt5/` (vendored fork), `deployment/live/`
(`node.py`, `strategy.py`, `risk.py`, `config_ib.py`, `config_mt5.py`, `run_live.py`).

### 3a. IB executor → Nautilus IB adapter
- Current `IBTradeClient` (raw `ibapi`) → Nautilus **IB `ExecutionClient`** via
  `LiveExecClientFactory` under a `TradingNode` (Nautilus ships the IB adapter — `integrations/ib.md`).
- `compute_order_intents` delta math + dead-band → expressed by `TargetRebalanceStrategy` working
  the daily `position_fraction` target (shared with WP-3). `submit_order`/`wait_for_terminal`/
  cancel-on-timeout → Nautilus `OrderFactory` + `ExecutionEngine` order lifecycle/reconciliation.
- `position_sizer.py` stays the single contract-sizing authority (both backtest + live).

### 3b. MT5 executor → vendored mt5connect fork (decision log)
- Fork `aulekator/mt5-connect` into `deployment/nautilus_mt5/`, **pin a commit, preserve MIT
  license, audit order lifecycle + reconnection, add tests**. Do **not** depend on the PyPI dev
  package for funded accounts.
- `mt5_trade_executor._market_order`/`place_market_order`/`close_ticket` (esp. the
  `position=<ticket>` opposite-side close) → vendored MT5 `ExecutionClient`. The hedging-mode close
  semantics MUST be reproduced exactly or partial-closes will open hedges.
- mt5connect advertises startup + continuous **account reconciliation** → pair with Nautilus
  `ExecutionEngine` reconciliation (`OrderStatusReport`/`FillReport`/`PositionStatusReport`).

### 3c. Validation ladder (the gate — NOT parity)
1. **Nautilus sandbox** — vendored MT5 live `DataClient` streams real quotes into the node; Nautilus
   `SandboxExecutionClient`/`SimulatedExchange` fills locally (orders never leave the machine). NDX
   (MT5 feed already in WP-2) is the natural first instrument. Validates runtime/data/strategy/
   reconnection at zero broker risk.
2. **MT5 demo** (broker-side virtual fills) — validates real order lifecycle + reconciliation +
   partial fills + weekend-close + error handling. IB **paper** is the equivalent IB rung. (FTMO
   demo/trial fits here.)
3. **Shadow mode** (across rungs 1–2) — node runs in parallel with the legacy executor on the same
   forecasts; log intended orders from both; assert agreement (side/qty/instrument within tolerance).
4. **Governance preserved** at every rung (approval/audit/Telegram fire identically).
5. Only after 1–4 pass the soak → legacy executor retired in **WP-5**.

---

## 4. Risk surface (Phase-3 traps for later)

- **Order-lifecycle divergence vs the current executors.** Two subtle, easy-to-break mechanics:
  (a) IB cancel-on-timeout + abort-remaining-batch on REJECTED (`run_execution.py:299-305`);
  (b) MT5 hedging-mode close via opposite-side IOC `position=<ticket>` and **oldest-first
  partial-close summing to `reduce_lots`** (`mt5_rebalancer.py:148`, `mt5_trade_executor.py:373`).
  Nautilus netting/OMS differs from MT5 hedging tickets — a naive NETTING OMS would not reproduce
  per-ticket closes. **Must reconcile semantics in shadow mode before any demo→funded step.**
- **Deployment forecast path breaking** (I3 violation) — see §2 list. Any change to the
  enigma→predict→position_fraction→dispatch chain or the `deployment/ops/*.bat` paths is a release blocker.
- **Silent live cutover before the paper gate** — the highest-severity trap. `--execute` + `--live`
  + port 7496 (IB) or a funded MT5 login arms real orders. A new TradingNode must NOT be wired into
  the funded path until rungs 1–4 pass. Keep the legacy executor as the live default until WP-5.
- **RiskEngine / TradingState mapping** — map every `order_safety` + `mt5_order_safety` gate to a
  Nautilus `RiskEngine` pre-trade check (no weakening). Add **`TradingState` HALTED/REDUCING** to
  enforce prop-firm drawdown: `check_daily_loss_floor` (`mt5_order_safety.py:101`, default 5% FTMO
  DLL) → REDUCING/HALTED near the limit; `check_margin_budget` (`:82`, default 95% margin_free) →
  pre-trade denial. A missing safety check is a release blocker.
- **mt5connect maturity** (≈6 commits, hobby, Windows-only) — biggest single risk; vendoring + audit
  + paper soak mandatory. Windows-only is acceptable (FTMO VPS is Windows).
- **One node per process** (Nautilus global-singleton). IB + MT5 likely need two processes.

---

## 5. Order-safety / governance that MUST be preserved (re-home in WP-6, keep firing)

| Component | How it gates/records today | Migration note |
|---|---|---|
| `execution/order_safety.py` `run_all_preflight_checks` `:99` | Pure IB pre-flight: lock-file (one run/day), live-port-needs-flag, account-match, market-hours, max-orders. Raises `SafetyViolation` → batch aborts. | → Nautilus `RiskEngine` checks (no weakening). Lock-file/market-hours may stay as a bespoke risk component. |
| `execution/mt5_order_safety.py` `run_preflight` `:125` | Pure MT5 per-account: login/server match (fail-closed), trade_allowed/expert, symbol tradeable, trade-mode compat, **margin budget**, **daily-loss floor**. Returns failure list → account skipped. | → `RiskEngine` + `TradingState` (drawdown). |
| `execution/approval_flow.py` `request_approval` `:124` / `request_batch_approval` `:278` | Telegram inline-button human approval; whitelist of `authorized_user_ids` (empty = fail closed); `default_on_timeout` gated by `probe_health` so a Telegram outage never auto-approves (→ `POLL_UNHEALTHY`). | Re-home as a Nautilus `Actor`/gate around order submission; preserve fail-closed semantics exactly. |
| `execution/audit_log.py` `AuditLog` `:26` | Append-only JSONL per run (`run_start`/`intents`/`preflight`/`approval`/`order_placed`/`order_result`/`run_end`) + `positions_latest.json` for next-run reconciliation. | Re-home as an Actor subscribing to order/fill events; keep the JSONL + closing-positions contract. |
| `deployment/telegram_notifier.py` `TelegramNotifier` `:54` | Sends signal posts, approval keyboards, error/status; `probe_health` `:515` backs approval fail-closed. | Optionally wrap as an Actor subscribing to `on_order_filled` (04 doc) — not required, but must keep firing. |

These re-home as Nautilus Actors per **WP-6**, but **must keep firing** at every ladder rung (04 doc
acceptance: "Governance (approval/audit/Telegram) verified equivalent").

---

## 6. Stale-path drift table (doc → live)

| 04-doc reference | Live location (verified) | Status |
|---|---|---|
| `execution/ib_trade_executor.py` | `execution/ib_trade_executor.py` (`IBTradeClient:49`) | ✓ unmoved |
| `execution/mt5_trade_executor.py` | `execution/mt5_trade_executor.py` (`MT5TradeExecutor`) | ✓ unmoved |
| `execution/{rebalancer,mt5_rebalancer}.py` | same; `compute_order_intents:24`, `plan_symbol_actions:187` | ✓ unmoved |
| `execution/{order_safety,mt5_order_safety}.py` | same; `run_all_preflight_checks:99`, `run_preflight:125` | ✓ unmoved |
| `execution/{run_execution,run_mt5_execution,run_mt5_weekend_close}.py` | same | ✓ unmoved |
| `execution/position_sizer.py` | same (`Position:72`, `PositionSizer:106`) | ✓ unmoved |
| `execution/approval_flow.py`, `execution/audit_log.py` | same | ✓ unmoved |
| `deployment/{forecast_server,forecast_prediction_runtime,forecast_live_inputs,mt5_data_connector,telegram_notifier}.py` | all present, top-level `deployment/` | ✓ unmoved |
| `execution/models.py`, `execution/mt5_models.py` | same | ✓ unmoved |
| doc: "rebalancer/`plan_symbol_actions`, partial-close, `_quantize`" | `mt5_rebalancer.py:187/:148/:77` | ✓ accurate |
| doc implies `run_mt5_execution` orchestrates | top entry is `run_cfd_prop_execution` (calls `_build_account_plan:194`, `_execute_account:479`); IB path is `run_auto_execution:62` | ⚠ name nuance only |
| `scripts/enigma_*_forecast.py` "live-order portions" | thin wrappers → `scripts/enigma_live_forecast.py:main:1221`, Step-12 dispatch `:1634` | ✓ accurate |
| `deployment/ops/*.bat` cron entries | `run_prop_forecast.bat:24`, `run_personal_forecast.bat:27` call `scripts\enigma_*` by path; `run_mt5_scrape.bat` (data, unrelated) | ✓ — scripts/ reorg HELD for this reason |

**Confirmed:** `execution/` (15 files) and `deployment/` (6 files) are **unmoved** at top level.
`scripts/enigma_*` are deployment entrypoints referenced by `deployment/ops/*.bat`; the scripts/ reorg was
deliberately held. No stale paths require correction for WP-4 beyond the orchestrator-name nuance.

---

## 7. Recommended edit units (additive-first; gate = paper ladder, NOT parity), ordered

> All units are **additive** under `deployment/nautilus_mt5/` + `deployment/live/`. **Do not delete
> any legacy executor file** — that is WP-5, gated on the paper soak. **Do not touch a Frozen module
> (00 §3).** Each unit's evidence is a ladder rung, not a parity diff.

1. **Vendor mt5connect** → `deployment/nautilus_mt5/`: fork `aulekator/mt5-connect`, pin a commit,
   preserve MIT license, audit the order-lifecycle + reconnection paths, add unit tests.
   *Gate:* code audit notes + adapter unit tests pass; import-only smoke.
2. **Build the sandbox `TradingNode`** (`deployment/live/node.py` + `config_mt5.py` + `run_live.py`):
   vendored MT5 live `DataClient` → `SandboxExecutionClient`/`SimulatedExchange`. Instruments from
   the WP-2 catalog. NDX (MT5 feed already available) as first instrument.
   *Gate (ladder rung 1):* sandbox runs on live MT5 data with local virtual fills; strategy reacts
   in real time; reconnection verified.
3. **`TargetRebalanceStrategy`** (`deployment/live/strategy.py`): re-export/parametrize the WP-3
   strategy to consume daily `position_fraction` as registered `CustomData`; reproduce MT5
   partial-close + direction-flip + weekend-flatten and IB cancel-on-timeout semantics.
   *Gate:* sandbox order behaviours match the legacy planners on the same forecasts.
   *(Note: WP-3 Unit-2 is BLOCKED on the WP-2 precision fork; coordinate.)*
4. **`risk.py`** (`deployment/live/risk.py`): map every `order_safety` + `mt5_order_safety` gate to
   `RiskEngine` checks; add `TradingState` HALTED/REDUCING from `check_daily_loss_floor` +
   `check_margin_budget`. *Gate:* each legacy gate has a covering Nautilus check (1:1 table); no
   weakening.
5. **Governance Actors** (WP-6 cross-cut, but wire here so they fire): approval-flow gate, audit-log
   Actor (subscribe order/fill events), Telegram `on_order_filled`. *Gate:* approval/audit/Telegram
   verified equivalent at every rung.
6. **Demo rung** (`config_ib.py` IB paper + MT5 demo `ExecutionClient`): route to broker demo/paper.
   *Gate (rung 2):* node connects to IB paper + MT5 demo; instruments load; partial-close /
   weekend-close / flatten reproduced.
7. **Shadow mode + runbook**: run node in parallel with legacy executor; log + diff intended orders;
   write start/stop/crash-recovery/rollback runbook. *Gate (rungs 3–4 + acceptance):* documented
   order-agreement (none / within tolerance) over the soak window; governance equivalent; runbook
   complete.

**Hand-back (per 04 §Subagent instructions):** the node, vendored+audited MT5 adapter, shadow-mode
agreement report, paper-soak results, and the runbook. Legacy cull is WP-5.

---

> **This WP cannot start implementation without explicit user go on the broker / paper-trade
> approach** (which broker rung first — IB paper vs MT5 demo / NDX sandbox — vendoring the
> mt5connect fork commit, and the soak duration). No funded account is touched at any point in this
> WP; cutover is reversible until WP-5.

---

## 8. Live adapter connection test — FTMO demo (2026-06-05) ✓ PASSED

Driver: `scripts/dev/mt5_adapter_test.py` (broker-agnostic; `--broker <PREFIX>` reads
`<PREFIX>_DEMO_*` from `.env`, `--path` selects the terminal install). Run:
`--broker FTMO --path "C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"`.
This is **partial ladder rung-2 evidence** (real broker demo, real order lifecycle) — NOT a
sandbox (rung 1) and NOT a full strategy/governance run.

**Environment finding (the blocker, now resolved):** MT5's Python bridge drives a *running
terminal*, and each terminal install ships only its own broker's server list. The single prior
install was a **Darwinex** terminal (`D0E8…075`, base `Darwinex-Live` only) → `mt5.login("FTMO-Demo")`
and `login("FundedNext-Server3")` both `-10005 IPC timeout` (server not resolvable). Fix: install
the broker's OWN MT5 terminal (FTMO Global Markets, data folder `81A933…3850`, base `FTMO-Demo`
present) and bind to it via `mt5.initialize(path=<that terminal64.exe>)`.

**ADAPTER GAP (additive fix for Unit 1):** `mt5connect.config.MT5Config` has **no `path` field**, and
`MT5Connection._initialize()` calls bare `mt5.initialize()`. On a multi-terminal machine (live
Darwinex + demo FTMO) that binds non-deterministically. The test driver works around it by
pre-`initialize(path=...)` before `MT5Connection.connect()` (a subsequent bare attach binds to the
same terminal). **Add a `path` field to `MT5Config` + thread it through `_initialize()`** when
vendoring (edit unit 1). One-node-per-process (§4) still means two brokers = two processes.

**Validated layers (against FTMO-Demo, $100k, lev 1:30, `trade_mode=DEMO`):**
- L1-3 `MT5Connection` initialize→login→`get_account_info()` ✓
- L4 `MT5InstrumentProvider.load_symbol` → Nautilus instruments: `US100.cash`→`Cfd` (pp=2),
  `EURUSD`→`CurrencyPair` (pp=5) ✓
- L5-6 live tick + H1 `copy_rates_range` ✓ (index CFDs only tick during session hours)
- L7-8 `MT5DataClient` + `MT5LiveExecutionClient` construct against real NT msgbus/cache ✓
- L9 **round-trip order via the adapter's exact `order_send` request shape** (DEMO-guarded): BUY
  0.01 EURUSD @ 1.16150 (IN) → SELL 0.01 @ 1.16145 (OUT), net ≈ -$0.11 (½-spread + 2×-$0.03 comm),
  **account flat, 0 residual** (verified via `history_deals_get`). ✓

**Gotchas captured (fixed in the driver):** (a) FTMO terminal needs **Algo Trading ON** or
`order_send`→`10027`; (b) under MT5 *market execution* the `order_send` result returns `deal=0
price=0` even on success — the fill only appears in `history_deals_get` a beat later (treat history
as authoritative + settle-delay before flat check). HARD DEMO guard (`trade_mode==DEMO`) aborts
before any order on a non-demo account.

**Symbol universe confirmed (166 symbols) → `configs/mt5_brokers.yaml` `ftmo` now `confirmed: true`:**
`NQ→US100.cash, ES→US500.cash, YM→US30.cash, DAX→GER40.cash, FTSE→UK100.cash, GC→XAUUSD,
SI→XAGUSD, CL→USOIL.cash, NG→NATGAS.cash`, FX 1:1. Every canonical matched exactly one native
symbol (enumerated, not guessed). `brokers.resolve('ftmo',…)` / `canonical_for` verified.

**Still pending for rung-2 proper:** index-CFD order during session hours (today closed), partial-close
/ direction-flip / weekend-flatten semantics, governance (approval/audit/Telegram) firing, and the
sandbox rung-1 (`SandboxExecutionClient`) which was skipped by going straight to the demo broker.
