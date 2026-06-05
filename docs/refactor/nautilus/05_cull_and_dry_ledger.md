# WP-5 — Cull & DRY Ledger (guarded deletion)

> **Depends on:** the relevant gate for each row. **Nothing here is deleted until its
> guard is green.** This is the "cull aggressively to stay DRY" work — done safely.

## Rules

1. A row may be culled **only** when its **Guard** column is satisfied (parity green /
   candle-equality green / paper-soak passed), and after the agreed bake-in period.
2. Cull = delete the redundant code **and** its now-dead tests/imports/configs; update
   call sites to the Nautilus replacement; run the parity harness again.
3. If removing something would force an edit to a Frozen module (alpha core), **stop** —
   it isn't redundant, it's load-bearing. Re-scope.
4. Prefer deletion over deprecation shims. Only keep a shim if an external consumer
   (deployment script, notebook) needs a transition window; track it for removal.

## Status legend

`PLANNED` (guard not yet met) · `READY` (guard met, safe to cull) · `DONE`.

## Ledger

### From WP-2 (data layer) — guard: candle-equality test + WP-1 parity green with Nautilus default

| Module / symbol | Replaced by | Guard | Status |
|---|---|---|---|
| `data_platform/loaders.py` (`load_data`, `load_stock_data`, `_normalize_loaded_frame`, `ohlc_data_dir`) | `data_platform/nautilus/candles.py` adapter over `ParquetDataCatalog` | candle-equality + parity | PLANNED |
| `data_platform/core/catalog.py` (`InstrumentCatalog` read/write) | Nautilus instruments in the catalog (`data_platform/nautilus/instruments.py`) | instrument round-trip + parity | PLANNED |
| `data_platform/core/source_priority.py` (`load`, `configs/source_priority.yaml`) | catalog ingest precedence in `data_platform/nautilus/ingest.py` | candle-equality | PLANNED |
| `data_platform/core/{enums,identifiers,instruments}.py` (homegrown domain types) | Nautilus `Instrument`/`InstrumentId`/`AssetClass`/`InstrumentClass` | all consumers migrated | PLANNED |
| `utils/.../helpers.load_data_multi_ticker` (legacy branch) | adapter routing (flag removed once default-on) | parity green default-on | PLANNED |

> Keep the **provider scrapers** (`data_platform/providers/norgate/*`,
> `data_platform/providers/mt5/*`) — they fetch raw vendor data. They are *re-pointed*
> to write into the Nautilus catalog (via `ingest.py`), not deleted. Only their
> bespoke read/serialize helpers that the catalog now owns are cull candidates.

### From WP-3 (P&L lane) — guard: `pnl_engine` protocol routed, WP-1 parity green

| Module / symbol | Replaced by | Guard | Status |
|---|---|---|---|
| Direct calls to `calculate_strategy_returns_from_positions` scattered across pipelines | `PnLEngine` protocol (default `VectorizedPnLEngine` wraps it verbatim) | parity green | PLANNED |

> The function itself is **NOT** culled — it *is* the vectorized lane. Only the
> ad-hoc direct call sites are unified behind the protocol. `futures_sim.py` and
> `quantfoundry_core.prop_firm` are **kept as-is** (user decision).

### From WP-4 (live execution) — guard: shadow-mode agreement + paper soak passed

| Module / symbol | Replaced by | Guard | Status |
|---|---|---|---|
| `execution/ib_trade_executor.py` | Nautilus IB `ExecutionClient` under `TradingNode` | paper soak (IB) | PLANNED |
| `execution/mt5_trade_executor.py` | vendored MT5 `ExecutionClient` (`deployment/nautilus_mt5/`) | paper soak (MT5 demo) | PLANNED |
| `execution/rebalancer.py`, `execution/mt5_rebalancer.py` | `TargetRebalanceStrategy` (shared with WP-3) | shadow-mode agreement | PLANNED |
| `execution/order_safety.py`, `execution/mt5_order_safety.py` | Nautilus `RiskEngine` checks + kept bespoke rules | every rule mapped + verified | PLANNED |
| `execution/run_execution.py`, `run_mt5_execution.py`, `run_mt5_weekend_close.py` | `deployment/live/run_live.py` + node config | paper soak incl. weekend-close | PLANNED |
| live-order portions of `scripts/enigma_*_forecast.py` | thin node launchers | paper soak | PLANNED |
| `execution/models.py`, `execution/mt5_models.py` (order/exec models) | Nautilus order/position/account model | all consumers migrated | PLANNED |

> **Keep:** `execution/position_sizer.py` (shared sizing authority),
> `execution/approval_flow.py`, `execution/audit_log.py` (bespoke governance — re-homed,
> not deleted). `deployment/forecast_*` + `telegram_notifier.py` stay (forecast gen +
> notifications). `prop_firms/` stays.

## Definition of done (whole refactor)

- All `READY` rows culled; repo has a single data path (Nautilus catalog), a single
  sizing/execution strategy shared by backtest + live, and a single metrics layer
  (`quantfoundry_core`).
- WP-1 parity harness green (vectorized lane identical to pre-refactor baseline).
- Nautilus P&L lane available via `pnl_engine="nautilus"` and documented.
- Live execution on `TradingNode` (IB + MT5) past paper soak, with rollback runbook.
- No dead code/imports/tests remain for culled modules; `grep` for the old symbols is clean.

## Subagent instructions

- Cull **one guard-group at a time**; run WP-1 parity after each group.
- Use `codegraph_impact <symbol>` before deleting any symbol to catch non-obvious
  consumers (notebooks, scripts, configs).
- If a "redundant" module turns out to have a consumer with no Nautilus equivalent,
  move it to a "Keep" note here with the reason — do not force-delete.
