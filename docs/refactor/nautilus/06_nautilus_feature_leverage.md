# WP-6 — Nautilus Feature-Leverage Audit

> Cross-cutting capability inventory: for each meaningful Nautilus feature, **are we
> using it, and where?** Verified against `docs/nautilustrader/` (concepts:
> orders/advanced, execution, continuous_futures, custom_data, reports, actors,
> backtesting, architecture). This is the "are we using the full feature set" check.

## Status legend

- **ADOPT** — use it; folded into a WP below.
- **ADOPT (parity-gated)** — use only if it reproduces existing numbers within tolerance.
- **DEFER** — valuable, not in the current scope; revisit.
- **DECLINE** — deliberately not used (reason given). Avoids scope creep / protects invariants.

---

## A. Data & persistence

| Feature | Status | Where / note |
|---|---|---|
| `ParquetDataCatalog` (instruments + bars + custom data) | **ADOPT** | WP-2 — canonical store. |
| Instrument types (`FuturesContract`/`Equity`/`Cfd`/`CurrencyPair`), tick scheme, precision/increment | **ADOPT** | WP-2 — from `InstrumentCatalog` rows. |
| `BarDataWrangler` / bar types / `ts_init=close` convention | **ADOPT** | WP-2 — ingest. |
| **Custom data** (`register_custom_data_class`, `CustomData`, `DataType`, catalog persistence, `on_data` routing) | **ADOPT** | WP-3 & WP-4 — the clean way to feed the daily `position_fraction`/forecast into the strategy (subscribe/`on_data`) **and** persist signals in the catalog for reproducible runs. Replaces ad-hoc "lookup table" plumbing. |
| **Continuous futures** (native splicing; `BACKWARD/FORWARD_SPREAD/RATIO`, transition metadata) | **ADOPT (parity-gated)** | WP-2 — directly relevant to the Norgate roll/back-adjustment work. **Caveat:** the engine does *not* discover rolls or infer prices — caller supplies transitions; and the adjusted series must match the existing back-adjusted series + EWSD σ handling within tolerance or research parity breaks. Safe default: keep the existing adjusted series as stored bars; use Nautilus continuous-future for the **live node's** roll handling and as an *optional* validated path for futures. |
| `request_aggregated_bars` / internal bar aggregation from ticks | **ADOPT** | WP-3 — aggregate M1/tick → higher bars on the fly for the realism lane. |
| Built-in vendor data adapters (Databento, Tardis) | **DECLINE** | You have Norgate + MT5 providers already; keep them feeding the catalog (WP-2). Revisit Databento only if you need intraday futures (see WP-3 futures prerequisite). |
| Redis/Postgres-backed catalog & state persistence | **DEFER** | Useful for live state durability (crash-only restart). Consider in WP-4 hardening. |

## B. Execution realism (backtest lane — WP-3)

| Feature | Status | Where / note |
|---|---|---|
| Order types: `MARKET`, `LIMIT`, `STOP_MARKET`, `STOP_LIMIT`, `MARKET_TO_LIMIT`, `*_IF_TOUCHED`, trailing stops | **ADOPT** | WP-3 `ExecutionPolicy` — the whole point (limit vs market, spread capture). |
| **Bracket orders + contingency (`OTO`/`OCO`/`OUO`, `reduce_only`)** via `OrderFactory.bracket()` | **ADOPT** | WP-3 & WP-4 — native entry+SL+TP; replaces homegrown partial-close/SL bracketing. Note default backtest venue uses **partial-trigger** OTO (`oto_trigger_mode` to change). |
| **Execution algorithms** (built-in `TWAP`, custom `ExecAlgorithm` spawning child orders) | **ADOPT** | WP-3 & WP-4 — "work a daily target intraday" (slice/limit-then-cross) maps exactly to an `ExecAlgorithm` or the `CROSS_AFTER` policy. |
| Order emulation (`OrderEmulator`) for venues lacking native types | **DEFER** | Use if the MT5 broker lacks a needed order type (e.g. native trailing). |
| `FillModel` family (`BestPriceFillModel`, `OneTickSlippage`, tiered, probabilistic, `SizeAware`, `MarketHours`, `VolumeSensitive`, ...) | **ADOPT** | WP-3 — frictionless `BestPriceFillModel` for the parity reconciliation; realistic models for the actual study. |
| `LatencyModel` | **ADOPT** | WP-3 — model order-arrival latency; also how close-to-close daily fills are achieved. |
| Bar `bar_adaptive_high_low_ordering`, `price_protection_points`, `liquidity_consumption`, `queue_position`, `trade_execution` | **ADOPT** (as knobs) | WP-3 — venue-config realism knobs; document the chosen settings. |
| FX rollover interest / funding `SimulationModule` | **DEFER** | Relevant to CFD overnight **swap** for `CLOSE_TO_CLOSE` holds; wire when modeling overnight cost. |
| OMS type `NETTING` | **ADOPT** | WP-3 & WP-4 — single position per instrument matches the `position_fraction` target model; deterministic `{instrument}-{strategy}` position id. |

## C. Risk & accounting

| Feature | Status | Where / note |
|---|---|---|
| **`RiskEngine`** pre-trade checks (`max_notional_per_order`, instrument `max_notional`, qty/price precision, `reduce_only` enforcement, rate limits) | **ADOPT** | WP-4 — maps `order_safety`/`mt5_order_safety`. Every existing safety rule must map here or to a kept bespoke component (no weakening). |
| **`TradingState` (`ACTIVE`/`HALTED`/`REDUCING`)** | **ADOPT** | WP-4 — natural fit for **prop-firm drawdown/limit enforcement**: flip to `REDUCING` (only exposure-reducing orders) or `HALTED` when nearing a daily/overall loss limit. High-value, previously unplanned. |
| Account types: `MARGIN`/`CASH`, single- & multi-currency | **ADOPT** | WP-3 (backtest venue) & WP-4 — margin account for futures/CFD prop sims; matches `FuturesSimConfig` margin intent. |
| Greeks / options accounting | **DECLINE** | Not traded (futures/CFD/equities). |

## D. Live runtime (WP-4)

| Feature | Status | Where / note |
|---|---|---|
| `TradingNode` + `LiveDataClientFactory`/`LiveExecClientFactory` | **ADOPT** | WP-4 — IB adapter + vendored mt5connect. |
| **Sandbox environment** (live data + local simulated fills) | **ADOPT** | WP-4 rung 1 + continuous staging env. |
| Execution **reconciliation** (`OrderStatusReport`/`FillReport`/`OrderWithFills`/`PositionStatusReport`, overfill handling, dedup) | **ADOPT** | WP-4 — robust live state recovery; replaces bespoke reconciliation. |
| Crash-only design + externalized state (Redis) | **DEFER** | WP-4 hardening — durable restart for the funded node. |
| `MessageBus` pub/sub (+ optional external/Redis streaming) | **ADOPT (light)** | WP-4 — decouple forecast publication → execution; also how Actors/strategy communicate. |
| Own order books / order-fill & cancel subscriptions | **ADOPT (light)** | WP-4 — fill/cancel subscriptions drive Telegram notifications + audit. |

## E. Components & orchestration

| Feature | Status | Where / note |
|---|---|---|
| **`Actor`** (timers, data/custom-data subscriptions, cache/portfolio access, msgbus, `on_order_filled`) | **ADOPT** | WP-4 — re-home deployment pieces: forecast publisher, **Telegram notifier** (fill/status), periodic **ReportingActor**, volatility refresh, weekend-close scheduler. |
| `Strategy` (shared `TargetRebalanceStrategy`) | **ADOPT** | WP-3 & WP-4 — one strategy, backtest→sandbox→demo→live. |
| `Clock` timers/alerts | **ADOPT** | WP-4 — session boundaries, scheduled forecasts, weekend close. |
| `BacktestNode` + `BacktestRunConfig` (high-level, multi-run) | **ADOPT** | WP-3 — parameterized/parallel realism runs across configs (vs single low-level `BacktestEngine`). |
| Rust-native strategies/actors | **DECLINE** | Python control plane is sufficient; no need for the Rust toolchain path. |

## F. Analytics & reporting

| Feature | Status | Where / note |
|---|---|---|
| **`ReportProvider`** (orders / order-fills / fills / positions / account reports as DataFrames; `liquidity_side` MAKER/TAKER, commission, realized PnL per position) | **ADOPT (additive)** | WP-3 & WP-4 — **execution-quality diagnostics** the vectorized lane cannot produce (realized spread, slippage, fill rate, commissions). This is *diagnostics*, not return-metrics, so it does **not** conflict with keeping `quantfoundry_core`. |
| Real-time `Portfolio` (live PnL, exposure, margin) | **ADOPT** | WP-4 — live monitoring/guardrails (feeds `TradingState` decisions, Telegram). |
| `PortfolioAnalyzer` / `PortfolioStatistic` (return metrics) | **ADOPT (additive reference)** | WP-3 & WP-4 — run **alongside** `quantfoundry_core` so its return stats can be compared side-by-side (user wants to evaluate whether they're better). QF stays **authoritative** for gating (I4); no replacement without a reviewed switch. A custom `PortfolioStatistic` can wrap QF metrics if you later want them inside Nautilus reports. |
| `create_tearsheet` (Plotly) | **ADOPT (optional reference)** | WP-3 — emit the Plotly tearsheet as an extra reference artifact next to QuantStats/Matplotlib; the repo viz policy still governs the primary research charts. |
| Nautilus `indicators` | **DECLINE** | Alpha core is frozen (50+ custom bias nodes are the IP); no parity-safe swap. |

---

## Decline log (explicit, so we don't re-litigate)

- **Nautilus return metrics / tearsheets** — now **ADOPT (additive reference)** per user request: run Nautilus `PortfolioAnalyzer` + tearsheet **alongside** `quantfoundry_core` for comparison. QF remains **authoritative** for gating (I4); QuantStats/Matplotlib remain the primary research charts.
- **Nautilus indicators** — alpha core frozen for parity.
- **Vendor data adapters (Databento/Tardis), options/greeks, betting/crypto adapters, Rust strategies** — out of asset scope or redundant with existing providers.

## Net new items this audit adds to earlier WPs

1. **Custom-data signal injection** (A) → tighten WP-3/WP-4 to pass `position_fraction` as registered `CustomData` (persistable, subscribable) instead of an ad-hoc table.
2. **Bracket/contingent orders + ExecAlgorithm/TWAP** (B) → expand WP-3 `ExecutionPolicy` and WP-4 live order construction.
3. **`TradingState` for prop-firm limits** (C) → add to WP-4 risk design (HALTED/REDUCING on drawdown).
4. **Execution reconciliation + Actors** (D/E) → WP-4 reconciliation + deployment re-home (Telegram/reporting as Actors).
5. **`ReportProvider` execution diagnostics + `PortfolioAnalyzer` return stats + real-time Portfolio** (F) → additive reports in WP-3/WP-4 **alongside** quantfoundry (QF authoritative for gating; Nautilus metrics for side-by-side comparison/evaluation).
6. **Continuous-futures engine** (A) → WP-2 optional, parity-gated, for the live node's roll handling.

## Subagent instructions

- Treat this as the capability checklist. When implementing a WP, confirm each ADOPT item
  for that WP is actually wired (or consciously DEFERRED with a note).
- Do not adopt a DECLINE item without raising it — several protect invariant I4 or the
  frozen alpha core.
