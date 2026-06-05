# Nautilus Trader Refactor — Overview & Operating Contract

> **Status:** PLAN (not yet implemented). These documents are the hand-off
> artifacts for implementation subagents. Read this file first, then the
> work-package file relevant to your task.

## 1. Goal

Adopt [NautilusTrader](../../nautilustrader/) as the core trading infrastructure
**at the edges** (data, execution simulation, live execution) while preserving the
firm's alpha IP and research results exactly. Cull homegrown code that Nautilus can
own, to keep the codebase DRY — but only behind hard validation gates.

## 2. Non-negotiable invariants (the whole plan is gated on these)

| # | Invariant | How it is enforced |
|---|-----------|--------------------|
| I1 | `feature_research` produces **near-identical** results before/after every change. | Golden-snapshot parity harness — see [01_parity_harness.md](01_parity_harness.md). |
| I2 | `portfolio_research` produces **near-identical** results before/after every change. | Same harness. |
| I3 | Deployment **keeps working** (forecasts + live orders) throughout. Live execution is *migrated*, not broken — and only cut over after a paper-trading gate. | See [04_live_execution_trading_node.md](04_live_execution_trading_node.md). |
| I4 | `quantfoundry_core` stays the **authoritative** metrics/robustness source for all gating decisions. Nautilus `PortfolioAnalyzer`/`ReportProvider` may run **alongside as an additive reference** for comparison; it does **not** replace QF unless a switch is explicitly reviewed. | Code review; QF drives the gates. |
| I5 | No cull/delete happens until the replacement passes the relevant gate (parity or paper-trade). | [05_cull_and_dry_ledger.md](05_cull_and_dry_ledger.md) — every row has a guard. |

If a change would violate I1–I5, it is out of scope for that work package; stop and
escalate rather than weaken a gate.

## 3. Layer map — what changes vs. what is frozen

```
                          ┌─────────────────────────────────────────────┐
 DATA (WP-2: REPLACE)     │ data_platform loaders/catalog/providers      │
                          │   →  Nautilus ParquetDataCatalog + Instrument │
                          │      + Bar wranglers, behind an adapter that  │
                          │      yields the SAME candle DataFrame shape   │
                          └───────────────────────┬─────────────────────┘
                                                  │ candles (unchanged schema)
                          ┌───────────────────────▼─────────────────────┐
 ALPHA CORE (FROZEN)      │ nodes/ → feature_selection/base_models/      │
   *** DO NOT TOUCH ***   │   → ensemble/ (DiversifiedEnsemble,          │
   This is the IP.        │      WeightLayer, TFPortfolio, GlobalPortfolio)│
   Nautilus has no        │   → position_fraction                        │
   equivalent.            └───────────────────────┬─────────────────────┘
                                                  │ ['ticker','datetime',
                                                  │  'forecast_score','position_fraction']
                          ┌───────────────────────▼─────────────────────┐
 P&L (DUAL LANE)          │  PnLEngine protocol (NEW):                   │
                          │   • "vectorized" (DEFAULT) = existing         │
                          │     calculate_strategy_returns_from_positions │
                          │     → EXACT parity, fast. [FROZEN behaviour]  │
                          │   • "nautilus"  (OPT-IN, WP-3) = BacktestEngine│
                          │     realistic fills/margin/contracts. ADDITIVE│
                          └───────────────────────┬─────────────────────┘
                                                  │ returns / equity series
                          ┌───────────────────────▼─────────────────────┐
 METRICS (FROZEN)         │ quantfoundry_core (metrics, robustness,      │
                          │ prop_firm, portfolio_gate) — UNCHANGED        │
                          └──────────────────────────────────────────────┘

 LIVE EXECUTION (WP-4: MIGRATE)
   execution/ib_*  + execution/mt5_* + deployment/  →  Nautilus TradingNode
   (IB adapter + vendored mt5connect fork), gated by paper trading.
```

**Frozen (must not change behaviour):** `nodes/`, `feature_selection/`,
`ensemble/` (incl. `weight_layer.py`, `weight_hierarchy.py`,
`portfolio_impl/`), `feature_extraction/feature_extractor.py` math, and the
`quantfoundry_core` integration points.

## 4. Decisions already made (decision log)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Research P&L | **Dual lane.** Vectorized stays the source of truth (default, exact parity). Nautilus `BacktestEngine` is an *additive, opt-in* realistic-execution lane. | Nautilus bar backtesting has **no native next-bar-open fill** (a bar's `ts_init` is its close; filling at the next open would be look-ahead). Our default research convention is `log_intraday` = enter `open[t+1]`, exit `close[t+1]`. So Nautilus cannot reproduce research numbers bit-for-bit; it must not overwrite them. See [03_backtest_validation_lane.md](03_backtest_validation_lane.md). |
| Discrete-contract / prop-firm sim | **Keep as-is for now.** `portfolio_research/futures_sim.py` and `quantfoundry_core.prop_firm` untouched. | User decision. Revisit after data + research lanes land. |
| Live execution / MT5 | **Vendor/fork `aulekator/mt5-connect`** (MIT Nautilus MT5 adapter), paper-trade gate, unify IB + MT5 under `TradingNode`. | Nautilus ships an IB adapter but **no MT5 adapter**; CFD prop runs on MT5. mt5connect is early-stage (≈6 commits, hobby, Windows-only) so it must be vendored, audited, pinned, and paper-validated before funded accounts. |
| Metrics | **Keep `quantfoundry_core` as authoritative; ADD Nautilus `PortfolioAnalyzer`/`ReportProvider` as an additive reference lane.** | QF is agnostic to how returns are produced and drives gating. Nautilus metrics run side-by-side so they can be evaluated (they may be better — user wants to compare). No replacement without a reviewed switch. |
| Alpha core | **Freeze.** No Nautilus indicators. | 50+ custom bias nodes are the IP; no parity-safe replacement exists. |
| Pre-Nautilus cleanup (WP-8) | **Do it first** (after WP-1): cull dead/QF-outsourced modules + reorganize. **Restructure approach = "consolidate flat top-level"** (approved 2026-06-04); single-`src/` and minimal-only declined. | Smaller, modular base makes every Nautilus WP cheaper; one parity-gated import churn up front beats interleaving it with the migration. |

## 5. The dual-lane P&L design (`PnLEngine`)

Both lanes consume the identical `position_fraction` frame and terminate at the same
`quantfoundry_core` metrics, so their reports are directly comparable.

```python
# Conceptual interface — see 03_backtest_validation_lane.md for the full spec.
class PnLEngine(Protocol):
    def returns_from_positions(
        self,
        positions_df: pd.DataFrame,   # ['ticker','datetime','position_fraction', ...]
        candles_df: pd.DataFrame,
    ) -> pd.Series: ...               # returns indexed by realization datetime

# Lane 1 (DEFAULT, frozen behaviour): wraps the existing function verbatim.
#   calculate_strategy_returns_from_positions(..., instrument_return_kind='log_intraday')
# Lane 2 (OPT-IN, additive): runs a BacktestEngine Strategy that rebalances to
#   position_fraction with frictions/margin/contracts, returns the account return series.
```

Selection is one config field, default `vectorized`:

```
pnl_engine: "vectorized" | "nautilus"   # default "vectorized"
```

This gives the "fast iterate on vectorized, switch to Nautilus for realism" workflow
without ever mutating the parity baseline.

## 6. Work packages & order of execution

Do them in this order. Each WP has its own file with tasks, file/symbol maps,
acceptance gates, and a guarded cull list.

| WP | File | Depends on | One-line scope |
|----|------|-----------|----------------|
| WP-1 | [01_parity_harness.md](01_parity_harness.md) | — | Build golden-snapshot parity harness; capture baselines **before any change**. The gate for everything else. |
| WP-2 | [02_data_layer.md](02_data_layer.md) | WP-1 | Replace homegrown data layer with Nautilus catalog/instruments/bars behind an identical-DataFrame adapter. |
| WP-3 | [03_backtest_validation_lane.md](03_backtest_validation_lane.md) | WP-1, WP-2 | Add the opt-in Nautilus `BacktestEngine` P&L lane (`PnLEngine`). Additive only. |
| WP-4 | [04_live_execution_trading_node.md](04_live_execution_trading_node.md) | WP-2 | Migrate IB + MT5 live execution to `TradingNode`; paper-trade gate. |
| WP-5 | [05_cull_and_dry_ledger.md](05_cull_and_dry_ledger.md) | all above | Execute the guarded cull/delete ledger once each replacement passes its gate. |
| WP-6 | [06_nautilus_feature_leverage.md](06_nautilus_feature_leverage.md) | cross-cuts all | Feature-leverage audit: which Nautilus capabilities we ADOPT / DEFER / DECLINE, and where. Use as the checklist while implementing each WP. |
| WP-7 | [07_docs_refactor.md](07_docs_refactor.md) | Phase A standalone; Phase B per-WP | Refactor `docs/`: Phase A fixes docs to the **current** codebase; Phase B updates docs as each Nautilus WP lands. Excludes vendored `docs/nautilustrader/`. |
| WP-8 | [08_repo_cleanup_and_restructure.md](08_repo_cleanup_and_restructure.md) | WP-1 | **Pre-Nautilus cleanup:** cull dead / `quantfoundry_core`-outsourced modules and reorganize the scattered folder layout. Parity-gated; runs before the Nautilus WPs. |

**Execution sequence:** `WP-1` (parity harness — the gate) → **`WP-8`** (cleanup &
restructure, on a tidy base before any Nautilus work) → `WP-2` → `WP-3` → `WP-4` → `WP-5`.
`WP-6` (feature audit) cross-cuts as the checklist. `WP-7` (docs) is continuous: **Phase A
is done**; a mechanical **path-refresh follows WP-8** (the reorg moves modules); **Phase B**
updates docs as each Nautilus WP lands.

## 7. How implementation subagents must work

1. **Read this file + your WP file fully before editing.**
2. **Use `codegraph` first** (`codegraph_explore`, `codegraph_callers`,
   `codegraph_impact`) to confirm exact call sites and blast radius before changing
   anything. The symbol/file references in these docs were accurate at plan time but
   verify against the live index.
3. **Never edit a Frozen module** (§3) to make something else work. If the only way
   to land a change is to alter the alpha core, stop and escalate.
4. **Run the parity harness (WP-1) before and after** every change to a research-path
   file. A non-trivial parity diff is a failure, not a "to be reviewed later".
5. **Do not delete anything** outside the guarded ledger in WP-5, and only when its
   guard condition is satisfied.
6. **Environment (Windows):** use the repo venv interpreter directly, e.g.
   `.\.venv\Scripts\python.exe -m pytest ...` (or `.\venv\Scripts\python.exe`
   if that is the folder that exists). Never create a new venv. See repo `CLAUDE.md`.

## 8. Glossary of Nautilus pieces referenced

- `ParquetDataCatalog` — Nautilus-format Parquet store for instruments + market data.
- `Bar` / `BarType` / `BarSpecification` — Nautilus OHLCV bar + its (instrument, step,
  aggregation, price-type) identity.
- `BarDataWrangler` — converts a DataFrame of OHLCV into Nautilus `Bar` objects
  (handles the `ts_init = close` convention via `ts_init_delta`).
- `Instrument` subtypes — `FuturesContract`, `Equity`, `Cfd`, `CurrencyPair`, etc.
- `BacktestEngine` (low-level) / `BacktestNode` (+`BacktestRunConfig`, high-level) —
  deterministic event-driven simulation.
- `FillModel`, `LatencyModel` — execution realism knobs for the backtest venue.
- `TradingNode` + `LiveDataClientFactory` / `LiveExecClientFactory` — live runtime and
  adapter plug points.
- `Strategy` — event-driven component receiving `on_bar`/`on_quote_tick` and submitting
  orders. In our design it is a *thin rebalancer* driven by precomputed `position_fraction`.
