# Data feed & execution — documentation index

> How we source price data for research and execution. **One-line policy:** trade
> and research **CFD-native by default** (only feed with tick data → realistic
> spreads); use **futures only for equity-index research** (longer history), then
> validate on CFD. Full rationale in [[feed_and_execution_decision]].

## Which feed for what

| Job | Feed | Doc |
|---|---|---|
| Execution (all instruments) | **Darwinex CFD** | [[feed_and_execution_decision]] |
| Realistic backtest / cost modelling | **CFD bars + on-demand ticks** | [[hybrid_tick_backtest]] |
| Research — equity indices | **ratio-adjusted futures** → validate on CFD | [[futures_research_data]] |
| Research — gold/silver + everything else | **CFD-native** | [[feed_comparison_and_adjustment]] |

**The three futures series (index research only):** `additive` → signals (point
moves), `ratio` → returns/σ (%), `unadjusted` → real prices. See [[futures_research_data]].

## Start here

- **[[feed_and_execution_decision]]** — the decision of record: signal source,
  execution venue, the research split, open-to-close convention, swap economics.

## Reference docs

### Feeds & adjustment
- [[feed_comparison_and_adjustment]] — CFD↔futures daily fidelity (0.99), commodity-cash
  clock-time quality, gold/silver future↔CFD transfer, the data-quality guardrail.
- [[futures_research_data]] — the futures research feed: three-series rule, additive
  distortion (k≈0.69) + ratio fix, roll mechanics, Norgate provider, Norgate→IB handover.

### CFD / MT5 infrastructure (the primary feed)
- [[mt5_timezones]] — **read first for anything MT5-relative.** Timestamps are broker
  EET, not UTC (ET = stored − 7h); rollover at 00:00 broker = 17:00 ET.
- [[hybrid_tick_backtest]] — bars spine + on-demand tick cache for realistic fills.
- [[mt5_data_scraper]] — the Darwinex scraper, storage layout, scheduled task.
- [[mt5_broker_config]] — per-broker symbols/sessions/rules (`configs/mt5_brokers.yaml`).
- [[darwinex_universe]] — the 844-symbol Darwinex trading universe + history depth.

### Storage architecture (schemas, invariants, the metadata DB question)
- [[storage_and_registry_plan]] — **proposal of record.** Payload stays parquet; a small
  rebuildable SQLite registry indexes the files + holds lineage. Phase-0 schema contracts,
  phased rollout, grounded DDL, the `del registry.db` rollback invariant.
- [[data_store_inventory]] — every on-disk store (28 of them): path, key, enforcement,
  ~file count, and the parquet-blob-vs-relational verdict that drives the plan.

## Decision register (what was decided)

1. **Execute exclusively on Darwinex CFDs.**
2. **Research CFD-native by default;** futures only for the equity indices (1997+
   history + multi-vendor cross-check), with a mandatory **CFD-native promotion gate**.
3. **Production signal = CFD-native at the 17:00 ET rollover** — signal + execution +
   swap on one clock, one nightly rebalance.
4. **Adjustment architecture:** `additive → signals`, `ratio → returns/σ`,
   `unadjusted → real prices`. **IMPLEMENTED** (σ + IDM repointed to ratio).
5. **Return convention:** research P&L is **open-to-close** (`log_intraday`) —
   already in place repo-wide; correct under the overlay.
6. **Swap:** real (−2.1%/yr on the net-long book) but recovered by the carry-aware
   overlay → CFDs become Sharpe-equivalent to futures.

## Open questions / next steps (engineering, no decisions left)

- **⚠️ Metals rollover-gap drift** (gold +4%/yr, silver +8.75%/yr in the reopen): real
  capturable drift or reopen-spread artifact? Resolve with the tick data in
  `research/rollover_cost`.
- **Productionize:** (a) CFD-native daily-bar builder (16:45-ET close); (b) carry-aware
  rollover overlay in `execution/mt5_rebalancer.py`; (c) CFD-native promotion-gate harness.
- **`silver_mr`** sleeve is weak on every feed (Sharpe ~0) and timing-sensitive (92%) —
  review whether it earns its slot.
- **Wire `ratio_driver` into `rebuild.py`** (currently a separate manual step).
- **Minor consistency:** point the IDM correlation (`_ticker_return_series`) at
  `log_intraday` to match P&L/selection.

All venv: `.venv\Scripts\python.exe`. MT5 `time` is broker tz — see [[mt5_timezones]].

> _Verified against current code via CodeGraph on 2026-06-07._
