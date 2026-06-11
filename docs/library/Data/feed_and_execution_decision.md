# Decision of record — signal source & execution venue

> The policy for which feed drives what. Evidence in [[feed_comparison_and_adjustment]];
> futures adjustment mechanics in [[futures_research_data]]; timezone rules in
> [[mt5_timezones]]. Reproduce with `research/feed_comparison/feed_*`, `research/feed_comparison/gs_*`.

## The decision (one paragraph)

We **execute exclusively on Darwinex CFDs** (the prop venue), and we **research
CFD-native by default** — because the CFD is faithful to the real market (0.99 daily),
it's the instrument we actually fill on, and only the CFD has **tick data**, so it's
the only feed that gives realistic spread/slippage in a backtest. The **one
exception** is the **equity indices** (ES/NQ/…): there we research on **ratio-adjusted
futures** for the ~11 extra years of clean, multi-vendor history (1997+ vs the CFD's
2008), then **validate on CFD before promotion**. Gold/silver and everything else are
CFD-native — their CFD history is ample and the futures are weakest there. Wherever
futures research *is* used, the ratio-adjustment fix (shipped) keeps it
distribution-consistent with CFD execution.

## Which feed for what

| Job | Feed | Why |
|---|---|---|
| **Execution (all instruments)** | **Darwinex CFD** | prop venue; faithful 0.99; tick data → real spreads |
| **Realistic backtest / cost modelling** | **CFD (bars + on-demand ticks)** | only feed with ticks; see [[hybrid_tick_backtest]] |
| **Research — equity indices** | **ratio-adjusted futures**, then validate on CFD | +11 yr history (incl. dot-com) + multi-vendor cross-check |
| **Research — gold/silver + everything else** | **CFD-native** | CFD history ample (28/24 yr); futures weakest + worst clock gap here |
| **Production signal** | **CFD-native at the 17:00 ET rollover** | signal + execution + swap share one clock, one nightly rebalance |

## Why CFD-native ≈ futures (proven on identical strategies)

With the *same vault params*, computing the signal on the CFD and executing on the CFD
gives essentially the same result as doing both on the futures — they're claims on the
same underlying (daily returns 0.97–0.99) and slow signals are more stable still.
Self-consistent backtests, gold/silver sleeves:

| Sleeve | window | Sharpe futures | Sharpe CFD-native | agreement |
|---|---|---|---|---|
| gc_breakout | 2004–26 | 0.64 | 0.63 | 97–99% |
| silver_trend | 2004–26 | 0.15 | 0.26 | 96% |
| silver_mr | 2004–26 | 0.35 | 0.37 | 92–98% |

Across ~5,900 days the two land within ±0.1 Sharpe, CFD-native marginally *better*
(it carries none of the additive vol-compression). Whole-book OOS: positions built on
futures, executed on CFD, preserved Sharpe (1.61 → 1.61 with the swap overlay).

> **Caveat (not a feed issue):** the silver sleeves are weak in absolute terms on
> *every* feed (`silver_trend` ~0.15–0.26, `silver_mr` ~0 recently). Review whether
> they earn their slot; the feed choice doesn't fix that.

## Production architecture (live, nightly — whole book)

**One event per instrument, at the 17:00 ET CFD rollover:**
1. Snapshot the CFD daily close at **~16:45 ET (T-15)** — before the rollover spread
   blows out — and compute the new target signal from it.
2. Run the **carry-aware swap overlay** as the same trade: flatten negative-carry
   (long) legs into the 17:00 rollover with passive limits; skip the 17:00–18:00 ET
   dead zone; **re-enter** the new target (`delta = target − current_actual`) after the
   ~18:00 reopen. Hold positive-carry (short) legs through to earn the swap.
3. Per-symbol fallback rules per `research/rollover_cost` (limits, eat-swap on silver,
   chase on indices/gold).

This collapses signal-rebalance + swap-avoidance into **one coordinated nightly
operation on one clock**, removing the cross-feed/cross-clock reconciliation that
caused every timing bug in the study. Swap economics on the actual (net-long,
vol-targeted) book: **−2.1%/yr if held through; ~0 with the overlay** (Sharpe
1.32 → 1.61). The overlay is the linchpin that makes CFDs cost-equivalent to futures.

## The indices "1-hour gap", quantified

A 16:00-ET futures settle vs the 17:00-ET CFD rollover: daily-return corr 0.976
(SP500) / 0.967 (NDX); the index moves ~10–14 bps in that hour (NDX more — US earnings
drop 16:00–16:30 ET, which the futures settle misses but the CFD captures). For slow
signals the result is ~99% identical, so it's immaterial — and the CFD-native 17:00
signal is if anything *better* aligned to the venue you actually trade.

## Research return convention — OPEN-TO-CLOSE, not close-to-close

Because we **flatten over the 17:00-ET rollover**, we're flat ~16:45→18:00 ET nightly
and never hold the close→reopen gap. Research P&L must use **open-to-close** (the
in-session, rollover-bounded return), not close-to-close — c2c credits a move we never
hold and lets you select strategies whose edge lives in the gap you don't hold.

> **STATUS: already implemented.** The pipeline uses open-to-close
> (`instrument_return_kind="log_intraday"` = `log(close/open)`) for both the
> feature-selection target and the portfolio P&L engine
> (`research/portfolio/pnl/pnl_engine.py`). **Caveat 1:** on Norgate futures
> `open` ≈ the Globex session open (~18:00 ET ≈ prior close), so `log(close/open)` is
> overnight-inclusive — which is *correct* under the overlay (we hold the overnight,
> skip only the ~1.25h rollover). **Caveat 2:** the futures proxy can't capture the
> CFD-specific rollover-gap drift (metals +4–8%/yr) — compute `log_intraday` on
> CFD-native rollover-bounded bars (18:00-ET open, 16:45-ET close) to make it exact.

**Mechanics:** signals are still computed from **closes**; only the return/P&L changes.
Rollover-bounded daily bar (broker tz, dead zone 00:00–00:59): `open` = first M1 close
≥ stored 01:00 (~18:00 ET reopen); `close` = last M1 close ≤ stored 23:59 (~16:59 ET).
Captured return = `close/open − 1`; skipped gap = `open/prior_close − 1`. Carry-aware
**shorts** held through the rollover capture close-to-close + swap; model that leg as a
separate credit.

**Rollover-gap magnitude** (`research/rollover_cost/rollover_gap_decomp.py`, 2018+): indices o2c ≈
c2c (corr 0.98, gap drift ≈ 0 — either convention is fine). **Metals carry large
positive drift** (gold +4.1%/yr, silver +8.75%/yr) which o2c correctly excludes — but
that flags a real tension:

> **⚠️ OPEN QUESTION (queued):** is the metals overnight-gap drift *real capturable
> return* or a *reopen-microstructure artifact*? The reopen spread spikes (XAUUSD ~1.3,
> silver 18–60 bps), so the marked reopen price may be biased. The research convention
> is the same either way (open-to-close), but the **metal rollover overlay's net
> benefit must be re-checked against the forgone gap**, not just swap-minus-cost.
> Resolve with the tick data in `research/rollover_cost`.

## Promotion gate (mandatory)

Before any strategy goes live, re-run it on **CFD-native** history and confirm it
holds (`silver_mr` is the one to watch — weakest edge, most timing-sensitive at 92%
agreement). Keep `data_platform/data_quality/reference_series.py` running on the
CFD feed — single-vendor production is the one real risk of this architecture.

## What ships next (engineering, not decisions)

1. **CFD-native daily-bar builder** — 16:45-ET (stored ~23:45) close from M1, one per
   instrument, as the production signal input.
2. **Carry-aware rollover overlay** in `execution/mt5_rebalancer.py` — pre-rollover
   flatten of long legs + per-symbol limit/eat-swap/skip rules + nightly state machine
   (`delta = target − current_actual`, never `target − 0`).
3. **Promotion-gate harness** — re-validate a (futures- or CFD-) discovered strategy on
   CFD-native history before go-live.

## Status of upstream fixes (done)

- Ratio-adjustment architecture implemented (additive→signals, ratio→returns/σ,
  unadjusted→prices); 69 ratio series generated; σ + IDM repointed with fallbacks. See
  [[futures_research_data]].
- Data-quality validator wired into preflight. Prop-config loader bug fixed.

## Related

- [[feed_comparison_and_adjustment]] · [[futures_research_data]] · [[mt5_timezones]] · [[hybrid_tick_backtest]] · `research/rollover_cost/README.md`
- [[Strategy_research/execution_architecture]] — consumes this decision: swap avoidance is a
  portfolio-level overlay (not a `StrategySpec` field); the two-bounds (c2c vs rollover-bounded
  o2c) method to size the forgone gap · [[Strategy_research/strategy_spec]]

> _Verified against current code via CodeGraph on 2026-06-07._
