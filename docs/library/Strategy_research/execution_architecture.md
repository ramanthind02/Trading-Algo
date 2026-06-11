# Execution architecture — two engines, one of them deferred

How a strategy's **signal** becomes a **realized position**, and where the line between alpha and
execution sits. The central decision is **not** "how do we bolt stops/limits onto a bias node" —
it is recognizing that fill-sensitive strategies are a *different machine* and keeping them apart.
Companion to [[Strategy_research/strategy_spec]].

> [!important]
> There are **two engines**, not one pipeline with a hard mode:
> - **Engine A — the signal book.** Bias nodes emit a *level* signal → vol-scaled → diversified →
>   combined → `position_fraction` → executed as **market or passive-limit**. No per-trade stops.
>   This is the whole production system today.
> - **Engine B — bracket / scalping bot (deferred).** A state machine that owns the full trade
>   lifecycle (brackets, stop, take-profit, reset) on **partitioned instruments**. Not a bias node,
>   not in the portfolio pipeline. Built only if/when we want fast mean-reversion.
>
> The awkward middle — *a bias node that also needs precise intrabar stop/limit execution* — is
> the zone we deliberately **do not build**. Strategies are either signal-driven (Engine A) or
> fill-driven (Engine B).

This mirrors how Robert Carver structures his own system: the slow signal book (`pysystemtrade`)
executes via a passive-limit-with-market-fallback order handler and uses **no per-trade stops**;
his scalping experiment is a **separate state machine** (subclassing the order stack handler) on a
**partitioned** instrument set — *"it would be too risky to do both on a given market."*

---

## Engine A — the signal book (what we run)

### The decoupling

- A bias node is **single-ticker, single-timeframe** (`BiasNode(ticker, tf)`, `nodes/__init__.py`)
  and emits a **signal**, never an order.
- The signal is validated `⊆ {-1, 0, +1}` by the ensemble
  (`ensemble/diversified_ensemble.py`). It is a *desired position state*.
- Vol scaling (`F = signal × τ/σ`, capped ±`forecast_cap`) happens in `DiversifiedEnsemble`; σ
  (EWSD) is passed in separately. The node never sees σ.
- Cross-timeframe combination is a portfolio concern: non-daily forecasts forward-filled to a
  daily grid before `WeightLayer`. σ and the combination grid are **daily**.
- The bridge to execution is **one scalar per (ticker, datetime): `position_fraction`**.

This `{-1,0,+1}`-signal + scalar-`position_fraction` design is what buys the unified cross-asset,
cross-timeframe book and clean vol scaling. **None of it changes.**

### The signal is always a level

The node emits a *level* ("be +1"), and the execution layer **continuously works toward that
target**. When the node flips toward 0, you exit. There is **no autonomous intrabar exit**, so:

- no stop / take-profit / trailing overlay,
- no edge-vs-level dual semantics,
- no re-entry latch,
- no `ExecutionIntent` risk-geometry payload,
- no node output columns for risk geometry.

The position simply tracks the signal. The "am I still in the position?" question never arises —
you are in iff the signal says so and the fills got you there.

### Execution = how to reach the target: market or passive limit

The only execution lever in Engine A is **how** to move to the signal's target. This is selected
per leg (Carver's point that entry is often signal-urgent while the exit can be patient):

```python
@dataclass(frozen=True)
class ExecutionSpec:
    entry_policy: OrderPolicy = OrderPolicy.MARKET_ON_OPEN   # urgent: take the signal
    exit_policy:  OrderPolicy = OrderPolicy.MARKET_ON_OPEN   # patient: a limit can scrape spread
    unfilled_limit: UnfilledLimitPolicy = UnfilledLimitPolicy.CROSS_AFTER  # CROSS_AFTER | CARRY
    holding: Holding = Holding.OVERNIGHT   # OVERNIGHT (hold multi-day) | INTRADAY (flat outside session)
    fill_feed: FillFeed = FillFeed.DERIVED   # fine (M1/tick) iff any leg uses a limit, else signal_bar
```

`OrderPolicy` and `Holding` map onto existing enums in
`research/portfolio/pnl/nautilus_engine.py`:

| Spec | Engine enum | Values |
|------|-------------|--------|
| `entry_policy` / `exit_policy` | `ExecutionPolicy` | `MARKET_ON_OPEN`, `LIMIT_AT_TOUCH`, `LIMIT_IMPROVE` |
| `holding` | `ExecutionWindowPolicy` | `OVERNIGHT → CLOSE_TO_CLOSE`, `INTRADAY → INTRADAY_OPEN_TO_CLOSE` |

`holding` is the **alpha's** intent (does the edge want overnight exposure), not a cost decision.
The engine's third value, `ROLLOVER_FLATTEN_REENTER`, is a *portfolio-level* concern, **not** a
per-strategy field — see [§ Swap avoidance](#swap-avoidance--portfolio-level-decision-of-record).

Most of this is **already built**: `LIMIT_AT_TOUCH` / `LIMIT_IMPROVE` + `CrossAfterPolicy` + the
synth M1-spread quotes from the CFD migration already do passive-limit execution with a taker
fallback. The genuine additions are: split the single policy into **per-leg** `entry`/`exit`, and
expose `CARRY` alongside `CROSS_AFTER`.

### The one subtlety: a limit that doesn't fill

If a passive limit doesn't fill while the signal still wants the target, the realized position
lags the target for that window. Two honest resolutions, both already precedented:

- **`CROSS_AFTER(window)`** — work the limit for N minutes, then cross with market to guarantee
  the move. Bounds the lag. Sensible default.
- **`CARRY`** — leave the position; it rides to the next signal bar where a fresh target is
  computed (the rollover overlay's "missed the fill → carry the position" behavior).

Either way **the node knows nothing** — the signal always dictates the target and the execution
layer chases it. No latch, because there is no autonomous exit to desync from.

### When passive limits actually matter

Not by default — only when **spread is a material fraction of the per-trade edge**: wide-spread
instruments (Darwinex CFDs), short holding periods, low edge-per-trade, and non-urgent
(mean-reversion) entries/exits. Evidence: the CFD/Nautilus migration lifted test Sharpe
1.01 → 1.78 → 2.27 by modeling realistic fills and working spread properly.

**Caveat — limits are not free.** The rollover overlay found *market wins* in the financing
dead-zone (*"limits lose 6 ways"*). So per-leg, per-strategy choice — never a blanket default.

### Why stops are absent here (and that is correct)

In a vol-targeted, diversified book the risk control **is** position sizing (`F = τ/σ`), applied
continuously across the book. A per-trade stop realizes noise losses, then misses the reversion
(brutal for mean-reversion sleeves), and double-counts risk vol-scaling already governs. Carver
argues against stops for diversified systematic trading for exactly these reasons. Hard loss
limits (prop-firm max-daily-loss / max-drawdown) are an **account-level kill switch**, handled in
the account/risk layer — not a per-trade alpha stop.

### Backtest accounting

The signal/forecast is the **intended** target; the realized position is what Nautilus holds; the
return series comes from **actual equity** (`equity_curve` / `returns_from_positions` in
`nautilus_engine.py`). Intended and realized can diverge between rebalances when a limit lags —
that divergence is the honest execution cost, marked from fills, not assumed.

The lanes that turn a signal into a return series — the fast vectorized baseline, the realistic
Nautilus lane, and the lookahead-free **final-validation** lane (the real live strategy with the
signal generated on-the-fly, for pre-promotion checks) — plus the no-lookahead causality contract
they share, are documented in [[Strategy_research/pnl_lanes_and_validation]].

### Swap avoidance — portfolio-level (decision of record)

Distinct from everything above: on CFDs you are **forced to rebalance daily** because **swap cost
> spread-of-re-entry**, *not because the strategy wants to*. So the flatten/re-enter around the
financing rollover is a **cost overlay**, not an alpha property, and it lives at the **portfolio
level** — never in a `StrategySpec`. Two reasons it must:

1. The swap rate and rollover time are a function of `(instrument, broker)`, not the strategy.
2. It acts on the **net** book position per instrument (aggregated across all strategies). You
   would never flatten/re-enter each strategy's offsetting legs independently.

This is already the **decision of record** — [[Data/feed_and_execution_decision]] — and already
implemented (`execution/rollover_overlay.py`, applied at the vault level; swap avoidance lifted
*vault* Sharpe ≈ 1.32 → 1.61 on the net book). It is **carry-aware**: flatten negative-carry
(long) legs into the 17:00-ET rollover with passive limits, skip the dead zone, re-enter the new
target after the ~18:00 reopen; **hold** positive-carry (short) legs through to *earn* the swap.

Because of the overlay, the default research return convention is **rollover-bounded
open-to-close** (`log(close/open)`, already wired in the P&L engine) — you must not credit the
close→reopen gap you never hold.

#### `ROLLOVER_FLATTEN_REENTER` vs `INTRADAY_OPEN_TO_CLOSE` — opposite overnight exposure

A common confusion. They are nearly opposites:

| Window policy | Overnight exposure | Captures the overnight gap? | Swap |
|---|---|---|---|
| `INTRADAY_OPEN_TO_CLOSE` | **flat all night** | no — never on the book | none (not holding when charged) |
| `ROLLOVER_FLATTEN_REENTER` | **held all night** | yes — full session incl. overnight | dodged (off the book only for the ~1.25h rollover) |

`ROLLOVER_FLATTEN_REENTER` is *not* "flat overnight" — it is the **opposite**: you hold through
the night because you want the overnight return, you just step out for the few minutes the swap is
levied. `INTRADAY_OPEN_TO_CLOSE` genuinely forfeits the whole overnight.

#### The two-bounds research method (decide flatten vs hold-through)

For an overnight-holding strategy, report P&L under **both** conventions:

- **close-to-close** = "I can hold over the rollover with no swap" (the ceiling — full overnight
  capture; the futures-like case).
- **rollover-bounded open-to-close** = "I flatten over the rollover to dodge swap" (forgoes the
  rollover gap).

Their **delta is the forgone rollover gap** — the overnight contribution you give up by
flattening. Weigh it against the actual swap cost, per instrument:

- forgone gap ≈ 0 → flatten freely (dodge swap, lose nothing). Default for indices (gap drift ≈ 0).
- forgone gap large, swap cheap → hold through.
- forgone gap large, swap expensive → genuine tension — this is the **open question flagged for
  metals** in the decision of record (gold +4.1%/yr, silver +8.75%/yr rollover-gap drift; is it
  real capturable return or a reopen-microstructure artifact?). Resolve with the tick data in
  `research/rollover_cost`.

So a strategy declares `holding`; research reports both bounds; and the broker-level swap overlay
(carry-aware) decides where the net book actually lands between them.

---

## Engine B — bracket / scalping bot (deferred, walled off)

Fast mean-reversion (≈4-8 minute horizon) is a **different machine**. It has no meaningful alpha
signal — the order placement *is* the strategy — so it does not belong in the bias-node →
portfolio pipeline. If we ever build it, it is a **separate state machine** that subclasses the
Nautilus execution strategy (the way Carver's scalper subclasses his order stack handler), running
on a **partitioned** set of instruments removed from the main book.

Shape (Carver's ten-state machine, for reference — **not built**):

```
A flat / no orders        → place symmetric bracket limits at ±(R/2)·F around equilibrium
B ready (buy & sell limit) → one side fills
C/F in position, unprotected → place stop at (R/2)·K from entry
D/G in position, protected   → take-profit limit + stop both working
  · TP hit  → flat, cancel stop, reset to A
  · stop hit → flat, cancel TP, reset to A
```

- **Stops are structurally required here** — there is no slow signal protecting the position and
  the strategy is negatively skewed. (The opposite of Engine A: the stop belongs to the engine
  that has no signal.)
- Brackets and stops are scaled to **R** (the high-low range over a horizon, a vol proxy) — the
  same ATR/σ-units discipline as the rest of the framework.
- Intraday only: flat overnight, soft-close (stop opening) then hard-close (force flat), rolls
  done flat overnight.

### Why it is deferred — and a permanent health warning

Carver's own conclusion is a caution, not an endorsement:

- He **could not backtest it** — only simulate (*"I don't have the data to backtest this"*).
- The double-digit simulated Sharpes were a **path artifact**: bid-ask bounce from tick rounding,
  plus the optimistic assumption that within a bar the take-profit fills before the stop. Whether
  TP-or-SL wins first inside a bar swings Sharpe from **+8 to −9**.
- The stop-fill assumption (limit vs next-price) swings annualized SR by ~140 points.
- It **failed live**: *"I tried it for a couple of days and it didn't work."*

Two hard requirements fall out, should we ever build Engine B:

1. **True tick data, not M1 bars** — only ticks resolve the TP-vs-SL-within-a-bar ordering. This
   is the strongest justification for the demand-driven tick path
   ([[Data/hybrid_tick_backtest]]). With bars you are *assuming* the intrabar sequence.
2. **Keep it out of the vault / portfolio system.** The fill-sensitive regime is where backtests
   lie most; Engine B can only be validated by trading it small, and must not contaminate the
   honest numbers of Engine A.

---

## What changes, what doesn't

**Unchanged:** `BiasNode` base (single-tf, signal-emitting), `DiversifiedEnsemble`, `WeightLayer`,
`TFPortfolio`, `GlobalPortfolio`, the `{-1,0,+1}` contract, daily σ and the daily combination grid.

**Built (Engine A spec side):**

- Per-leg `entry_policy` / `exit_policy` — modelled in `ExecutionSpec` (both the Python dataclass
  and the JSON serialization are complete). `NautilusPnLEngine` still applies one policy for both
  legs; the engine split is the remaining open item.
- `CARRY` — in `UnfilledLimitPolicy` enum; serialized correctly.

**Remaining open items (Engine A):**

- Split the single execution policy in `NautilusPnLEngine` into per-leg `entry` / `exit`.
- Two-bounds reporting (close-to-close vs rollover-bounded open-to-close) as a standard research
  diagnostic, to size the forgone rollover gap per instrument.

**Already in place (portfolio level, not the spec):** the carry-aware swap-avoidance overlay
(`execution/rollover_overlay.py`) and the rollover-bounded open-to-close research convention — the
[[Data/feed_and_execution_decision]] decision of record.

**Deleted from the earlier draft** (all of it was machinery for autonomous intrabar exits, which
Engine A does not have): stop / take-profit / trailing overlay, `ExecutionIntent` geometry, node
risk-geometry output columns, edge-vs-level dual semantics, the re-entry latch and `ReentryPolicy`.

**Deferred (Engine B):** the bracket/scalping state machine, on partitioned instruments, with
tick-data fills — built only if we pursue fast mean-reversion, and kept out of the main book.

> _Rewritten 2026-06-06 to the two-engine model. Updated 2026-06-07: per-leg spec built; engine
> split is open. Grounded against `nodes/__init__.py`, `ensemble/diversified_ensemble.py`,
> `research/spec/strategy_spec.py`, and `research/portfolio/pnl/nautilus_engine.py` via CodeGraph._

> _Verified against current code via CodeGraph on 2026-06-07._
