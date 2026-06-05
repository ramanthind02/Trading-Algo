# Nautilus BacktestEngine — speed benchmark & resolution strategy

Measured throughput of the WP-3 Nautilus P&L lane (`research/portfolio/pnl/nautilus_engine.py`)
on the **NDX intraday catalog**, to answer: *how long does a daily-model backtest on 1-min /
tick data take, and how much data do we need for statistically significant results?*

Reproduce: [`scripts/dev/bench_nautilus_backtest.py`](../../../scripts/dev/bench_nautilus_backtest.py)
(`--tick-budgets` configurable). Benchmark machine: this dev box, single-threaded, `bypass_logging=True`,
`BestPriceFillModel`, OMS NETTING. Run on 2026-06-05.

## Use case driving this

Daily model on a CFD prop account: we **cannot hold across the daily financing rollover** (≈17:00
NY) because of swaps. Idea: exit the position on a **limit** just before rollover, re-enter on a
**limit** just after — dodging the swap *and* (by resting passive) earning rather than paying the
spread. The thing being simulated is an **execution overlay**, not new alpha; the realism that
matters is the **fill probability + spread at the rollover window**, which only sub-daily data can
represent.

## Measured results

| Case | events | `engine.run()` | throughput |
|---|---:|---:|---:|
| 1-min bars only (`MARKET_ON_OPEN`) | 100,033 | 8.40 s | **11,912 evt/s** |
| bars + 1,000,000 quote ticks (`LIMIT_AT_TOUCH`) | 1,100,033 | 13.33 s | 82,523 evt/s |
| bars + 4,000,000 quote ticks (`LIMIT_AT_TOUCH`) | 4,100,033 | 34.10 s | 120,220 evt/s |

### "events/sec" is not one number — it depends on which callback fires

Decomposing the runs yields two very different per-event costs:

| Event | Rate | Per-event | Why |
|---|---:|---:|---|
| **Bar** (`on_bar`) | ~12k/s | ~84 µs | heavy: session roll, sizing via `PositionSizer`, order mgmt, equity mark every bar |
| **Quote** (`on_quote_tick`) | ~145k/s | ~7 µs | light: just stores bid/ask (marginal cost from the 1M→4M step: 3M quotes / 20.8 s) |

A bar costs **~12× a quote**. That is why the bars-only run shows the *lowest* evt/s — every event is
an expensive one. As the quote fraction rises, aggregate evt/s climbs toward the ~145k/s quote ceiling.

## Data volumes (NDX, ~3.4-month window 2026-02-23 → 06-05, annualized)

| Series | rows (window) | per instrument-year | ratio |
|---|---:|---:|---:|
| 1-min bars | 100,033 | ~345k | 1× |
| **ticks** | **54,204,267** | **~187M** | **~540×** |

Ticks are ~540× the 1-min event volume for the same period — the entire cost question in one number.

## Extrapolated `engine.run()` wall-clock (single-threaded)

| Scope | 1-min bars | Full tick | **Hybrid** (tick only in rollover window) |
|---|---:|---:|---:|
| 1 instrument-year | ~0.5 min | ~22 min | ~1 min |
| **5 instruments × 10 yr** | **~24 min** | **~18 hours** | **~47 min** |

- **Hybrid** = 1-min bars across the whole holding period + tick **only inside a ~30-min rollover
  window** (~2.2% of a ~23 h CFD session → ~4M ticks/inst-yr instead of 187M). **~23× cheaper than
  full-tick**, with tick fidelity preserved *exactly where the strategy lives* (the exit/re-entry
  fills around 17:00). Full-tick across the whole day is mostly wasted compute — 23 h of every day the
  position just sits there.
- **Parallelism:** backtests are embarrassingly parallel across instruments (independent), so divide
  by cores. On 8 cores: 1-min ≈ 3 min, hybrid ≈ 6 min, full-tick ≈ 2.3 h.
- **Ingest** (raw MT5 → catalog) ran at ~143k ticks/s; it is a one-time cost that persists in the
  catalog, so research iteration pays only load + run.

## Statistical significance — count rollover events, not ticks

The unit of evidence is **one rollover round-trip** (exit + re-enter), **not** a tick or a bar. A
billion ticks describing 200 rollovers is still only 200 trials.

- Rollovers ≈ **252 per instrument-year**. A 5-instrument book held ~70% of nights over 10 yr ≈
  **~8,800 rollover events**.
- Power: `n ≈ (z_α+z_β)² (σ_event/Δ)²`. Here `Δ` (net edge ≈ a few bp swap + ½-spread) is **small**
  and `σ_event` (gap / missed-fill risk) is **large**, so a small-but-real edge needs **~3,000–5,000
  events** — comfortably met by 5×10 yr, *not* by one instrument over 2 yr (~350 events).
- Events are **not i.i.d.** (consecutive nights share regime + the same open position) → cluster /
  block-bootstrap by week; effective n is lower than the raw count.
- The result is only as real as the **fill model**: because `Δ` is small, validate under ≥2 fill
  assumptions (best-price touch **and** require trade-through) — an edge that exists only under
  optimistic fills is an artifact.

**Conclusion:** more tick resolution sharpens per-event fill realism; it does **not** buy more trials.
The **hybrid** lane delivers both — full statistical power (every rollover, every year) *and* honest
rollover-window fills — for ~47 min instead of ~18 h. Target **≥3,000–5,000 rollover round-trips
across multiple instruments and years, validated under ≥2 fill models.**

## Caveats

- Throughput depends on `on_bar` weight; `TargetRebalanceStrategy` does realistic per-bar work
  (sizing, equity mark, position trace). A lighter strategy would raise bar throughput.
- `bypass_logging=True` and a frictionless `BestPriceFillModel` are assumed; a stochastic fill model
  or logging adds overhead.
- Extrapolations assume per-event costs hold at scale (linear in events — confirmed across the
  1M→4M step) and a ~23 h CFD session calendar. Per-instrument tick density varies (index CFDs are
  denser than the rollover window assumes; the hybrid figure is conservative).

## Hybrid lane — built & measured

The hybrid is implemented entirely on the **existing** Nautilus engine (no bespoke harness):

- **Windowed-tick ingest:** `data_platform.nautilus.ingest.ingest_mt5_intraday_windowed(symbol,
  catalog, *, rollover, half_width_minutes, tz)` — ingests full 1-min bars + quote ticks **only**
  within `[rollover − Δ, rollover + Δ]` daily (a recurring-window mask; Nautilus' catalog filters one
  contiguous range, so the recurring selection is ours — ~15 lines of pandas).
- **Strategy:** a third `ExecutionWindowPolicy.ROLLOVER_FLATTEN_REENTER` on the existing
  `TargetRebalanceStrategy` (`research/portfolio/pnl/nautilus_engine.py`) — flatten to flat just before
  the rollover, restore the daily target just after, reusing the validated `_limit_price` passive
  anchoring. Run via `NautilusPnLEngine(window_policy=ROLLOVER_FLATTEN_REENTER,
  execution_policy=LIMIT_AT_TOUCH, rollover_minute=…, rollover_half_width_min=…)`.
- **Runner:** `scripts/dev/run_hybrid_rollover.py`.

**Measured (NDX, rollover 21:00 UTC ± 20m):** 100,033 bars + **2,116,943 window quotes (3.91% of all
ticks, 26× fewer)**; `run_with_diagnostics` in **22.4 s** incl. load → consistent with **~1 min /
instrument-year**. The flatten/re-enter legs filled **140/140 as MAKER**, **+61.45 price-units of
half-spread captured**, 0 missed-fill rejects, finite returns.

> ⚠️ That all-maker result is the **optimistic** fill case (`BestPriceFillModel` + `post_only` fills
> whenever the bar/quote touches the resting price). Before trusting any swap-avoidance edge, re-run
> under a **trade-through-required** fill model (and measure the missed-fill / carried-overnight tail),
> per the significance section above. The plumbing + speed are proven; the *economics* still need the
> adversarial fill assumption.
