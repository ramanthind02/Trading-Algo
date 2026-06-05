# WP-3 — Nautilus BacktestEngine P&L Lane (realistic execution)

> **Depends on:** WP-1 (parity harness), WP-2 (catalog/instruments/bars available).
> **Parity expectation:** NONE — this lane is **additive**. It must never overwrite
> the vectorized research baseline. The *difference* between the two lanes is the
> deliverable (the cost/edge of real execution).

## Why this lane exists (the actual motivation)

The vectorized pipeline can only fill at a candle's own `open`/`close`. It has **no
order types, no spread, no notion of where in the day a fill happened**. This lane
exists to answer the questions the vectorized lane structurally cannot:

- Market vs **limit** orders for daily strategies.
- **Capturing/limiting spread** by resting limit orders at/inside the touch.
- Whether an order would actually **fill intraday**, and at what realized price.
- Realistic slippage, partial fills, margin, and integer-contract effects.

The mechanism is a **signal/execution timeframe split**: the daily strategy emits a
daily `position_fraction` *target*; a Nautilus `Strategy` **works that target intraday**
against sub-daily data using real order types; `BacktestEngine` decides the fills.

## The dual-lane contract (`PnLEngine`)

Both lanes consume the identical `position_fraction` frame and terminate at the same
`quantfoundry_core` metrics.

```python
# NEW — e.g. ensemble/evaluation/pnl_engine.py (confirm home with codegraph)
class PnLEngine(Protocol):
    def returns_from_positions(
        self,
        positions_df: pd.DataFrame,    # ['ticker','datetime','position_fraction', ...]
        candles_df: pd.DataFrame,      # daily candle contract (signal timeframe)
    ) -> pd.Series: ...                # returns indexed by realization datetime

class VectorizedPnLEngine:   # DEFAULT — wraps existing function verbatim, FROZEN behaviour
    # delegates to calculate_strategy_returns_from_positions(
    #     ..., instrument_return_kind=<convention>)

class NautilusPnLEngine:     # OPT-IN — runs BacktestEngine, returns account return series
    def __init__(self, execution_policy, window_policy, fill_model, venue_cfg, intraday_data): ...
```

Selection is one config field; default `vectorized`:

```
pnl_engine: "vectorized" | "nautilus"      # default "vectorized"
```

`portfolio_research` and `feature_research` configs gain this field. When
`vectorized`, the code path and numbers are **exactly today's**. When `nautilus`, the
positions are routed into `NautilusPnLEngine` and the result is reported **alongside**
(never replacing) the vectorized report.

## Convention ⇒ data granularity (the overnight-gap issue)

Daily bars in Nautilus can only fill at `close[N]` / `close[N+1]` — i.e. they **hold
the overnight rollover gap**. To trade only the open→close session (flat overnight)
you must feed **sub-daily** data so entry and exit are distinct events.

`ExecutionWindowPolicy`:

| Policy | Economics | Matches vectorized | Data required |
|--------|-----------|--------------------|---------------|
| `CLOSE_TO_CLOSE` | hold overnight | `instrument_return_kind='log'` | daily bars OK (enter/hold; latency → close fills) |
| `OVERNIGHT_GAP` | hold only the gap | `log_overnight` | intraday (open must be a distinct event) |
| `INTRADAY_OPEN_TO_CLOSE` | **flat overnight** | `log_intraday` (DEFAULT research) | **intraday bars / ticks** (enter on session-open bar, flatten on session-close bar) |

## Order types & spread (`ExecutionPolicy`)

To simulate limit/spread behaviour, the matching engine needs **bid/ask**, which means
**quote data** is strongly preferred:

| Data fed to the lane | What it can simulate |
|----------------------|----------------------|
| `QuoteTick` (bid/ask) — **best** | True spread capture: a resting `LIMIT` at the bid fills when the market trades through; you save the half-spread. Required for "limit spreads with limit orders". |
| Trade ticks / M1 bars | Intraday timing + slippage via `FillModel` (`prob_fill_on_limit`, `prob_slippage`); spread only approximate (single price level). |
| Daily bars | Idealized close fills only — not usable for this lane. |

`ExecutionPolicy` (composable recipe the `Strategy` applies to each daily target):

- `MARKET_ON_OPEN` — cross at the session open (baseline aggressive fill).
- `LIMIT_AT_TOUCH` — rest a limit at the near touch; capture spread if it fills.
- `LIMIT_IMPROVE(ticks)` — rest inside the spread by N ticks.
- `CROSS_AFTER(cutoff)` — if unfilled by `cutoff` within the session, convert to MARKET.
- `FLATTEN_AT_CLOSE` (for `INTRADAY_OPEN_TO_CLOSE`) / `HOLD` (for `CLOSE_TO_CLOSE`).

Build these on **native Nautilus primitives**, not bespoke logic (see
[06_nautilus_feature_leverage.md](06_nautilus_feature_leverage.md) §B):

- `OrderFactory.bracket()` for entry + stop-loss + take-profit with `OCO`/`OUO`
  contingency and `reduce_only` legs (replaces homegrown SL/partial-close bracketing).
- Built-in **`TWAP` `ExecAlgorithm`** (or a custom `ExecAlgorithm`) to *work a daily
  target intraday* by spawning child orders — this is the engine-native form of
  `CROSS_AFTER`/slicing. Drive via `exec_algorithm_id` + `exec_algorithm_params`.
- OMS `NETTING` (single position per instrument, matches the `position_fraction` model).
- `FillModel` (`BestPriceFillModel` for the frictionless reconciliation; realistic models
  for the study) + optional `LatencyModel`, `price_protection`, `liquidity_consumption`,
  `queue_position` venue knobs — document the chosen settings.

Data availability: **MT5 ticks carry bid/ask** and are already scraped
(`data_platform/providers/mt5/`, `scripts/fetch_mt5_data.py`) → CFD/MT5 instruments are
ready for the quote-driven lane. Norgate **futures are daily only** → an intraday/quote
feed (IB) is a *prerequisite* for an intraday futures realism run; until then, futures
in this lane are limited to `CLOSE_TO_CLOSE`. Flag missing intraday data explicitly
rather than silently degrading.

## Reference test dataset — NDX 2026 (MT5 1-min + tick)

Use the **NDX MT5 1-minute and tick data for 2026** as the canonical fixture to prove
out this lane end to end. It is the ideal first test because:

- **Tick data carries bid/ask** → exercises the real motivation: a `LIMIT_AT_TOUCH`
  order resting at the bid that fills (and saves the half-spread) vs. a `MARKET_ON_OPEN`
  that pays it. This is unobservable in the vectorized lane.
- **1-minute bars** → exercise intraday timing, `INTRADAY_OPEN_TO_CLOSE` flat-overnight
  behaviour, and `CROSS_AFTER(cutoff)` logic.
- It maps to an instrument the book actually trades (NQ futures / NDX), so results are
  meaningful, not synthetic.

Tasks for this fixture:

- Locate/validate the files (start from `scripts/_check_ndx_data.py`,
  `data_platform/providers/mt5/`; confirm exact paths with codegraph). Record the path,
  symbol, date span, and whether tick rows include bid+ask.
- Ingest into the WP-2 `ParquetDataCatalog` as `QuoteTick` (ticks) and `Bar`
  (1-minute, `ts_init=close`) for the NDX instrument.
- Drive `NautilusPnLEngine` with a constant/known daily `position_fraction` target so the
  execution behaviour is isolated from signal noise, then with a real daily signal.

## Target design

```
ensemble/evaluation/                  # NEW (confirm package home with codegraph)
  pnl_engine.py                       # Protocol + VectorizedPnLEngine + NautilusPnLEngine
  nautilus_lane/
    strategy.py                       # TargetRebalanceStrategy: daily target -> intraday orders
    venue.py                          # BacktestVenueConfig (oms NETTING, MARGIN/CASH, fees)
    build.py                          # assemble BacktestEngine from catalog + policies
    report.py                         # account equity/returns -> quantfoundry_core
                                      #   (AUTHORITATIVE) + Nautilus PortfolioAnalyzer
                                      #   return stats side-by-side (additive reference,
                                      #   for evaluation) + ReportProvider execution
                                      #   diagnostics (fills, realized spread, MAKER/TAKER,
                                      #   commissions) + optional create_tearsheet. WP-6 §F.
```

`TargetRebalanceStrategy` outline:

- Receives intraday `Bar`s / `QuoteTick`s for the instruments.
- Knows the daily `position_fraction` target per (ticker, session-date), precomputed by
  the frozen alpha core and injected as a **registered `CustomData` signal**
  (`register_custom_data_class` → catalog-persistable → delivered via `on_data`); **no
  signal recompute** in the strategy. See [06](06_nautilus_feature_leverage.md) §A.
- At each session boundary, computes target contracts from `position_fraction` (reuse
  `execution/position_sizer.py` sizing math for consistency with live) and emits orders
  per `ExecutionPolicy`; manages SL/working-order lifecycle; flattens or holds per
  `ExecutionWindowPolicy`.
- The account's realized return series is the lane output.

Reuse, don't reinvent: pull contract/multiplier/fx from the WP-2 Nautilus instruments;
pull sizing from `execution/position_sizer.py`.

## Acceptance criteria (gate)

- With `pnl_engine="vectorized"`, WP-1 parity is **unchanged** (this WP added code only).
- With `pnl_engine="nautilus"` + `CLOSE_TO_CLOSE` + a `BestPriceFillModel` (frictionless,
  cross at close), the lane reproduces the vectorized **close-to-close** (`log`) returns
  within a documented tolerance — this is the sanity check that the plumbing is correct.
- On the **NDX 2026 MT5 fixture** (1-min + tick), with `INTRADAY_OPEN_TO_CLOSE`:
  positions are provably **flat overnight** (none carried across session boundaries),
  and a `LIMIT_AT_TOUCH` policy shows **measurable, signed spread capture** vs.
  `MARKET_ON_OPEN` (fill prices and realized spread reported per trade).
- A comparison report is produced with two side-by-side views: (a) vectorized-lane vs
  nautilus-lane **returns**, and (b) on the same return series, `quantfoundry_core` vs
  Nautilus `PortfolioAnalyzer` **metrics** — so the Nautilus metrics can be evaluated
  against QF (QF stays authoritative for gating).

## Risks / notes

- **No native next-bar-open** (docs). Achieve open/close fills via intraday data + the
  session-boundary strategy, not via daily-bar tricks.
- **Determinism:** pin `FillModel` `random_seed`; one `BacktestEngine`/`BacktestNode` per
  process (Nautilus global-singleton constraint — see concepts/architecture docs).
- **Scope discipline:** this lane must not change `calculate_strategy_returns_from_positions`
  behaviour or any research default. It is selected only by opt-in config.
- **Performance:** intraday/tick backtests are far heavier than the vectorized lane; this
  is expected — it is the "switch on for realism" lane, not the sweep lane.

## Subagent instructions

- Build `VectorizedPnLEngine` first as a verbatim wrapper and route the existing call
  sites through the protocol with `vectorized` default — prove WP-1 still green. Only
  then build `NautilusPnLEngine`.
- Do not couple the lane to a specific instrument set; drive everything from WP-2
  instruments + policy objects.
- Hand back: the protocol + both engines, the frictionless close-to-close reconciliation
  numbers, one worked intraday limit-vs-market example, and the comparison report format.
