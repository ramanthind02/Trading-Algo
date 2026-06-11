# Futures research data (Norgate) — index research only

> **Scope.** Futures are now used for **one job: research on the equity indices**
> (ES/NQ/…), where they buy ~11 extra years of clean, multi-vendor history
> (1997+ vs the CFD's 2008). **Everything else — execution, and research on
> gold/silver and the rest — is CFD-native** (see [[feed_and_execution_decision]]).
> This doc is the complete reference for the futures research feed: the
> three-series adjustment rule, the back-adjustment distortion and its fix, roll
> mechanics, the Norgate provider, and the Norgate→IB handover plan. It replaces
> the former `back_adjustment_percentage_distortion`, `futures_backtesting_data_guide`,
> `futures_rolling_carver_alignment`, `norgate`, and `multi_source_update_architecture`
> docs.

---

## 1. The one rule: three series, one job each

Continuous futures splice successive contracts; each roll leaves a price gap (cost
of carry, dividends, term structure) that is **real economic information, not a
price move**. There are three ways to remove the gap, and using the wrong one for
the wrong job is the classic silent bug in systematic futures backtesting.

| Series (file) | Splice rule | Preserves | Use for |
|---|---|---|---|
| **`additive`** (`D_{T}.parquet`, Norgate `_CCB`) | subtract cumulative roll gaps (Panama) | absolute **point** differences | **Signals** on price levels/points: EWMAC (fastMA−slowMA in points), Donchian/breakout distances, ATR, and that signal's own points-based vol normalization |
| **`ratio`** (`D_{T}_ratio.parquet`, derived) | multiply each segment by ∏ later `new/old` ratios; **never negative** | **percentage** returns | **All %-based math:** σ for vol-targeting (`F=τ/σ`), backtest P&L / Sharpe / drawdown, return features, IDM / WeightLayer correlations |
| **`unadjusted`** (`D_{T}_unadj.parquet`) | raw stitched front; has roll-day jumps | **actual traded prices** | Real price levels: contract sizing, margin, fills, roll diagnostics. *Not* returns/σ |

**The rule:** *signals see `additive`; risk & money see `ratio`; real prices see
`unadjusted`.* Because we execute on CFDs, the `unadjusted` series matters only for
roll diagnostics in research — live fills mark against the CFD.

> **Two vols, don't conflate them:** the *signal-normalizing* price-vol inside an
> EWMAC-style forecast is in **points** → compute on `additive`. The
> *position-sizing* return-vol σ in `F=τ/σ` is in **%** → compute on `ratio`.

---

## 2. Why additive distorts %-returns (the k bug)

Additive back-adjustment shifts every historical bar by the cumulative sum of all
*later* roll gaps, so it inflates the historical price **level** while keeping point
**differences** correct. Any percentage off that inflated level is too small:

```
r_adj[t] = ΔP[t] / P_adj[t-1] = r_true[t] · k[t]      k = P_true / P_adj   (k≈0.69 for ES, k≈1.0 for TLT/ETF-like)
σ_adj    = k · σ_true
```

**It cancels in a self-consistent backtest** (size on `σ_adj`, earn `r_adj`):

```
strategy_return = (τ / σ_adj)·signal·r_adj = (τ / (k·σ_true))·signal·(k·r_true) = (τ / σ_true)·signal·r_true     ← k cancels ⇒ Sharpe ~unbiased
```

**It bites live** (size on `σ_adj`, earn the *real* % at real fills):

```
live vol = (τ / σ_adj)·σ_true = τ / k  > τ            ← over-leveraged by 1/k (≈1.46× for ES: a 15% target runs ~21%)
```

Empirically (SPY/TLT study, 2002–2021): equity leg `k=0.69`, bond/ETF leg `k=1.0`;
back-adj backtest shows 10.7% vol but sizing on it and trading real prices realizes
**14.9%**. The cancellation is only *approximate* because `k` drifts with rolls/price
and the long-run σ component spans a wide `k` range — and cross-asset signals compare
two series with different `k`. Crude is worse still: additive passes through the
**negative** Apr-2020 WTI print, making `%`/`log` undefined.

---

## 3. The ratio fix — ✅ IMPLEMENTED (2026-06)

The fix is **Option 2** (a percentage-faithful series), now shipped. A `ratio`
(proportional) continuous series exists in the canonical store and the two %-math
consumers read it (with graceful fallback).

- **Generator:** `data_platform/providers/norgate/backadjust/ratio_adjuster.py`
  (pure multiplicative back-adjust) + `ratio_driver.py` (I/O shell). It recovers the
  exact roll events from the stored additive/unadjusted pair (no in-repo roll
  detection on the live `_CCB` path) and multiplies each segment by ∏ later `new/old`
  ratios. **Regenerate:**
  `python -m data_platform.providers.norgate.backadjust.ratio_driver`
  → 23 tickers × D/W/M = **69** `*_ratio.parquet` files (nothing overwritten).
- **Catalog:** `Instrument.price_adjustments` carries `{"D_ratio":"RATIO", …}` (new
  `PriceAdjustment.RATIO` enum). Re-seed: `python -m data_platform.providers.norgate.to_catalog`.
- **Consumer 1 — EWSD σ:** `nodes/volatility/ewsd/ewsd.py::_load_unadj_close_series`
  prefers `D_{T}_ratio.parquet`, falls back to `_unadj`, then to the candle close.
- **Consumer 2 — IDM/weight corr:** `ensemble/portfolio_impl/portfolio_returns.py::_ticker_return_series`
  uses `ratio.pct_change()` (≥95% date coverage), falls back to the additive close.
- **Validator-in-preflight:** `data_platform.data_quality.reference_series.validate_consumed_series()`
  runs inside `run_portfolio_research_cache_preflight`, scoped to consumed σ/return
  series. Loud WARN by default; hard FAIL only when a consumed series is genuinely
  broken (negative-on-positive-source, or additive-classified). CL's real WTI
  Apr-2020 negative is a WARN, never a FAIL.
- **Tests:** `tests/back_adjustment/test_ratio_adjuster.py` (pure logic) +
  `test_ratio_repoint_regression.py` (no negatives ex-CL, ratio corr ≥ additive,
  EWSD/IDM read ratio, scoped validator). 60 passed / 2 skipped.

**Ratio matches or beats additive on every ticker** (daily %-return corr to the
matching ETF, full history) — the distortion only accumulates on deep history, which
is why the bug hid (recent windows carry few rolls):

| ticker | rolls | additive corr | **ratio corr** | ratio min |
|---|---|---|---|---|
| ES | 44 | 0.981 | **0.986** | 757 (≥0) |
| **NQ** | 39 | **0.910** | **0.987** | 1113 (≥0) |
| GC | 156 | 0.863 | 0.877 | 411 (≥0) |
| **CL** | 353 | **0.351** | **0.832** | −77 (real source print) |
| **HO** | 397 | **0.357** | **0.789** | +0.05 (≥0) |

**σ impact (before `_unadj` → after `ratio`, annualized):** ES −0.6% full / −0.9% 5y ·
NQ −0.3% / −0.4% · GC −2.1% / −5.0%. The recent (sizing) anchor is identical; the
delta is the additive-distortion correction on deep history. *(Metals' ~0.88–0.91
ceiling is genuine continuous-future-vs-ETF basis/roll-yield noise, not an
adjustment defect — additive sits at the same level.)*

---

## 4. Decision table — which series for which computation

| What you're computing | Series | Extra input |
|---|---|---|
| Research Sharpe / CAGR / drawdown | **ratio** | — |
| Daily % return for σ (EWSD) | **ratio** `pct_change` | — (fallback: `ΔP_adj / P_unadj[t]`) |
| Vol-targeted position fraction | correct σ from above | — |
| Donchian / ATR / breakout distance | **additive** | — |
| EWMAC (points) | **additive** | — |
| Cross-asset % relative strength | **ratio** | both legs ratio |
| Signal direction only | either | — |
| Actual fill price / contract sizing | **unadjusted** | point value (multiplier) |
| Dollar P&L simulation | `contracts × ΔP_adj × multiplier` | additive diffs + point value |
| Margin / capital | — | `norgatedata.margin(symbol)` |

---

## 5. Dollar vs percentage P&L (research vs realism)

Research runs in **percentage mode** (`daily_return = position_fraction[t-1] ×
pct_return[t]`, `pct_return` off the ratio series) — capital-independent, right for
strategy work. **Dollar mode** goes through contracts and matters for one thing
above all: **rounding at small capital.**

```
contracts[t]    = round( position_fraction[t] × capital / (price[t] × multiplier × fx) )   # price = unadjusted close
daily_pnl[$][t] = contracts[t-1] × ΔP_adj[t] × multiplier × fx                              # ΔP_adj = additive diff
```

One futures contract controls a large notional (ES ≈ $250k, NQ ≈ $400k), so at
realistic prop capital many tickers round to **0 contracts** — invisible in
percentage research, which happily earns the fractional position. For this 23-ticker
book that's ~$300k–$500k before most tickers size to ≥1 contract at a 15% target.
This is a *futures*-execution concern; on CFDs you size in fractional lots, so it
largely disappears — another reason execution is CFD-native.

**Contract specs** are pre-fetched (no Norgate needed):

```python
from data_platform.providers.norgate import load_contract_specs
specs = load_contract_specs().set_index("ticker")   # point_value, tick_size, tick_value, margin, ...
specs["point_value"]["ES"]   # 50.0   (ES $50/pt, NQ $20/pt, CL $1000/pt, GC $100/pt)
```

Refresh after a tick-size/margin change: `python -m data_platform.providers.norgate.fetch_specs`.
`norgatedata.margin()` is an initial-margin snapshot, not a time series — for a
vol-targeted book margin is rarely the binding constraint (the vol target is).

---

## 6. Roll mechanics vs Carver — what aligns, what's a gap

The repo uses Norgate `_CCB` (additive Panama), which Carver also recommends; rolls
are volume-based and detected from the **`Delivery Month`** column
(`backadjust/roll_detector.py`), with per-instrument timing offsets in
`backadjust/roll_rules.py` (e.g. ES −5d before quarterly expiry, GC +2d from month
end). The σ-denominator fix matches Carver's *"I don't use percentage returns with
the spliced price as the denominator."*

| Topic | Carver | This repo | Status |
|---|---|---|---|
| Stitching | Panama (additive) | Norgate `_CCB` (additive) | Aligned |
| σ denominator | real price, not adjusted | ratio series (was `_unadj`) | Aligned |
| Roll detection | `Delivery Month` | `Delivery Month` | Aligned |
| Which contract | avoid front when possible | always front (volume-based) | Simplified (front-only markets ⇒ no choice; matters only for seasonal energy/grains) |
| **Carry/rolldown signal** | core signal | **not implemented** | **Gap** |
| Contract-month tracking in execution | PRICE/FORWARD/CARRY contracts | none | Gap (none for daily close-only; needed if carry added) |

The **carry signal** is the one material gap: the raw material is on disk
(`data/norgate/archive/contracts/{TICKER}/` + the `Delivery Month` column), but
nothing computes `carry = (carry_price − front_price)/front_price` annualized and
feeds it as a `BiasNode`. It's a signal-implementation gap, not a data gap. For a
daily close-only system the simplifications (always-front, no in-flight roll state
machine, roll cost baked into the adjusted series) are standard and don't affect
research Sharpe.

---

## 7. Norgate provider

Norgate is the futures source, accessed via the `norgatedata` package reading data
synced by the Norgate Data Updater (NDU).

> **Windows only.** NDU must be running on a Windows host; `norgatedata.status()`
> must return `True`. The research venv **cannot** import `norgatedata` — market
> series are consumed from cached parquet.

| Task | Command |
|---|---|
| Full rebuild | `python -m data_platform.providers.norgate.rebuild` |
| Fetch continuous only | `python -m data_platform.providers.norgate.fetch_continuous` |
| Archive individual contracts | `python -m data_platform.providers.norgate.fetch_contracts` |
| Migrate to ohlc_data | `python -m data_platform.providers.norgate.migrate` |
| Regenerate ratio series | `python -m data_platform.providers.norgate.backadjust.ratio_driver` |
| Fetch US stocks | `python -m data_platform.providers.norgate.stocks --universe both --workers 8` |
| Fetch market series (indices/cash/forex) | `python -m data_platform.providers.norgate.market_series` |
| Seed instrument catalog | `python -m data_platform.providers.norgate.to_catalog` |

Databases captured (US Stocks + Futures subscription): Continuous Futures 224 →
`ohlc_data/` (23 trading) + `norgate/archive/continuous/`; Individual Futures 27,201
→ `norgate/archive/contracts/`; US Equities+Delisted 14,223+20,988 → `stock_data/`;
US Indices 1,615, World Indices 31, Cash Commodities 100, Forex Spot 57 →
`norgate/market_series/…`; Economic 148 → `events/econ/`. Full adapter layout/CLI:
`data_platform/providers/norgate/README.md`.

> ⚠️ **Commodity-cash series are mis-timed** (LBMA/LME morning-London fix, not a
> 16:00-ET close) — do **not** use them as same-day return references. Details and
> the data-quality guardrail are in [[feed_comparison_and_adjustment]].

---

## 8. Norgate→IB handover (future work)

When the Norgate subscription ends, IB must become the primary index-futures updater.
The design (Phases 1–3 implemented; 4+ future):

- **Source priority is config-driven** (`configs/source_priority.yaml`):
  `norgate_active: true → false` is a single flag flip that routes futures daily to
  IB — no code change. `data_platform/core/source_priority.py` evaluates the rules.
- **The splice** (`prepare_ib_rows_for_central_cache_append` in
  `cache/runtime/ib_candle_ratio_align.py`) is already correct and needs no
  change: at the boundary, `ratio = anchor_close / ib_first_close`, applied to the IB
  block; each later session recomputes a fresh junction ratio so CONTFUT roll jumps
  are absorbed per-session.
- **σ-denominator continuity (Option C, recommended):** build a ratio-adjusted
  continuous from **individual contracts** — Norgate `archive/contracts/` for history
  + IB individual contracts going forward. Permanently clean, percentage-faithful,
  source-independent. IB ships the raw materials (individual expiries + real
  front-month price) even though it has no pre-built unadjusted continuous symbol.
- **Identity:** the same instrument has different native symbols per source
  (`ES.XCME` = Norgate `&ES_CCB` = IB `ES` CONTFUT = MT5 `SP500`); the catalog's
  `info["source_symbols"]` + `data_platform/core/symbol_map.py` are the single
  authority. **No new scheduler** — extend the existing Windows Scheduled Task.

| Phase | Goal | Status |
|---|---|---|
| 0–3 | Identity, priority config, reconciler, provenance/conflict | ✅ done |
| 4 | Norgate handover (final rebuild, `handover_manifest.json`, flip flag, splice verify) | future |
| 5 | Clean σ denominator from individual contracts | future |
| 6–7 | Stocks Norgate→IB; Nautilus `ParquetDataCatalog` migration | future |

---

## Reproducibility

| Topic | Script |
|---|---|
| Ratio-vs-additive prototype | `scripts/ratio_backadjust_prototype.py` |
| Data-quality guardrail (CI/preflight) | `data_platform/data_quality/reference_series.py` |
| Norgate reference-series lag scan | `research/feed_comparison/norgate_reference_lag_scan.py` |

## Related

- [[feed_and_execution_decision]] — the decision of record (why futures = index research only)
- [[feed_comparison_and_adjustment]] — CFD↔futures fidelity, commodity-cash clock-time quality, gold/silver evidence
- Carver, R. *Systematic Trading* (2015) ch. 10–11; "Systems building — futures rolling" (2015)

> _Verified against current code via CodeGraph on 2026-06-07._
