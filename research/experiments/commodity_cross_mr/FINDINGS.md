# Commodity-Cross Intraday Mean-Reversion — Research Findings

**Status:** in-sample + cost analysis complete; **pending live-demo fill validation** before any capital.
**Last updated:** 2026-06-08
**Code/outputs:** this directory (`research/experiments/commodity_cross_mr/`), outputs in `outputs/`.

---

## TL;DR

We set out to replicate the Robot Wealth / Ernie Chan AUD/NZD mean-reversion blog and, more
broadly, to find a **forex sleeve orthogonal to our futures book**. The daily and spread
versions are dead; the surviving strategy is a **walk-forward-reoptimized, event-driven
intraday (1H) mean-reversion basket on commodity-currency crosses**, traded as a **maker**
(limit entries), flat overnight.

- **Modeled basket Sharpe ≈ 0.7–1.4** after real spread (1.5 pip) and the real
  **$2.50/lot commission** (range = per-side vs round-turn commission × optimistic/conservative fills).
- **It is a market-making strategy, not a mean-reversion alpha** — the entire net edge is the
  captured spread (~0.5–0.8 pip/round-trip); the price reversion roughly only pays for the exits.
- A **90-day realistic tick backtest** (real fills + commission, static params) was net positive
  (+3%/yr annualized), consistent with the model — but too short to be a verdict.
- **The one remaining unknown is real fill quality** (the maker capture `cf`), which backtesting
  cannot settle. Demo fills are the next step.

---

## Objective

A diversifier for the prop/futures portfolio. Forex is the only asset class we don't trade, so a
genuinely orthogonal FX sleeve adds portfolio Sharpe even if standalone-thin (Carver
diversification logic; mean-reversion is short-gamma and complements a trend book).

## What was tested and KILLED

| Variant | Result | Why dead |
|---|---|---|
| **Daily outright AUD/NZD z-MR** (the blog) | Sharpe 0.14–0.75 but feed-fragile | Cross-feed daily-return corr only **0.36** (same path, different daily wiggle); the +0.75 was a 1990s artifact; modern net-of-swap ~0.16 |
| **AUD/NZD–AUD/CAD cointegration spread** (blog pt 2) | Modern net Sharpe **negative** | Unstable/sign-flipping hedge ratio; cross-feed corr **0.21** (worse — differencing amplifies feed noise); **two-leg swap ≈ 4%/yr** > edge |
| **Continuous 1H z-MR** | Gross Sh 1.54 → **net −1.04** | Hourly rebalancing × spread; apparent edge was bid-ask bounce in thin hours, not Asian-session MR |

Key structural fact: AUD/NZD's mean-reversion half-life is **~150–190 days at *every* sampling
frequency** (30m/1h/4h) — there is no fast intraday reversion to harvest. You either hold for
weeks (pay swap) or churn intraday (pay spread).

## What WORKS

**Event-driven 1H mean-reversion, maker entries, walk-forward reoptimized, basket of commodity crosses.**

- **Signal:** EMA "fair price" + z-score of deviation. Enter at ±2σ (fade), exit at ±0.5σ, hard
  stop ±3.5σ. Flat overnight (no swap). ~300 round-trips/yr/pair.
- **Maker entries** flip the spread from a cost to a credit (MR = liquidity provision). This was
  the unlock — market-order versions are net-negative; limit-entry versions survive.
- **Walk-forward reoptimization** (monthly, 6-month trailing lookback, tiny 18-combo grid
  L∈{24,48,96} × entry∈{1.5,2,2.5} × exit∈{0,0.5}) is the **breakthrough**: it roughly *doubles*
  the OOS Sharpe vs static (0.35→0.80 on AUD/NZD) and flattens regime drawdowns (the 2017
  trend-year loss flips positive). This matches the practitioner method (Stefan Friedrichowski,
  JFD — re-optimizes every 2–4 weeks).
- **Basket** of AUDNZD/AUDCAD/NZDCAD/AUDCHF: pairwise corr only **0.08–0.34** → real
  diversification. EW basket modeled Sharpe ~1.0–1.8 pre-commission.

**Negative control (the key credibility check):** the same walk-forward method *fails* on majors
(EURUSD ~0, GBPUSD negative, EURGBP weak). It cannot manufacture an edge where there is no
mean-reversion → the commodity-cross result is **real MR, not curve-fitting**. (Matches Stefan:
commodity pairs revert; not all pairs do.)

## The cost investigation (the important part)

**Cost waterfall (AUD/NZD, %/yr, cf=0.6, 1-pip assumption):**

```
  gross price move            +2.24
  − stop + overnight exits    −2.28   ← market exits ~cancel the gross move
  price edge net of exits     ≈ 0.00
  + entry maker credit        +1.22
  + take-profit credit        +0.27
  = NET                       +1.45      (cf=0 → ~0)
```

→ **The entire edge is the maker spread capture.** This is a market-making strategy where the MR
signal just decides where to post passive liquidity.

**Tick-data validation (4.1M real AUD/NZD ticks, 24 days):**
- Real spread is **1.5 pip** median (not 1.0) and *stable* (p99 3.2 pip). The M1 `spread` field's
  fat tail (p90 5.8, p99 24 pip) is a **data artifact** — don't use it; use real tick spread.
- Wider spread *helps* (more to capture as a maker) — corrected basket rises to ~1.5–1.8 pre-commission.
- **Maker fill rate ~82%**; captured ~0.7 pip when filled (cf≈0.9 *conditional on fill*).
- **Adverse selection:** −2.75 pip median in the 15 min after fill — you fill the continuations and
  *miss the immediate reverters*. This selection bias is **structurally outside the `cf` model**.

**Commission ($2.50/lot, confirmed):** material but survivable.

| scenario | basket Sharpe |
|---|---|
| no commission (ceiling, cf 0.8) | 1.82 |
| $2.50 round-turn, cf 0.8 | 1.40 |
| $2.50 per-side ($5 r/t), cf 0.8 | 0.97 |
| $2.50 per-side, cf 0.6 (conservative) | **0.68** |

**Realistic 90-day tick backtest** (real fills + commission, static params): basket **net positive,
~+3%/yr annualized**, fill rates 72–83% — consistent with the model. One quarter only (per-pair
dispersion −2.4 to +3.1); a calibration point, not a performance verdict.

**Conservative YTD-2026 tick backtest** (158d, real ticks, **trade-through maker fills** [behind
queue] + **2pt stop slippage** + $5 r/t commission, static L24): basket Sharpe 3.93 (touch) →
**3.57 (conservative)** — i.e. engine-grade conservatism barely dents it; execution risk is
smaller than feared. **BUT the 3.5 Sharpe is sample-inflated** — 5 months / one favorable
(ranging) regime; per-pair dispersion AUDNZD −0.26 vs AUDCAD +3.94 *in the same window* is the
tell. Forward expectation = the full-history walk-forward (~0.8–1.3 modeled), not 3.5.

**On Nautilus:** a Nautilus `BacktestEngine` run over the same recorded demo ticks would ≈ the
conservative sim above — the live-only effects (last-look, real queue, latency, partial fills) do
not exist in recorded ticks, so a backtest can't surface them. Nautilus's real value is the
**live demo**. Put the engine effort into the demo harness, not another backtest.

## Current verdict

A **plausibly real ~0.7–1.0 Sharpe market-making strategy**, swap-free and orthogonal, that
survives every cost we can measure (spread, commission). It is **thin** — the edge is ~0.5–0.8
pip/round-trip, the same order of magnitude as commission — so it lives or dies on execution.

## Risk register / open questions

1. **Real fill capture (`cf`)** — THE open question. Demo limit-fills flatter reality (no last-look,
   no queue/latency); and the fill *selection bias* (miss reverters) isn't modeled. Only live fills settle it.
2. **Single feed** — all pairs are Darwinex M1; cross-*pair* diversification is real but feed-robustness is untested.
3. **Commission interpretation** — confirm $2.50 is per-side vs round-turn (basket ~0.7 vs ~1.4).
4. **Short live/tick samples** — regime-limited; the strategy bleeds in trending regimes (walk-forward
   and the basket mitigate, but don't eliminate).

## Recommendation / next step

Stop backtesting — it's hit the ceiling of what in-sample data can answer. **Build the walk-forward
commodity-cross basket into `deployment/live/` and run on demo**, instrumented to log *realized fill
price vs modeled fill* on every order, so the demo **measures** the last unknown (`cf`) instead of
assuming it. Stack with the GBP/CAD time-zone-premium signal for a combined thin-FX sleeve.

## Artifacts (kept)

All research scripts were exploratory scratch and have been removed; this memo is the record, plus
the key result charts in `outputs/`:

- `outputs/ensemble_equity.png` / `.csv` — the headline **Sharpe~1 commodity-cross basket** equity
  curve (walk-forward, 1.5pip / cf0.8 / $2.50-side commission).
- `outputs/intraday_walkforward.png` / `intraday_wf_equity.csv` — the **walk-forward vs static**
  result (the breakthrough: static ~0.35 → walk-forward ~0.80 OOS on AUD/NZD).

The full numbers (every backtest stage, cost waterfall, tick validation, commission) are captured
in this memo and in the persistent memory note `project_audnzd_mean_reversion_replication.md`.

Related: `[[project_fx_timezone_premium]]` (GBP/CAD intraday time-zone premium), the live runtime
in `deployment/live/`, and the maker-capture model in `research/rollover_cost/`.
