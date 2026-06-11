# Do the vaulted DAILY strategies retain an edge on intraday timeframes?

**Date:** 2026-06-08 · **Phase:** 0 (frictionless screen + measured-spread cost haircut) ·
**Window:** 2018-01-01 → 2026-06 (true-minute MT5 CFD M1, resampled) · **TFs:** H4, H2, H1, M30

## TL;DR

**No vaulted daily strategy delivers a robust, standalone intraday *alpha*.** The daily sleeves'
edges are daily-horizon phenomena; almost nothing ports cleanly to intraday bars once you require
beating buy-and-hold, surviving measured spread, holding across both halves of the sample, sitting on
a parameter plateau, and passing a drift-stripped null.

The single robust, reproducible, cost-surviving pattern is the **long-only breakout trend filter**
(`robust_trend_breakout`, `strategy_mode=long`) on trending instruments (gold cleanest; Nasdaq/S&P
weaker). But hard scrutiny shows it is a **de-risking / exposure-selection overlay, not a
return-generating alpha**: its in-market timing skill is ≈ 0 (a binary all-in/all-out signal earns
exactly the instrument's return while in), 100% of its Sharpe "lift" comes from *when to be in vs
flat*, the lift over buy-hold is **statistically inside the noise** (95% CI crosses zero), and it
**gives up ~half of buy-hold's cumulative return** to get a smoother ride (see
`outputs/equity_vs_buyhold.png`). It beats buy-hold on *risk-adjusted* terms by a thin margin, mostly
via volatility reduction — a known trend-overlay result, not new alpha.

**The actual vaulted directions fare worse than the long-only proxy:** the CL/SI sleeves'
**`long_short` breakout does NOT extend** (0 robust survivors — the short leg degrades passive beta),
and the **mean-reversion oscillator sleeves** (`ibs_lower_band`, `regime_lrsi`, mostly `algomatic`
RSI-2) **largely do not survive**. The only MR-family bright spot is **`double7s` on NDX H4** (clean,
low-turnover). Silver `donchian` shows a lift but with deep drawdowns, cost-fragility, and unmodelled
swap.

## What was tested, and why (the selection call)

Only sleeves whose thesis is horizon-transferable AND whose node is driven purely by the OHLC stream
(not the wall-clock calendar) were tested. **Tested:** `robust_trend_breakout` (both directions),
`donchian_breakout`, `regime_lrsi` (Laguerre-RSI MR), `algomatic_momentum` (RSI-2), `ibs_lower_band`,
`double7s`. **Skipped (thesis does not transfer):** `turnaroundtuesday` + seasonal `calendar_ensemble`
(day-of-week/calendar — undefined intraday; confirmed `candle.datetime.weekday()` logic),
`sma_regime` period-252 (long-horizon regime → a few days intraday = a different strategy),
`rebalancing_es_tlt` (low-frequency cross-asset premium), `buy_hold` (monthly).

Instruments (native + generalization): SP500(ES), NDX(NQ), XTIUSD(CL), XAUUSD(GC), XAGUSD(SI), TLT.
Pre-2018 M1 is sparse (daily/hourly-as-M1), so the intraday window is floored at 2018.

## Method (and why it is trustworthy)

- **Real vaulted nodes, streamed.** Each signal is the *exact* vaulted bias node built via
  `helpers.create_fresh_bias_node` and streamed bar-by-bar (`node.add_candle`) — the same path
  `BaseModel._instantiate_bias_node` uses. **No node logic was re-implemented**, so there is no
  replication risk. A **control** proved the streamed signal is byte-identical to `BaseModel.predict`
  on daily bars (`run_screen.py --mode control`).
- **Lookahead-guarded frictionless P&L.** `ret = log(close).diff()`, `pos = signal.shift(1)`,
  `pnl = pos·ret`; re-derived a second way (manual numpy lag) and asserted equal on every config.
  Annualized by the **realized** bars/year (n_bars / span_years), not a nominal 252.
- **De-staled intraday bars.** The first M1 bar of each broker session (~01:00) opens at the prior
  close carried across the 00:00–01:00 financing dead zone — a fabricated open/high/low. It is
  collapsed to its own close before resampling (mirrors `cfd_candles._first_tradeable_open`); the
  close series is untouched.
- **Robustness battery:** buy-and-hold benchmark per (instrument, TF); parameter plateau (fraction of
  the param axis with net>0); sub-period split (2018–2021 / 2022–2026); drift-stripped circular-shift
  null (demean to remove net exposure a roll can't break, then 95th-pct |Sharpe| over many shifts).
- **Cost haircut:** measured per-fill half-spread `= median over spread>0 bars of spread·point/2/close`,
  deducted as `half_spread·|Δposition|`; reported net Sharpe at 0.5× / 1× / 2× the measured spread.

### Validation chain (this is the part that makes the conclusions credible)
1. **Control passed** — engine == `BaseModel.predict` on daily CL (4309 bars, exact match).
2. **Adversarial code audit (workflow)** found and I fixed **4 real bugs** before trusting any number:
   (a) the breakout family was silently run **long-only but labelled long_short** (missing
   `strategy_mode`) — fixed by running both directions explicitly; (b) **SP500 cost was identically
   zero** (65% of SP500 M1 spreads are 0 → median 0) — fixed by conditioning the cost on live
   (spread>0) quotes; (c) the **dead-zone filter was a no-op** that let the stale 01:00 session-open
   bar contaminate OHLC nodes — fixed with a real de-stale; (d) the **shuffle null was a single noisy
   draw that couldn't strip market drift** — fixed (demean + many shifts).
3. **Independent reproduction (workflow): 3/3 headline survivors reproduced** to within 0.003 Sharpe
   with *fresh* P&L code (not the harness). No lookahead — corroborated by the lag-decay signature
   (breakout lag0=5.31→lag1=1.17→lag2=0.64) and the double7s un-lagged Sharpe of **−5.62** (the
   correct signature of a *lagged* dip-buyer; a peek would be large positive).
4. **Skeptic + completeness critic (workflow)** drove the calibrated framing and the gate fix below.

## Results — per-family verdict

Buy-hold intraday Sharpe (2018–26), for reference: **XAUUSD 0.88, NDX 0.77, SP500 0.64, XAGUSD 0.49,
XTIUSD 0.10, TLT −0.30.** A config is a "robust survivor" only if it clears ALL of: net@1× ≥ 0.5,
both sub-periods > 0, gross > drift-stripped null, ≥ half the param axis net-positive, **and
lift-over-buy-hold ≥ 0.20** (applied to *both* directions — a book that can't beat passively holding
the instrument is not an edge). **20 of 784 configs survive.**

| Family (dir) | Robust | Best net@1× | Verdict |
|---|---|---|---|
| `robust_trend_breakout_long` | 11 | 1.16 | **Modest, real, but it's a DE-RISKING overlay, not alpha** (see below). XAUUSD is the genuine broad plateau; NDX/SP500 headlines are favorably-selected cells. |
| `double7s` (long) | 2 | 1.08 | **Cleanest MR survivor.** NDX H4 short=5 (net 1.08, lift +0.34, turnover 0.08) and marginally SP500 H4. Low turnover, both sub-periods positive. |
| `algomatic_rsi2` (long) | 2 | 0.96 | Only **NDX/SP500 M30** survive, and cost-fragile (SP500 falls to 0.26 net at 2×). The eye-catching XTIUSD "1.11 / lift +1.01" is a **single non-plateau spike** (1 of 6 axis cells net>0) — not a real edge. |
| `regime_lrsi` (long_short) | 1 | 0.88 | **Fragile.** Only NDX H4; family **median net −0.68**; turnover 0.24 (~10× the breakout families, exit_bars=1 churns); edge concentrated in 2022–26. Not robust. |
| `donchian_breakout` (long_short) | 4 | 0.75 | **Silver only**, with genuine lift (+0.21…+0.43) but **deep DD (−0.36…−0.43)**, cost-fragile (2 of 4 fall below 0.5 net at 2×), inferred `point`, and **no swap modelled** (negative-carry). Marginal. |
| `robust_trend_breakout_ls` (long_short) | **0** | — | **The CL-vaulted long_short breakout does NOT extend.** Its apparent "survivors" all have negative/low lift vs buy-hold — the short leg degrades passive beta. |
| `ibs_lower_band` (long) | **0** | — | **Does not extend.** No config clears the gates (best net 0.64, fails plateau/sub-period). |

### The breakout_long result, stated honestly
- **In-market timing skill ≈ 0.** The signal is binary all-in/all-out, so while in-market it earns
  exactly the instrument's return — the in-market-bars Sharpe equals buy-hold's to machine precision.
  **100% of the "lift" is bar *selection*** (when to be in vs flat), not skill while holding.
- **Selection is drift-aware, not pure beta.** In-market bars have 3–6.5× the mean return of flat
  bars; it beats a random-flat null 98–100% of the time; and a dumb vol-target captures only 13–38%
  of the lift — so a real (if small) drift-selection component exists.
- **But it's de-risking, not return generation.** In cash 70–88% of the time; ann-return only
  0.44–0.56× buy-hold while ann-vol is 0.29–0.40×. Cumulative `strat − buyhold` is **monotonically
  negative** — it underperforms buy-hold's dollars in nearly every year (`equity_vs_buyhold.png`).
- **The lift is inside the noise.** Block-bootstrap of the Sharpe *difference*: mean +0.28…+0.41 but
  sd ~0.35–0.41, **P(lift ≤ 0) ≈ 0.16–0.25, 95% CI crosses zero** for every config. Positive-lift
  years are only 4/9 (NDX/SP500), 6/9 (XAUUSD) — carried by strong-trend years (2018/2021/2024).
- **Selection risk:** these are the top of an ~800-config screen; a multiple-testing/deflated-Sharpe
  correction would likely render the individual headlines non-significant.

## Caveats (must be read with the numbers)
- **Frictionless of financing.** Only spread is charged. **No swap/carry, slippage, market impact, or
  commission.** For the `long_short` and overnight-held metals/oil books this is *material* — CFD swap
  on negative-carry positions would very likely push the silver-donchian / regime configs net-negative.
- **2× cost fragility:** 9 of 20 survivors fall below 0.5 net at 2× the measured spread (all silver
  donchian, the algomatic indices, regime). "Survives cost" means *median half-spread only*.
- **Weekend/overnight gaps** are folded into a single close-to-close bar (empty bins dropped), so the
  per-bar std understates gap-tail risk — Sharpe is mildly optimistic for positions held over gaps.
- **Broker timezone:** bars sit on broker EET/EEST wall-clock (ET = stored − 7h), not UTC. Fine for
  these non-time-of-day-gated signals, but bin boundaries are on that clock.
- **Inferred `point`** for XAGUSD/XTIUSD/TLT (1st-pct of positive close-diffs; only SP500/NDX/XAUUSD
  are terminal-probed) — the silver/oil cost rests on an inferred increment.
- **Single feed** (one MT5 CFD broker's quotes/spreads); spreads and roll differ across brokers.

## So what (recommendation)
- **Do not promote anything to the vault from this.** There is no robust standalone intraday alpha here.
- **Only two configs are even plausible Phase-1 (Nautilus realism + swap) candidates:**
  `robust_trend_breakout_long` on **XAUUSD** (broad plateau, +0.2–0.3 lift, net≈gross, shallow DD) and
  **`double7s` on NDX H4** (low turnover, clean). And even these should be evaluated **as a risk
  overlay**, benchmarked head-to-head against a plain vol-target and a 200-day trend filter on the same
  instrument — the breakout must beat *those*, not just buy-hold, to justify itself.
- The honest headline: **the daily vault's edges are daily-horizon edges.** The only thing that ports
  is the trend-following *idea* as a volatility-reduction overlay — a thin, known result, not a new
  intraday signal.

## Artifacts (`outputs/`)
- `screen_master.csv` (784 configs, all metrics) · `screen_annotated.csv` (+ benchmark, lift, null
  margin, cost decay, plateau, robust flag) · `survivors.csv` (the 20) · `family_verdict.csv` ·
  `benchmark.csv` (buy-hold per instrument/TF) · `plateau_pivots.csv`.
- `plateau_heatmaps.png` (net@1× Sharpe by family — plateau vs spike) ·
  `equity_vs_buyhold.png` (breakout_long is smoother but trails buy-hold's cumulative return).

## Reproduce
```powershell
.\.venv\Scripts\python.exe -m research.experiments.vault_intraday.run_screen   # control + screen
.\.venv\Scripts\python.exe -m research.experiments.vault_intraday.benchmark    # buy-hold
.\.venv\Scripts\python.exe -m research.experiments.vault_intraday.analyze      # survivors + verdict
.\.venv\Scripts\python.exe -m research.experiments.vault_intraday.plots        # figures
```
The harness (`data_io.py`, `engine.py`, `run_screen.py`, `benchmark.py`, `analyze.py`, `plots.py`) is
**retained** rather than deleted as scratch: it is a validated, documented, reusable intraday-screen
harness (control-checked, audited, independently reproduced), isolated in this experiment dir. The
`outputs/_bars_cache/` (regenerable from M1) and the run log were removed.
