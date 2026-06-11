# Industry-Timing (Zarattini & Antonacci 2025) on OUR Instruments — Research Findings

**Status:** Phase-0 frictionless POC + robustness battery + adversarial verification **complete.**
**KILLED** as a standalone return strategy on the instruments we trade. Viable only as a
drawdown/crisis *overlay*. No Phase-1 (Nautilus cost realism) was warranted — the edge dies in
Phase-0 on our universe.
**Last updated:** 2026-06-08
**Code/outputs:** this directory (`engine.py`, `run_poc.py`, `run_robustness.py`, `outputs/`).

---

## TL;DR

The user asked whether the 2025 Dow-Award "Timing Industry" method is robust and effective **on the
instruments we actually trade** (futures/CFD book) — *not* a replication of the paper's 48 Kenneth-French
industry portfolios. We ported the **method** (asymmetric Keltner+Donchian long-only channel breakout,
ratcheted trailing-stop exit, vol-target sizing, 200% cap) to our book with real OHLC (so a real ATR
instead of the paper's close-only 1.4×MAD proxy), and ran it lookahead-free over full history.

- **On our book it is weak and the timing overlay SUBTRACTS risk-adjusted return.** Frictionless
  full-history Sharpe: **core8 0.40 / equity-only 0.34**, versus the *same vol-targeting with timing
  OFF* (always-long) at **0.50 / 0.46**. Timing costs ~0.10–0.12 Sharpe.
- **Robust negative:** the timed strategy loses to always-long in **0 of 27** (entry × exit × k)
  parameter configs and in **every** sub-era. Not a bad-parameter or one-window artifact.
- **What timing actually buys is drawdown reduction, not Sharpe** — core8 MDD ~42% vs ~51%; it
  protects in crises (2008 **+9%**, 2022 **+8%** vs always-long) but gives up huge upside in up-years
  (2004/2019/2024 ≈ **−16%** vs always-long). It is a **crisis/DD overlay, not a return engine.**
- **Breadth is the paper's real mechanism, and we don't have it.** At maximum cross-asset breadth
  (15 futures) the frictionless number flips to a small edge (timed 0.66 vs 0.58, ~half the DD) — **but
  that edge is *entirely* pre-1997**, a no-equity bonds+commodities trend book. In the modern,
  actually-tradeable era (1997+, where our equity book exists) always-long wins **gross and net at every
  cost level** (1997+ gross 0.65 vs 0.75; 2bp 0.55 vs 0.68; 5bp 0.40 vs 0.57).
- The paper's **Sharpe 1.39** needs ~48 diversifying long-only **equity** streams sharing one positive
  risk premium. Our book is ~4 near-identical equity indices (0.65–0.87 correlated) + non-drifting
  commodities/bond ≈ **3–4 independent bets** — no comparable diversification multiplier, and no
  risk-premium premise on metals/energy/FX.
- **Verified lookahead-free** by two fully independent clean-room reimplementations (every headline
  Sharpe reproduced; a final-bar corruption test changed 0 earlier positions).

**Bottom line:** the method does not provide a robust standalone Sharpe edge on the instruments we
trade; its only durable property (drawdown reduction) overlaps with our existing
`[[project_equity_curve_filter]]` work ("risk overlay, not return engine"). Do **not** pursue as a vault
sleeve. If we want trend diversification, the lever is **breadth** (more uncorrelated markets), not this
entry/exit machinery — which underperforms plain always-long vol-targeting on our universe.

---

## Objective

Assess robustness & effectiveness of the Industry-Timing method **on our tradeable universe**
(Darwinex/FTMO canonical book: equity indices ES/NQ/YM, metals GC/SI, energy CL, bond TLT, FX). The
question is not "does the published result replicate" (it's on a different asset class) but "does the
**method** port to what we trade, and is it a candidate sleeve for the prop/futures book?"

## The method, ported (faithful, adapted to real OHLC)

- **Long-only per instrument.** Daily bars, decision at close *t*, position held into *t+1* (weights
  `.shift(1)`).
- **Entry (from flat):** `close_t ≥ UpperBand_{t-1}`, `UpperBand = min(DonchianUp(20), KeltnerUp(20,k=2))`;
  `KeltnerUp(n,k) = EMA(close,n) + k·ATR(n)`, ATR = simple-mean true range (real OHLC — *not* the paper's
  close-only 1.4×MAD proxy, which exists only because French data is close-only).
- **Exit (ratcheted trailing stop, never moves down):** `LowerBand = max(DonchianDown(40), KeltnerDown(40,k=2))`;
  `stop_t = max(stop_{t-1}, LowerBand_{t-1})`; exit when `close_t < stop_t`. Asymmetry (20-bar entry /
  40-bar exit) → structural long bias / slow exit.
- **Sizing:** each currently-long instrument `w_j = (target_vol/N)/σ_j`, σ = 14-day stdev of returns,
  `target_vol = 1.5%` daily total, gross capped at 200%. Futures are collateralised → rf cancels (rf=0);
  Sharpe is the scale-invariant comparison metric.
- **Benchmark = always-long vol-target (timing OFF):** identical sizing machinery, position pinned to 1.
  This isolates the *value of the timing rules*. (A true static equal-notional hold is a *worse*
  benchmark — core8 Sharpe 0.30, MDD −87%, dominated by the most volatile leg — so vol-targeting is what
  keeps the comparison honest.)
- **Data:** ratio-adjusted continuous futures from `data/ohlc_data` (~1997 for equity indices, ~1978 for
  metals/energy, bonds 1980s, TLT 2002). Adjustment matters for channel geometry — ratio-adjusted, not
  additive.

---

## Results (lookahead-free, frictionless unless noted)

### Per-instrument (standalone, sized as 1/8 of the book)

| | NQ | TLT | GC | YM | RTY | ES | SI | CL |
|---|---|---|---|---|---|---|---|---|
| Sharpe | **0.52** | 0.39 | 0.30 | 0.24 | 0.24 | 0.17 | 0.17 | **−0.08** |

FX negative control (no long-drift premise): EU 0.18, BP 0.15, JY 0.16, SF 0.08, AUDNZD 0.04, CD −0.12 —
all ≈ 0. The control behaves: the method manufactures no edge where there is no risk premium.

### Portfolio — timed vs. always-long (the timing-attribution test)

| Universe | N | Timed Sh | Always-long Sh | Δ (timing) | Timed MDD | Always-long MDD |
|---|---|---|---|---|---|---|
| equity-only (ES/NQ/YM/RTY) | 4 | 0.34 | **0.46** | −0.12 | −34% | −53% |
| core8 (+GC/SI/CL/TLT) | 8 | 0.40 | **0.50** | −0.11 | −42% | −51% |
| wide15 (max cross-asset breadth) | 15 | **0.66** | 0.58 | +0.09 | −18%…−28%* | −42% |

\* timed wide15 MDD is band-convention-dependent: our engine reports −18%, an independent clean-room
build −28%. Both are ≈ half the always-long DD; the point survives either way.

- **Param robustness (core8):** 0 of 27 configs (entry∈{10,20,40} × exit∈{20,40,80} × k∈{1.5,2,3}) beat
  always-long; best (20/80/k3) = 0.46 < 0.50.
- **Sub-era (core8):** always-long wins **every** era — 1997-07 0.62 vs 0.28; 2008-14 0.62 vs 0.57;
  2015-26 0.79 vs 0.63.

### The wide15 "edge" decomposed — it's pre-1997, not modern, and not cost-driven

| Window | Timed Sh | Always-long Sh |
|---|---|---|
| pre-1997* (no-equity bonds+commodities) | **0.70** | 0.32 |
| 1997-2007 | 0.76 | **0.90** |
| 2008-2014 | 0.72 | **0.94** |
| 2015-2026 | 0.52 | 0.49 |
| **1997+ (the era we can actually trade)** | 0.65 | **0.75** |

\* all equity indices start ≥ 1997-09 (ES) and TLT 2002, so the pre-1997 window is a *different,
no-equity* universe. The full-history edge is a 1978–1996 commodity/bond trend "golden age," not a
property of the 15-asset mix we'd trade.

**Costs do NOT differentiate** (both legs rebalance daily): timed turns over 27.6×/yr, always-long
23.4×/yr. Fair both-net wide15 full-history: 2bp 0.57 vs 0.51, 5bp 0.44 vs 0.42 — timed even edges
slightly. But post-1997 both-net, always-long wins at every level (2bp 0.55 vs 0.68; 5bp 0.40 vs 0.57).
**Era/breadth defeats the edge, not transaction costs.** (An earlier draft claimed costs killed it; that
compared net-timed against gross-always-long — corrected here.)

### Why it doesn't port: diversification

core8 per-instrument timed-stream correlations: the 4 equity indices are **0.65–0.87** with each other
(≈ one bet), GC–SI 0.59; only cross-asset-class pairs diversify. Avg pairwise corr 0.18, but effectively
**3–4 independent bets**, not 8. The paper's 48 industries — even at higher pairwise corr — give a far
larger diversification (FDM) multiplier on many positively-drifting streams. **The 1.39 is mostly a
breadth result on a uniform-risk-premium universe.** We have neither the breadth nor the uniform premium.

---

## Verdict (adversarially verified)

> The Industry-Timing method gives **no robust standalone Sharpe edge on the instruments we trade**; it
> acts as a **drawdown/crisis overlay**. Its apparent return edge at wide breadth is **entirely pre-1997**
> — a no-equity bonds+commodities trend book — and at our available **post-1997 breadth the timed
> strategy loses to always-long vol-targeting both gross and net of costs** (both 1997–2007 and 2008–2014
> won by always-long; only the recent era a tie). It is **era/breadth, not transaction costs**, that
> defeats the edge.

Three independent reviewers (two clean-room reimplementations + an engine audit) confirmed C1–C4; no
conclusion-flipping bug was found; lookahead-freedom verified by construction, a two-derivation check
(max diff 6.9e-18), a no-shift "cheat" probe (+1.03 Sharpe, diagnostic-only), and a final-bar corruption
leak test.

## The one genuinely useful property

Across every test the method's *robust* effect is **drawdown reduction**: lower MDD, crisis-year
protection (2008 +9%, 2022 +8%), positive monthly/limited downside. This is exactly a lookahead-free P&L
**regime/risk overlay** — the same conclusion as `[[project_equity_curve_filter]]` ("robust maxDD cut;
risk overlay not return engine"). If a drawdown overlay is wanted, that existing tool is the cheaper,
more general lever than bolting the asymmetric-channel machinery onto the book.

## Risk register / caveats

1. **Single feed** — ratio-adjusted continuous futures (`data/ohlc_data`); not re-checked on the live CFD
   feed. Irrelevant given the edge dies in Phase-0 frictionless (no point spending Nautilus realism).
2. **MDD convention** — wide15 timed MDD is −18% (our engine) to −28% (clean-room) depending on
   band/EMA/stop convention; the "~half the always-long DD" conclusion is convention-robust.
3. **Crisis-edge averaging** — +4.5%/yr is the 4-crisis-year average (2008/2018/2020/2022); the per-year
   2008 +9% / 2022 +8% are not in dispute.
4. **N = fixed universe size** is faithful to the spec; early years run low exposure because few
   instruments exist yet. The benchmark uses identical machinery, so this does not bias the comparison.

## Recommendation / next step

**Do not pursue as a standalone vault sleeve.** On our instruments the timing overlay underperforms plain
always-long vol-targeting in every era and parameterization; its only value is drawdown reduction, which
we already address via `[[project_equity_curve_filter]]`. The repo already harvests per-instrument trend
(`gc_breakout`, `cl_breakout`, `robust_trend_breakout`). If portfolio-level trend diversification is the
goal, the actionable lesson from this study is that **breadth (many uncorrelated markets), not this
specific entry/exit construction, is the lever** — and our cross-asset futures book tops out at ~3–4
independent bets, well short of what makes the published 1.39 work.

## Artifacts (kept — clean reproducible harness, not scratch)

- `engine.py` — the ported method + always-long benchmark + metrics (lookahead-free, verified).
- `run_poc.py` — Phase-0: lookahead proof, per-instrument, portfolio, timing attribution, cost sweep.
- `run_robustness.py` — breadth / crisis / sub-era / param-grid / diversification / wide15 deep-dive.
- `outputs/core_equity.csv`, `outputs/rolling3y_sharpe.csv`.

Run from repo root with `.\.venv\Scripts\python.exe research\experiments\industry_timing\run_poc.py`
(and `run_robustness.py`); set `$env:PYTHONIOENCODING='utf-8'`.
