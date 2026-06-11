# LAFO / KAMA fair-value mean-reversion — FINDINGS

**Date:** 2026-06-09 · **Instrument:** NDX (Darwinex Nasdaq-100 cash CFD), real M1 2018-2026 ·
**Status:** SURVIVOR (right-sized) — a real, lookahead-clean, cost-robust long-only intraday MR
edge, but low-frequency, event-concentrated and regime-contingent → **capped diversifier sleeve
only, not a standalone.** Intraday, so **not vault-promotable** (the StrategySpec/vault rail is
daily-only).

> All scripts here are reproducible analysis (engine + run scripts, like the concretum/orb_ibs
> studies). This memo + `outputs/` is the record.

---

## 1. The idea (and what the EA actually is)

From Xu et al. 2025 (UCL) "Advanced Signal Filtering for Mean Reversion". The shared NinjaTrader EA
("LAFO_MeanReversion_Share") is **not** the paper's neural filter — it's a simple **KAMA
(Kaufman Adaptive MA) fair-value** instantiation:

```
fair_value = KAMA(period=20, fast=5, slow=30)
delta      = (close - KAMA) / KAMA                 # relative mispricing
LONG  delta < -thr   ·   SHORT delta > +thr         # EA: thr = 1%
EXIT: 3.5x ATR(14) stop + fixed R:R target          # EA: 0.5R (current) / 1.5R (overall)
```

**The EA's own defaults LOSE.** Faithful port, M5 RTH, 2018-2026, **frictionless**:
Sharpe **−0.43**, PF 0.94, −38% total (60% win rate, but the 0.5R-target / 3.5x-stop = 1:2
reward:risk takes tiny profits and eats rare huge stops — a classic curve-fit). The EA's "1% on
MNQ 5-min" settings sit squarely in the no-edge zone. The user's instinct ("a lot of potential for
improvement") was right — but the improvements are the **opposite** of the EA defaults.

## 2. Verified Darwinex costs (NOT assumed — recorded from 54.2M ticks)

| fact | value | source |
|---|---|---|
| symbol | **NDX** (not "NAS100" — that's FTMO's name) | configs/mt5_brokers.yaml NQ:NDX |
| price increment **POINT** | **0.1** (the Nautilus catalog's 0.01 is WRONG) | live-probed; research/rollover_cost/config.py |
| round-trip spread (RTH) | **~0.9 idx pts** (median 0.90, mean 0.906), ~0.45/side | 54.2M bid/ask ticks data/mt5_data/NDX/ticks |
| in bps | ~0.35 bps RT / ~0.17–0.20 bps per side | same |
| commission | **0** | data/broker_cache/darwinex/slippage.csv |
| swap | long −1.75 bps/day (pay), short +0.72 bps/day (earn) | live-probed; **RTH-only → 0 swap** |

> The prior concretum FINDINGS note "recorded CFD spread ≈ 0 (feed artifact)" was a `point_size=0.01`
> bug (charged 1/10th spread). The real recorded RTH round-trip is ~0.9 pt. This study uses POINT=0.1.

## 3. Phase 0 (frictionless) — where the edge actually lives

Swept exit method × RR × threshold × entry mode × direction × tf (M5/M15) × session.

**The edge is: LONG-ONLY, DEEP dislocations, REVERT-to-fair-value exit.**

- **Long-only.** The short side is negative *everywhere* (−0.4 to −0.8). Shorting an up-drifting index
  on MR doesn't work (the EA author warned of this). The diversifier value is the long side.
- **Deep dips only.** Edge is monotonic in threshold — median Sharpe (across all tf/stop/exit):
  1.5%→0.14, 1.75%→0.52, 2.0%→0.58, 2.25%→0.74, 2.5%→0.76, 3.0%→0.78, with **every** combo at
  thr≥1.75% positive. Routine 0.5–1% wiggles have no edge; the alpha is in the tail.
- **Revert exit > fixed target.** Exit when price crosses back through KAMA beats the EA's 0.5R target.
- Stop multiplier (2.5–4.5×ATR) barely matters (a reverting trade rarely stops). M15 ≥ M5 at the top.

Best cell: **M15, thr 2.5%, long, revert exit → frictionless Sharpe 1.01** (146 trades, maxDD 14%).

### Make-or-break: is it real timing or buy-the-dip beta?

- **Random-entry null (decisive):** on the *same days* it trades, a *random-time* long entry has null
  Sharpe mean **−0.13** / p95 +0.28; the real dip-timed entry is **0.83** (+0.55 over p95). The null
  mean is *negative* → these are down-skewed days where a random long loses, and the KAMA entry flips
  it positive. corr to always-long-RTH is only 0.25. **Timing is real, not beta.**
- **Gap-fade decomposition (rebuts "it's just overnight reversal"):** 42%/68% of entries are in the
  first 30/90 min ET, trade-day overnight return −0.87% vs +0.11% — so entries *are* gap-down
  correlated. BUT a raw "buy big gap-down open, hold to EOD, no KAMA" control scores only **0.08**
  (corr 0.46 to LAFO). Passive gap-down holding is flat-to-negative; the **KAMA entry + revert exit
  is the alpha**, not the gap. KAMA is not cosmetic.

## 4. Phase 1 (real cost) — survives, because the edge is large & low-turnover

| config | gross Sh | net @1pt slip | net @3pt | net ann% | maxDD% | tr/yr | avg gross/trade | cost/gross |
|---|---|---|---|---|---|---|---|---|
| **M15 2.5% revert** | 1.01 | **0.96** | 0.87 | 7.2% | 14.7 | 17 | 57 pt | ~4.5% |
| M5 2.0% revert | 0.83 | 0.77 | 0.66 | 4.6% | 10.4 | 15 | 42 pt | ~6% |
| M5 1.75% revert | 0.89 | 0.80 | 0.66 | 5.5% | 16.8 | 25 | 34 pt | ~8% |
| z3.5 (vol-norm) | 0.66 | 0.54 | 0.36 | 3.4% | 9.6 | 35 | 13 pt | ~19% |

Recorded spread + 1pt/side ≈ 2.6 pt cost vs a 42–57 pt reversion → cost is ~5% of edge. **Open-auction
stress:** even +6 pt punitive RT spread on the 42% of first-30-min entries → 0.96 only falls to **0.91**.

**Targeting / vol-scaling (user's question):**
- **Revert-to-mean exit** beats every fixed R:R target. The EA's 0.5R is the worst.
- **Vol-targeting HURTS** (M5 2.0%: 0.77→0.55; z3.5: 0.54→0.07) — it throttles size in exactly the
  high-vol regimes where the deep-dip edge lives. **Use fixed sizing.**
- The **vol-normalised z-entry is strictly worse** — smaller per-trade edge → cost-fragile, and it
  trades into low-vol regimes where the edge is absent. The "more principled" version loses on both.

## 5. Robustness — holds off the chosen cell

- **KAMA params:** broad plateau — median net Sharpe 0.70 over 45 combos, 76% net>0.5, base (20,5,30)
  near the top. Only `fast=8` (over-smoothed) is weak. Not param-overfit.
- **SP500 generalization (cross-sectional OOS):** the long-only deep-dip revert is positive on SP500
  too, same shape but a **lower threshold** (~1.5% peak Sharpe 0.57) — exactly right for a lower-vol
  index. It's a real index-MR phenomenon, not NDX data-mining; NDX is the higher-vol expression.
- **Session:** RTH (0.96) > 24h (0.70) but both positive — RTH isn't a cherry-pick, the edge just
  expresses best in US cash hours (economically sensible). RTH-only also = zero swap.
- **Train/test (2018-21 vs 2022-26):** positive both halves; 3 of 4 configs stronger OOS
  (M15 2.5%: train 0.76 → **test 1.17**).
- **Leave-one-year-out:** Sharpe range **0.82–1.07**; no single year is load-bearing.

## 6. The honest caveats (right-sizing the headline)

The 1.01 is the best cell of a grid on a flattering day-grid metric. Honest forward expectation:

- **PnL is concentrated in ~10–20 trades.** Drop top 5 (=44% of net pts) → 0.65; top 10 (=71%) →
  **0.37**; top 20 → negative. A fat-tailed, event-driven edge — a handful of big reversions per cycle
  carry it. This is the genuine fragility.
- **Bootstrap CI:** stationary block bootstrap of the net daily series → mean 0.98 / 5th-pct **0.59** /
  P(SR<0.5)=3%. (A more pessimistic trade-level resample centers ~0.70.) **Honest number ≈ 0.6–0.9, not 1.0.**
- **Low frequency + regime-contingent:** ~15–17 trades/yr; active/profitable in chop (2020, 2022),
  **dormant (0 trades, flat) in low-vol trend years (2023, 2026).**
- **Single-market, no internal breadth.**

## 7. Diversification value (the user's actual goal)

vs an ORB breakout proxy (trend/breakout-EA stand-in) on NDX RTH, net daily:

- **Full-sample correlation = −0.03** (≈ zero / slightly negative). Per-year mostly negative.
- **2020: breakout proxy −21.5% while LAFO +18.5%** (corr −0.21). 2018: breakout −9.3% / LAFO +2.5%.
  LAFO prints exactly when a trend/breakout book gets chopped — matches the author's lived experience
  ("Keltner slots chopped up, LAFO best quarter").
- It's dormant in trend years (2023/2026) when breakouts print → genuinely complementary.

## 8. Recommended config (vs the EA)

| | EA default (loses) | **Recommended** |
|---|---|---|
| threshold | 1% | **2.0–2.5% deep dip** |
| direction | long + short | **long only** |
| exit | 0.5R fixed target | **revert to KAMA** (3.5×ATR protective stop) |
| sizing | 1 contract | **fixed** (NOT vol-target) |
| timeframe / session | M5–M15 | **M15 (or M5) RTH** |
| frictionless Sharpe | −0.43 | +1.01 (honest ~0.6–0.9) |

**KAMA(20,5,30) is well-chosen — keep it.** Trade it as a small, capped, fixed-size diversifier
sleeve alongside the trend/breakout EAs; size it for the fat-tailed, low-frequency event profile (a
few big reversions/yr), not for steady income.

## 9. Verification record

- **Clean-room reimplementation (independent from-scratch code): EXACT replication** — 146 trades,
  identical exit-kind mix (eod 117 / revert 19 / sl 10), Sharpe 1.014 vs engine 1.010, trade-for-trade.
  No implementation/lookahead bug drives the result.
- **Adversarial code audit: cosmetic only** — one cost-timing nit (entry- vs trigger-bar spread), fixed.
  Entry-at-next-open, revert-at-close, KAMA/ATR/δ causality, the multi-trade state machine all
  lookahead-free; the 2-way PnL re-derivation asserts on every run.
- **Red-team: "plausible-but-fragile"** — its sharpest claims (just-a-gap-fade / open-auction-cost /
  3-year-luck) were tested and rebutted (§3,4,5); its real point (event concentration, wider CI) confirmed (§6).

## 10. Next step if traded

Productionize as an **intraday Nautilus CFD lane** (adapt `orb_ibs_nas100/nautilus_lane.py` or
`peter_nautilus_lane.py`: POINT=0.1, recorded-spread QuoteTicks, market fills, reconcile corr>0.99)
to confirm fills before any live sandbox. Not promotable through the daily StrategySpec/vault rail.

## Files

`engine.py` (harness: load/resample/KAMA/ATR + bracket-order event sim + metrics + lookahead assert) ·
`sweeps.py` (Phase-0 exit/RR/threshold/z grids) · `deepen.py` (long plateau, per-year, random-entry
null) · `costs.py` (real-cost + slippage + sizing) · `robustness.py` (KAMA params, SP500, session,
train/test) · `diversification.py` (vs breakout) · `verify.py` (gap-fade, bootstrap, open-auction,
LOO) · `plots.py` → `outputs/*.png` (headline equity, diversification, threshold plateau, cost decay,
annual) + `outputs/*.csv`.
