# Concretum / Zarattini intraday momentum — robustness & stability

Two related Concretum papers, one experiment dir:
- **Part 1 — "Beat the Market" (Zarattini–Aziz–Barbon, 2024):** the Noise-Area intraday momentum
  strategy. Verdict: faithfully replicated, parameter-robust, **friction-fragile and net-negative OOS**
  — **REVISED in Part 4**: the "net-negative" call used a 1 bps/turn cost that is ~5× the real CFD
  spread; at the true recorded spread the in-sample edge is net-positive (~1.0), and a lower-turnover
  variant is OOS-viable on NDX. The cost verdict was overstated; the decay is the real (separate) issue.
- **Part 2 — "Improving Performance with Fast Alphas" (QuanTips #2, Zarattini–Pagani, 2026):** a
  5-min reversal *fast alpha* used as an **execution overlay** on a slower breakout. Verdict: the
  "informational alpha" idea **works** — a real, significant, OOS-stable net improvement (the
  *opposite* outcome to Part 1's standalone).

**Date:** 2026-06-08 · **Phase:** 0 (vectorized, frictionless-first) + a Phase-1 cost preview.
Durable artifacts: `engine.py` + `run_matrix.py` + `analyze.py` + `_smoke.py` (Part 1);
`fast_alpha.py` + `run_fast_alpha.py` (Part 2); `outputs/`. This memo is the record.

---

# PART 1 — "Beat the Market" Noise-Area intraday momentum

**Status:** edge faithfully replicated and characterised; **friction-fragile** and **not established
net-of-cost out-of-sample**. The decisive untested factor is event-driven fill realism (Nautilus lane).

---

## 0. The strategy (decoded from the paper)

Time-of-day **Noise Area**: boundaries from the average |move-from-open| over the previous 14 days,
*per time-of-day mark*, so they widen through the session:
```
move_{t-i,HH:MM} = |close_{t-i,HH:MM}/open_{t-i,9:30} - 1|;  sigma = mean_{i=1..14} move (strictly past)
UB = max(open_t, close_{t-1}) * (1 + VM*sigma);  LB = min(open_t, close_{t-1}) * (1 - VM*sigma)
```
Decisions **only at semi-hourly marks 10:00–15:30 ET**, force-flat at 16:00. Enter long>UB / short<LB.
Three stops: **opp** (hold to opposite band, reverse = base model), **curr** (exit to flat on
current-band re-entry), **curr+VWAP** (trailing stop max(UB,VWAP)/min(LB,VWAP) = headline). Vol-target
sizing: `leverage = min(4, 0.02/sigma_daily)`, `sigma_daily` = 14-day sample stdev of daily returns.

## 1. Method (frictionless-first, lookahead-disciplined)

Faithful M1 event-replay on `data/mt5_data/<SYM>/bars_M1` (broker EET, ET = stored − 7h; US cash
session 09:30–16:00 ET = 16:30–23:00 broker). **Instruments deliberately chosen for cross-representation
robustness**, with the longest clean M1 coverage we have:

| Symbol | Wrapper | Underlying | Clean M1 |
|---|---|---|---|
| **SPY** | ETF (paper's exact instrument) | S&P 500 | 2010–2026 |
| **SP500** | index CFD | S&P 500 | 2018–2026 |
| ES | "futures" CFD | S&P 500 | **excluded — broken series** (see §7) |
| **QQQ** | ETF | Nasdaq-100 | 2010–2026 |
| **NDX** | index CFD | Nasdaq-100 (the blog's "NQ") | 2018–2026 |

2010–2026 on SPY/QQQ overlaps the paper (ends Apr 2024) **and** extends ~2 yrs past publication — a
genuine out-of-sample persistence test. Lookahead discipline: time-of-day sigma / prev-close /
leverage all use `.shift(1)` (strictly past); the position chosen at mark *k* earns the **forward**
return mark *k*→*k+1*; a from-scratch minute-grid **log-return** replay matches the engine mark-grid to
**2.1e-15** (`_smoke.py`), proving alignment. An independent re-implementation (verification §8)
reproduced the SPY base-model Sharpe (engine 0.343 brackets between its bar-open 0.487 / bar-close 0.296
anchor variants). The **binary** (leverage = 1) series is reported alongside the vol-target series
everywhere as the implementation-invariant anchor.

---

## 2. Headline verdict

**A genuine, faithfully-replicated intraday-momentum edge that is parameter-robust and
cross-representation consistent — but friction-fragile and not established net-of-cost
out-of-sample.** It replicates the paper's *mechanism* cleanly; it does **not** clear a realistic
cost bar as a standalone forward strategy.

1. **Replication is faithful (mechanism, not magnitude).** SPY 2010–2024 frictionless reproduces the
   paper's qualitative ladder: base opp-band Sharpe **0.34** (skew −0.34) → +current-band/VWAP **0.72**
   (skew flips **+0.87**) → +vol-target **0.94**, with vol **14.4%** / maxDD **26%** essentially equal
   to the paper's 14.3% / 25%. Absolute Sharpe is below the paper's 1.33 because we miss **2007–2009**
   (the GFC high-vol window the paper says the strategy thrives in) and run a frictionless CFD proxy.

2. **Parameter-robust — smooth plateaus, no knife-edge.** Volatility-Multiplier peaks at **VM ≈ 1.0–1.25**
   and declines smoothly (the paper's claim that VM≈1.5 is best risk-adjusted does *not* replicate — we
   peak earlier); time-of-day lookback flat across 7–28 days; Sharpe is essentially **invariant** to the
   vol-target level (1–3%) and leverage cap (2–6×), confirming vol-targeting is a risk overlay, not the
   edge. Decision frequency: **hourly ≈ semi-hourly** (robust), 15-min slightly worse, and the academic
   **last-30-min-only baseline is weak/negative** (SPY 0.50, NDX −0.73) — which *validates the paper's
   central thesis* that confining momentum to the last 30 min is suboptimal.

3. **Cross-representation consistent.** The edge appears in **both** wrappers of **both** underlyings:
   S&P via SPY (ETF, gross Sharpe 0.87) and SP500 (index, 1.07); Nasdaq via QQQ (0.90) and NDX (1.12).
   Same signal, independent data series → not a single-series artifact.

4. **The VWAP refinement — the paper's headline innovation — adds essentially nothing.** Decomposing the
   stop: **opp → curr** is the entire jump (SPY 0.31→0.87, SP500 0.65→1.10, NDX 0.70→1.18); **curr →
   curr+VWAP** adds ~0 and sometimes slightly *hurts* (NDX 1.18→1.12). The Sharpe "doubling" is the band
   **tightening**, not VWAP. A cleaner spec drops VWAP. (Robust across all four instruments.)

5. **Cost is the binding constraint — breakeven ≈ 1 bps one-way (≈2 bps round-trip).** At the paper's own
   assumed SPY-ETF cost (~1 bps one-way) the **in-sample net** Sharpe is **marginal**: SPY **0.05**, QQQ
   **0.26**; only the high-index-level names clear it (NDX **0.84**, SP500 **0.67**). The recorded CFD
   spread in this feed is ~0 (a feed artifact, frequently zero — *not* real cost), so the honest cost is
   the paper's ~1 bps + slippage/impact, which sits **at breakeven**. This is the same gate that
   threatened the repo's prior intraday studies (ORB, channel-breakout).

6. **Out-of-sample, net of cost, the edge is not established.** Gross post-publication (May 2024–Jun 2026,
   ~526 days) Sharpe roughly **halves** in point estimate (SPY 0.94→0.45, QQQ 0.97→0.45, NDX 1.46→0.30,
   SP500 1.44→0.05). **But the decay is statistically significant only for SP500** (block-bootstrap
   P(real decay)=0.99); for SPY/QQQ the 2-year window is too short (P≈0.78, post-pub Sharpe CI includes 0)
   — *not confirmed dead, but not confirmed persistent either*. Decisively: **at 1 bps net the post-pub
   Sharpe is negative on every instrument** (SPY −0.36, QQQ −0.18, SP500 −0.76, NDX −0.27), and 2025–2026
   are the worst calendar years across the board.

---

## 3. Robustness detail

### 3a. Escalation ladder + OOS persistence (`outputs/escalation__*.csv`)
Headline **vol-target** Sharpe (full / paper ≤2024-04 / post ≥2024-05):

| sym | full | paper-era | post-pub | binary curr+VWAP (full/paper/post) |
|---|---|---|---|---|
| SPY | 0.87 | 0.94 | 0.45 | 0.70 / 0.72 / 0.58 |
| QQQ | 0.90 | 0.97 | 0.45 | 0.78 / 0.84 / 0.44 |
| SP500 | 1.07 | 1.44 | 0.05 | 0.79 / 0.99 / 0.09 |
| NDX | 1.12 | 1.46 | 0.30 | 1.04 / 1.24 / 0.48 |

The index CFDs (SP500/NDX, 2018+ only) post the highest paper-era Sharpe — partly period selection
(their data starts in the strong 2018–24 window) — and decay the most. The longer-history ETFs
(SPY/QQQ, 2010+) post lower paper-era Sharpe and a gentler binary-core decay.

### 3b. Bootstrap significance of the decay (`outputs/bootstrap_significance.csv`)
Block-bootstrap (block 10, n 4000) 95% Sharpe CIs and the paper→post decay test:

| sym | paper Sharpe [95% CI] | post Sharpe [95% CI] | decay = paper−post [CI] | **P(decay real)** |
|---|---|---|---|---|
| SPY | 0.94 [0.44, 1.38] | 0.45 [−0.89, 1.54] | 0.54 [−0.68, 1.89] | 0.78 (NS) |
| QQQ | 0.97 [0.47, 1.43] | 0.45 [−0.98, 1.62] | 0.56 [−0.74, 2.00] | 0.79 (NS) |
| SP500 | 1.44 [0.73, 2.06] | 0.05 [−1.25, 1.09] | 1.44 [0.20, 2.84] | **0.99** |
| NDX | 1.46 [0.83, 2.07] | 0.30 [−1.18, 1.52] | 1.20 [−0.26, 2.76] | 0.94 |

### 3c. Net-of-cost escalation (`outputs/net_escalation.csv`) — gross vs net @ 1.0 bps one-way

| sym | paper gross→net | post gross→net |
|---|---|---|
| SPY | 0.94 → **0.05** | 0.45 → **−0.36** |
| QQQ | 0.97 → 0.26 | 0.45 → −0.18 |
| SP500 | 1.44 → 0.67 | 0.05 → −0.76 |
| NDX | 1.46 → 0.84 | 0.30 → −0.27 |

Cost waterfall (`outputs/cost__*.csv`): breakeven one-way bps ≈ SPY 1.0, QQQ 1.3, SP500 1.4, **NDX 1.9**
(most cost-robust — highest index level → lowest relative turnover cost).

### 3d. Long/short symmetry (`outputs/longshort__*.csv`) — vol-target Sharpe
Long-tilted but the short side is **independently positive everywhere** → a real symmetric component, not
pure long-beta, but with a directional tilt (paper claims β≈0; ours leans long):

| sym | both | long-only | short-only |
|---|---|---|---|
| SPY | 0.87 | 0.92 | 0.42 |
| QQQ | 0.90 | 0.96 | 0.40 |
| SP500 | 1.07 | 1.19 | 0.52 |
| NDX | 1.12 | 1.06 | 0.65 |

### 3e. Stability over time (`outputs/yearly__*.csv`, `rolling_stability.csv`)
SPY 13/17 calendar years positive (best-2-years = 32% of profit → **not** lottery-concentrated, unlike the
repo's ORB study at 57%). Rolling 252-day Sharpe: median ~1.0–1.3 gross, **79–92% of windows > 0**, but a
worst rolling-12m Sharpe of **−1.3 to −1.8** — a meaningful underwater capacity. **2025–2026 are weak/
negative across all instruments** (the OOS softness in calendar form).

### 3f. Day-of-week & vol regime (`outputs/dow__*.csv`, `regime__*.csv`)
- **DOW replicates the paper (§4.3):** Wed & Fri strongest and significant (SPY Wed t+2.5, Fri t+2.3),
  Monday weakest/insignificant (t+0.5). (Per-DOW Sharpe annualized by √52, not √252 — verification fix.)
- **The paper's "Sharpe rises with VIX" claim (§4.1) does NOT cleanly replicate** on realized-vol
  quartiles: the by-vol Sharpe profile is non-monotonic/noisy (SPY full: Q1 0.98, Q3 0.35, Q4 1.22;
  SP500 *low*-vol Q1 is strongest at 2.04), and post-pub the high-vol buckets often turn negative.
  Caveat: realized-vol proxy ≠ VIX-at-open exactly.

---

## 4. Robustness scorecard

| dimension | result |
|---|---|
| Faithful replication of paper's mechanism | ✅ ladder, skew-flip, vol≈14% / maxDD≈25% reproduced |
| Lookahead-clean | ✅ minute-grid log-PnL = mark-grid to 2e-15; independent recompute brackets engine |
| Parameter plateau (VM / lookback / sizing / freq) | ✅ smooth, no knife-edge |
| Cross-representation (ETF vs index, S&P & Nasdaq) | ✅ consistent on 4 of 5 series |
| Beats academic last-30-min baseline | ✅ all-day semi-hourly >> last-30 |
| Long/short symmetry (not pure beta) | ⚠️ short positive but long-tilted |
| VWAP refinement adds value | ❌ marginal/none over current-band stop |
| Survives realistic cost | ⚠️ breakeven ~1 bps one-way; at-breakeven in-sample, **negative OOS** |
| OOS persistence (gross) | ⚠️ halves in point estimate; significant only for SP500 |
| OOS persistence (net @1bps) | ❌ negative on every instrument |
| "Better in high VIX" (paper §4.1) | ❌ does not cleanly replicate on realized-vol buckets |
| Subperiod stability | ⚠️ 79–92% rolling windows>0 but worst-12m Sharpe −1.3…−1.8; 2025–26 weak |

---

## 5. Threats to validity / caveats

- **No 2007–2009.** Our data starts 2010 (2015/2018 for some), missing the GFC — exactly the high-vol
  regime the paper credits for much of the edge. Our cost-pessimism is *partly* because we miss the
  strategy's best window; the paper's full-sample net edge is plausibly stronger than our 2010+ proxy.
- **Cost is a flat bps proxy, not event-driven fills.** The recorded CFD spread is ~0 (feed artifact), so
  the bps sweep is the honest cost axis; but it omits true per-fill spread, market-order slippage on the
  breakout itself, impact at ~2.6× median leverage, and commission. The real net number needs the
  **Nautilus realistic lane** — this is the decisive open question (§7).
- **maxDD / tot_ret are additive (cumsum) not compounded** — overstates DD on the leveraged curve
  (SPY 26% additive vs ~17.8% compounded). Internally consistent; read as additive (verification CONCERN).
- **SPY/QQQ open anchor is the 09:31 bar** (first RTH bar is 16:31 broker), ~1 min off 09:30. Applied
  consistently to both the historical move and the live band → ~cancels; index CFDs have a true 09:30 open.
- **Post-pub window is ~2 years** → wide Sharpe CIs; "decay" is a point estimate, significant only for
  SP500. Open-anchor and RSI/NR4-pattern (paper §4.5/§4.2) sensitivities were not run (lower priority).

## 6. Recommendation

1. **Do not promote as a standalone.** Gross it is a real, robust, well-replicated edge; **net of a
   realistic ~1 bps one-way it is at-breakeven in-sample and negative out-of-sample.** As a forward
   net proposition it is not established.
2. **The one credible path is execution-quality + the right instrument.** It is only viable if all-in
   one-way cost is held **well below ~1 bps** — i.e. institutional execution on the tightest, highest-
   index-level instrument (NDX/SP500, or **real ES/NQ futures economics** — *not* this CFD), where the
   in-sample net Sharpe (~0.7–0.8) is worth trading. Confirm with the Nautilus realistic lane before
   anything else.
3. **If pursued, simplify:** drop the VWAP stop (no value over the current-band stop), use VM≈1.0–1.25,
   hourly or semi-hourly marks — fewer parameters, same edge.
4. **Potential value is as a low-correlation intraday-equity sleeve**, not a standalone — its daily
   stream is plausibly orthogonal to the daily trend/MR vault. That only matters if it clears cost.
5. **Phase-1 gate (next step):** run NDX + SP500 + SPY through the Nautilus realistic lane (true
   per-fill spread/slippage/commission + the rollover overlay) on a common 2018+ window to nail the
   net Sharpe at the spread it actually pays. The flat-bps proxy says "at breakeven"; only event-driven
   fills resolve go/no-go.

## 7. The ES casualty (data quality)
The `ES` CFD is **negative across every window and dimension** while SPY *and* SP500 (same underlying)
are strongly positive — a red flag. Its price ranges **$45–98** over 2015–2026 with wild non-monotonic
swings (peak ~$98 in 2020 → ~$56 in 2024), while SPY (×3.2) and SP500 (×3.2) track the real S&P. The ES
series does **not track the S&P 500** (a mis-rolled / decayed / mislabelled product) and carries sporadic
huge spreads. **Excluded as a data-quality casualty** — the blog's ES claim cannot be evaluated on this
series; the S&P edge is validated via SPY + SP500 instead. (Flagged for the data team.)

## 8. Verification (adversarial — 5-agent workflow)
- **Lookahead lens: PASS.** No forward leak; the minute-grid log-PnL proof is genuine and dispositive
  (different aggregation grids, exact telescoping). 3 low-severity notes, all confirmed safe.
- **Fidelity lens: PASS.** Bands, per-time-of-day sigma, all three stops, semi-hourly marks, force-flat,
  and vol-target sizing all faithful to the paper. 2 low-severity, non-biasing notes (1-min SPY anchor;
  no same-mark VWAP re-entry — a faithful choice).
- **Cost/metrics lens: CONCERN→addressed.** Core cost model correct (cost scales with leverage; reversal =
  2 units; point_size 0.01). Two real defects fixed/caveated: **dim_dow Sharpe over-annualized by √5**
  (fixed → √52; t-stats were always correct) and **additive maxDD overstates the compounded curve**
  (caveated). Breakeven ~1 bps is "effectively at-breakeven, not comfortable."
- **Independent recompute: PASS.** From-scratch SPY base model reproduced the engine (0.343 brackets
  between bar-open 0.487 / bar-close 0.296; vol% matches; trade-days% exact). No sign flip.
- **Completeness critic: CONCERN→addressed.** Demanded (1) bootstrap significance on the decay, (2)
  net-of-cost headline, (3) VIX/vol-regime, (4) rolling stability — **all four added** (`analyze.py`,
  `dim_regime`). Remaining un-run: open-anchor sensitivity, RSI/NR4 conditioning (lower priority).

## 9. Artifacts (`outputs/`)
- `escalation__*.csv` (ladder × window) · `bootstrap_significance.csv` (CIs + decay p) ·
  `net_escalation.csv` (gross vs net@1bps) · `rolling_stability.csv`
- `vm/lookback/sizing/freq/stop/longshort/cost/dow/regime__*.csv` (per-dimension grids, 5 syms)
- `yearly__*.csv` (calendar-year stability)
- Reproduce: `python -m research.experiments.concretum_intraday_momentum.run_matrix <SYM>` then
  `… .analyze`. Lookahead proof: `… ._smoke`.

---

# PART 2 — "Improving Performance with Fast Alphas" (QuanTips #2) execution overlay

**Status:** the paper's central thesis — that a fast, *un-tradeable* alpha can add value as an
**execution overlay** on a slower monetizable strategy ("informational alpha" ≠ "monetizable alpha")
— **replicates and holds up on our data.** This is the *opposite* outcome to Part 1's standalone:
here the conditioning genuinely, significantly, and OOS-stably improves the slower strategy net of cost.

## P2.0 The three pieces (decoded)
- **Fast alpha:** 5-min one-bar reversal `S_t = −sign(R_t)` (+ streak-N conditioning). Extreme turnover.
- **Baseline:** ATR(14)-band breakout (Kaufman) — bands = `session_open ± 0.5·ATR(14d)`; long if a
  15-min-mark close > upper, short if < lower; **stop = return to session open**; 15-min marks
  (HH:00/15/30/45); EOD flat; vol-target 2%/day.
- **Overlay:** delay the baseline's *execution* until the fast alpha prints an opposite 5-min bar —
  long entry waits for a down-bar (pullback), long stop-exit waits for an up-bar (counter-move).

`fast_alpha.py` (engine) + `run_fast_alpha.py` (robustness), 5-min bars resampled from the same M1 RTH
store; SPY/QQQ (2010+), SP500/NDX (2018+). Lookahead-clean by construction: the fast alpha at bar *i*
uses `R_i` (known at close *i*) and the position earns the **forward** 5-min return; ATR and leverage
use `.shift(1)` (strictly past).

## P2.1 Fast alpha replicates and is universal (`outputs/fa_streaks.csv`, `fa_standalone.csv`)
5-min mean-reversion (avg next-bar return after ≥N same-sign bars), all 4 instruments, highly significant:

| sym | Down N≥1 | Up N≥1 | Down N≥4 | Up N≥4 |
|---|---|---|---|---|
| SPY | +0.17 (t7.0) | −0.12 (t−5.3) | +0.35 (t3.7) | −0.24 (t−3.5) |
| QQQ | +0.15 (t5.0) | −0.11 (t−4.0) | +0.43 (t3.7) | −0.19 (t−2.3) |
| SP500 | +0.13 (t3.3) | −0.09 (t−2.6) | +0.22 (t1.6) | −0.18 (t−1.8) |
| NDX | +0.13 (t2.8) | −0.06 (t−1.4) | +0.44 (t2.9) | −0.16 (t−1.5) |

(QQQ +0.43 after 4 down-bars matches the paper's +0.43 exactly.) **Standalone** the fast alpha is
strong gross (Sharpe SPY **2.26**, QQQ 1.61, SP500 1.46, NDX 1.06 — paper ">2") but **dies on cost**:
at 1 bps one-way, net Sharpe **−11 to −15**, ann ~−170%, ~78 turns/day. → *informational, not
monetizable.* ✓

## P2.2 The overlay works — and the ENTRY leg is the whole value (`outputs/fa_overlay.csv`)
Net @1 bps one-way Sharpe, full sample:

| sym | baseline | entry-only | exit-only | both | **placebo** |
|---|---|---|---|---|---|
| SPY | 0.142 | **0.373** | 0.150 | 0.379 | 0.081 |
| QQQ | 0.262 | **0.452** | 0.315 | 0.504 | 0.229 |
| SP500 | 0.305 | **0.461** | 0.176 | 0.369 | 0.195 |
| NDX | 0.541 | **0.720** | 0.496 | 0.709 | 0.531 |

- **The entry overlay (wait for a pullback) is the entire gain; the exit overlay is neutral-to-slightly-
  negative** (entry-only ≈ both; exit-only ≈ baseline). The paper bundles both — our decomposition says
  drop the exit overlay.
- **It is genuine price-timing, not cost reduction:** the overlay improves **gross** Sharpe too
  (SPY 0.61→0.85, NDX 0.88→1.05) at ~unchanged turnover (~1.1/day).
- **Placebo falsification PASSES decisively:** flipping the overlay to wait for a *same*-direction bar
  is **≤ baseline everywhere** (SPY 0.081 < 0.142). The gain is the documented 5-min reversion being
  harvested at the fill — not a generic delay/turnover or look-ahead artifact (a structural leak would
  help the placebo too).

## P2.3 Significant, OOS-stable, and lifts the cost ceiling
- **Paired block-bootstrap on (overlay − baseline) net Sharpe** (`outputs/fa_boot.csv`): SPY Δ**0.237**
  [0.13, 0.35] **P=1.00**; QQQ Δ**0.239** [0.13, 0.35] **P=1.00**; NDX Δ0.174 [−0.11, 0.39] P=0.89;
  SP500 Δ0.084 [−0.24, 0.31] P=0.71. **Significant on the two highest-data instruments** (SPY/QQQ);
  borderline/insignificant on the short-history index CFDs.
- **Persists out-of-sample** (post ≥2024-05): SPY 0.179→**0.514**, QQQ 0.209→**0.606**, NDX 0.405→0.487.
  The overlay's *value* is more OOS-stable than the underlying strategy's *level*.
- **Cost sweep** (`outputs/fa_cost.csv`): the overlay shifts the whole net-Sharpe curve up ~0.2–0.25 at
  every level, buying **~0.4–0.6 bps of cost headroom** (breakeven SPY 1.2→1.6 bps, NDX 2.5→3.0). Still
  a friction-sensitive intraday strategy — the overlay *improves* implementability, it is not a free lunch.
- **ATR-mult sensitivity** (`outputs/fa_atr.csv`): the baseline breakout is somewhat fragile (best at
  0.25–0.5 ATR, weak/negative at ≥0.75); the overlay is positive across more band-widths — it also adds
  parameter robustness.

## P2.4 Caveats
- **Our baseline breakout is materially weaker than the paper's** (net Sharpe ~0.14 vs the paper's 0.87;
  gross 0.61). Most likely the paper computes ATR(14) + daily sizing-vol from **Norgate daily bars (incl.
  overnight gaps)**, while we derive the daily range from RTH-only 5-min bars → tighter bands → noisier
  breakouts; plus we lack 2007–09. **The robust deliverable is the *relative* overlay improvement**, which
  replicates in direction and significance; the absolute baseline level is convention-dependent.
- Cost is still a flat-bps proxy (recorded CFD spread ~0 in this feed) — the true net needs event-driven
  fills (the spread the breakout *actually* pays). But unlike Part 1, the overlay's *incremental* value is
  a price-improvement that should, if anything, look better with realistic microstructure.

## P2.5 Verification (adversarial — 3-lens workflow, all PASS)
- **Overlay-lookahead lens: PASS.** No leak — position at bar *i* uses `R_i` (closed bar) and earns the
  strictly-forward 5-min return; ATR/leverage `.shift(1)` (strictly past); the breakout-confirming bar is
  never captured. Decisive counterfactual: forcing the position to earn the *same* bar yields Sharpe
  **−0.903** — a same-bar leak would *hurt* this construction, so the genuine forward 0.379 is real.
  The placebo-hurts result is confirmed valid evidence against an accounting leak.
- **Overlay-fidelity lens: PASS.** 5-min resample, 15-min marks (exactly ET :00/:15/:30/:45, 09:45–15:45),
  ATR band/stop, per-leg overlay + placebo, vol-target — all faithful. The RTH-vs-Norgate ATR gap explains
  the absolute-level shortfall (net 0.14 vs paper 0.87) and is direction-symmetric → relative conclusion intact.
- **Independent recompute lens: PASS.** A from-scratch SPY reimplementation reproduced **all three**
  falsification criteria *convention-invariantly*: baseline ≪ overlay, a positive material net improvement
  (+0.145 to +0.152), and placebo ≤ baseline — confirming the fast-alpha reversal timing is the causal driver.
  (Absolute levels differ by convention, as expected.)
- Two low-severity items: a misleading inline comment (**fixed**) and a conservative NaN-first-bar entry
  delay (intended, lookahead-free). Neither affects correctness.

## P2.6 Verdict & recommendation (Part 2)
The "informational alpha" framework is **validated**: a fast signal that is worthless standalone
(net −15 Sharpe) **significantly and robustly improves** the net performance of a slower monetizable
strategy by conditioning *when* it executes — confirmed across 4 instruments, significant on the two
with most data, OOS-stable, and survives a placebo falsification. Two concrete takeaways:
1. **Adopt the entry overlay, drop the exit overlay** — the entry-timing (wait for a 5-min pullback
   before taking a breakout) is the entire, significant gain; the exit overlay adds nothing.
2. **This is the more promising of the two papers for us.** The execution-overlay idea is *general* — the
   same 5-min-reversion entry-timing could be bolted onto **any** of our intraday breakout/momentum
   sleeves (incl. Part 1's Noise-Area model, and the channel-breakout / ORB studies) to recover a few
   tenths of net Sharpe of cost headroom. The natural next step is to test the entry overlay on those
   existing sleeves and, if it holds, build it into the Phase-1 Nautilus execution layer (a fill-timing
   rule), not as a standalone strategy.

## P2.7 Artifacts (Part 2)
- `fa_streaks__/fa_standalone__` (fast-alpha existence + standalone) · `fa_overlay` (variant × window ×
  gross/net panel incl. placebo) · `fa_cost` (cost sweep) · `fa_atr` (band-width) · `fa_boot` (paired
  bootstrap significance).
- Reproduce: `python -m research.experiments.concretum_intraday_momentum.run_fast_alpha`.

---

# PART 3 — Improving the Noise-Area model (3 frictionless levers tested)

Tested the three ideas net @1 bps one-way on a 5-min grid (2018+ index / 2010+ ETF). To let the 5-min
overlay interpose, the Noise-Area model is rebuilt natively on 5-min bars (`improvements.py`); its mark
prices are ~4 min coarser than the 1-min engine, so the **5-min baseline sits a bit below the 1-min
engine** (NDX 0.39 vs 0.52) — improvements are judged *relative to the 5-min baseline* (same grid).

## P3.1 #1 Fast-alpha overlay — the clear winner (apply BOTH legs)
Net @1 bps Sharpe, full sample:

| sym | baseline | + overlay (both) | gross baseline→both |
|---|---|---|---|
| NDX | 0.39 | **0.76** | 1.01 → 1.40 |
| QQQ | 0.35 | **0.56** | 1.04 → 1.28 |
| SP500 | 0.21 | **0.40** | 0.95 → 1.17 |
| SPY | 0.06 | **0.40** | 0.93 → 1.29 |

- Improves all 4 by +0.19 to +0.34 net, and **improves gross too** → genuine fill-price timing, not just
  cost. **Unlike the QuanTips ATR-breakout (exit overlay neutral), here BOTH legs help** (NDX entry-only
  0.60 → both 0.76): the Noise-Area curr+VWAP stops exit at dislocated prices that benefit from waiting
  for a counter-move.
- **Helps OOS too** (lifts every instrument post-pub) but does **not reverse** the base signal's OOS
  decay — NDX/SP500 still ~flat/negative post-pub; SPY overlaid is solidly positive (0.34).

## P3.2 #2 Frequency / cooldown — a cost & drawdown hedge, not a free Sharpe lift
- **Hourly vs semi-hourly is cost-dependent:** semi wins ≤0.5 bps, ~tie at 1 bps, **hourly wins ≥2 bps**.
  Stacked on the overlay, hourly cuts turnover ~25% and maxDD sharply, and *raises* Sharpe on the weaker
  names (SPY 0.40→**0.49**, SP500 0.40→**0.44**) at a small cost on the strong (NDX 0.76→0.70, maxDD
  **32→20**). Adopt hourly if real costs run ≥~2 bps or to tame drawdown.
- **Cooldown:** cooldown=1 is marginal (NDX 0.60→0.61), ≥2 hurts. Not worth it.

## P3.3 #3 Equity-curve kill-switch — too fragile; a niche maxDD tamer that can blow up
Window- and instrument-sensitive. On NDX *overlay-entry* w50 it held Sharpe while halving maxDD — but on
*overlay-both* it cut Sharpe (0.76→0.51), on SPY it killed Sharpe, and on **SP500 overlay-both it went to
−0.42**. Inconsistent and risks more than it saves on this strategy. **Do not adopt systematically;** at
most a discretionary drawdown circuit-breaker on one clean stream (`execution/equity_curve_filter.py`).

## P3.4 Recommendation
**[NOTE: these numbers are at 1 bps/turn — ~5× the real CFD cost. See P4.7 for the real-cost re-pricing
(roughly doubles everything) + the long-only cut.]**
**Core win: Noise-Area + fast-alpha overlay (both legs)** — net @1 bps NDX 0.39→0.76, SPY 0.06→0.40,
QQQ 0.35→0.56, SP500 0.21→0.40; robust across 4 instruments, gross-confirmed, OOS-stable *improvement*.
**Add hourly marks** as a cost/drawdown hedge (helps the weaker names, ~halves NDX maxDD). **Drop**
cooldown and the equity filter. This roughly **doubles the net Sharpe** of the headline model and is the
most productionizable result of the study — but it is still an intraday strategy at ~1 bps breakeven, so
the real go/no-go remains the **Nautilus fill-realism lane** (where limit-vs-market also gets answered).

## P3.5 Artifacts (Part 3)
- `imp_overlay` (overlay × window × gross/net, 4 syms) · `imp_freqcost` · `imp_cooldown` · `imp_eqfilter` ·
  `ndx_equity_overlay_improvement.png`.
- Reproduce: `python -m ...run_improvements` then `... .plot_improvements`. Engine: `improvements.py`
  (5-min Noise-Area + overlay/cooldown/equity-filter).

---

# PART 4 — Peter/CrackingMarkets live variant + the CFD cost correction

Triggered by: a practitioner (CrackingMarkets "Peter") trades a *simpler* variant of this family live
with a real dated track record (since May 2023, combined Sharpe ~1.57). Two outcomes: (a) built and
validated that simpler model, and (b) discovered our Part-1 cost penalty was **~5× too harsh**, which
materially revises the "net-negative OOS" verdict.

## P4.1 The cost correction (the headline of Part 4)
Part 1 charged **1 bps/turn**. The **real recorded MT5 CFD spread** is ~**0.7 idx pts round-trip on NDX
= ~0.4 bps/day** at 1.77 turns/day (point=0.1; the catalog's 0.01 increment is wrong for spread). The
Nautilus lane confirms this (`avg_rt_cost_bps` ≈ 0.43–0.58). Re-running the **academic Noise-Area model**
(semi, curr_vwap, vol-target) at the true cost, net Sharpe **full / paper / post**:

| cost | NDX | SP500 |
|---|---|---|
| frictionless | 1.12 / 1.46 / 0.30 | 1.07 / 1.44 / 0.05 |
| **real recorded spread** | **0.98** / 1.30 / 0.21 | **0.99** / 1.41 / −0.16 |
| real spread + 0.5 pt/side slip | 0.75 / 1.07 / −0.00 | 0.05 / 0.48 / −1.13 |
| *(old 1 bps/turn)* | *0.52 / 0.84 / −0.26* | *0.29 / 0.67 / −0.76* |

So the academic model is **net-positive in-sample (~1.0)** at real cost, NOT net-negative. The remaining
problems are **separate from spread**: (1) genuine **OOS decay** (frictionless post collapses 1.46→0.30
NDX, 1.44→0.05 SP500; NDX decay was *not* statistically significant per Part-1 bootstrap, SP500 *was*);
(2) **turnover × slippage** — at 1.77 turns/day a fixed 0.5-pt slippage is 0.37 bps/turn on NDX but
**1.2 bps/turn on SP500** (low index price ~4,150), which annihilates SP500 (full 0.05).

## P4.2 Lower-turnover academic variant rescues NDX (not SP500)
At real spread + 0.5 pt/side slip, sweeping marks × stop:

| sym | marks | stop | turn/d | full | paper | post |
|---|---|---|---|---|---|---|
| NDX | semi | curr_vwap | 1.77 | 0.75 | 1.07 | −0.00 |
| NDX | **hour** | **curr** | 1.28 | **0.85** | 1.06 | **0.30** |
| NDX | hour | opp | 1.10 | 0.53 | 0.52 | **0.58** |
| NDX | semi | opp | 1.21 | 0.50 | 0.51 | 0.48 |
| SP500 | hour | curr_vwap | 1.32 | 0.17 | 0.62 | −1.08 |
| SP500 | semi | opp | 1.24 | 0.09 | 0.27 | −0.42 |

**hour+curr beats the baseline on both in-sample AND OOS** (lower turnover, drop VWAP). The **`opp` stop
(lowest turnover) shows NO OOS decay** (full≈paper≈post≈0.5) — so part of the "decay" was the curr_vwap
**whipsaw instability**, not pure alpha death. **SP500 stays dead at any turnover** (price×slippage).

## P4.3 Peter's live variant (`peter_breakout.py`) — the simple model, validated
Rules (distinct from the academic one): ATR(5) bands off the **open** (open ± 0.4·ATR5); entry =
**resting STOP at the band**; stop = retrace to the **open** (risk = 0.4·ATR = exactly 1R); **one long +
one short attempt/day** (no semi-hourly re-eval); EOD flat; fixed-fractional risk (PnL in R-multiples).
Why it survives where the academic one struggles: **~0.9 trades/day** (vs 1.77), clean asymmetric payoff,
**no OOS decay**. Results (R-Sharpe, full/paper/post):
- NDX frictionless 0.87 / 0.84 / **0.96** (NO decay); @1 bps one-way 0.41 / 0.37 / 0.51.
- Parameter-insensitive (entire ATR_n×k grid 0.41–0.66 net). Long/short legs **regime-anti-correlated**
  (long carried 2018–24, short carried post-2024 +1.00) → keep BOTH sides (opposite to Part 3).
- GLD ~0.05 corr to equities (real diversifier); DIA/GLD weak standalone. Basket cost-fragile (low-edge
  ETFs drag); single-market NDX is the cleaner vehicle. Peter's own low-vol-day filter did NOT reproduce
  as a naive natr-percentile gate.

## P4.4 Nautilus CFD lane (`peter_nautilus_lane.py`) — the realistic answer
Event-driven Nautilus lane: entries are **resting STOP_MARKET** at the bands (one-shot/side is automatic
— an entry order is consumed on fill), protective STOP at the open, EOD flat; real recorded half-spread
(taker both legs, point=0.1) + slippage charged on the ledger; PnL in R to reconcile vs the POC.
- **Reconciliation PASS**: frictionless lane vs `peter_breakout` daily-R corr **0.9998** (2yr) / **0.9755**
  (full). The full-history gap (lane sumR 148.6 > POC 123.7) is the **same-minute whipsaw convention**
  (POC pessimistically books entry+stop same bar as −1R; lane rests the protective stop to the next bar)
  → treat the realistic Sharpe as a RANGE.
- **NDX net (full history)**: recorded spread only **Sharpe 0.98** (cost 0.58 bps); + 0.5 pt/side slip
  **0.77** (cost 1.41 bps). The real CFD spread is **NOT the binding constraint — stop-entry slippage is.**

## P4.5 Verdict (Part 4)
- **CFDs work** for the single-market NDX breakout: real recorded spread keeps Sharpe ~1.0, realistic
  slippage ~0.77, OOS-stable. SP500 is borderline (low price → proportionally larger slippage).
- The **academic vs practitioner gap narrowed** once costs were corrected: at real cost both are ~0.8–1.0
  in-sample on NDX; Peter wins on **OOS stability + slippage-robustness (low turnover) + more markets**.
- Open: SP500 Peter lane (interrupted), futures economics (MNQ/MES — even cheaper), bootstrap CI on the
  Peter no-decay claim, per-leg per-year on the Peter engine, curated low-corr basket vs single.

## P4.6 Artifacts (Part 4)
- `peter_breakout.py` (vectorized model + R-multiple accounting) · `peter_nautilus_lane.py` (Nautilus CFD
  lane + POC reconcile) · `plot_peter.py` → `outputs/peter_breakout_equity.png` ·
  `outputs/peter_nautilus_NDX.csv` + `..._recon_NDX.csv`.
- Reproduce: `python -m research.experiments.concretum_intraday_momentum.peter_nautilus_lane --sym NDX --start 2018`.
- Verified: 3-agent adversarial pass (lookahead audit, independent from-scratch recompute matched NDX 2024
  to 4 dp / 50 clean −1R stops, completeness critic). One EOD optimistic-fill bug found + fixed (immaterial).

## P4.7 Part-3 overlay re-priced at REAL cost + long-only (revises the P3.4 table)
The P3.4 overlay table was built at **1 bps/turn**; at the **real CFD cost (~0.25 bps/side)** the whole
Noise-Area + fast-alpha-overlay model **roughly doubles**. Net Sharpe **full / post** (maxDD additive):

| sym | baseline (real) | **overlay-both (real)** | overlay-both @1bps *(old P3.4)* |
|---|---|---|---|
| NDX | 0.85 / 0.18 | **1.24 / 0.43** (mdd 25%) | 0.76 |
| QQQ | 0.86 / 0.33 | **1.10 / 0.54** (mdd 27%) | 0.56 |
| SP500 | 0.77 / 0.32 | **0.98 / 0.45** (mdd 19%) | 0.40 |
| SPY | 0.71 / 0.67 | **1.07 / 0.97** (mdd 27%) | 0.40 |

The overlay still adds its genuine ~0.2–0.4 (gross-confirmed; frictionless gross NDX 1.40), but the big
lift is the cost correction — at ~0.25 bps the net sits just under gross, so **cost is no longer binding**
for this model on CFDs. **LONG-ONLY** cut (net Sharpe full / post, maxDD):

| sym | long-only + overlay | both-sides + overlay |
|---|---|---|
| NDX | **1.17 / 0.52**, mdd **10%** | 1.24 / 0.43, mdd 25% |
| SP500 | 0.86 / 0.37, mdd 9% | 0.98 / 0.45, mdd 19% |
| QQQ | 1.04 / 0.50, mdd 14% | 1.10 / 0.54, mdd 27% |
| SPY | 0.97 / 0.56, mdd 10% | 1.07 / 0.97, mdd 27% |

**Long-only ~thirds the drawdown for a ~0.05–0.15 Sharpe give-up** → best return-per-DD. OOS split is
instrument-dependent: **NDX long-only is better OOS** (short leg is the weak one lately, post 0.52 > 0.43);
**SPY keeps both sides** (its short leg carries OOS, 0.97 vs 0.56). Standout for a DD-limited/prop context:
**NDX long-only overlay, Sharpe 1.17 at ~10% maxDD.** Caveats: maxDD is additive `cumsum` (read the
long-vs-both *ratio* ≈⅓, not the level); only NDX has a feed-validated spread (0.26 bps/side ≈ the 0.25
used) — SPY/QQQ/SP500 recorded spread is the known 0.0 feed artifact, so those three use a representative
tight cost. Artifacts: `plot_overlay_realcost.py` → `outputs/ndx_overlay_longonly_realcost.png`; engine
`improvements.py` (now carries `long_only` + `ma_filter`). Reproduce with `IParams(long_only=…,
overlay_entry=True, overlay_exit=True, cost_bps_oneway=0.25)`.

## P4.8 Cost realism — how real is 0.25 bps/side? (live FTMO + Darwinex spread check)
Validated the cost basis against **live broker quotes**, not just the recorded-spread column.
- **FTMO `US100.cash`, last 2 weeks** (13,650 M1 bars, 2026-05-26→06-08, queried read-only from the live
  terminal; `point=0.01`): spread **median 0.246 bps/side**, mean 0.262, p99 0.347. Stable (~145 pts midday),
  widens modestly at the cash open (mean 158, max 190) and close (median 178, max 208). **So 0.25 bps/side
  is essentially exact for FTMO's *current* spread.** Darwinex is tighter (live snapshot NQ 1.1 idx pts ≈
  **0.19 bps/side**).
- **But the backtest numbers are still optimistic, for 3 reasons:** (1) **spread-only — no stop-entry
  slippage**, and this strategy stop-enters *into* momentum (fills at ask-or-worse); the lane showed
  +0.5 idx-pt/side slip → NDX 0.98→0.77, and 0.5 pt is arguably light for breakout stops. (2) **Price-level
  effect** — the spread is ~1.5 idx *points*; FTMO is ~30k *today* but the backtest's median price is ~13.4k,
  where the same 1.5 pts = **~0.5 bps/side, not 0.25** → the honest historical spread is ~0.4–0.5 bps/side.
  (3) **Demo feed + a calm 2-week window** (no CPI/FOMC spread blow-out or gap-through).
- **Honest central case: ~0.5–0.75 bps/side all-in.** Use the **0.5 bps/side** column, not 0.25: NDX overlay
  both **1.08** / long-only **1.04**; Peter NDX lane +0.5pt slip **0.77**. Edge SURVIVES (realistic NDX
  Sharpe ~0.8–1.1, OOS-stable) but the 1.2+ headline is the optimistic floor. The one genuinely unmeasured
  cost is **stop-entry slippage on breakouts** — only a fills-based test (real, or a slippage model in the
  lane) resolves it. Diagnostics: `/tmp` spread-realism + `ftmo_spread.py` (live read-only; not committed —
  re-run against the terminal). Recorded-spread reality: NDX recorded spread DOES vary (66 vals, 0–95 pts,
  widens at open/close, tightened 2018-19 → 7-8 from 2020) — it's real, not a placeholder; SPY/QQQ/SP500
  recorded spread is 0 (feed artifact).

---

# STUDY BOTTOM LINE (Parts 1–4)

Two CrackingMarkets-adjacent intraday breakout families, one experiment dir. Net of the whole study:

1. **The Noise-Area academic model was MISPRICED, not broken.** Our original "friction-fragile / net-negative
   OOS" verdict charged 1 bps/turn — ~5× the real CFD spread. At realistic cost (~0.5 bps/side) it is
   net-POSITIVE in-sample (~0.8–1.0) on NDX. Its real weaknesses are SEPARATE from spread: genuine OOS decay
   (partly curr_vwap whipsaw — the low-turnover `opp` stop doesn't decay) and turnover×slippage (kills SP500
   at its low index price).
2. **The fast-alpha overlay is the best single improvement** and at real cost ~doubles the model
   (NDX both 1.24 / long-only 1.17 @0.25; ~1.08 / 1.04 @0.5). Gross-confirmed (genuine fill timing), OOS-stable.
3. **Peter's simpler live-traded variant is the more robust vehicle** — ~0.9 trades/day, no OOS decay,
   validated in a Nautilus CFD lane (recon corr 0.98–0.9998) at NDX net 0.77–0.98 on real costs.
4. **CFDs work** for the single-market NDX breakout; the binding constraint is **stop-entry slippage**, not
   spread. **Best DD-efficient config: NDX long-only + overlay (~Sharpe 1.0–1.2 @ ~10% maxDD).** SP500 is
   borderline (low price → proportionally larger slippage).

**Open / next:** SP500 Peter lane (interrupted); put the overlay model through its own Nautilus fills lane
(it currently uses a flat-bps proxy, so its 1.08–1.24 lack the lane's stop-entry realism); futures economics
(MNQ/MES — even cheaper than CFD); block-bootstrap CI on the Peter no-decay claim; curated low-corr basket
(NDX+GLD, GLD ~0.05 corr) vs single-market.
