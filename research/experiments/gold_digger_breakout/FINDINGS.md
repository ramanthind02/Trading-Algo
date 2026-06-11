# Gold Digger / Gen_Breakout — Asian-session range breakout on XAUUSD & USDJPY

**Date:** 2026-06-09 · **Status:** Phase 0 PASS (both) · Phase 1 PASS (gold modestly, JPY strongly
ex-2026) · **No Phase-2 vault slot** (intraday; vault rail is daily-only).

This memo is the record. The reproducible harness is `poc.py` (base engine + verification) and
`analysis.py` (cost + robustness battery); raw outputs in `outputs/`. Inline scratch one-liners
used during the study were not saved as files.

---

## The strategy (faithful to the MQL5 EA)

Asian-session range breakout, one independent episode per broker calendar day:

1. **Range** = `[max(high), min(low)]` of M1 bars in the **broker-time** window `[03:00, 06:00)`
   (EET/EEST). `width = high − low`.
2. After the window closes, arm stop orders at both edges. The **first post-06:00 M1 bar** to trade
   through an edge enters that side **at the edge** (buy-stop @ range_high / sell-stop @ range_low).
3. **Stop-loss = the opposite edge** ⇒ `SL distance = width` ⇒ **every loss is exactly −1R**.
   **No take-profit.**
4. Force-flat at the **daily close 18:00** broker ⇒ time-exit at an R-multiple. Never holds overnight.
5. At most **one filled trade/day** (the opposite pending is cancelled on fill).

Percent-risk sizing makes risk-per-trade constant in account terms, so the natural P&L unit is the
**R-multiple** (PnL ÷ range width). This also makes gold and JPY directly comparable and normalizes
away the range-width-dependent lot scaling.

**Broker-time note:** MT5 stamps are tz-tagged UTC but are actually broker EET/EEST wall-clock. We
strip the tz and the hour thresholds map 1:1 to the EA's "server time" — **no shift**. Real dense M1
starts **2018**; the study floors there (2018-01-02 → 2026-06-05, ~8.4 yrs).

---

## Verification (this is why the numbers are trustworthy)

- **Independent clean-room re-derivation** (a separate agent, forbidden from reading `poc.py`,
  re-implemented the spec from scratch on the same data) reproduced the base headline **to 5–6
  decimals** on both symbols (XAU Sharpe 1.19315 vs 1.193; JPY 1.67356 vs 1.674; all of
  n/win/avg_R/total_R/PF identical). That rules out a lookahead or logic bug in the base path.
- **Adversarial code audit** verdict: *minor_issues*, all harmless — same-bar BE optimism
  (BE-variant only, since fixed, conservative direction), gap optimism (already handled: all cost
  work uses the gap-aware base, and gap-aware == fill@level here), straddle/Sharpe-denominator
  (conservative or immaterial: 0 straddles XAU, 1 JPY).
- Lookahead is structural: the range uses **only** `[03:00,06:00)` bars; entries/exits **only**
  `[06:00,18:00)` bars. M1 intrabar high/low is faithful for the real intrabar stop.

---

## Phase 0 — frictionless gross edge (range 03–06, flat 18)

| sym | n | win% | avg_R | total_R | PF | Sharpe(d,ann) | maxDD_R |
|---|---|---|---|---|---|---|---|
| **XAUUSD** | 2159 | 39.6 | **+0.124** | +269 | 1.235 | **1.19** | −22.7 |
| **USDJPY** | 2153 | 45.1 | **+0.155** | +334 | 1.342 | **1.67** | −25.3 |

Classic breakout profile: low win rate, **−1R-capped losers**, runners held to the 18:00 flat.
Positive **nearly every year**:

- **Gold:** +ve every year except 2019 (≈flat, −2R). 2018 +58R, 2020 +44, 2021 +51, 2024 +46. **2026
  YTD healthy (+19R).**
- **JPY:** +ve every year **except 2026** (see decay below). 2020 +72R, 2022 +57, 2025 +62.

**Symmetry (the non-beta test):**
- **Gold is genuinely symmetric** — long avg **+0.1255R** / short **+0.1235R** (PF 1.25 / 1.22),
  50.8% long. A real breakout edge, **not long-gold beta** despite the 2024–25 bull run.
- **JPY real both ways but long-tilted** — long +0.210R (PF 1.48) / short +0.096R (PF 1.20). The
  short side works (not pure beta), but the historical magnitude was lifted by 2021–25 yen weakness.

---

## Phase 1 — cost / realism (the decision for gold)

Cost in R = `cost_price / width`, so a fixed price cost hurts **tight-range days hardest** (they
carry the largest size). Closed-form total-R breakeven: `cost* = Σ gross_R / Σ(1/width)`.

| sym | breakeven rt cost | recorded entry spread (median) | net Sharpe @ realistic cost |
|---|---|---|---|
| **XAUUSD** | **$0.70** | $0.05 (25% record $0) | spread-only **1.10** → +2×slip/side ($0.44 rt) **0.71** (PF 1.13) |
| **USDJPY** | **0.037 (≈3.7 pip)** | 0.001 (42% record $0) | spread-only **1.56** → +2×slip/side (0.013 rt) **1.09** (PF 1.21) |

- **Gold is cost-fragile.** It survives only with tight execution: net Sharpe ~**0.7–1.0** if
  round-trip cost stays under ~$0.40–0.50; **dead by ~$0.80**. Recorded spread is tiny ($0.05–0.09)
  but **25% of entries record $0** (feed artifact ⇒ true cost is understated) and stop-entry
  **slippage**, not spread, is the binding constraint on gold breakouts.
- **JPY has wide cost headroom** (breakeven ≈3.7 pip vs realistic ~1–2 pip round-trip) — net Sharpe
  stays >1 even at 2× slippage.

**Width-quintile fragility (gold, ref cost $0.30)** — explains the user's "min-range filter hurt PF":

| width quintile | med width | gross avg_R | net avg_R |
|---|---|---|---|
| q1 (tightest) | $2.86 | **+0.220** | +0.107 |
| q2 | $4.60 | +0.137 | +0.071 |
| q3 | $6.52 | **−0.011** | −0.057 |
| q4 | $9.60 | +0.129 | +0.097 |
| q5 (widest) | $23.5 | +0.149 | +0.135 |

The gross edge lives in the **tightest** ranges (q1) and widest (q5); the middle is dead. So a
**min-range filter removes the most profitable gross bucket** — even though q1 is also the most
cost-fragile, it stays net-positive. Confirms the user's empirical result.

---

## ⚠️ USDJPY 2026 out-of-sample decay (live red flag)

| 2026 month | n | win% | total_R |
|---|---|---|---|
| Jan | 21 | 29% | **−5.5** |
| Feb | 20 | 40% | −1.8 |
| Mar | 22 | 41% | −0.8 |
| Apr | 22 | 27% | **−7.7** |
| May | 20 | 25% | **−8.8** |

**Five straight losing months** (−24R YTD). Not noise — it coincides with the post-2024 BoJ-hike /
carry-unwind yen regime, and it hit exactly when the long-USD/JPY tilt reversed. The 2018–2025 JPY
Sharpe (1.67) **should not be assumed forward.** Gold shows **no** such decay (2026 +19R).

---

## SL → Breakeven (the user's best-reported feature) — reconciled

BE (move SL to entry once price runs `be_R × width` in favor), with same-bar pessimism modeled:

- **Gold:** BE never beats "none" on Sharpe (1.19) or total R; PF ticks up marginally (1.235→1.247).
- **JPY:** tight BE (0.5R) **raises PF 1.342→1.441 and cuts maxDD −25.3→−21.8R** (this *is* the
  user's "PF increase"), but **lowers total R 334→248 and Sharpe 1.674→1.611**.

**SL→BE is a risk-smoothing overlay — higher PF, lower drawdown, higher hit-rate — not an
alpha-adder** (it trades expectancy for hit-rate). The user's PF improvement is real; it just isn't
a free lunch.

---

## Robustness — broad plateaus, default sits near the top

- **Range window** (rs×re sweep): flat plateau; default **03–06** is near-optimal (gold Sharpe 1.19,
  JPY 1.67). No knife-edge — not overfit to the timing.
- **Close hour:** monotone-improving 12→18; default **18:00** near the gold peak; JPY marginally
  prefers 20–22 (Sharpe 1.69–1.73).

---

## Assessment of the user's other proposed features

- **Optional EOD close (hold overnight):** the close-sweep already shows holding *later* helps
  (gold→18, JPY→22), but going past the 18:00 flat introduces overnight swap drag + gap risk and
  breaks the one-episode-per-day structure — test separately; expect carry drag to bite multi-bar holds.
- **Campaign mode (add a 2nd position day-2 after SL→BE):** pyramiding; only sensible if the base
  edge is strong and trends persist multi-day — gold's edge is intraday and modest, so this likely
  adds variance more than return. Low priority.
- **Friday EOD flatten:** the EA already flats daily at 18:00, so this only matters if EOD close is
  made optional. Sensible *if* overnight holding is added.

---

## Verdict & next steps

**This is a real, simple, non-overfit intraday breakout edge** — verified to the decimal by an
independent re-implementation, symmetric on gold, robust across timings, positive nearly every year.

- **Gold:** genuine symmetric edge, but **modest and execution-sensitive** (net Sharpe ~0.7–1.0,
  contingent on keeping round-trip cost under ~$0.40–0.50). A **diversifier, not a standalone**.
- **JPY:** historically the stronger, more cost-robust leg, **but currently decaying** (2026 YTD
  broken). Do not deploy on the 2018–25 Sharpe.
- Both are **orthogonal to the daily futures/CFD vault** (intraday session breakout on FX/metal).
  There is **no Phase-2 vault slot for intraday** (the StrategySpec/validate/vault rail is daily-only),
  so this stays an experiment / candidate for a separate intraday lane rather than a vault sleeve.

**If pursued:** (1) nail the *real* gold execution cost with live-demo stop-entry fills (slippage is
the swing factor); (2) add the high-impact-USD news-filter overlay (ignored here — makes the POC
slightly optimistic); (3) monitor whether the JPY decay is transient or a regime break before
trusting JPY again.

---

## Follow-up (2026-06-09): is the yen model dead? · ensembling · economic rationale

### Is the USDJPY model dead? — 2026 is a genuine regime break, not a normal drawdown
Three independent tests agree (`decay.py`, `outputs/usdjpy_decay.png`):
1. **Statistically unprecedented.** Block-bootstrap of the 2018-2025 per-trade R series: the 2026-YTD
   result (−24.6R / 110 trades) is below the **0.06th percentile** (`pct_below=0.0006`) — the worst
   110-trade window in the whole history. (Prior deep dips — mid-2023 rolling Sharpe ≈ −2 — recovered
   *within a rising equity*; this one rolls over from the all-time high.)
2. **Both directions broke.** The edge was **long-USDJPY-tilted** (long side +ve every year; short the
   weak leg). In 2026 the shorts did NOT catch the yen rally — **long −12.4R AND short −12.2R**. Both
   sides failing ⇒ the *breakout mechanism* stopped working, not merely a directional tilt flipping.
3. **Mechanism signature.** 2026 has the **highest stop-out rate (50.9%)** and the **lowest avg winner
   (1.09R vs 1.14-1.50)** — more false breakouts, smaller runners ⇒ breakouts stopped following through.

**Cross-market localization:** USDJPY-epicentered but not universal — USDJPY −24.6R is the worst of all;
GBPUSD/AUDUSD/CADJPY/GBPJPY also soft; **NZDJPY +12, CHFJPY +31, gold +19, silver +20 are fine**.
Crucially the **full-history edge only ever existed on JPY crosses + metals** (USDJPY +0.155R, gold
+0.124R) — *not* on EURUSD/GBPUSD (≈0). The edge is structurally a **Tokyo-session** phenomenon.

**Verdict:** not "permanently arbitraged away", but the **directional yen tailwind that made USDJPY the
standout is gone** and the current regime is actively hostile. **Cannot be traded at 2018-25 sizing;**
base case = impaired / regime-dependent / stand-aside until a directional yen regime resumes or a
regime filter is proven.

### Can we add other tickers to make an ensemble? — gross yes, net only gold survives
`ensemble.py`, `outputs/ensemble.png`. Basket selected by **structural rationale** (Tokyo-session names),
not realized Sharpe: JPY crosses {USDJPY,GBPJPY,AUDJPY,NZDJPY,CADJPY,CHFJPY} + metals {XAUUSD,XAGUSD}.
- **Correlation structure is favorable**: JPY crosses ~0.32 among themselves (partly one yen bet),
  metals 0.49 with each other, **metals vs JPY ≈ 0.0-0.11 (orthogonal)**. Gold is the real diversifier
  (USDJPY↔XAUUSD corr **0.11**).
- **Gross**, the 8-name ensemble looks great: Sharpe **1.54**, div-ratio **1.84×**, maxDD halved to −19.8R.
- **Net of realistic cost it collapses**: only **USDJPY (1.33) and XAUUSD (0.90)** clear their spread;
  the 5 secondary JPY crosses + silver all flip **net-negative** (CHFJPY 0.35→−1.10, NZDJPY 0.91→−0.52)
  — thin gross edges vs wider synthetic-cross spreads. So a naive equal-weight ALL-8 ensemble (**net
  −0.27**) is **WORSE than trading USDJPY alone** — it dilutes the one winner with cost-losers.
- **The only cost-real ensemble is {USDJPY, XAUUSD}**: net Sharpe **1.48** (> either alone), maxDD −20R,
  positive every year 2018-25. **2026: USDJPY −29.8R but gold +15.6R ⇒ pair only −7.1R** — gold absorbs
  most of the yen-regime hit *because they're uncorrelated*. That decay-year cushion is the real
  argument for ensembling here, not higher return.
- **Better diversification is cross-STRATEGY, not cross-ticker**: same-strategy FX names co-move and
  co-decay; the orthogonal lever is combining this breakout with [[project_lafo_kama_mr]] (corr −0.03 to
  a breakout proxy) / [[project_orb_ibs_nas100]] mean-reversion + momentum sleeves.

### Economic rationale (web-grounded, fact-checked — `yen-macro-research` workflow)
**Why a Tokyo-range → London/NY breakout works:** FX has a well-documented intraday activity/volatility
seasonality — Tokyo is the low-volatility, range-bound home session; London open + NY/US data bring the
volatility and directional order flow that breaks the Asian range (Ito & Hashimoto 2006, EBS data:
U-shaped Tokyo/London activity; Andersen & Bollerslev 1998 on macro-announcement vol; Crabel 1990 on
contraction-precedes-expansion / opening-range breakout). The Tokyo "fixing" (9:55 JST) shows a
structural importer USD-buying imbalance (Ito & Yamada 2017). This is why the edge lives on **JPY pairs
and gold** (Tokyo is their home session) and **not** on EURUSD/GBPUSD (03-06 EET is dead-of-night for
Europe) — matching the cross-market data exactly.
**Why USDJPY was the standout, and why 2026 broke it:** USDJPY had a persistent **directional driver** —
the yen carry trade / wide US-Japan rate differential — that gave Asian-range breakouts *follow-through*
(2021-24 trend, ~108→162). The regime then turned: **BoJ ended NIRP/YCC on 19 Mar 2024**, hiked
31 Jul 2024 (and into 2025), the **Aug 2024 carry unwind** (BoJ hike + weak 2 Aug US payrolls → USDJPY
~161→142 in ~3 weeks, Nikkei −12.4% on 5 Aug), MoF yen-buying interventions (2022, Apr-Jul 2024), and the
US-JP rate gap narrowed as the Fed cut. The positive rate-differential↔USDJPY link even weakened/flipped
around Apr 2025. Net: the carry/trend tailwind that powered the breakout's follow-through is gone, leaving
a two-way, mean-reverting, intervention-prone yen where intraday breakouts get faded — exactly the
both-sides-failing / higher-whipsaw signature seen in 2026. *(Caveat: academic work documents the
seasonality but does NOT establish net-of-cost profitability — that rests on this study's own cost lane.)*

## Reproduce

```powershell
# base headline + by-year (XAU & JPY), with built-in verification asserts
.\.venv\Scripts\python.exe -m research.experiments.gold_digger_breakout.poc
# cost sensitivity + width-bucket fragility + window/close/BE sweeps
.\.venv\Scripts\python.exe -m research.experiments.gold_digger_breakout.analysis
# is-it-dead decay diagnostics (rolling Sharpe, long/short, bootstrap, cross-market)
.\.venv\Scripts\python.exe -m research.experiments.gold_digger_breakout.decay
# ensemble: correlations, diversification, cost-survivors, jackknife
.\.venv\Scripts\python.exe -m research.experiments.gold_digger_breakout.ensemble
```

Data: `data/mt5_data/{XAUUSD,USDJPY}/bars_M1/year=*/part.parquet` (2018+ real M1). Outputs:
`outputs/equity_R.png`, `outputs/by_year_*.csv`, `outputs/battery.txt`.
