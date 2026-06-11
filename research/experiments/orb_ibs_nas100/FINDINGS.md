# ORB + IBS on NAS100 — robustness & effectiveness review

**Verdict:** a *legitimate, non-overfit, symmetric* intraday breakout edge on NAS100 — but **modest,
friction-sensitive, and regime-concentrated**. Frictionless Sharpe ≈ **1.0–1.06** (2018–2026);
realistic net Sharpe ≈ **0.82–0.94** depending on execution; it dies once round-trip cost reaches the
~10 index-point gross edge. Unlike the repo's other standalone intraday studies it **survives**
realistic cost (long hold-to-close period, not a scalper), but it is not enough on its own — its only
path to mattering is as a low-correlation **diversifying sleeve**.

**Date:** 2026-06-08 · **Phases run:** 0 (vectorized frictionless POC) + 1 (event-driven Nautilus
lane, reconciled). Durable code: `poc.py` (faithful M1 replay) + `nautilus_lane.py` (BacktestEngine);
`outputs/`. This memo is the record.

---

## 0. The strategy (decoded from the MQL5 EA)

An Opening-Range Breakout gated by a *contrarian prior-day* Internal-Bar-Strength filter. Broker time
is GMT+3 (= ET+7), so the session maps to the **US cash session**:
- **Opening range (OR)** = bars 16:30–16:59 broker (= 09:30–10:00 ET) → OR high/low; `OR_width = hi − lo`.
- **Breakout entry:** first M1 *close* beyond the OR, on a closed bar, only while 17:00 ≤ t < 21:00
  broker (10:00–14:00 ET). One trade/day.
- **IBS day-filter (the twist):** `IBS = (close − low)/(high − low)` over the **prior day's** 16:30–22:59
  session. Longs allowed only if `prevIBS < 0.5` (yesterday closed weak); shorts only if `prevIBS > 0.5`
  (yesterday closed strong). So: after a weak day, take *upside* breakouts; after a strong day, *fade*
  with downside breakouts — a coherent contrarian-day × intraday-momentum combination.
- **Exit:** SL at the opposite OR extreme, TP at `3 × OR_width`, force-flat at session end (22:59).

---

## 1. Method (frictionless-first, lookahead-disciplined)

Faithful M1 event-replay (`poc.py`) on `data/mt5_data/NDX/bars_M1` (NDX = the Nasdaq-100 CFD, our
NAS100 equivalent). Real M1 coverage **2018-01 → 2026-06**, ~2,172 session days; pre-2018 partitions
are sparse daily snapshots and excluded. NDX point = **0.1** (1-decimal prices; the live-probed
increment, not the catalog's 0.01).

**Lookahead discipline:** trigger reads a **closed** bar, entry is the **next bar's open**; intrabar
SL/TP resolved **SL-first** (pessimistic); the per-trade PnL is derived two independent ways and
asserted equal (`assert |a−b| < 1e-6`). MT5 timestamps are broker EET/EEST mislabelled UTC — the
wall-clock hh:mm *is* the broker time the EA keys off (confirmed by the 01:00 daily reopen), so the
session boundaries apply to the stored timestamps directly.

**Metric:** per-trade notional return `(exit − entry)·dir / entry`; daily Sharpe over every session
day (flat days = 0, one trade/day max); R-multiple = net / (entry − SL).

---

## 2. Headline verdict

A genuine, faithfully-replicated intraday edge that is **parameter-robust and symmetric** and **clears
a realistic cost bar** — but is **modest in level and concentrated in a few years**, so a diversifier
rather than a standalone.

1. **The IBS filter is doing real work, not noise** — it ~halves the trades and roughly doubles Sharpe
   while cutting maxDD 60% (§3a).
2. **It is NOT disguised long-NDX beta.** Long *and* short are independently profitable (~0.73 / 0.76
   Sharpe); the short side made money across a net-strongly-up decade and contributed in the 2022 bear
   (§3b). This is the single strongest robustness signal.
3. **Parameters are smooth plateaus, no knife-edge** across ORB window, RR ratio, and IBS threshold
   (§3c). The advertised "3:1 RR" is **cosmetic** — the TP fills only 5% of the time; economically it's
   a *hold-to-EOD breakout with an OR-extreme stop*, and the RR knob barely moves Sharpe for RR ≥ 2.
4. **Cost is the binding constraint, but the edge survives it** (§3d). Gross edge ≈ 10.1 idx pts/trade;
   at realistic ~1.5–2.5 idx round-trip cost the net Sharpe is ~0.76–0.94. It only dies near ~10 idx.
5. **Profit is regime-concentrated** (§3e): ~57% of all return came from 2 of 8.6 years; 2 losing years.
6. **Phase-1 event-driven fills confirm Phase-0** with per-trade PnL corr **1.0** — the vectorized
   numbers are not a shortcut artifact (§4).

---

## 3. Robustness detail

Base config: ORB 30 min / RR 3 / IBS thr 0.5 / both sides, frictionless unless stated.

### 3a. The IBS filter ablation (`outputs/cost_sensitivity.csv` is the with-cost waterfall)
| config | trades | win% | PF | Sharpe | maxDD% |
|---|---|---|---|---|---|
| ORB-only (no IBS) | 2086 | 47.0 | 1.11 | **0.47** | 36.1 |
| **ORB + IBS** | 1271 | 49.3 | 1.27 | **1.06** | 13.5 |

### 3b. Long/short — not pure beta (each side run independently)
| side | trades | win% | PF | Sharpe |
|---|---|---|---|---|
| both | 1271 | 49.3 | 1.27 | 1.06 |
| long-only | 572 | 54.2 | 1.31 | **0.73** |
| short-only | 699 | 45.2 | 1.24 | **0.76** |

### 3c. Parameter plateaus (frictionless Sharpe)
- **ORB window** 15 / 30 / 45 / 60 / 90 min → 0.91 / 1.06 / 0.99 / 1.19 / 1.23 (longer is *better*; more EOD exits, higher win%).
- **RR ratio** 1 / 2 / 3 / 4 / 6 / 10 → 0.71 / 1.01 / 1.06 / 1.06 / 1.09 / 1.08 — **flat for RR ≥ 2** (TP is vestigial).
- **IBS threshold** 0.35 / 0.4 / 0.5 / 0.6 / 0.65 → 0.96 / 0.95 / 1.06 / 0.95 / 1.00 — smooth, peak at the natural midpoint 0.5.

### 3d. Cost sensitivity — the binding constraint
Per-trade **gross edge ≈ 10.1 idx pts** on ~99-pt risk → breakeven round-trip cost ≈ 10 idx pts.
Recorded spread is tight (median **0.7 idx pts**); entries are post-10:00 ET (not the open-minute
spike), so the realistic band incl. breakout/stop slippage is ~1.5–3 idx pts round-trip.

| round-trip cost (idx pts) | Sharpe | ann% | PF | maxDD% |
|---|---|---|---|---|
| 0.0 frictionless | 1.06 | 10.8 | 1.27 | 13.5 |
| 0.6 recorded spread | 0.99 | 10.1 | 1.25 | 14.4 |
| 1.6 rec + 0.5 slip/side | 0.87 | 8.9 | 1.22 | 17.5 |
| 2.5 (1.5 sprd + 0.5 slip) | 0.76 | 7.7 | 1.20 | 21.3 |
| 4.0 (2.0 sprd + 1.0 slip) | 0.58 | 5.9 | 1.16 | 25.9 |
| 6.0 (3.0 sprd + 1.5 slip) | 0.34 | 3.4 | 1.10 | 32.1 |
| 10.0 brutal | −0.15 | −1.5 | 1.00 | 47.0 |

### 3e. Stability over time & concentration (`outputs/yearly.csv`)
t-stat on per-trade R = **3.52** (N = 1271) — naively significant, but returns are **regime-clustered**.

| year | trades | win% | ret% | Sharpe | long ret% | short ret% |
|---|---|---|---|---|---|---|
| 2018 | 141 | 36.2 | **−11.0** | −1.16 | −13.8 | +2.7 |
| 2019 | 144 | 46.5 | +6.2 | 0.92 | +4.4 | +1.8 |
| 2020 | 156 | 55.8 | **+29.4** | 2.17 | +19.0 | +10.5 |
| 2021 | 140 | 54.3 | +14.1 | 1.71 | +5.0 | +9.1 |
| 2022 | 149 | 50.3 | +18.8 | 1.24 | +9.0 | +9.8 |
| 2023 | 159 | 51.6 | +14.0 | 1.67 | +8.9 | +5.1 |
| 2024 | 156 | 42.9 | **−1.6** | −0.21 | −1.0 | −0.6 |
| 2025 | 164 | 54.9 | **+23.0** | 2.13 | +7.6 | +15.4 |
| 2026* | 62 | 50.0 | −0.0 | −0.01 | +5.6 | −5.6 |

Positive in **6 of 8** full years; 2 losers (2018, 2024). ~**57% of all profit came from just 2 years**
(2020 +29%, 2025 +23%) — excluding them, total return drops 93% → 40% (~6%/yr frictionless → ~3–4%/yr
net). *2026 partial.

---

## 4. Phase 1 — event-driven Nautilus lane (DONE)

`nautilus_lane.py` runs the strategy in a NautilusTrader `BacktestEngine` over the full 2018–2026
history: M1 bars + synth bid/ask QuoteTicks from the recorded `spread` (point 0.1), a slippage
`FillModel`, native bracket (MARKET entry / STOP_MARKET SL / LIMIT TP). Reuses
`data_platform.nautilus.{ingest,instruments}`; touches no core package.

**Reconciliation vs the Phase-0 POC (the point of the exercise):** 1271 POC trades / 1273 lane trades,
1271 matched days, **same direction 100%**, **same exit reason 100%**, **per-trade PnL corr 1.0**,
daily-return corr 0.9937 → the lane is an exact replica.

**Net metrics (Nautilus lane) vs Phase-0:**
| lane | NL Sharpe | POC Sharpe | NL ann% | NL maxDD% | rt cost (idx) |
|---|---|---|---|---|---|
| frictionless | **1.018** | 1.06 | 10.4 | 13.4 | 0.0 |
| recorded spread | **0.935** | 0.99 | 9.6 | 15.6 | 0.7 |
| + 0.5 idx-pt/side slip | **0.815** | 0.76–0.87 | 8.3 | 18.7 | 1.7 |

Net Sharpe lands ~**0.82–0.94**, in the expected band; the lane is marginally more conservative
(trigger-bar-close entry + taker-both-legs spread). **Conclusion holds under true fills, no surprise.**

---

## 5. Robustness scorecard
| dimension | result |
|---|---|
| Filter adds value | ✅ Sharpe 0.47 → 1.06, maxDD 36% → 13.5% |
| Long/short symmetry (not beta) | ✅ both ~0.75 Sharpe independently |
| Parameter plateau (window / RR / IBS) | ✅ smooth, no knife-edge |
| Works in bear (2022) & high-vol (2020) | ✅ both sides positive |
| Lookahead-clean | ✅ next-bar-open, SL-first, double-derived PnL; event-driven recompute corr 1.0 |
| Survives realistic cost | ✅/⚠️ yes at ~1.5–2.5 idx (Sh 0.76–0.94); dies by ~10 idx |
| Subperiod consistency | ⚠️ 2 losing years; ~57% profit in 2 years |
| Standalone net Sharpe | ⚠️ ~0.8 — modest, single-instrument single-session |
| VWAP/TP refinement adds value | ❌ TP is vestigial (5% fill); drop it |

---

## 6. Threats to validity / caveats
- **Regime concentration is the real weakness.** A ~0.8 net Sharpe over 8.6 years with 57% of profit in
  2 years and 2 losing years is a *thin, lumpy* edge. The point estimate is significant but the
  effective sample is far below N=1271 (returns cluster by vol regime); a 2-year drawdown is plausible.
- **The recorded `spread` is a quiet-period snapshot**, not the price the breakout *actually* pays at
  the entry minute. The Phase-1 lane charges it as a deterministic ledger debit (see below), so the
  cost band is a faithful proxy but not a tick-level measurement of open-window slippage.
- **Phase-1 fill mechanics (documented in `nautilus_lane.py` header):** (1) Nautilus 1.227 forbids
  next-bar-open fills as look-ahead, so the lane enters at the **trigger-bar close** vs the POC's
  next-bar open — verified economically identical (daily corr 1.0, identical dir/exit, 0.3-idx-pt
  median entry gap). (2) With **bar** data the matching engine fills MARKET orders at the bar's OHLC
  point; synth QuoteTicks don't take precedence and `FillModel.prob_slippage` moves a fill only one
  price increment (0.01, negligible), so the recorded half-spread is applied as a deterministic debit
  (= the POC's `use_recorded_spread`). The fills stay event-driven; only the spread cost is
  deterministic. **Verified: ZERO of 1271 trades have an exit bar straddling both SL and TP**, so the
  POC's SL-first assumption never binds. Net: the lane is an *independent validation of the trade/exit
  mechanics*, not a richer cost model.
- **DST edge.** ET = broker − 7h holds year-round *except* the ~2–3 weeks/yr the EU and US DST
  transitions misalign; on those days the session opens ~1h off the US cash open. Minor, not modelled.
- **Single instrument, single session** — no cross-sectional diversification within the strategy itself.

---

## 7. Recommendation
1. **Do not promote as a standalone** — ~0.8 net Sharpe, single-instrument, regime-concentrated is not
   enough on its own.
2. **The one credible path is a low-correlation intraday-equity sleeve.** Its daily return stream is
   plausibly orthogonal to the daily trend/MR vault; if a correlation study confirms low corr *and* net
   Sharpe holds ~0.8, it earns a small weight. **Note there is no standard Phase-2 gate path** — the
   portfolio-addition gate is built for daily node/`StrategySpec` features and cannot host an intraday
   ORB, so this requires a manual orthogonality check (correlate the daily stream vs the vault sleeves).
3. **If pursued, simplify:** drop the vestigial TP (RR sweep says no loss), use a 30–60 min OR — fewer
   parameters, same edge.
4. **For a tighter cost number:** a tick-level check of the actual fill spread at the 10:00–14:00 ET
   entry minutes (vs the quiet-period recorded snapshot) would resolve the residual cost uncertainty.

---

## 8. Artifacts (`outputs/`)
- `equity_curve.png` — frictionless vs net (~1.6 idx) vs ORB-only cumulative return.
- `yearly.csv` — per-year trades/win/return/Sharpe with long & short split.
- `cost_sensitivity.csv` — the Phase-0 round-trip-cost → Sharpe waterfall.
- `nautilus_lane_metrics.csv` / `nautilus_reconciliation.csv` — Phase-1 net metrics + POC↔lane reconciliation.
- `nautilus_trades_{frictionless,realistic}.parquet` — Phase-1 per-trade detail.
- Reproduce: `.\.venv\Scripts\python.exe research\experiments\orb_ibs_nas100\poc.py` (Phase 0) ·
  `… nautilus_lane.py [--start 2023 --end 2024]` (Phase 1).

## Related studies
Part of the repo's **intraday US-equity** family, all sharing the same intraday-cost-gate theme:
- `research/experiments/concretum_intraday_momentum/` — Noise-Area momentum (friction-fragile,
  net-negative OOS standalone) + the 5-min fast-alpha execution overlay (works). That memo already
  cites *this* ORB study as the "57% profit-concentration" comparison. **Contrast:** ORB+IBS *survives*
  cost where Concretum's standalone Part 1 did not, but ORB is the more concentrated of the two.
- The 5-min **entry overlay** from Concretum Part 2 (wait for a pullback before taking a breakout) is a
  general execution-timing rule that could be bolted onto this ORB to buy a few tenths of net Sharpe of
  cost headroom — a natural follow-up if this sleeve is pursued.
- Infra: `docs/library/Strategy_research/`, the realistic lanes in `research/validation/` +
  `scripts/run_sandbox_sim.py`, and the FX-intraday microstructure work (`project_fx_timezone_premium`).
