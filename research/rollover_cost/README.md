# Rollover-cost execution study

> ## ⭐ BOTTOM LINE (read this first — decision-grade, on the ACTUAL vault)
>
> The CFD prop portfolio (ES/NQ/GC → Darwinex `SP500 NDX XAUUSD`, all **daily** models so
> *every* trade is a rollover trade) pays an overnight **swap** on every position held through
> the 17:00-ET (= 00:00-broker) financing rollover. The strategy is **low vol (Sharpe 1.01 @
> 3.7 %/yr)**, so financing is huge in *relative* terms. A **swap-avoidance overlay executed
> with market orders** is the answer:
>
> | Scenario | ann return | **Sharpe** | |
> |---|---:|---:|---|
> | GROSS (frictionless backtest) | 3.72 % | **1.01** | the alpha |
> | HOLD (pay the swap) | 1.20 % | **0.33** | swap costs −0.68 Sharpe |
> | OVERLAY market — swap avoidance + optimised execution | 2.93 % | **0.80** | the execution-cost view |
> | **OVERLAY market — FULL (incl. drift-timing)** ✅ | 2.71 % | **0.74** | the realised live edge |
> | OVERLAY market — + per-night spread gate | 2.83 % | **0.77** | the last free +0.03 |
>
> - **The swap is the whole problem: −0.68 Sharpe (1.01 → 0.33)** on real positions.
> - **A MARKET-order overlay recovers it: 0.33 → 0.74** (+0.41), pure **swap avoidance**.
> - **Limit "market-making" does NOT help — tested SIX ways, it loses on both legs** (see
>   [Why limits lose](#why-limit-market-making-loses-six-ways)). The reopen is uniquely adverse
>   for passive orders: a transient spike spread, thin liquidity (~30–60 % passive fill), and a
>   post-reopen up-drift so the nights you miss are your **best** long nights.
> - **The execution work paid off via *timing*, not limits:** exit-market at **T-15** (dodge the
>   down-into-close drift), enter-market at the reopen (gold waits ~5 min for the spike). That
>   lifted the overlay from ~0.62 (naive) to **0.80**.
> - **FTMO swap is ~1.4× worse** → HOLD Sharpe **0.07** (near-dead); the overlay rescues it to
>   **0.70** and *equalises* the venues. **Stay on Darwinex; on FTMO the overlay is mandatory.**
> - **Execution is essentially maxed.** The remaining ~0.24 Sharpe gap to frictionless is a
>   **structural** spread floor (~0.16–0.18) + drift-timing (~0.06) — not an inefficiency.

The engine policy that produces this is encoded in
[`execution/rollover_overlay.py`](../../execution/rollover_overlay.py)
(market orders, T-15 exit, immediate reopen entry, per-night `swap > round-trip-spread` gate,
carry-aware, delta-based). Everything below is the evidence.

---

## The settled execution policy

**The one universal rule (no per-ticker config):** *overlay a leg — flatten before the rollover,
re-enter after — only when the round-trip half-spread you'd pay is smaller than the swap you'd
save that night. Otherwise hold the leg through and eat the swap. Hold positive-carry legs to
earn the swap.* On triple-swap nights the swap is ×3, so the threshold rises and more legs
overlay — automatically.

1. **Exit — one MARKET order at T-15 min** before the 00:00-broker rollover, for each
   negative-carry leg the gate selects. Early on purpose: the price drifts **down** into the
   close *and* the spread widens, so T-15 pays a tighter spread **and** sells before the dip.
   Waiting to T-1 costs **+1.0–1.4 bps**. A passive sell-limit is *worse* than market (it waits
   for an up-tick that doesn't come into a falling book).
2. **Dead zone** 00:00–01:00 broker: market closed, no action.
3. **Entry — one MARKET order at the reopen** (≈ 01:00 broker), sized **delta = target −
   current** (position-aware; a held leg has delta 0 and is never doubled). *Immediate* for
   tight-reopen names (indices); for wide-reopen names (**gold**) wait ~5 min for the spike to
   decay (measured **+0.70 bps**). **Never rest a limit; never skip a miss.**
4. **Carry-aware:** only overlay negative-carry legs (longs on these CFDs). Hold shorts through
   to collect the positive swap.
5. **Per-night gate (the last +0.03 Sharpe):** the swap is deterministic (rate × triple-day
   calendar); the spread varies nightly — so skip the overlay on a leg whose live round-trip
   spread blows out above its swap that night.

This took the actual vault book from **0.33 → 0.80** on Darwinex (0.74 after the unavoidable
drift-timing, 0.77 with the per-night gate), and **0.07 → 0.70** on FTMO.

---

## Full-edge attribution (both brokers, real positions)

`integrated_backtest.py` / the full-edge ladder drives the **actual vault position path**
(`tests/parity/.../portfolio_test_default__test__positions.parquet`, CL/ES/GC/NQ 2023–2026) and
charges, per leg per night: the **real scraped rollover spread** (exit @ T-15 + reopen @ +5 min),
the **real nightly swap** (contractual POINTS rate at the real daily price), and the
**drift-timing** (the price move over the flat [T-15 → reopen] window × the position).

| component | **Darwinex** | **FTMO** |
|---|---:|---:|
| swap avoided | +2.52 %/yr | +3.47 %/yr |
| market execution cost (exit T-15 + enter post-decay) | −0.79 %/yr | −0.79 %/yr |
| drift-timing | −0.22 %/yr | −0.36 %/yr |
| **GROSS / HOLD / OV-market / OV-full** | 1.01 / 0.33 / 0.80 / **0.74** | 1.01 / 0.07 / 0.80 / **0.70** |
| spread cost vs frictionless (GROSS − OV-market) | **+0.22 Sharpe** | +0.21 Sharpe |
| swap cost if you HOLD (GROSS − HOLD) | +0.69 Sharpe | +0.95 Sharpe |

**Reading it:**
- **The spread cost is ~0.2 Sharpe and is the same on both brokers** — Darwinex's wide *gold
  reopen spike* offsets FTMO's wider *index* spreads (FTMO reopens ~5 min later, so its reopen is
  calmer). Spread is **not** what separates the venues.
- **The swap is what separates them** (Darwinex 0.69 vs FTMO 0.95 Sharpe). The overlay removes it,
  so **OV-market is 0.80 on both** — the overlay *equalises* the brokers.
- **Drift-timing is a small drag, not a bonus** (−0.22 to −0.36 %/yr). Earlier I speculated the
  overlay would *capture* the down-into-close drift by being flat — it doesn't: the reopen gaps
  back up *above* T-15, so sitting out the flat window misses a small net up-move. Unavoidable
  (the dead-zone is closed; you can't trade through it).

---

## Why limit market-making loses (six ways)

Passive "market-making" at the rollover was tested six independent ways; **every *measured*
method says market orders win the re-entry.** The reason is structural, not a modelling gap.

| method | result |
|---|---|
| parametric `capture_frac` model | "limit +1.1–1.4" — **withdrawn**, it *assumed* the capture |
| static tick-replay (at the touch) | capture **−0.04 to −2.67 bps** (loses) |
| Nautilus matching engine (NDX) | maker +0.60 bps/fill, **total return 0.125 < 0.149** (loses) |
| entry-price + chase replay (full year) | capture **negative** on all legs |
| optimistic repricing model | +0.9 to +3.3 — but assumes fills the book doesn't give you |
| **Carver passive→aggressive + threshold sweep** | **−0.15 to −2.8 bps at every patience setting** |

**The mechanism, quantified.** On the actual positions, the held next-day P&L on the nights the
entry limit **misses is +1.4 bps vs +0.1 bps on the nights it fills** — i.e. the limit captures
spread on the flat nights that don't matter and **misses your best long nights** (the up-drift
nights a market order rides). Two corollaries, both measured:
- **Skip-on-miss is the *worst* policy** (Sharpe 0.23 vs market 0.37) — skipping forgoes the
  +1.4 bps winners. Chasing beats skipping (you stay in your winners).
- **Placement doesn't fix it** — moving the limit from the bid to the mid to through-the-bid is
  negative at every level; the up-drift is the binding constraint, and no passive placement beats
  being in immediately.

> Why Carver's celebrated passive→aggressive algo (80 % cost reduction) doesn't transfer: it's
> built for **normal, liquid** markets (tight standing spread, ~2/3 passive fill, spread *is* the
> cost). The rollover **reopen** breaks all three assumptions — the spread is a transient *spike*,
> liquidity is thin (~30–60 % passive fill), and there's a directional up-drift. Right tool, wrong
> moment.

**Use market orders. The edge is swap avoidance, not spread capture.** (Caveat: the up-drift is
partly *regime* — measured present every quarter 2025–26 but it varies; a flat/down market would
hurt the limit less. The robust call is market orders, and live limit fills would be *worse* than
the sim, which models no queue position.)

---

## Is it maxed out?

Essentially yes. The big cost (swap) is fully neutralised; the small one (spread) is mostly
structural.

| Darwinex scenario | ann return | Sharpe |
|---|---:|---:|
| OVERLAY-all | 2.71 % | 0.74 |
| **per-night gate (realistic: overlay iff swap > spread)** | 2.83 % | **0.77** |
| per-night gate (hindsight ceiling) | 3.23 % | 0.88 |

- **Per-leg gating gives nothing** — every leg already clears the bar (even gold: overlay 0.80 <
  swap 1.17, once the reopen-wait optimisation is applied). The old "don't overlay gold" was from
  the *spike* spread; waiting 5 min flips it.
- **The per-night spread gate gives +0.03 Sharpe** (0.74 → 0.77) and is genuinely implementable
  (swap known, spread estimable pre-trade). This is the last free lever.
- **The 0.77 → 0.88 ceiling is a mirage** — it comes entirely from predicting the night's
  drift-timing, i.e. forecasting the overnight move. That's *alpha*, not execution.
- **The remaining ~0.24 Sharpe gap to frictionless is structural:** a spread floor (~0.16–0.18,
  the irreducible cost of round-tripping a low-vol book nightly) + drift-timing (~0.06). Closing
  more requires **lower turnover** (a strategy/sizing change) or **overnight alpha** — neither is
  execution.

---

## How the swap is calculated (and from where)

- **Source:** live `mt5.symbol_info(symbol)` — `swap_long`, `swap_short`, `swap_mode`,
  `swap_rollover3days` (Darwinex probed 2026-06-05 via `research/rollover_cost/spec_probe.py`; FTMO via
  `build_events_ftmo.py`'s live `ftmo_spec`). All legs use **`swap_mode = POINTS`**: a fixed
  points charge per 1.0 lot at each rollover.
- **Snapshot (points/lot/day, long / short):** Darwinex SP500 −10.85/+4.59 · NDX −45.51/+18.79 ·
  XAUUSD −63.6/+38.9. FTMO US500 −154.67/−3.09 · US100 −619.48/−12.41 · XAUUSD −86.40/−15.30.
- **Conversion to bps** (`config.py: SymbolSpec.swap_bps`): `swap_points × point / mid × 1e4`,
  recomputed **per event with that event's own mid** (a fixed point charge is a larger bps cost
  when price is lower, so the 2023–24 cheaper-gold years carry a bigger drag).
- **Triple-swap nights** (`swap_rollover3days`: Fri for indices, Wed for metals) apply ×3.
- **Sign:** `swap_long` < 0 on every leg → **longs pay, shorts earn**. The vault is ~93–100 % long
  ES/NQ/GC, so it pays nearly every night.
- **Not modelled:** commission (these CFDs are spread-only — verify your tier) and FX (all settle
  USD = account ccy).

> **Limitation:** the swap is a **single snapshot** applied across the year (price drift handled,
> *point* drift not). The conclusion is robust to ±30 % swap variation; pull a fresh rate before
> going live. The FTMO/Darwinex ratio (~1.4×) is from live probes of both terminals.

---

## Position-state awareness (a hard correctness requirement)

The overlay is **stateful and delta-based** — it must never assume it starts flat at re-entry.
Under carry-aware we deliberately *hold* some legs through the rollover, so those legs are **not
flat** afterwards. The universal rule (`reentry_delta` in the decision core):

> **At re-entry, order `delta = target − current_actual_position` (read live from MT5) — never
> `target − 0`.** Re-entering a full target on a still-held leg doubles the position.

| Leg state at reopen | current position | correct order |
|---|---|---|
| Flattened (exit filled) | 0 | full target |
| Held (positive-carry / not-overlaid) | held size | **target − held** |
| Exit missed → ate swap | yesterday's size | target − yesterday's |

Implementation: the post-rollover step **is** the existing delta-based rebalancer
(`execution/mt5_rebalancer.py`), which already trades `target − current` from live positions; the
overlay only adds the **pre-rollover flatten of the gate-selected negative-carry legs**, and
persists per-symbol nightly state so the re-entry knows the true starting position.

---

## Drift around the rollover (the two effects that drive every decision)

Measured from the real ticks, the price path around the rollover is systematic:
- **Down into the close** (T-15 → 00:00): −0.9 to −1.1 bps on the indices → **exit early** (T-15),
  sell before the dip, and a passive sell-limit won't fill into the fall.
- **Up after the reopen** (+2 to +4 bps in the first hour, present every quarter in this 2025–26
  sample) → **enter immediately** (market) to ride it; a waiting limit misses it on the up-nights.

On **both** legs the price drifts *away* from a passive order, which is exactly why limits lose
both legs and aggressive, well-timed market orders win. The regime caveat (the up-drift could be a
bull-market artifact) is real but doesn't change the call: in a flat/down regime limits would hurt
*less*, but market orders are still the robust choice.

---

## FTMO vs Darwinex (real scraped ticks from BOTH brokers)

`build_events_ftmo.py` scrapes FTMO's own rollover ticks (US500.cash/US100.cash/XAUUSD, 1 yr);
`ftmo_compare.py` runs the same overlay on the same positions. The reopen spread is measured at
**+5 min** on both — FTMO reopens ~5 min later, so first-tick spreads would compare a decayed
spread vs Darwinex's spike (a trap caught and corrected).

| leg | broker | exit ½-spr | reopen ½-spr | reopen lag | exit fill | entry fill | **swap/night** |
|---|---|---:|---:|---:|---:|---:|---:|
| ES | Darwinex | 0.34 | 0.35 | 0m | 91% | 85% | 1.75 |
| ES | FTMO | 0.40 | 0.43 | 5m | 76% | 94% | **2.49** |
| NQ | Darwinex | 0.18 | 0.28 | 0m | 98% | 83% | 1.63 |
| NQ | FTMO | 0.35 | 0.38 | 5m | 92% | 98% | **2.21** |
| GC | Darwinex | 0.61 | 1.36 | 1m | 89% | 91% | 1.93 |
| GC | FTMO | 0.75 | 0.67 | 5m | 78% | 96% | **2.62** |

Full-edge ladder (market overlay, real positions): **Darwinex** HOLD 0.33 / OV-full **0.74**;
**FTMO** HOLD **0.07** / OV-full **0.70**.

**Verdict — Darwinex is the better venue; on FTMO the overlay is mandatory.**
1. **FTMO swap ~1.4× worse** → holding through is near-dead (0.07 vs 0.33). The overlay is the
   only thing keeping the strategy alive on FTMO.
2. **More missed exits** — FTMO exit-limit fill 76–92 % vs 89–98 % (wider exit spread + it closes
   ~10 min earlier). With market orders on the exit (our policy) this is moot; with limits it
   would matter.
3. **Once overlaid, FTMO ≈ Darwinex** (0.70 vs 0.74) — the overlay avoids the swap (the FTMO
   penalty) and FTMO's reopen is comparable (wider on indices, *tighter* on gold). The overlay
   equalises the venues; the swap is what separated them.

---

## Methodology & honest limitations

- **Real positions throughout** — the signed `position_fraction` from the vault backtest
  (ES 92.7 %, GC 100 %, NQ 98.2 % long; |frac| 0.07–0.13). CL is in the book but untraded
  (|frac| 0.002) so the execution tests run on ES/NQ/GC.
- **Spreads are real scraped bid/ask** (`events_*.parquet`, exit @ T-15, reopen @ +5 min). The
  fill model is **pessimistic** — a resting limit counts as filled only when the market trades
  *through* it (no queue-priority credit), so the limit's *measured* loss is, if anything,
  understated.
- **Execution-mechanism tests (skip/chase, limit replays) are bounded to the ~1-year tick window**
  (555 leg-nights, 2025-06 → 2026-05) — that's how far back real bid/ask exists. The **full-edge
  swap ladder uses the full 3-year `strategy_returns`** series; the two are different samples (the
  skip/chase ladder's absolute Sharpe levels are not directly comparable to the swap ladder, but
  its *ordering* — skip is worst — is robust).
- **Held P&L in the skip/chase test = CFD close-to-close × position** (the right choice for an
  *execution* question — you realise CFD returns), not the backtest's own return series.
- **FTMO** uses its own scraped ticks for spreads/fills and live-probed swaps; its swap is exact,
  its spreads are from real FTMO bid/ask.
- **Frictionless backtest assumption** — the research return is frictionless by convention; the
  sim adds the real swap + spread on top (no double-counting).
- **Scraper timezone** — the rollover windows are fetched in correct **broker time** via
  `ensure_ticks`; the legacy `rollover_tick_scraper.py` / `build_symbol_sessions.py` /
  `nautilus/ingest.py` windowing assumed real UTC and is offset by the broker-EET gap. See
  [docs/library/Data/mt5_timezones.md](../../docs/library/Data/mt5_timezones.md).

---

## Evolution of the numbers (why they changed)

Refined as each proxy was replaced by real data. Only the last row is decision-grade.

| Framing | Result | Why superseded |
|---|---|---|
| Per-leg medians, full notional | "+4–6 %/yr, limits required" | medians hid fat reopen-spread tails; per-leg not portfolio |
| Buy/hold, equal-weight 4 legs, 15 % vol | "limits +0.53 Sharpe" | a **silver** artifact — vault doesn't trade silver; wrong vol |
| Parametric `capture_frac` overlay | "limit 1.1–1.4" | **assumed** the maker capture; real fills refute it |
| **Actual returns + signed positions, real ticks, market orders** | **swap −0.68 Sharpe; overlay → 0.74 (Dwx) / 0.70 (FTMO)** | ✅ decision-grade |

Two corrections drove it: (1) the vault is **net-long ES/NQ/GC and does not trade silver**, and
(2) the strategy is **low vol (3.7 %)**, so the swap matters far more in Sharpe terms than the
absolute %/yr suggested — and every *measured* fill simulation killed the limit story.

---

## Reproduce

```powershell
# Build per-event features (fetches correct broker-time rollover windows via the tick cache)
.\.venv\Scripts\python.exe -m research.rollover_cost.build_events --days 365
.\.venv\Scripts\python.exe -m research.rollover_cost.build_events_ftmo --days 365   # FTMO (binds FTMO terminal)

# Real-fill checks + ladders
.\.venv\Scripts\python.exe -m research.rollover_cost.capture_sim          # tick-replay limit vs market
.\.venv\Scripts\python.exe -m research.rollover_cost.ftmo_compare         # both-broker per-leg econ
```

**Code:** `config.py` (specs + window geometry), `build_events.py` / `build_events_ftmo.py`
(fetch + feature-ize), `capture_sim.py` (limit-capture tick replay), `ftmo_compare.py`
(both-broker economics), `integrated_backtest.py` (full-edge ladder). The live decision core is
[`execution/rollover_overlay.py`](../../execution/rollover_overlay.py) with unit tests in
`tests/unit-tests/execution/test_rollover_overlay.py`. All timestamps are **broker time**
(rollover = 00:00); see the timezone doc above. Relates to the hybrid tick backtest and the MT5
timezone reference.

> _Verified against current code via CodeGraph on 2026-06-07._
