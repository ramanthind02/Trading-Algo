# Channel Breakout (MQL5 `ChannelBreakOutStrategy_MultiSymbol`) on intraday CFDs — CLOSED

**Date:** 2026-06-08 · **Status: closed.** Six investigation steps (§1–§13): frictionless POC →
robustness/grid → node comparison → adversarial verification → costed ensemble → TF/4th-instrument sweep →
M1 event-driven fills → bar-close re-optimisation.

**All research scripts were exploratory scratch and have been removed; this memo (+ `outputs/`) is the
record.** The Methodology & reproduction appendix (§A) carries enough detail to rebuild the engine; data is
the uncommitted M1 CFD store (`data/mt5_data/<SYM>/bars_M1/`).

> ## Executive summary
>
> 1. **The MQL5 EA *is* `nodes/breakout/donchian/robust_trend_breakout.py`** — stripped of its %-risk
>    sizing it's a long-only Donchian-breakout + ATR-chandelier trend follower; P&L corr **0.87–0.91** to
>    the node. **Do not build a new node.** (§2, §4)
> 2. **Frictionless, the edge is real on gold / nasdaq / usdjpy and absent on silver / crude** (the latter
>    flip negative out-of-sample). Robust *plateau* on the winners (90–100 % of a 40-cell grid positive
>    both splits), but **val is regime-inflated** (2023–24 trend bull) — trust the **train** number. (§2–§3a)
> 3. **As a costed, diversified ensemble it survives** — and the decisive realism was **execution, not
>    spread/swap**. The EA's **intrabar trailing-stop order whipsaws** (fills on M1 wicks a close-based rule
>    held through → ~7× turnover), cutting net train Sharpe to ~0.88. **Switching to a bar-close exit
>    recovers it.** (§5, §12–§13)
> 4. **FINAL deliverable — the 3-instrument winners basket (gold+nasdaq+usdjpy), traded as a *bar-close*
>    breakout** (enter on a close above the **L20** Donchian high, exit on a close below the
>    **ATR(14)×4** chandelier trail), **1h+2h**, vol-scaled (Carver EWSD, F=τ/σ cap 2.0) + IDM, nets
>    **≈ 1.25 train / 1.87 val, net of realistic M1-lane fills** (spread×2 + 3 % swap). Robust to 3×
>    spread + 6 % swap. (§13)
> 5. **Two design rules that matter:** (a) **use a bar-close exit, never the EA's intrabar stop** (nor the
>    node's `low < chandelier_stop`, which wicks identically) — worth ~0.4 train Sharpe; (b) **size at
>    entry, do not band-rebalance the vol target intraday** (that fabricates ~10× phantom turnover). (§10, §13)
> 6. **It does not scale and has no production rail.** Adding any 4th CFD (FX majors negative-edge, other
>    indices redundant with nasdaq, natgas wide-spread) **hurts** — the edge is concentrated in 3 markets
>    that trended hard 2018–24 (§11). And the repo's `StrategySpec`→vault→`validate_candidate.py` rail is
>    **daily-only** — an intraday CFD breakout has no slot; productionising needs intraday rails or a
>    standalone system (§12). That infra call is the only open item.
> 7. **It's a long-biased risk-premium harvester, not symmetric trend-following.** The **short side has no
>    edge on any asset** tested — down-moves in this era were sharp/reverting (not slow/persistent), and
>    shorting fights the upward drift — so **long/short is strictly worse than long-only** and doesn't
>    rescue silver/crude (Appendix B). The edge is the *upward* drift+momentum of a few specific markets.

---

## 1. What was tested

The user's MQL5 EA, **stripped of its `RiskPercent` / fixed-currency lot sizing** (we use our own
vol-scaling instead), reduces to a **long-only signed signal**:

- **Entry (flat):** `close > Highest(High, Length)` of the prior `Length` bars (Donchian breakout).
- **Exit (long):** ATR-chandelier trailing stop `close ≤ running max(close − mult·ATR(14))`
  **OR** channel break `close ≤ Lowest(Low, Length)` (when `UseChannelExit`).

Run **frictionlessly** (no spread/commission/swap — user's explicit request: prove the edge first),
**2018→2024 on our CFD M1 feed** (`data/mt5_data/<SYM>/bars_M1`), resampled to **30 min / 1 h / 2 h**:

| instrument | CFD symbol | node Ticker |
|---|---|---|
| gold | XAUUSD | GC |
| silver | XAGUSD | SI |
| nasdaq | NDX | NQ |
| usdjpy | USDJPY | JY |
| crude (WTI) | XTIUSD | CL |

**Splits:** train `2018-01-01 … 2022-12-31`, val `2023-01-01 … 2024-12-31`. **2025+ never loaded** (locked).

### Vol-scaling = our infra (not the EA's % risk)
- **σ:** Carver EWSD replicated from `nodes/volatility/ewsd/ewsd.py` — `0.7·EWMA(λ=0.06061)·σ_short + 0.3·expanding-std(≤2520 bars)`, on simple bar returns, annualised by `√(bars_per_year)` measured empirically per (symbol, TF) (so the cap binds like the daily node's ×16 does).
- **Position:** `F = clip(target_vol/σ_annual, 0, 2.0)`, `target_vol = 0.15` (DiversifiedEnsemble defaults); `position = signal · F`.
- **P&L:** `pos.shift(1) · r` (position known at close of *t−1*, return earned over bar *t*), re-derived a second way with an explicit held-position loop and `assert np.allclose` (lookahead guard).

---

## 2. Headline verdict

1. **It is the same strategy as `nodes/breakout/donchian/robust_trend_breakout.py`.** Configuring the
   real node to mimic the EA (Donchian `lookback=5`, regime EMA ≈ off, `atr_mult=2`) gives a
   **P&L correlation of 0.87–0.91 to the channel breakout across *every* instrument × timeframe**
   (signal corr 0.76–0.82). The EA contributes nothing the node lacks; its differences (a
   channel-break exit, intrabar stop fills) are second-order. **→ Do not build a new node.**

2. **Frictionless, it is effective and *robust* on trending instruments, and fails on the rest.**
   Over a 40-cell param grid (Length ∈ {5,10,20,40,55} × atr_mult ∈ {1.5,2,3,4} × channel_exit ∈ {on,off}),
   pooled across the 3 timeframes:

   | instrument | grid train-Sharpe median | grid val-Sharpe median | % grid train>0 | % grid val>0 | both>0 |
   |---|---|---|---|---|---|
   | **nasdaq** | +0.68 | +1.80 | 100% | 100% | **100%** |
   | **gold** | +0.54 | +1.43 | 98% | 100% | **98%** |
   | **usdjpy** | +0.57 | +0.74 | 98% | 92% | **91%** |
   | silver | −0.14 | +0.15 | 31% | 70% | 22% |
   | crude | +0.30 | −0.21 | 87% | **27%** | 23% |

   gold/nasdaq/usdjpy show a genuine **plateau** (almost the whole grid is positive in both windows,
   and the val-Sharpe surface is smooth — see `outputs/heatmap_val_sharpe_1h.png`), not a lucky cell.
   **silver** is barely positive even in-sample; **crude** is positive in train but **flips negative
   out-of-sample** (overfit / regime break — train-selected crude configs go to −0.63/−0.10 in val).

3. **The high val Sharpes overstate the edge — regime, not skill.** 2023–24 was an exceptionally
   trend-friendly, long-biased window (gold bull, AI/tech bull, USDJPY/yen-carry uptrend), which a
   **long-only** breakout is built to exploit. The **train** Sharpes are the honest forward estimate;
   the val Sharpes are a tailwind. Stripped of vol-scaling leverage, the implementation-invariant
   **binary** edge is *modest* even on the winners (gold-1h train **0.34**/val 0.83; nasdaq-1h
   0.14/1.41) — see §3a.

4. **Not yet costed — and that is the real gate.** This is pure trend-following on intraday bars; the
   user is right that costs will chunk it. See §5.

5. **Our 0.15 *daily* vol target saturates the F=2.0 cap on low-vol intraday names** (§3a). At intraday
   cadence σ_annual runs well below 0.15 for gold (~0.087) and especially usdjpy (~0.05), so F levers to
   the cap on **38% / 75%** of bars — the system runs near fixed 2× leverage there, and the vol-scaled
   *absolute* Sharpe becomes sensitive to the exact EWSD estimator (±~0.2). An intraday deployment would
   need to re-target vol. The **binary** Sharpe is the implementation-invariant anchor.

---

## 3. Robustness detail (per symbol × timeframe)

`default` = the EA's shipped params (L5 / atr_mult 2.0 / channel-exit ON). `best-train→val` = pick params
by **train** Sharpe, then read their **val** Sharpe (honest out-of-sample selection). Full table:
`outputs/summary_by_symbol_tf.csv`.

| sym | TF | default tr/val | grid %train>0 / %val>0 | best-train cfg | best-train tr→val | trades/5y (train) |
|---|---|---|---|---|---|---|
| gold | 30m | +0.46/+1.89 | 0.95 / 1.00 | L55/m4/ce-off | +0.76→+2.14 | 353 |
| gold | 1h | +0.42/+1.07 | 1.00 / 1.00 | L40/m3/ce-off | +0.91→+1.17 | 251 |
| gold | 2h | +0.53/+1.92 | 1.00 / 1.00 | L20/m4/ce-off | +0.96→+1.49 | 150 |
| nasdaq | 30m | +0.44/+2.27 | 1.00 / 1.00 | L5/m4/ce-off | +0.87→+2.75 | 848 |
| nasdaq | 1h | +0.07/+1.78 | 1.00 / 1.00 | L40/m3/ce-on | +1.28→+2.13 | 278 |
| nasdaq | 2h | +0.77/+1.13 | 1.00 / 1.00 | L20/m4/ce-off | +1.09→+1.54 | 162 |
| usdjpy | 30m | +0.54/+0.06 | 0.95 / 0.90 | L40/m4/ce-on | +1.01→+0.95 | 439 |
| usdjpy | 1h | +0.26/+0.73 | 1.00 / 0.85 | L20/m4/ce-on | +0.97→+0.47 | 341 |
| usdjpy | 2h | +0.10/+1.10 | 0.98 / 1.00 | L20/m4/ce-off | +0.92→+0.93 | 144 |
| silver | 30m | −1.11/+0.16 | 0.15 / 0.92 | L55/m3/ce-on | +0.25→+0.50 | 409 |
| silver | 1h | −0.62/−0.07 | 0.10 / 0.40 | L55/m2/ce-on | +0.04→**−0.74** | 266 |
| silver | 2h | +0.38/+0.15 | 0.68 / 0.78 | L10/m2/ce-on | +0.50→+0.29 | 339 |
| crude | 30m | −0.28/−0.78 | 0.70 / 0.22 | L40/m1.5/ce-on | +0.70→**−0.63** | 790 |
| crude | 1h | +0.15/−0.34 | 0.90 / 0.08 | L10/m3/ce-on | +0.75→**−0.10** | 653 |
| crude | 2h | +0.67/+0.11 | 1.00 / 0.50 | L5/m3/ce-on | +0.68→+0.11 | 586 |

Robustness reads:
- **Timeframe is not where the edge lives.** The *sign* of the edge is consistent across 30m/1h/2h for
  each instrument (gold good at all three, crude bad at all three). The TF mostly changes **turnover**:
  30m trades ~2–4× more than 2h for the same config. So the edge isn't a fragile single-TF artifact —
  but the cost-survivable TF is the slowest (2h).
- **The EA's default params are the *noisy* corner.** Every train-selected winner is **longer-lookback
  (L20–55), wider-stop (m3–4), and frequently channel-exit OFF** — i.e. the data prefers slower, fewer
  trades than the L5/m2/channel-on default. The default's value is mostly that it trades a lot.
- The channel-break exit (`UseChannelExit`) helps a little on the choppy names (silver) and hurts a
  little on the clean trenders (gold/nasdaq prefer it OFF — let the ATR trail run the winners).

### 3a. Implementation-invariant binary core + vol-target cap saturation (`outputs/binary_core.csv`)

`bin_*` = default signal P&L with **no** vol-scaling (the implementation-invariant edge). `vs_*` =
vol-scaled. `pct_capped` = fraction of bars where `F` hits the 2.0 cap; `sigma_ann_med` = median EWSD
annual vol.

| sym | TF | bin train/val | vs train/val | pct_capped | σ_ann med |
|---|---|---|---|---|---|
| gold | 30m | 0.23/1.16 | 0.46/1.89 | 0.39 | 0.086 |
| gold | 1h | **0.34/0.83** | 0.42/1.07 | 0.38 | 0.087 |
| gold | 2h | 0.49/1.35 | 0.53/1.92 | 0.38 | 0.087 |
| nasdaq | 30m | 0.14/1.59 | 0.44/2.27 | 0.17 | 0.125 |
| nasdaq | 1h | 0.14/1.41 | 0.07/1.78 | 0.13 | 0.130 |
| nasdaq | 2h | 0.36/1.17 | 0.77/1.13 | 0.10 | 0.134 |
| usdjpy | 30m | 0.48/0.11 | 0.54/0.06 | **0.75** | 0.050 |
| usdjpy | 1h | 0.31/0.86 | 0.26/0.73 | **0.75** | 0.050 |
| usdjpy | 2h | 0.15/0.99 | 0.10/1.10 | **0.75** | 0.051 |
| silver | 30m | −0.74/−0.39 | −1.11/0.16 | 0.05 | 0.170 |
| silver | 1h | −0.37/0.20 | −0.62/−0.07 | 0.04 | 0.173 |
| silver | 2h | 0.64/0.12 | 0.38/0.15 | 0.04 | 0.172 |
| crude | 30m | −0.33/−1.03 | −0.28/−0.78 | **0.00** | 0.221 |
| crude | 1h | 0.17/−0.42 | 0.15/−0.34 | **0.00** | 0.227 |
| crude | 2h | 0.37/−0.12 | 0.67/0.11 | **0.00** | 0.242 |

- **The binary core tells the same story** (gold/nasdaq robust-positive, usdjpy modest, silver/crude
  fail OOS), so the verdict is **not** a vol-scaling artifact. But it also shows the *honest* edge is
  modest in-sample (binary train 0.1–0.5 on the winners); the impressive vs-val numbers are
  regime tailwind + leverage.
- **Cap saturation is the vol-scaling caveat.** The production 0.15 *daily* target, applied to low-vol
  intraday bars, drives `F` to the 2.0 cap on 38% (gold) / 75% (usdjpy) of bars — there the strategy is
  effectively fixed-2×-leverage and the vs-Sharpe is EWSD-estimator-sensitive (the recompute gap, §8).
  For higher-vol names (crude σ≈0.22) the cap never binds and vol-scaling behaves as designed.

---

## 4. Channel breakout vs `robust_trend_breakout` (the comparison the user asked for)

`outputs/summary_node_vs_cb.csv` (train/val Sharpe). Three node configs were streamed through the
**same** intraday candles + **same** vol-scaling:

- `node_default` = shipped (L55 / EMA-100 regime / atr_mult 3.5)
- `node_L20` = L20 / EMA-100 / 3.0
- `node_mimic` = L5 / EMA≈off / 2.0 (built to imitate the EA)

| sym·TF (sample) | cb default | node_default | node_L20 | node_mimic | sig corr | **pnl corr** |
|---|---|---|---|---|---|---|
| gold 1h | +0.42/+1.07 | +0.60/+1.13 | **+0.72/+1.49** | +0.18/+1.04 | 0.79 | **0.90** |
| nasdaq 1h | +0.07/+1.78 | +0.93/+2.07 | +0.86/+2.15 | +0.34/+1.29 | 0.80 | **0.88** |
| usdjpy 30m | +0.54/+0.06 | +0.69/+0.56 | **+0.93/+1.07** | +0.64/−0.00 | 0.79 | **0.89** |
| silver 30m | −1.11/+0.16 | **+0.32/+0.95** | −0.27/+0.67 | −0.76/+0.52 | 0.78 | 0.89 |

- **Same edge.** pnl corr 0.87–0.91 to `node_mimic` everywhere → the EA *is* a short-lookback
  parameterisation of `RobustTrendBreakout`. The **entry mechanic is byte-identical**
  (`rolling(L).max().shift(1)` == the node's `_donchian_entry_level`); the **exit is *approximate*** —
  the node stops on the intrabar **low** (`low < chandelier_stop`) whereas this engine (matching the
  EA's bar-close branch) stops on the **close**. That exit-basis difference (plus the node permitting
  same-bar re-entry, and an EMA≈off residual) is why the signal corr is 0.78–0.82 rather than ~1.0.
- **The node's robustness upgrades work as advertised.** Its EMA-100 regime filter + longer Donchian
  (the `node_default` / `node_L20` configs) generally **match or beat** the raw EA on the trend
  instruments while **trading ~3–4× less** (gold 30m: node_default 449 trades vs EA 2106), and the
  regime filter even rescues silver 30m (node_default +0.32/+0.95 vs EA −1.11/+0.16). For costs, this
  is a large advantage. `node_L20` is the most consistently strong single config across instruments.

**Implication:** there is no reason to add a new node. If this idea is pursued, it is `robust_trend_breakout`
(already `signed_signal`-wireable via `create_base_model_from_config`), tuned to L20–55 + regime + wide stop.

---

## 5. Cost vulnerability (the decisive untested factor)

Frictionless only. Turnover (above) is the warning: the EA default does **100–450 round-trips/yr** on
30m/1h. On CFDs a round-trip costs spread (gold ≈ a few bp; indices/oil more) **plus overnight swap**
on every multi-bar hold (holds here are hours→days). A rough ~2–4 bp/round-trip + carry on ann returns
of only ~5–9% (train) at 15% vol means a plausible **1–2%/yr drag**, i.e. several tenths of Sharpe.
Expectation:
- **30m variants and the raw L5/m2 default → likely net-negative after costs.**
- **2h, longer-lookback (L20–55), wider-stop, channel-exit-OFF on gold/nasdaq (and maybe usdjpy) →
  the only cost-plausible survivors.** The `spread` column is present in the M1 store, so a realistic
  Phase-1 (Nautilus realistic lane: modeled CFD spread + swap + the rollover-flatten overlay) is the
  natural next step and the real go/no-go.

---

## 6. Threats to validity / caveats

- **Long-only into a long-biased, trend-rich OOS window** — val Sharpes are regime-inflated; trust train.
- **No costs** (§5) — the headline.
- **Close-based entry/exit**, not the EA's intrabar BUY-STOP fill + intrabar SL. This is the *conservative*
  frictionless choice (no intrabar look-ahead); the EA's intrabar fills would change fill prices, not the
  existence of the edge — but they belong in the costed lane.
- **`bars_per_year` measured on the full 2018–24 sample** (annualisation scalar) rather than frozen on
  train — a cosmetic detail; it scales train and val Sharpe by the same ~constant and does not move the
  cross-sectional conclusions. (Flagged by the vol-scaling audit.)
- **Vol-scaled absolute Sharpe carries ~±0.2 of estimator uncertainty on the low-vol names** because the
  F=2.0 cap binds 38–75% of bars there (§3a, §5); the **binary** core is the implementation-invariant
  number. (This is the source of the §8 recompute train gap.)
- **Drawdowns/Sharpe are additive (cumsum return-unit), not compounded** — standard frictionless
  convention, internally consistent; read maxDD as additive.
- **Broker-EET timestamps treated as a consistent monotonic clock** (tz dropped, wall-clock kept) — fine
  for a breakout on resampled bars; absolute session alignment is irrelevant to the rolling high/low logic.
- **`n_trades`/`avg_hold` can be off by ±1 per split** (`np.roll` boundary wrap in the turnover counter) —
  cosmetic, no P&L impact.
- **silver/crude verdicts are robust to params** (whole-grid negative/flipping), so they aren't a tuning
  artifact — the edge genuinely isn't there intraday for those two.

---

## 7. Recommendation

1. **Do not build a new bias node.** This is `robust_trend_breakout`. Any productionisation reuses it.
2. **Frictionless edge confirmed only on gold / nasdaq / usdjpy** (robust plateau), **rejected on
   silver / crude** (no OOS edge). Prefer the **slow** corner: L20–55 + EMA regime + wide ATR stop + 2h.
3. **Gate = costs.** Before any `StrategySpec` / vault consideration, run the survivors through the
   Nautilus realistic lane (CFD spread + swap + rollover overlay). The frictionless val Sharpes (1–2)
   are not the number that matters; the **net-of-cost 2h, long-lookback** number is.
4. If/when it survives costs, express as a `StrategySpec` over `robust_trend_breakout` and run the normal
   `exploration → validation (portfolio-addition gate)` flow — note the gate baseline is the **daily**
   portfolio, so an intraday sleeve is a genuinely new axis for this repo (its own integration question).

---

## 8. Verification (adversarial)

A background workflow ran three independent reviewers (distinct lenses) over `engine.py` + `run_matrix.py`
plus an independent from-scratch recompute of one cell:
- **Lookahead lens:** no forward-information leak; inputs independently causal; the dual-derivation
  `assert` is a genuine shift-consistency guard. (Nit: "lookahead-free" is more precisely "shift-consistent".)
- **Vol-scaling lens:** EWSD replication faithful to the node (`ewm(alpha=1−λ)` algebra correct;
  `rolling(2520,min_periods=2).std(ddof=1)` reproduces the deque semantics); `F=clip(0.15/σ,0,2)` matches
  DiversifiedEnsemble. Only note = full-sample `bpy` (§6).
- **Signal/metric lens:** channel-breakout state machine faithful to the EA; entry mechanic
  byte-identical to the node's Donchian; `ema_period=2` does effectively disable the regime filter;
  metrics correct.
- **Independent recompute (gold 1h default, from scratch, no shared code):** reproduced the **binary**
  signal P&L *exactly* — train **0.344 / 0.344**, val **0.830 / 0.830** (recompute vs this engine) — so
  the signal + P&L alignment is confirmed by two independent implementations. Vol-scaled **val** also
  matched (0.978 vs 1.07, within tolerance). Vol-scaled **train** differed (recompute 0.61 vs this engine
  0.42); **localised** (`outputs/_diag` since removed) entirely to the EWSD estimator under a
  frequently-binding cap: gold-1h σ_annual≈0.087 ⇒ F caps on 38% of bars (§3a), so the vs-Sharpe is
  estimator-sensitive while the binary core is stable. This engine's EWSD is **byte-faithful to
  `nodes/volatility/ewsd`** (audit-confirmed); the recompute used a reasonable EWSD variant — the two
  straddle. **No bug; the binary core is the number to trust.**

## 9. Artifacts (`outputs/`)
- `grid.csv` (600 rows: 5 sym × 3 TF × 40 params, train+val metrics) · `node.csv` (45) · `compare.csv` (15)
- `summary_by_symbol_tf.csv` · `summary_by_symbol.csv` · `summary_node_vs_cb.csv` · `binary_core.csv`
- `heatmap_val_sharpe_1h.png` (param plateau) · `equity_1h.png` (cum frictionless P&L, cb vs node)
- **Phase 1:** `ensemble_summary.csv` · `ensemble_cost_sensitivity.csv` · `ensemble_equity_net.png`
- **Sweep (§11):** `sweep_tf.csv` · `sweep_4th.csv` · `sweep_4th_equity.png`
- **M1 fills (§12):** `m1_fills_configs.csv` · `m1_fills_equity.png`
- **Bar-close re-opt (§13):** `m1_optimize.csv` · `m1_optimize_equity.png`
- Reproduce: the engine was removed (house close-out); rebuild from **Appendix A** against `data/mt5_data/<SYM>/bars_M1/`, then re-run the lanes. The CSV/PNG artifacts above are the frozen evidence.

---

## 10. Phase 1 — costed ENSEMBLE (the result, and it survives costs)

**Construction (`ensemble.py`):** per-instrument vol-scaled position **sized at entry and held**
(a breakout system does not micro-rebalance the vol target every bar — doing so fabricates ~10× phantom
turnover; an earlier band-rebalance version made that mistake and was corrected) → per-bar gross + NET
P&L → aggregate to **daily** per-instrument returns → **equal-weight × IDM** (instrument diversification
multiplier `min(√(1/mean_corr+0.01), 2.5)` from the train-window cross-instrument daily-return corr;
IDM caps at 2.5 for these low-correlated trend streams — net Sharpe is IDM-invariant, IDM only sets leverage).

**Costs (data-driven):** SPREAD = per-instrument round-trip bp from the M1 `spread` column
(gold 0.39 / nasdaq 0.64 / usdjpy 0.18 / silver 2.98 / crude 5.87 bp), charged on entry+exit turnover ×
`cost_mult` (2 = raw + commission/slippage buffer). SWAP = `swap_annual/bars_per_year` on the held
leveraged notional (3% default — **conservative**: a real long-USDJPY *earns* carry, so usdjpy net is understated).

### Net Sharpe (daily, train / val; NET = spread×2 + 3% swap) — `ensemble_summary.csv`

| strategy | universe | 1h | 2h | 1h+2h (gross→net) |
|---|---|---|---|---|
| **cb L20/m3/off** | **winners** | 1.29 / 1.66 | 0.86 / 1.68 | **1.65/2.25 → 1.19 / 1.85** |
| rtb L55/ema100/m3.5 (regime) | winners | 1.12 / 1.58 | 0.42 / 1.48 | 1.21/2.02 → 0.85 / 1.69 |
| rtb L20/ema100/m3.0 (regime) | winners | 0.99 / 1.74 | 0.69 / 1.16 | 1.37/1.99 → 0.93 / 1.61 |
| rtb L20/ema2 (NO regime) | winners | 0.98 / 1.56 | 0.98 / 1.21 | 1.55/1.96 → 1.09 / 1.55 |
| cb L5/m2/on (EA default) | winners | −0.28 / 1.34 | 0.39 / 1.68 | 0.68/2.26 → **0.06 / 1.67** |
| cb L20/m3/off | all5 | 0.76 / 0.66 | 0.35 / 1.25 | 1.18/1.60 → 0.60 / 1.04 |
| rtb L20/ema100 (regime) | all5 | 0.51 / 0.87 | 0.44 / 0.47 | 1.12/1.36 → 0.52 / 0.74 |

**Reads:**
1. **The winners ensemble survives costs comfortably.** Net **train ~0.9–1.2 / val ~1.6–1.85** for the
   robust configs — a *strong* ensemble Sharpe net of realistic costs, with shallow drawdowns
   (net maxDD ~10% train / 5% val of a ~10–12% ann return; `ensemble_equity_net.png` compounds smoothly).
2. **The slow config is mandatory.** The EA's L5 default nets **0.06 train** (1h alone: −0.28) — its
   2–4× turnover is eaten by costs. L20+/wide-stop is the cost-survivable corner (confirms Phase 0).
3. **Drop silver & crude.** all5 net (~0.5–0.7 train) ≪ winners (~1.0–1.2). Their wide spreads
   (crude 3.6%/yr, silver 2.3%/yr drag) + weaker edge outweigh the diversification benefit.
4. **The EMA regime filter is ≈neutral on an all-trending basket.** winners 1h+2h NET:
   regime 0.93/1.61 vs **no-regime 1.09/1.55** — within noise, and the plain channel breakout (1.19/1.85)
   is right there too. The regime filter earns its keep on *marginal* instruments (its design intent) —
   but those (silver/crude) are cost-killed anyway. So "robust_trend_breakout + regime" ≈ "channel
   breakout" ≈ "rtb no-regime" net; the edge is the breakout + ATR trail + diversification, not the filter.

### Cost decomposition (rtb L20 regime, 1h+2h, mult=2 swap=3%) — per instrument
gold 0.55%+0.99% · nasdaq 0.76%+0.80% · usdjpy 0.42%+1.74% · **silver 2.27%**+0.49% · **crude 3.61%**+0.39%
(spread%/yr + swap%/yr). Winners are **swap-dominated** (sub-1bp spreads); silver/crude are spread-killed.

### Cost sensitivity — winners 1h+2h, **val** net Sharpe (`ensemble_cost_sensitivity.csv`)

| spread×→ / swap↓ | 0% | 3% | 6% |
|---|---|---|---|
| **×0 (frictionless)** | 1.99 | 1.72 | 1.45 |
| ×1 | 1.93 | 1.66 | 1.39 |
| **×2 (base)** | 1.88 | **1.61** | 1.33 |
| ×3 (harsh) | 1.82 | 1.55 | **1.27** |

Even at 3× spread + 6% swap the winners ensemble holds **val net 1.27** — the conclusion is robust to the
cost assumptions, not balanced on them.

### Caveats specific to Phase 1
- **Same regime tailwind:** net val (1.6–1.85) ≫ net train (0.9–1.2). The honest forward number is the
  **train ~1.0–1.2**, still a strong ensemble Sharpe.
- **IDM (and the 3% usdjpy swap) computed conservatively;** real usdjpy carry is positive → upside.
- **Daily aggregation** of intraday P&L sidesteps the intraday cross-instrument clock-alignment problem
  and matches how the production book is ultimately scored; entries/exits still fire at intraday timing.
- **Still a research lane, not the Nautilus event lane.** The natural next step before any vault move is
  `scripts/validate_candidate.py` / the realistic Nautilus lane on the winners basket — but the cost gate
  the user was worried about is **cleared**.

---

## 11. Winners-basket sweep — timeframe + 4th-instrument (`sweep_tf.csv`, `sweep_4th.csv`)

Net daily Sharpe (channel breakout L20/m3; spread×2 + 3% swap), train / val.

### (A) Timeframe — **1h+2h is the sweet spot; 2h-only is NOT cleaner**
| TF set | 30m | 1h | 2h | **1h+2h** | 30m+1h+2h |
|---|---|---|---|---|---|
| winners net | 0.27 / 1.56 | **1.30** / 1.66 | 0.86 / 1.68 | 1.20 / **1.86** | 0.95 / 1.93 |

- **30m is the worst** (train 0.27) — too noisy / cost-heavy; the hypothesis "faster is fine" fails.
- **2h-only is *noisier* on train (0.86)**, not cleaner — fewer bars = a weaker estimate, despite lower turnover.
- **1h has the best single-TF train (1.30)**; **1h+2h** gives ~top train AND best val (1.20/1.86) by smoothing
  the two. Adding 30m (30m+1h+2h) lifts val to 1.93 but drags train to 0.95 — not worth its turnover.
- **Use 1h+2h** (or 1h-only for simplicity / max train). **2h-only is the wrong call.**

### (B) A 4th trender — **none of the obvious candidates help** (3-basket baseline 1.20 / 1.86)
| candidate | class | spread | standalone tr/va | corr→winners | 4-basket tr/va | **Δtrain** |
|---|---|---|---|---|---|---|
| sp500 | EQ-idx | 0.45bp | 0.59 / 0.38 | 0.49 | 1.11 / 1.53 | −0.09 |
| nikkei | EQ-idx | 1.55bp | 0.27 / 0.58 | 0.30 | 1.03 / 1.66 | −0.17 |
| dow | EQ-idx | 0.62bp | 0.26 / 0.63 | 0.33 | 1.02 / 1.75 | −0.18 |
| eurusd | FX | 0.18bp | **−0.47** / −0.37 | 0.16 | 0.82 / 1.43 | −0.38 |
| gbpusd | FX | 0.32bp | **−0.59** / −0.65 | 0.16 | 0.73 / 1.27 | −0.47 |
| usdcad | FX | 0.37bp | **−0.76** / −0.43 | −0.14 | 0.73 / 1.61 | −0.47 |
| audusd | FX | 0.44bp | **−0.81** / −1.89 | 0.22 | 0.59 / 0.78 | −0.61 |
| estoxx | EQ-idx | 1.46bp | −0.79 / 0.14 | 0.27 | 0.57 / 1.57 | −0.63 |
| natgas | COMM | **30.6bp** | −1.21 / −0.66 | 0.02 | 0.40 / 1.32 | −0.80 |

**Every candidate lowers the basket's train Sharpe.** Two distinct reasons:
- **FX majors (EUR/GBP/AUD/CAD) have *negative* standalone breakout edge** intraday — they range/mean-revert,
  they don't sustain trends. USDJPY worked only because it had a singular multi-year BoJ-driven trend; it is
  not representative of FX. Adding edgeless instruments dilutes regardless of low correlation.
- **Other equity indices (SP500/Dow/Nikkei/EStoxx) have a real but weaker edge AND are correlated with the
  nasdaq leg** (corr 0.30–0.49) — redundant equity exposure, not diversification. SP500 is the least-bad
  (Δtrain −0.09, ≈neutral) but still doesn't add. Natgas: negative edge + 30bp spread = worst.

5-basket (winners + sp500 + nikkei) = 0.98 / 1.44 < 3-basket. **`sweep_4th_equity.png`: the 3-basket sits
at/above the 4- and 5-baskets throughout.**

**Implication (important caveat):** the ensemble's strength is **concentrated in three instruments that each
trended hard in 2018–24**; it does **not** scale by bolting on more CFDs from this pool. That reinforces the
regime-tailwind caveat — a genuinely broad, robust trend ensemble would need *other markets with real trend
edge* (e.g. rates / a futures trend universe), which aren't in this liquid-CFD set. The product is the
**3-instrument winners basket at 1h+2h**, not a scalable many-instrument book.

---

## 12. M1 event-driven fills — the intrabar-stop reality check (`m1_fills.py`)

`scripts/validate_candidate.py` and the StrategySpec/vault rail are **daily-only** (the live
`VaultRebalanceStrategy`/`VaultForecastEngine` load daily ensemble JSONs and query `TimeFrame.D`; the
realistic `NautilusPnLEngine` is also one-target-per-session). **None can run a 1h/2h multi-trade-per-day
CFD breakout.** So instead of the daily Nautilus lane, this replaces the Phase-1 daily-aggregated cost model
with an **M1 event-driven fill simulation** (`m1_fills.py`): decisions anchored to the 1h/2h bar, but the
**BUY-STOP entry and chandelier-STOP exit rest and fill at M1 resolution** (entry on the first M1 high ≥
level, exit on the first M1 low ≤ stop; gap-through → fill at the M1 open; real spread embedded; swap/night).
Lookahead-free; faithful to the EA's real pending-stop + attached-SL behaviour.

### The finding: intrabar stop fills haircut the edge (3-basket, 1h+2h, spread×2 + 3% swap)

| config | net train | net val | trades (3 inst) |
|---|---|---|---|
| L20/m3/ce-off *(close-based winner)* | 0.78 | 1.47 | 3,387 |
| **L20/m4/ce-off** *(wider stop)* | **0.88** | 1.28 | 2,584 |
| L20/m4/ce-on *(adds bar-close channel exit)* | 0.69 | 1.37 | 2,841 |
| L55/m5/ce-on *(slow)* | 0.45 | **1.96** | 1,405 |
| **Phase-1 close-based reference (L20/m3)** | **1.20** | **1.86** | ~500 |

- **Realistic fills cut net train Sharpe from ~1.2 → ~0.8–0.9** (val ~1.3–1.5 vs 1.86). The Phase-0/1
  *close-based* exit (exit only if the **bar closes** below the stop) was **optimistic**: a real stop
  **order** fills on **intrabar wicks** the close-based rule held through, so the system trades **~7×
  more** (3,387 vs ~500), stopping out on a wick and re-entering higher — **whipsaw**, the true cost.
- **It is NOT slippage:** mean exit fill is −0.5…−0.9 bp vs the bar close (i.e. the stop touch is *slightly
  better-priced* than the close, since the bar usually closes lower). The damage is the **frequency** of
  stop-outs + the extra spread on 7× turnover, not the per-fill price.
- **Config preference shifts under realistic fills: wider stops win** (L20/**m4** 0.88 > L20/m3 0.78 on
  train, by wicking less). The close-based sweep's preference for tight-ish `m3` was an artefact of not
  modelling intrabar stops. **Actionable design insight:** the whipsaw comes entirely from the *intrabar
  ATR trailing stop*; a **bar-close exit rule (channel/EMA break) avoids wicking altogether** and is the
  fill-robust design — worth testing a pure close-based-exit variant (note `robust_trend_breakout`'s own
  `low < chandelier_stop` exit would wick the **same** way, so the node is not immune).

### Verdict
The edge **survives realistic event-driven execution** (best config nets **~0.88 train / ~1.3 val**, all
configs compound positively — `m1_fills_equity.png`), and diversification still does the heavy lifting
(weak gold-standalone 0.19 → basket 0.78). **But the honest, fill-realistic number is ~0.8–0.9 train, not
the 1.2 the close-based model showed** — Phase-1 overstated it by ~0.3–0.4 Sharpe. This is a **respectable
but modest** net-of-realistic-cost intraday system, more **stop-design-sensitive** than earlier phases
implied.

### On the StrategySpec / vault path
**Not available for this strategy.** `StrategySpec` → adapter → exploration/validation → vault → live is a
**daily (D/W/M)** rail end-to-end (`TimeFrame` has no intraday member; the vault stores daily ensembles;
`validate_candidate.py` runs the daily live strategy). An intraday 1h/2h CFD breakout has **no slot** in it.
Productionising it would require **either** building intraday rails (a real new-infra project — intraday
Nautilus harness + an intraday position/forecast path), **or** running it as a **separate intraday system**
alongside the daily vault — a deliberate call (see the recovered edge in §13).

---

## 13. Bar-close exit (wick-robust) + re-optimization under the M1 lane — the edge is recovered

The §12 whipsaw came entirely from the **intrabar trailing STOP**. So I added a second engine
(`m1_fills.py::simulate_bc`): the trailing/channel exit is evaluated **only at the decision-bar close**
(no intrabar stop → no wick); it needs only the resampled OHLC (intrabar *entry* fills still modelled via
the bar high/open). Then I **re-optimised 64 configs** (lookback × atr_mult × entry-mode × channel-exit)
**under this lane** — the honest re-tune the close-based Phase-0/1 sweep couldn't do (`m1_optimize.csv`).

### Result: the edge comes back (3-basket net, 1h+2h, spread×2 + 3% swap)

| variant | config | net train | net val |
|---|---|---|---|
| **bar-close exit + bar-close entry** | **L20 / m4 / ce-off** | **1.25** | **1.87** |
| bar-close exit + bar-close entry | L20 / m4 / ce-on | 1.26 | 1.52 |
| bar-close exit + bar-close entry | L20 / m3 / ce-off | 1.17 | 1.84 |
| bar-close exit + **intrabar** entry | L20 / m4 / ce-off | 0.84 | 1.68 |
| intrabar STOP (EA as written), §12 | L20 / m4 | 0.88 | 1.28 |
| close-based Phase-1 reference | L20 / m3 | 1.20 | 1.86 |

- **Switching the exit from an intrabar stop to a bar-close rule recovers train 0.88 → 1.25 and val 1.28 →
  1.87** — i.e. essentially **all** of the edge the whipsaw destroyed, now *net of realistic M1-lane fills*
  (real spread embedded, swap/night). The alpha was never the problem; the **EA's intrabar stop order was**.
- **Both sides must be bar-close.** `entry=bar_close` beats `entry=intrabar` across the board (L20/m4/ce-off:
  **1.25 vs 0.84**): an intrabar BUY-STOP also whipsaws by filling on failed-breakout wicks that close back
  below the channel. Entering only when the bar **closes** above the channel removes that. ce-off edges
  ce-on on val. L20 + wider stop (m4) dominates the grid (`m1_optimize_equity.png` — the bar-close curve is
  smoother and higher through train).
- This is a genuine **design fix, not a curve-fit**: it's the same L20/wide-stop region the close-based
  sweep liked, and the improvement is structural (remove intrabar stop-outs), confirmed across the whole grid.

### Final verdict (whole study)
The 3-instrument winners basket (gold/nasdaq/usdjpy), traded as a **bar-close** breakout (enter on a close
above the L20 Donchian high, exit on a close below the ATR(14)×4 chandelier trail), 1h+2h, vol-scaled +
IDM, nets **≈1.25 train / 1.87 val net of realistic spread + swap + intrabar entry fills**. Standing
caveats unchanged: **trust the ~1.25 train** (val is 2023–24 trend tailwind); the edge is **concentrated in
3 markets** and doesn't scale by adding CFDs (§11); and there is **no daily-rail slot** — productionising
means an intraday harness or a standalone intraday system. **Do NOT run the EA's intrabar trailing stop**
(nor `robust_trend_breakout`'s `low < chandelier_stop`, which wicks the same way); use a bar-close exit.

---

## Appendix A — Methodology & reproduction (engine removed; this is the spec)

**Data.** `load_data` is D/W/M-only; intraday is read directly from the M1 CFD store
`data/mt5_data/<SYM>/bars_M1/year=YYYY/part.parquet` (cols `time, open, high, low, close, tick_volume,
spread`). `time` is **broker EET/EEST mis-tagged UTC** — drop the tz, keep the wall-clock (absolute offset
is irrelevant to rolling-high/low logic). Resample to 30m/1h/2h OHLC (`open=first, high=max, low=min,
close=last, vol=sum(tick_volume)`), drop empty bars (NaN close or `vol==0`). Symbols: gold=XAUUSD,
nasdaq=NDX, usdjpy=USDJPY, silver=XAGUSD, crude(WTI)=XTIUSD. Years loaded **2018–2024** (2025+ locked).

**Vol-scaling (faithful to `nodes/volatility/ewsd` + `DiversifiedEnsemble` defaults).** Per-bar simple
return `r=close.pct_change()`. EWSD σ = `0.7·σ_short + 0.3·σ_long`, σ_short = `sqrt( ewm(alpha=1−λ,
adjust=False) of r² )` with `λ=0.06061`; σ_long = `r.rolling(2520, min_periods=2).std(ddof=1)`. Annualise:
`σ_ann = ewsd_daily · √(bars_per_year)` where `bars_per_year = n_bars / years_spanned` (per symbol×TF).
Position size `F = clip(target_vol/σ_ann, 0, 2.0)`, `target_vol=0.15`. (NB: the 0.15 *daily* target
saturates the F=2.0 cap on low-vol intraday names — usdjpy 75 %, gold 38 % of bars — so report the
implementation-invariant *binary* Sharpe as the anchor; §3a.)

**Signal (channel breakout = the EA as a signed signal).** Decision bar `t`, all levels from bars `< t`:
`upper_prev = high.rolling(L).max().shift(1)`, `lower_prev = low.rolling(L).min().shift(1)`, Wilder
`ATR(14) = tr.ewm(alpha=1/14, adjust=False).mean()`. Entry (flat → long) when `close_t > upper_prev_t`;
chandelier `stop = max(prev_stop, close − atr_mult·ATR)`; exit on stop and/or channel break
`close ≤ lower_prev`. Entry mechanic is byte-identical to `RobustTrendBreakout._donchian_entry_level`.

**P&L lanes (each lookahead-guarded — `pnl = pos.shift(1)·r`, re-derived a 2nd way + asserted equal).**
- *Frictionless* (§1–§9): vol-scaled position × bar return, no costs.
- *Costed daily-aggregated ensemble* (§10): per-instrument vol-scaled position **sized at entry, held**
  (do **not** band-rebalance vol drift → ~10× phantom turnover); spread charged on entry+exit turnover
  using the M1 `spread` column (round-trip bp: gold 0.39 / nasdaq 0.64 / usdjpy 0.18 / silver 2.98 /
  crude 5.87) × `cost_mult` (2 = +commission/slippage); swap `= |held|·swap_annual/bars_per_year`
  (3 %/yr, conservative — real long-USDJPY earns carry). Aggregate per-bar P&L → **daily** per instrument.
- *Portfolio*: equal-weight × **IDM** `= min(√(1/(mean_corr+0.01)), 2.5)` from the **train-window**
  cross-instrument daily-return correlation (net Sharpe is IDM-invariant; IDM only sets leverage).
- *M1 event-driven fills* (§12–§13): decisions on the 1h/2h bar, orders fill at M1 resolution. Two exits:
  **intrabar stop** (SELL-STOP fills on first M1 `low ≤ stop`; the faithful-but-whipsawy EA behaviour) vs
  **bar-close** (trail/channel evaluated only at the decision-bar close — no wick; needs only resampled
  OHLC). Entry likewise `intrabar` (BUY-STOP on M1 `high ≥ level`, gap → M1 open) or `bar_close`
  (`close > level`). Spread embedded in each fill price; swap per broker-night. **Best = bar_close entry +
  bar_close exit.**

**Splits.** train `2018-01-01…2022-12-31`, val `2023-01-01…2024-12-31`, **2025+ never loaded** (locked).
Sharpe = `mean/std·√(bars_per_year)` (intraday lanes) or `·√252` (daily portfolio); drawdowns additive
(cumsum). Adversarial verification (3 reviewers + an independent from-scratch recompute that reproduced the
binary signal P&L exactly) found no lookahead/correctness bug; the only nuance is the cap-saturation above.

**Final config (the deliverable).** Universe gold+nasdaq+usdjpy; signal channel-breakout `L=20`,
`atr_mult=4`, channel-exit off; **bar-close entry + bar-close exit**; TFs 1h+2h (equal blend); vol-scaled
(above) + equal-weight×IDM. Net of spread×2 + 3 % swap: **≈ 1.25 train / 1.87 val**.

---

## Appendix B — Long vs long/short (post-closeout test; `outputs/long_short_test.csv`)

Tested whether the short side adds (standalone net daily Sharpe, bar-close L20/m4, 1h+2h, real spread+swap):

| asset | long-only tr/va | **short-only** tr/va | long/short tr/va |
|---|---|---|---|
| crude | −0.07 / −0.35 | −0.54 / −0.95 | −0.43 / −1.10 |
| silver | −0.06 / +0.10 | −0.48 / −0.65 | −0.24 / +0.11 |
| gold | +0.62 / +0.99 | −0.51 / −0.99 | −0.08 / +0.39 |
| nasdaq | +0.70 / +1.34 | −0.72 / −1.34 | −0.04 / +0.19 |
| usdjpy | +0.70 / +0.79 | −0.26 / −1.08 | −0.04 / +0.05 |
| eurusd | −0.39 / −0.53 | +0.19 / −1.23 | −0.25 / −1.42 |
| sp500 | +0.64 / +0.50 | −0.48 / −1.28 | +0.27 / −0.03 |

- **The short side has NO edge on any asset** (short-only negative everywhere). Long/short is therefore
  **worse than long-only** wherever the long side works (shorts drag the winners to ~0), and it does **not**
  rescue silver or crude. (Earlier conjecture that crude might benefit from shorting the 2020/2022 crashes
  was **wrong** — crude short −0.54/−0.95.)
- **Why no short edge:** (1) trend *asymmetry* — 2018–24 up-moves were slow/persistent (ideal for breakouts),
  down-moves were sharp V-shaped panics that reverse fast, so a breakdown-entry system enters late and eats
  the snap-back; (2) shorting fights the structural upward drift (equity risk premium; gold/USDJPY carry);
  (3) long/short ~doubles turnover → cost eats any marginal short edge (worst on wide-spread silver/crude).
- **Economic reframing:** this is **not** symmetric trend-following — it's a **long-biased momentum / risk-
  premium harvester** (equity drift for nasdaq/sp500, real-rate/CB-buying bull for gold, rate-diff + carry
  for USDJPY). That's why it concentrates in a few up-trending markets, why long-only is correct, and why it
  neither scales nor benefits from shorts.
