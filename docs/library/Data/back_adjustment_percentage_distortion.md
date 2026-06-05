# Additive Back-Adjustment Distorts Percentage Returns & Vol Sizing

> ⚠️ Slated for rewrite under the NautilusTrader migration (WP-2 data layer). See docs/refactor/nautilus/.

> [!warning] Known issue — not yet fixed
> `data/ohlc_data` futures series are **additively** back-adjusted continuous contracts
> (Norgate `&XX_CCB`). Additive adjustment preserves *point* moves but inflates the
> historical price *level*, so any **percentage** computed from these prices is distorted.
> This silently biases **volatility-based position sizing** (and, mildly, cross-asset
> percentage signals). It does **not** materially bias risk-adjusted research metrics
> (Sharpe), but it makes the backtest run **colder than live** — a book tuned to a 15%
> vol target on this data will run ~30–45% hotter when traded at real prices.
> Discovered 2026-05-30 while verifying the SPY/TLT rebalancing nodes. Bond/ETF-like
> series (e.g. `TLT`) are unaffected.

## TL;DR

- **Affected**: `σ` used for vol-targeting (EWSD), and percentage relative-strength
  signals — for **futures legs that are additively back-adjusted with a non-trivial roll
  adjustment** (e.g. `ES`). The damage is asymmetric across instruments.
- **Not affected**: risk-adjusted research conclusions (Sharpe is ~unbiased because the
  adjustment factor cancels), and ETF/cash-like series (`TLT`, `k≈1`).
- **The real hazard**: backtest vol understates live vol by `1/k` (≈ **1.46×** for `ES`).
  The backtest looks calm; live is over-leveraged.
- **Fix (option E)**: compute `σ` and percentage relative-strength signals from a
  **percentage-faithful** price series — a *ratio-adjusted* continuous contract or a
  *cash/ETF* series — not the additive back-adjusted series. Keep the back-adjusted series
  only for point-based logic (absolute trend/breakout distances, dollar P&L).

## Background: continuous-contract back-adjustment

A continuous futures series splices successive contracts; each roll leaves a price gap
(cost of carry, dividends, term structure). Two ways to remove the gap:

| Method | Preserves | Distorts | Right for |
|---|---|---|---|
| **Additive** ("Panama"/difference) | absolute *point* moves (→ correct dollar P&L) | the price *level* → **% returns** | futures point/dollar P&L systems |
| **Proportional** (ratio) | *percentage* returns | absolute point moves / dollar P&L | %-return strategies, vol estimation, ratio spreads |
| Unadjusted | actual traded prices | has artificial roll jumps | roll metadata only |

This repo implements Robert Carver's methodology, which is **percentage-return and
volatility-target based** — so its `σ` and relative-strength comparisons need *percentage*
faithfulness, i.e. the proportional/cash convention, not additive.

## What the repo does today (confirmed)

1. **Data source = additive back-adjusted.**
   - [`data_platform/providers/norgate/_constants.py`](../../../data_platform/providers/norgate/_constants.py)
     maps `"ES": "&ES_CCB"`. The `_CCB` suffix = *Continuous Contract, Back-adjusted*
     (additive). Fetched with `padding_setting = PaddingType.NONE`.
   - [`data_platform/providers/norgate/migrate.py`](../../../data_platform/providers/norgate/migrate.py)
     builds `data/ohlc_data/{TICKER}` from `data/norgate/working/continuous/adjusted/`.
   - [`data_platform/providers/norgate/backadjust/gap_calculator.py`](../../../data_platform/providers/norgate/backadjust/gap_calculator.py)
     is also additive (`cumulative_adjustment = sum of gap_points`).
   - The *unadjusted* series is fetched to `data/norgate/working/continuous/unadjusted/` and
     migrated to `data/ohlc_data/{TICKER}/D_{TICKER}_unadj.parquet` for use as the σ
     denominator (see fix below).
   - **`TLT` is ETF-like**: it starts 2002-07 (TLT ETF inception) and its returns match the
     TLT ETF (corr 0.9995, identical vol) — i.e. it is *not* additively distorted. So the
     repo currently mixes a distorted equity leg with a faithful bond leg.

2. **`σ` fix implemented** — Carver form in
   [`nodes/volatility/ewsd/ewsd.py`](../../../nodes/volatility/ewsd/ewsd.py):
   ```python
   price_diff = candle.close - self.prev_close          # ΔP from back-adjusted (correct $)
   ref_close  = unadj_close[candle.date]                # P_unadj (correct % denominator)
   daily_return = price_diff / ref_close                # true % return
   ```
   EWSD blends 70% EWMA-32 + 30% expanding 10-yr stdev, ×16 annualized. This feeds
   `forecast = min(τ/σ, 2.0) · signal` in
   [`ensemble/diversified_ensemble.py`](../../../ensemble/diversified_ensemble.py) (≈ L1218/L1388).
   It is **not** Carver's distortion-proof form (numerator = price *difference* ÷ *current*
   price).

3. **Research P&L is computed in the same (back-adjusted) % convention.**
   [`utils/evaluation/walkforward/portfolio_evaluator.py:142-145`](../../../utils/evaluation/walkforward/portfolio_evaluator.py)
   ```text
   forecast_score    = min(τ / EWSD[t], 2.0) × signal
   position_fraction = forecast_score × instrument_weight × IDM
   strategy_return   = position_fraction × log_return(close)     # back-adjusted close
   ```
   The research path does **not** go through contracts/dollars — it is
   `position_fraction × log_return`. This is why the distortion *cancels* in research (below).

4. **Live execution uses the real price.**
   [`execution/position_sizer.py`](../../../execution/position_sizer.py):
   `contracts = position_fraction × capital / (price × multiplier × fx)`, where `price` is
   the *current market price*. This is where the cancellation **breaks**.

## The mechanism (why it cancels in research but bites live)

Additive adjustment preserves the point move `ΔP` but shifts the level, so for an instrument
whose back-adjusted level is inflated by factor relative to the true price:

```
r_adj[t] = ΔP[t] / P_adj[t-1] = r_true[t] · k[t],    k = P_true / P_adj   (k<1 for ES, k≈1 for TLT)
σ_adj    = k · σ_true
```

**Research (size and earn in the same back-adjusted % units):**
```
strategy_return = (τ / σ_adj) · signal · r_adj
                = (τ / (k·σ_true)) · signal · (k·r_true)
                =  (τ / σ_true) · signal · r_true        ← k cancels  ⇒ Sharpe ≈ unbiased
```

**Live (size on back-adjusted σ, earn the real % at real fills):**
```
live vol = (τ / σ_adj) · σ_true = τ / k  > τ            ← over-leveraged by 1/k
```

The cancellation is only *approximate* because `k` drifts over time (rolls + price moves)
and the 30% expanding-10yr σ component spans a wide range of `k`; and because the
cross-asset signal compares two series with *different* `k` (ES `k≈0.7`, TLT `k≈1`).

## Empirical evidence (SPY/TLT rebalancing strategy, 2002–2021)

Reference = Robot Wealth "Stock Bond Reversal" spreadsheet using **SPY/TLT ETF** adjusted
closes (in `data/32. Stock Bond Reversal-*/`). Repo = `ES`/`TLT` from `ohlc_data`.

**Data divergence (ES back-adj vs SPY cash):**

| | repo ES | true S&P / SPY |
|---|---|---|
| Level 2002-07-26 | 1467 | ~852 |
| Level 2021-05-20 | 4903 | ~4155 |
| Growth 2002→2021 | 3.3× | ~4.9× |
| Daily vol | 0.86% | 1.22% |
| Worst day 2008-10-13 | 9.1% | 14.5% |
| Return corr | — | 0.986 |

**Volatility & sizing (EWSD σ, annualized):**

| Leg | repo (back-adj) | cash | k = σ_adj/σ_cash |
|---|---|---|---|
| Equity (ES vs SPY) | 0.084 | 0.121 | **0.69** |
| Bond (TLT_fut vs TLT_etf) | 0.100 | 0.100 | **1.00** |

**Equity leg through the full sizing pipeline (target 15%-when-active):**

| Run | Realized vol | Sharpe | CAGR |
|---|---|---|---|
| (i) back-adj — size ES, earn ES (self-consistent backtest) | 10.7% | 0.71 | 7.7% |
| (ii) cash / **FIX** — size SPY, earn SPY (true %) | 11.6% | 0.77 | 8.9% |
| (iii) live-like — size on ES σ, earn real SPY | **14.9%** | 0.80 | 11.9% |

`1/k = σ_cash/σ_adj ≈ 1.46×`. Combined strategy Sharpe: back-adj **0.85** vs cash **0.90**.

**Signal logic (separately verified):** with the correct prior-month-close basis the
first-15-day decision matches the reference 225/225 months; on repo ES vs cash SPY it agrees
217/225 (96.4%) — the ~3.6% flips are the cross-asset `k`-asymmetry near zero-spread months.

### Interpretation

- **Research / "does it work" → not inflated.** Back-adj vs cash Sharpe is 0.71 vs 0.77
  (equity) and 0.85 vs 0.90 (combined). Same edge; `k` cancels.
- **Backtest understates risk; live inflates it.** The back-adj backtest shows 10.7% vol,
  but sizing on that σ and trading at real prices realizes 14.9% (≈ **1.4× hotter**). A 15%
  target becomes ~21% live. This is the actionable hazard.
- **Asymmetric:** equity leg distorted (`k=0.69`); bond/ETF leg clean (`k=1.0`).

## Impact assessment

| Area | Affected? | Notes |
|---|---|---|
| Sharpe / IR / "edge exists" research conclusions | ~No | `k` cancels in `position_fraction × log_return` |
| Realized **volatility target** (live) | **Yes** | over-leveraged by `1/k` (~1.46× for ES) |
| Reported backtest vol / CAGR (absolute) | Yes (understated) | looks calmer/smaller than live truth |
| Cross-asset % relative-strength signal | Yes (mild) | ES vs TLT on different `k` → ~3.6% decision flips |
| Long-run σ (30%, 10-yr expanding) | Yes | spans wide `k`; doesn't cancel cleanly |
| IDM / cross-asset correlations | Mild | computed on distorted returns |
| ETF/cash-like instruments (`TLT`) | No | `k≈1` |
| Single-instrument futures point/dollar P&L | No | additive is correct for points/dollars |

## Fix (option E) and constraints

**Goal:** make `σ` and percentage signals reflect *true* % so backtest and live agree and the
vol target is honest.

- **Option 1 — Carver in-node σ.** Compute `σ% = stdev(price differences) / reference_price`,
  where the numerator uses back-adjusted price *differences* (correct under additive) and the
  denominator is the **actual/unadjusted price at time `t`** — *not* the lagged back-adjusted
  close. Requires the unadjusted series as the denominator (the back-adjusted close is anchored
  to the series' end date, so even recent historical bars are inflated; a naive "current
  close" tweak does **not** remove `k`).
- **Option 2 — percentage-faithful feed (recommended for %-strategies).** Drive `σ` and the
  relative-strength signal from a **ratio-adjusted** continuous contract or a **cash/ETF**
  series. For the SPY/TLT strategy specifically: trade/measure on SPY (CFD/ETF) + TLT ETF
  (your `TLT` is already ETF-like). Keep additive back-adjusted prices only for point-based
  rules and dollar P&L.

**Constraints / why not fixed yet:**
- The unadjusted/ratio series are not in `ohlc_data` and the raw `data/norgate/.../unadjusted/`
  folders are purged post-migration. Re-fetching requires the **Windows Norgate host** (see
  [[norgate]] and [[multi_source_update_architecture]]).
- `norgatedata` continuous-contract symbols expose back-adjusted (`_CCB`) and unadjusted
  (`&ES`); confirm the ratio-adjusted symbol/option before wiring it.

## Remediation checklist

- [x] Decide convention per instrument: keep additive for point/dollar logic; add a
      percentage-faithful series for `σ` + % signals.
- [x] Fetch unadjusted continuous series and migrate to `D_{TICKER}_unadj.parquet`
      (`data_platform/providers/norgate/migrate.py` writes this automatically).
- [x] Carver percentage-vol mode in `nodes/volatility/ewsd/ewsd.py`:
      `r = ΔP_adj / P_unadj[t]`. Active when `D_{TICKER}_unadj.parquet` exists;
      falls back to previous behaviour otherwise.
- [ ] Route the relative-strength / cross-asset percentage comparisons (e.g.
      `nodes/pairs/rebalancing_flow.py`) to the faithful series.
- [ ] Re-validate: realized vol of a vol-targeted book ≈ target; backtest vol ≈ live vol;
      `k = σ_adj/σ_cash` per instrument ≈ 1 after the fix.
- [ ] Audit other futures tickers for `k` magnitude/drift (plot `k[t] = P_true/P_adj`).

## How to reproduce

The two experiments below were run against `data/32. Stock Bond Reversal-*/` (SPY/TLT ETF
reference) and `data/ohlc_data` (`ES`/`TLT`). Run from repo root with the venv active:
`PYTHONPATH="$PWD" python <script>`.

**1. Strategy/logic verification** (decision base, reversal/continuation windows, cumulative
P&L vs the reference's arithmetic aggregation):
```python
# Drive RebalancingFlowNode with the reference SPY/TLT adjusted closes and reconcile:
#   - TD15 decision vs "First 15 Outperformance?"  (prior-month-close basis -> 225/225)
#   - "Long Only Reversal" / "Game Strat" vs reference (corr ~0.99999, cumulative 202% vs 202%)
# Key findings: signal logic is correct; reference sums DAILY RETURNS arithmetically.
```

**2. Back-adjustment impact (experiment B + C)** — full sizing pipeline on back-adj ES/TLT vs
cash SPY/TLT, reporting σ, realized vol, Sharpe, and the `1/k` live overshoot:
```python
import glob, os, numpy as np, pandas as pd
import utils.core.helpers as h
from nodes import BiasNode
from nodes.pairs.rebalancing_flow import RebalancingFlow, RebalancingFlowNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore

TAU, ANN = 0.15, np.sqrt(252.0)
def repo(tk):
    df = h.load_data(tk, TimeFrame.D).reset_index(); df['date'] = pd.to_datetime(df['datetime'])
    return df.set_index('date')['close'].sort_index()
es_adj, tlt_adj = repo(Ticker.ES), repo(Ticker.TLT)
D = glob.glob('data/32. Stock Bond Reversal*')[0]
sp = pd.read_csv(os.path.join(D, '32. Stock Bond Reversal-SPY_TLT.csv'))
sp['date'] = pd.to_datetime(sp['timestamp']); sp = sp.sort_values('date')
spy = sp.set_index('date')['SPY adjusted Close'].astype(float)
tlt_etf = sp.set_index('date')['TLT adjusted close'].astype(float)
idx = es_adj.index.intersection(tlt_adj.index).intersection(spy.index).intersection(tlt_etf.index).sort_values()
es_adj, tlt_adj, spy, tlt_etf = [s.reindex(idx) for s in (es_adj, tlt_adj, spy, tlt_etf)]

def candles(series, tk):
    return [Candle(datetime=d.to_pydatetime(), open=c, high=c, low=c, close=c, volume=0.0,
                   ticker=tk, tf=TimeFrame.D) for d, c in zip(idx, series.to_numpy())]
def ewsd_sigma(series, tk):  # annualized fraction
    n = h.create_fresh_bias_node('ewsd', tk, TimeFrame.D, {})
    return pd.Series([n.add_candle(c)[1] / 100.0 for c in candles(series, tk)], index=idx)
def signal(self_tk, peer_tk, self_series, peer_series, flow):
    CrossTickerDataStore.reset(); BiasNode._instances.clear()
    st = CrossTickerDataStore.get_instance()
    st.set_data(peer_tk, TimeFrame.D, pd.DataFrame(
        {'datetime': idx, 'open': peer_series.values, 'high': peer_series.values,
         'low': peer_series.values, 'close': peer_series.values, 'volume': 0.0}).set_index('datetime'))
    node = RebalancingFlowNode(self_tk, TimeFrame.D, cross_tickers=[peer_tk.name], flow=flow)
    return pd.Series([node.add_candle(c)[0] for c in candles(self_series, self_tk)], index=idx)
def run(sigma, sig, earn, warmup=60):
    r = earn.pct_change().fillna(0.0).to_numpy()
    fc = np.minimum(TAU / np.maximum(sigma.to_numpy(), 1e-8), 2.0) * sig.to_numpy()
    s = pd.Series(np.concatenate([[0.0], fc[:-1]]) * r, index=idx).iloc[warmup:]
    return s.std()*ANN, (s.mean()/s.std()*ANN if s.std() > 0 else float('nan'))

sig_es, sig_spy = ewsd_sigma(es_adj, Ticker.ES), ewsd_sigma(spy, Ticker.ES)
sig_eq_adj  = signal(Ticker.ES, Ticker.TLT, es_adj, tlt_adj, RebalancingFlow.BOTH)
sig_eq_cash = signal(Ticker.ES, Ticker.TLT, spy, tlt_etf, RebalancingFlow.BOTH)
print('k = sigma_adj/sigma_cash (equity):', round((sig_es/sig_spy).median(), 3))
print('(i)  back-adj :', run(sig_es,  sig_eq_adj,  es_adj))
print('(ii) cash/FIX :', run(sig_spy, sig_eq_cash, spy))
print('(iii)live-like:', run(sig_es,  sig_eq_adj,  spy))   # size distorted sigma, earn real %
```
Expected: `k≈0.69`; (i) ≈10.7%/0.71, (ii) ≈11.6%/0.77, (iii) ≈14.9%/0.80.

## Related
- [[futures_backtesting_data_guide]] — practical guide: which series to use for which
  computation, decision table, checklist for new strategies.
- [[norgate]] — Norgate provider, symbols, Windows-only extraction.
- [[multi_source_update_architecture]] — repository data layering.
- Carver vol-target / forecast scaling: `ensemble/diversified_ensemble.py`,
  `nodes/volatility/ewsd/ewsd.py`.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
