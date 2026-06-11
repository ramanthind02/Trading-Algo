r"""Portfolio Sharpe impact of the limit entry mechanism vs naive market orders.

Both policies flatten-before / re-enter-after the rollover (so they avoid the SAME
swap — it cancels); the difference is purely the execution-cost drag and its variance.
We build a per-night, per-instrument execution-cost series from the REAL rollover ticks
(research/rollover_cost/outputs/events_*.parquet — the same bid/ask Nautilus fills
against), aggregate to an equal-risk 4-leg portfolio, and report annualized Sharpe under:

  * MARKET   — flatten with a market order immediately before the rollover (T-1) and
               re-enter with a market order at the reopen (cross the wide open spread).
  * LIMIT    — passive limit exit (mid, market fallback) + limit re-entry at the touch
               with a chase-or-skip fallback (the universal rule).
  * (ref) HOLD — never trade the rollover; pay the swap every night.

Sharpe is computed as  (mu_gross - mu_cost) / sqrt(sigma_gross^2 + sigma_cost^2) * sqrt(252)
with the portfolio gross daily vol pinned to the 15% annual target (config) and the gross
Sharpe swept across a plausible range (the cost delta is what we isolate). Costs are in
bps of notional; equal-risk weighting => portfolio nightly cost = mean across the 4 legs.

Run:
  .\.venv\Scripts\python.exe -m research.rollover_cost.sharpe_impact
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.rollover_cost.config import SPECS, SYMBOLS

OUT = _REPO / "research" / "rollover_cost" / "outputs"
TARGET_ANNUAL_VOL = 0.15           # config target_volatility
TRADING_NIGHTS = 252

# Actual per-instrument risk weights from the MAIN prop vault/ (Test-phase summed
# stream_weight, research/portfolio/results/weight_layer_weights_all_phases.csv).
# Full vault: ES 0.323, GC 0.315, NQ 0.247, CL 0.116 (SI/silver is NOT traded).
# We map ES->SP500, NQ->NDX, GC->XAUUSD and use these three (88.5% of the book) at
# their renormalised weights; CL/XTIUSD (11.5%) is excluded — its swap is mode-3
# (margin-currency), not the POINTS model used here, and it has no rollover events yet.
VAULT_WEIGHTS_RAW = {"SP500": 0.3230, "NDX": 0.2467, "XAUUSD": 0.3148}


def _half(spread, mid):
    return (spread / 2.0) / mid * 1e4


def per_night_costs(sym: str) -> pd.DataFrame:
    """Per-rollover-night execution cost (bps) for MARKET vs LIMIT, + swap, one instrument."""
    spec = SPECS[sym]
    df = pd.read_parquet(OUT / f"events_{sym}.parquet").copy()
    df = df[df["open_mid"].notna() & df["exit_mid_t1"].notna()].reset_index(drop=True)
    open_mid = df["open_mid"]

    # ---- MARKET policy: cross both legs immediately around the rollover ----
    mkt_exit = _half(df["exit_spread_t1"], df["exit_mid_t1"])     # market-out at T-1 (immediately before)
    mkt_entry = _half(df["open_spread"], open_mid)                # market buy at the reopen
    cost_market = mkt_exit + mkt_entry

    # ---- LIMIT policy (the new mechanism) ----
    # Exit: passive limit @ mid from T-15; if unfilled, market fallback at T-1.
    exit_filled = df["exit_sell_limit_mid_fill_min"].notna()
    limit_exit = np.where(exit_filled, 0.0, mkt_exit)            # captured (~0) else fallback cross
    # Entry: limit 5 ticks below open, chase market @30m if unfilled (universal rule).
    n = 5
    improve = _half(2 * (n * spec.tick_size), open_mid)          # +n ticks improvement if filled (== n*tick/mid*1e4)
    improve = (n * spec.tick_size) / open_mid * 1e4
    filled30 = df.get("entry_buylim_t5_filled_30m", pd.Series(False, index=df.index)).astype(bool)
    # chase cost vs open at +30m (mid drift + half spread)
    chase = _half(df["entry_spread_p30"], df["entry_mid_p30"]) + (df["entry_mid_p30"] - open_mid) / open_mid * 1e4
    limit_entry = np.where(filled30, -improve, chase)
    cost_limit = limit_exit + limit_entry

    # ---- swap (HOLD reference): long swap magnitude, x3 on triple nights ----
    triple_py = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[spec.triple_weekday]
    is_triple = (df["weekday"] == triple_py)
    swap = np.where(is_triple, 3.0, 1.0) * df.apply(
        lambda r: abs(spec.swap_bps(r["open_mid"], is_long=True)), axis=1)

    return pd.DataFrame({
        "date": pd.to_datetime(df["rollover"]).dt.date,
        "cost_market": cost_market.to_numpy(),
        "cost_limit": np.asarray(cost_limit, dtype=float),
        "swap": np.asarray(swap, dtype=float),
    }).set_index("date")


def portfolio_cost_series(weights: dict[str, float] | None = None):
    """Portfolio nightly cost (bps), aligned by date across instruments.

    weights: instrument -> weight (normalised internally). None => equal weight over all
    four study instruments. The portfolio cost is the weighted sum of per-leg costs (cost
    is bps of each leg's notional, weighted by the leg's portfolio weight).
    """
    syms = list(weights.keys()) if weights else list(SYMBOLS)
    w = {s: weights[s] for s in syms} if weights else {s: 1.0 for s in syms}
    wsum = sum(w.values())
    w = {s: v / wsum for s, v in w.items()}                       # normalise

    frames = {s: per_night_costs(s) for s in syms}
    mkt = pd.concat({s: f["cost_market"] for s, f in frames.items()}, axis=1)
    lim = pd.concat({s: f["cost_limit"] for s, f in frames.items()}, axis=1)
    swp = pd.concat({s: f["swap"] for s, f in frames.items()}, axis=1)
    # SMART per-leg (REALISTIC, decided upfront — no per-night foresight): overlay a leg
    # for the whole period iff its EXPECTED limit-overlay cost beats its swap; else hold
    # that leg every night and pay the swap (same universal spread-vs-swap rule, one level up).
    smart = pd.concat(
        {s: (lim[s] if lim[s].mean() < swp[s].mean() else swp[s]) for s in syms},
        axis=1,
    )

    def wsumcol(frame):
        return sum(w[s] * frame[s] for s in syms)

    port = pd.DataFrame({
        "cost_market": wsumcol(mkt),
        "cost_limit": wsumcol(lim),
        "cost_smart": wsumcol(smart),
        "swap": wsumcol(swp),
    }).dropna()
    return port, frames


def sharpe(mu_gross_bps: float, sigma_gross_bps: float, mu_cost_bps: float,
           sigma_cost_bps: float) -> float:
    net_mu = mu_gross_bps - mu_cost_bps
    net_sig = np.sqrt(sigma_gross_bps ** 2 + sigma_cost_bps ** 2)
    return net_mu / net_sig * np.sqrt(TRADING_NIGHTS)


def analyze(label: str, weights: dict[str, float] | None) -> None:
    port, frames = portfolio_cost_series(weights)
    sigma_gross_bps = TARGET_ANNUAL_VOL / np.sqrt(TRADING_NIGHTS) * 1e4   # daily vol in bps

    syms = list(weights.keys()) if weights else list(SYMBOLS)
    wnorm = {s: (weights[s] if weights else 1.0) for s in syms}
    tot = sum(wnorm.values())
    print(f"\n############ {label} ############")
    print("weights: " + ", ".join(f"{s}={wnorm[s]/tot:.3f}" for s in syms))
    print("=== Per-instrument execution cost (bps/night): MARKET vs LIMIT, + swap ===")
    print(f"{'sym':7s} {'mkt_mean':>9s} {'mkt_std':>8s} {'lim_mean':>9s} {'lim_std':>8s} {'save_mean':>9s} {'swap':>6s}")
    for s, f in frames.items():
        print(f"{s:7s} {f['cost_market'].mean():9.2f} {f['cost_market'].std():8.2f} "
              f"{f['cost_limit'].mean():9.2f} {f['cost_limit'].std():8.2f} "
              f"{(f['cost_market']-f['cost_limit']).mean():9.2f} {f['swap'].mean():6.2f}")

    print("\n=== PORTFOLIO nightly cost (bps) ===")
    for k in ("cost_market", "cost_limit", "cost_smart", "swap"):
        print(f"  {k:12s} mean={port[k].mean():6.3f}  std={port[k].std():6.3f}  "
              f"annualized_drag={port[k].mean()*TRADING_NIGHTS/1e2:6.2f}%/yr")

    mu_c_mkt, sd_c_mkt = port["cost_market"].mean(), port["cost_market"].std()
    mu_c_lim, sd_c_lim = port["cost_limit"].mean(), port["cost_limit"].std()
    mu_c_smt, sd_c_smt = port["cost_smart"].mean(), port["cost_smart"].std()
    mu_c_hold, sd_c_hold = port["swap"].mean(), port["swap"].std()

    print(f"\nPortfolio gross daily vol (15% target): {sigma_gross_bps:.1f} bps/day")
    print("\n=== Annualized SHARPE by execution policy, swept over gross Sharpe ===")
    print(f"{'grossSh':>7s} {'HOLD':>7s} {'MARKET':>7s} {'LIMIT_all':>9s} {'SMART':>7s} "
          f"{'d(LIM-MKT)':>11s} {'d(SMART-HOLD)':>14s} {'d(SMART-MKT)':>13s}")
    for gs in (0.5, 0.75, 1.0, 1.25, 1.5):
        # gs is the ANNUAL gross Sharpe → daily gross mean = gs * daily_vol / sqrt(252)
        mu_g = gs * sigma_gross_bps / np.sqrt(TRADING_NIGHTS)
        s_hold = sharpe(mu_g, sigma_gross_bps, mu_c_hold, sd_c_hold)
        s_mkt = sharpe(mu_g, sigma_gross_bps, mu_c_mkt, sd_c_mkt)
        s_lim = sharpe(mu_g, sigma_gross_bps, mu_c_lim, sd_c_lim)
        s_smt = sharpe(mu_g, sigma_gross_bps, mu_c_smt, sd_c_smt)
        print(f"{gs:7.2f} {s_hold:7.3f} {s_mkt:7.3f} {s_lim:9.3f} {s_smt:7.3f} "
              f"{s_lim - s_mkt:+11.3f} {s_smt - s_hold:+14.3f} {s_smt - s_mkt:+13.3f}")

    # Persist the portfolio cost series + a summary (label-tagged so runs don't clobber)
    tag = label.split()[0].lower()
    port.to_csv(OUT / f"E_portfolio_cost_series_{tag}.csv")
    summary = pd.DataFrame([
        {"policy": "HOLD_pay_swap", "mean_cost_bps": mu_c_hold, "std_cost_bps": sd_c_hold,
         "annualized_drag_pct": mu_c_hold * TRADING_NIGHTS / 1e2},
        {"policy": "MARKET_overlay", "mean_cost_bps": mu_c_mkt, "std_cost_bps": sd_c_mkt,
         "annualized_drag_pct": mu_c_mkt * TRADING_NIGHTS / 1e2},
        {"policy": "LIMIT_overlay", "mean_cost_bps": mu_c_lim, "std_cost_bps": sd_c_lim,
         "annualized_drag_pct": mu_c_lim * TRADING_NIGHTS / 1e2},
        {"policy": "SMART_overlay", "mean_cost_bps": mu_c_smt, "std_cost_bps": sd_c_smt,
         "annualized_drag_pct": mu_c_smt * TRADING_NIGHTS / 1e2},
    ])
    summary.to_csv(OUT / f"E_sharpe_impact_summary_{tag}.csv", index=False)


def main() -> None:
    analyze("EQUAL-weight 4-leg (SP500/NDX/XAUUSD/XAGUSD)", None)
    analyze("VAULT-weight (ES/NQ/GC at actual vault risk weights; SI not traded, CL excluded)",
            VAULT_WEIGHTS_RAW)


if __name__ == "__main__":
    main()
