"""
Can we add tickers to make an ensemble? The question is DIVERSIFICATION, not
"which names are positive". JPY crosses share the yen leg -> they may be ~one bet
(and co-decayed in 2026). Metals are the candidate orthogonal diversifier.

Candidate basket selected by STRUCTURAL RATIONALE (Tokyo-session names where the
03:00-06:00 EET range -> London/NY breakout edge can exist), NOT by realized Sharpe:
  JPY crosses: USDJPY GBPJPY AUDJPY NZDJPY CADJPY CHFJPY   (EURJPY lacks 2018+ M1)
  Metals:      XAUUSD XAGUSD
USD majors are excluded on PRIOR (full-history avg_R ~0 in this window) — not survivorship.

Each name trades <=1/day in R-units (constant risk), so an equal-RISK ensemble is the
mean of the per-name daily-R series. Diversification benefit = ensemble Sharpe vs the
average single-name Sharpe, governed by the pairwise R-correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.gold_digger_breakout.poc import load_m1, prep_days, build_trades

JPY = ["USDJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
METAL = ["XAUUSD", "XAGUSD"]
POINT = {**{s: 0.001 for s in JPY}, "XAUUSD": 0.01, "XAGUSD": 0.001}
SLIP_MULT = 1.0  # round-trip cost = recorded_spread * (1 + 2*SLIP_MULT)


def name_daily(sym: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return (gross daily R, net daily R, trades) for one ticker."""
    df = load_m1(sym)
    days = prep_days(df)
    tr = build_trades(days, 3, 6, 18, gap_fill=True)
    idx = pd.Index(np.unique(df["day"].values))
    g = pd.Series(0.0, index=idx)
    gg = tr.groupby("day")["R"].sum()
    g.loc[gg.index] = gg.values
    # net: per-trade cost = recorded entry spread (price) * (1 + 2*slip), in R
    cost_price = tr["entry_sp"].values * POINT[sym] * (1.0 + 2.0 * SLIP_MULT)
    trn = tr.copy()
    trn["Rn"] = trn["R"].values - cost_price / trn["width"].values
    n = pd.Series(0.0, index=idx)
    nn = trn.groupby("day")["Rn"].sum()
    n.loc[nn.index] = nn.values
    return g, n, tr


def sharpe(s: pd.Series) -> float:
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0


def maxdd(s: pd.Series) -> float:
    eq = s.cumsum().values
    return float((eq - np.maximum.accumulate(eq)).min())


def by_year(s: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(s.index)
    return s.groupby(idx.year).sum().round(1)


def ann_year_table(daily: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: by_year(v) for k, v in daily.items()})


def main():
    pd.set_option("display.width", 220)
    syms = JPY + METAL
    gross, net, trades = {}, {}, {}
    for s in syms:
        g, n, tr = name_daily(s)
        gross[s], net[s], trades[s] = g, n, tr

    # align on common index
    G = pd.DataFrame(gross).dropna(how="all").fillna(0.0)
    N = pd.DataFrame(net).dropna(how="all").fillna(0.0)

    print("=== single-name (gross / net) ===")
    rows = []
    for s in syms:
        rows.append(dict(sym=s, grp=("JPY" if s in JPY else "metal"),
                         n_tr=len(trades[s]),
                         gross_Sharpe=round(sharpe(G[s]), 2), gross_totR=round(G[s].sum(), 0),
                         net_Sharpe=round(sharpe(N[s]), 2), net_totR=round(N[s].sum(), 0),
                         maxDD_R=round(maxdd(N[s]), 0)))
    print(pd.DataFrame(rows).to_string(index=False))

    # correlation of gross daily R, only over days where at least one traded (avoid 0-0 inflation)
    active = (G != 0).any(axis=1)
    C = G[active].corr()
    print("\n=== pairwise correlation of daily R (gross) ===")
    print(C.round(2).to_string())

    def ens(cols, series=G):
        port = series[cols].mean(axis=1)
        return port

    def report(name, cols, series, label):
        port = ens(cols, series)
        sub = C.loc[cols, cols].values
        mc = (sub.sum() - len(cols)) / (len(cols) * (len(cols) - 1))  # avg off-diag
        avg_single = np.mean([sharpe(series[c]) for c in cols])
        print(f"\n  [{label}] {name}: K={len(cols)} avg_pair_corr={mc:.2f} "
              f"single_Sharpe_avg={avg_single:.2f} -> ENSEMBLE Sharpe={sharpe(port):.2f} "
              f"(divers.ratio {sharpe(port)/avg_single:.2f}x)  totR={port.sum():.0f} maxDD={maxdd(port):.1f}")
        return port

    print("\n=== ensembles (equal-risk = mean of daily R) ===")
    print("  -- GROSS --")
    report("JPY-only", JPY, G, "gross")
    report("metals-only", METAL, G, "gross")
    p_all_g = report("JPY+metals (ALL 8)", syms, G, "gross")
    print("  -- NET (recorded spread + slippage) --")
    report("JPY-only", JPY, N, "net")
    report("metals-only", METAL, N, "net")
    p_all_n = report("JPY+metals (ALL 8)", syms, N, "net")
    SURV = ["USDJPY", "XAUUSD"]  # the only names whose gross edge clears its spread
    report("COST-SURVIVORS {USDJPY,XAUUSD}", SURV, N, "net")
    print("\n  cost-survivor pair by-year (net):")
    print(ann_year_table({"USDJPY": N["USDJPY"], "XAUUSD": N["XAUUSD"],
                          "ENS_SURV": ens(SURV, N)}).to_string())

    print("\n=== by-year total R ===")
    yt = ann_year_table({**{s: N[s] for s in syms},
                         "ENS_JPY": ens(JPY, N), "ENS_METAL": ens(METAL, N), "ENS_ALL": ens(syms, N)})
    print(yt.round(1).to_string())

    # jackknife: leave-one-out ensemble Sharpe (net) -> is any single name load-bearing?
    print("\n=== jackknife: leave-one-out ENS_ALL net Sharpe ===")
    base = sharpe(ens(syms, N))
    jk = []
    for s in syms:
        rest = [x for x in syms if x != s]
        jk.append(dict(dropped=s, ens_Sharpe=round(sharpe(ens(rest, N)), 2), delta=round(sharpe(ens(rest, N)) - base, 2)))
    print(f"  full ENS_ALL net Sharpe = {base:.2f}")
    print(pd.DataFrame(jk).to_string(index=False))

    # plot ensemble vs USDJPY alone, highlight 2026
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, lab, lw in [(N["USDJPY"], "USDJPY alone (net)", 1.1),
                         (ens(syms, N), "naive ALL-8 ensemble (net)", 1.1),
                         (ens(["USDJPY", "XAUUSD"], N), "{USDJPY,XAUUSD} cost-survivors (net)", 1.8)]:
        ax.plot(col.index, col.cumsum().values, lw=lw, label=lab)
    ax.axvspan(pd.Timestamp("2026-01-01"), N.index.max(), color="orange", alpha=.15, label="2026")
    ax.axhline(0, color="k", lw=.5); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Asian-range breakout: single name vs ensembles (net, cum R)")
    plt.tight_layout()
    plt.savefig("research/experiments/gold_digger_breakout/outputs/ensemble.png", dpi=110)
    print("\nwrote outputs/ensemble.png")


if __name__ == "__main__":
    main()
