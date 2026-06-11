"""Equity curves for NDX (2018+) — Part 1 (Noise-Area momentum) and Part 2 (fast-alpha overlay).
Compounded equity (start=1.0) on a log axis; Sharpe (daily series, sqrt(252)) in the legend; the
2024-05 post-publication boundary marked. Usage: python -m ...concretum_intraday_momentum.plot_ndx
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import fast_alpha as FA

OUT = "research/experiments/concretum_intraday_momentum/outputs"
SYM, Y0, Y1 = "NDX", 2018, 2026
POST = pd.Timestamp("2024-05-01")


def _sharpe(r):
    r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def _eq(series):
    r = series.to_numpy()
    r = np.where(np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + r)


def plot(curves, title, fname, idx):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, r, color, lw in curves:
        sh = _sharpe(r.to_numpy())
        ax.plot(idx, _eq(r), label=f"{label}  (Sharpe {sh:.2f})", color=color, linewidth=lw)
    ax.axvline(POST, color="grey", linestyle="--", linewidth=1)
    ax.text(POST, ax.get_ylim()[1], "  paper ends / post-pub", color="grey",
            va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("compounded equity (start = 1.0, log scale)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")


def main():
    # ---- Part 1: Noise-Area momentum ----
    win = E.load_sessions(SYM, Y0, Y1)
    dt = E.build_day_table(win, E.MARKS_ET_SEMI)
    nmk = len(E.MARKS_ET_SEMI)
    g = E.simulate(dt, E.Params(stop_mode="curr_vwap", sizing="vol_target"), nmk)
    n = E.simulate(dt, E.Params(stop_mode="curr_vwap", sizing="vol_target", cost_bps_per_turn=1.0), nmk)
    b = E.simulate(dt, E.Params(stop_mode="curr_vwap", sizing="binary"), nmk)
    gv = g[g.valid]; idx1 = gv.index
    curves1 = [
        ("vol-target gross", gv["lev_ret"], "C0", 1.8),
        ("vol-target net @1bps", n[n.valid]["net_lev"], "C3", 1.8),
        ("binary curr+VWAP (lev=1, invariant)", b[b.valid]["unlev_ret"], "C2", 1.3),
    ]
    plot(curves1, "NDX — Part 1: Noise-Area intraday momentum (headline curr+VWAP)",
         "ndx_equity_part1_momentum.png", idx1)

    # ---- Part 2: fast-alpha overlay (net @1bps) ----
    b5, d5 = FA.load_5m(SYM, Y0, Y1)
    def run(**kw):
        return FA.simulate_bo(b5, d5, FA.BParams(sizing="vol_target", cost_bps_oneway=1.0, **kw))
    base = run(); ovl = run(overlay_entry=True, overlay_exit=True)
    ent = run(overlay_entry=True); plac = run(overlay_entry=True, overlay_exit=True, placebo=True)
    idx2 = base.index
    curves2 = [
        ("baseline breakout", base["net"], "C7", 1.5),
        ("overlay both", ovl["net"], "C0", 1.8),
        ("overlay entry-only", ent["net"], "C2", 1.5),
        ("placebo (same-dir wait)", plac["net"], "C3", 1.2),
    ]
    plot(curves2, "NDX — Part 2: fast-alpha execution overlay (ATR breakout, net @1bps)",
         "ndx_equity_part2_overlay.png", idx2)


if __name__ == "__main__":
    main()
