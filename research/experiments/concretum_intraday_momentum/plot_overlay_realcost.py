"""NDX Noise-Area + fast-alpha overlay at REAL CFD cost (~0.25 bps/side): long-only vs both-sides.
Usage: python -m research.experiments.concretum_intraday_momentum.plot_overlay_realcost"""
from __future__ import annotations

import functools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import improvements as I

OUT = "research/experiments/concretum_intraday_momentum/outputs"
POST = pd.Timestamp("2024-05-01")
RC = 0.25  # bps/side ~ real CFD cost

_orig = E.load_sessions
_c = functools.lru_cache(maxsize=16)(lambda s, a, b: _orig(s, a, b))
E.load_sessions = lambda s, a, b: _c(s, a, b)


def _sh(r):
    r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def main():
    sym = "NDX"
    runs = [
        ("long-only baseline", I.IParams(long_only=True, cost_bps_oneway=RC), "C7", 1.5),
        ("long-only + overlay", I.IParams(long_only=True, overlay_entry=True, overlay_exit=True,
                                          cost_bps_oneway=RC), "C0", 2.0),
        ("both-sides + overlay (ref)", I.IParams(overlay_entry=True, overlay_exit=True,
                                                 cost_bps_oneway=RC), "C2", 1.4),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, p, color, lw in runs:
        res = I.simulate(sym, 2018, 2026, p)
        r = np.where(np.isfinite(res["net"].to_numpy()), res["net"].to_numpy(), 0.0)
        m = I.metrics(res, "net")
        ax.plot(res.index, np.cumprod(1 + r), color=color, linewidth=lw,
                label=f"{label}  (Sharpe {_sh(res['net'].to_numpy()):.2f}, maxDD {m.get('maxDD%',0):.0f}%)")
    ax.axvline(POST, color="grey", linestyle="--", linewidth=1)
    ax.text(POST, ax.get_ylim()[1], "  post-pub", color="grey", va="top", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("compounded equity (start = 1.0, log)")
    ax.set_title(f"NDX — Noise-Area + fast-alpha overlay @ real CFD cost ({RC} bps/side): long-only vs both-sides")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/ndx_overlay_longonly_realcost.png", dpi=130)
    print(f"  wrote {OUT}/ndx_overlay_longonly_realcost.png")


if __name__ == "__main__":
    main()
