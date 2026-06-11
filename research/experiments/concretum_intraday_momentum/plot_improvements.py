"""NDX equity curve: Noise-Area baseline vs the fast-alpha entry overlay (#1), net @1bps.
Usage: python -m research.experiments.concretum_intraday_momentum.plot_improvements"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import improvements as I

OUT = "research/experiments/concretum_intraday_momentum/outputs"
POST = pd.Timestamp("2024-05-01")


def _sh(r):
    r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def main():
    from research.experiments.concretum_intraday_momentum import engine as E
    sym = "NDX"
    base = I.simulate(sym, 2018, 2026, I.IParams())
    ovl = I.simulate(sym, 2018, 2026, I.IParams(overlay_entry=True, overlay_exit=True))
    ovh = I.simulate(sym, 2018, 2026, I.IParams(overlay_entry=True, overlay_exit=True,
                                                marks_et=tuple(E.MARKS_ET_HOUR)))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, res, color, lw in [
        ("Noise-Area baseline (net @1bps)", base, "C7", 1.5),
        ("+ fast-alpha overlay (both legs)", ovl, "C0", 1.9),
        ("+ overlay + hourly marks", ovh, "C2", 1.7),
    ]:
        r = np.where(np.isfinite(res["net"].to_numpy()), res["net"].to_numpy(), 0.0)
        ax.plot(res.index, np.cumprod(1 + r), label=f"{label}  (Sharpe {_sh(res['net'].to_numpy()):.2f})",
                color=color, linewidth=lw)
    ax.axvline(POST, color="grey", linestyle="--", linewidth=1)
    ax.text(POST, ax.get_ylim()[1], "  post-pub", color="grey", va="top", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("compounded equity (start = 1.0, log)")
    ax.set_title("NDX — Noise-Area momentum: fast-alpha entry overlay vs baseline (net @1bps, 5-min grid)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/ndx_equity_overlay_improvement.png", dpi=130)
    print("  wrote ndx_equity_overlay_improvement.png")


if __name__ == "__main__":
    main()
