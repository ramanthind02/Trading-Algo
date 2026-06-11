"""Equity curves for Peter's volatility-breakout model (R-account, non-compounded, fixed risk/trade).
Usage: python -m research.experiments.concretum_intraday_momentum.plot_peter"""
from __future__ import annotations

import functools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import peter_breakout as P
from research.experiments.concretum_intraday_momentum import engine as E

OUT = "research/experiments/concretum_intraday_momentum/outputs"
POST = pd.Timestamp("2024-05-01")
RISK = 0.0033  # per-R account risk (the original post's 0.33%/trade); Sharpe-invariant

_orig = E.load_sessions
_cache = functools.lru_cache(maxsize=64)(lambda s, a, b: _orig(s, a, b))
E.load_sessions = lambda s, a, b: _cache(s, a, b)


def _sh(r):
    r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def _eq(res):
    r = np.where(np.isfinite(res["R"].to_numpy()), res["R"].to_numpy(), 0.0)
    return res.index, np.cumsum(r * RISK), _sh(res["R"].to_numpy())


def main():
    bk = ["SPY", "QQQ", "IWM", "GLD", "DIA"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- NDX single market, cost ladder ---
    for bps, color, lw in [(0.0, "C7", 1.4), (1.0, "C0", 1.9), (2.0, "C3", 1.4)]:
        idx, eq, sh = _eq(P.simulate("NDX", 2018, 2026, P.PParams(cost_bps_oneway=bps)))
        ax1.plot(idx, 100 * eq, color=color, linewidth=lw,
                 label=f"NDX @{bps:g}bps/side  (Sharpe {sh:.2f})")
    ax1.axvline(POST, color="grey", linestyle="--", linewidth=1)
    ax1.text(POST, ax1.get_ylim()[1], "  post-pub", color="grey", va="top", fontsize=8)
    ax1.set_title("Peter volatility-breakout — NDX single market (R-account, 0.33% risk/trade, non-compounded)")
    ax1.set_ylabel("cumulative account return (%)")
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(True, alpha=0.25)

    # --- Basket, futures-like (0.5bps) vs ETF-like (1bps) ---
    for bps, color, lw in [(0.0, "C7", 1.4), (0.5, "C2", 1.9), (1.0, "C1", 1.5)]:
        idx, eq, sh = _eq(P.simulate_basket(bk, 2010, 2026, P.PParams(cost_bps_oneway=bps)))
        ax2.plot(idx, 100 * eq, color=color, linewidth=lw,
                 label=f"basket @{bps:g}bps/side  (Sharpe {sh:.2f})")
    ax2.axvline(POST, color="grey", linestyle="--", linewidth=1)
    ax2.text(POST, ax2.get_ylim()[1], "  post-pub", color="grey", va="top", fontsize=8)
    ax2.set_title("Peter volatility-breakout — 5-ETF basket SPY/QQQ/IWM/GLD/DIA (summed R, fixed risk/trade)")
    ax2.set_ylabel("cumulative account return (%)"); ax2.set_xlabel("date")
    ax2.legend(loc="upper left", fontsize=9); ax2.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(f"{OUT}/peter_breakout_equity.png", dpi=130)
    print(f"  wrote {OUT}/peter_breakout_equity.png")


if __name__ == "__main__":
    main()
