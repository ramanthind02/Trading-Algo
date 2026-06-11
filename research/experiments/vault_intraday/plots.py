"""Plots for the vault-intraday screen (matplotlib, per repo visualization policy).

  1. plateau_heatmaps.png  — per family: net@1x Sharpe over (instrument x TF) rows x param-axis
     cols. Shows WHERE the edge lives and whether it is a stable plateau or an isolated spike.
  2. equity_vs_buyhold.png — for the breakout_long headline survivors: cumulative log-return of
     the strategy vs buy-hold. Makes the "de-risking, not return-generation" story visible
     (lower vol, but underperforms buy-hold's cumulative dollars).

Run: .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.experiments.vault_intraday import data_io, engine
from lib.core.enums import Ticker

OUT = Path(__file__).resolve().parent / "outputs"
TFS = ["H4", "H2", "H1", "M30"]
INSTR_TICKER = {"SP500": Ticker.ES, "NDX": Ticker.NQ, "XTIUSD": Ticker.CL,
                "XAUUSD": Ticker.GC, "XAGUSD": Ticker.SI, "TLT": Ticker.ES}


def plateau_heatmaps() -> None:
    df = pd.read_csv(OUT / "screen_annotated.csv")
    fams = list(df.groupby(["family", "direction"]).groups.keys())
    n = len(fams)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n))
    if n == 1:
        axes = [axes]
    for ax, (fam, direction) in zip(axes, fams):
        g = df[(df["family"] == fam) & (df["direction"] == direction)]
        piv = g.pivot_table(index=["instrument", "tf"], columns="axis_value", values="net_sharpe_1.0x")
        # order rows instrument-major, TF in canonical order
        piv = piv.reindex(sorted(piv.index, key=lambda x: (x[0], TFS.index(x[1]) if x[1] in TFS else 9)))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.2)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i}/{t}" for i, t in piv.index], fontsize=7)
        ax.set_title(f"{fam} ({direction}) — net@1x Sharpe   [x={g['axis'].iloc[0]}]", fontsize=9)
        for (r, c), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.015)
    fig.suptitle("Intraday net@1x Sharpe plateau by family (green=edge, red=none)", y=1.001)
    fig.tight_layout()
    fig.savefig(OUT / "plateau_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'plateau_heatmaps.png'}")


def _strategy_equity(module, ticker, params, direction, bars):
    sigs = engine.build_signals_fanout(module, ticker, [params], bars)
    raw = list(sigs.values())[0]
    sig = engine.project(raw, direction)
    close = pd.Series(bars["close"].to_numpy(), index=pd.DatetimeIndex(bars["datetime"]))
    ret = np.log(close).diff()
    pos = sig.shift(1)
    strat = (pos * ret).fillna(0.0).cumsum()
    bh = ret.fillna(0.0).cumsum()
    return strat, bh


def equity_vs_buyhold() -> None:
    # breakout_long headline survivors on their plateau instruments
    targets = [
        ("XAUUSD", "H4", 80), ("NDX", "M30", 80), ("SP500", "H1", 10),
    ]
    base = {"ema_period": 126, "atr_mult": 1.5, "atr_len": 14, "atr_vol_filter": 0,
            "cooldown_bars": 0, "strategy_mode": "long"}
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
    for ax, (sym, tf, lb) in zip(axes, targets):
        bars = data_io.load_bars(sym, tf)
        strat, bh = _strategy_equity("robust_trend_breakout", INSTR_TICKER[sym],
                                     {**base, "lookback": lb}, "long", bars)
        ax.plot(strat.index, strat.values, label="breakout_long", lw=1.2)
        ax.plot(bh.index, bh.values, label="buy & hold", lw=1.0, alpha=0.8)
        ax.set_title(f"{sym} {tf} lookback={lb}", fontsize=10)
        ax.set_ylabel("cumulative log-return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("breakout_long vs buy-hold: lower vol / shallower DD, but trails cumulative return", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "equity_vs_buyhold.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'equity_vs_buyhold.png'}")


if __name__ == "__main__":
    plateau_heatmaps()
    equity_vs_buyhold()
