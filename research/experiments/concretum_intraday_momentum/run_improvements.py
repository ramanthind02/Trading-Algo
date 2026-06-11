"""Evaluate the 3 frictionless improvements to the Noise-Area model across instruments.
Usage: python -m research.experiments.concretum_intraday_momentum.run_improvements
Writes outputs/imp_*.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import improvements as I
from research.experiments.concretum_intraday_momentum import run_matrix as RM

OUT = RM.OUT
PAPER_END, POST_START = RM.PAPER_END, RM.POST_START
SYMS = ["SPY", "QQQ", "SP500", "NDX"]


def overlay_panel():
    rows = []
    variants = [("baseline", I.IParams()),
                ("entry_only", I.IParams(overlay_entry=True)),
                ("both", I.IParams(overlay_entry=True, overlay_exit=True))]
    wins = [("full", None, None), ("paper", None, PAPER_END), ("post", POST_START, None)]
    for s in SYMS:
        for label, p in variants:
            res = I.simulate(s, RM.DATA_START[s], 2026, p)
            for wl, lo, hi in wins:
                for gn in ["gross", "net"]:
                    m = I.metrics(res, gn, lo, hi)
                    m.update(sym=s, variant=label, window=wl, pnl=gn)
                    rows.append(m)
    pd.DataFrame(rows).to_csv(OUT / "imp_overlay.csv", index=False)
    print("  imp_overlay")


def eqfilter_sweep():
    rows = []
    for s in ["NDX", "SPY"]:
        for base_label, base in [("baseline", {}), ("overlay", dict(overlay_entry=True))]:
            for w in [0, 20, 50, 100, 200]:
                res = I.simulate(s, RM.DATA_START[s], 2026, I.IParams(eq_filter_window=w, **base))
                m = I.metrics(res, "net")
                rows.append({"sym": s, "base": base_label, "eq_window": w,
                             "net_sharpe": m["sharpe"], "maxDD%": m["maxDD%"], "vol%": m["vol%"]})
    pd.DataFrame(rows).to_csv(OUT / "imp_eqfilter.csv", index=False)
    print("  imp_eqfilter")


def freq_cost():
    rows = []
    for s in ["NDX", "SPY"]:
        for fl, marks in [("semi", tuple(E.MARKS_ET_SEMI)), ("hourly", tuple(E.MARKS_ET_HOUR))]:
            for bps in [0.5, 1.0, 2.0, 3.0]:
                res = I.simulate(s, RM.DATA_START[s], 2026,
                                 I.IParams(marks_et=marks, overlay_entry=True, cost_bps_oneway=bps))
                rows.append({"sym": s, "freq": fl, "oneway_bps": bps, "net_sharpe": I.metrics(res, "net")["sharpe"]})
    pd.DataFrame(rows).to_csv(OUT / "imp_freqcost.csv", index=False)
    print("  imp_freqcost")


def cooldown_sweep():
    rows = []
    for s in ["NDX", "SPY"]:
        for cd in [0, 1, 2, 3]:
            res = I.simulate(s, RM.DATA_START[s], 2026, I.IParams(overlay_entry=True, cooldown=cd))
            rows.append({"sym": s, "cooldown": cd, "net_sharpe": I.metrics(res, "net")["sharpe"],
                         "turn/day": I.metrics(res, "net")["turn/day"]})
    pd.DataFrame(rows).to_csv(OUT / "imp_cooldown.csv", index=False)
    print("  imp_cooldown")


def main():
    overlay_panel()
    eqfilter_sweep()
    freq_cost()
    cooldown_sweep()


if __name__ == "__main__":
    main()
