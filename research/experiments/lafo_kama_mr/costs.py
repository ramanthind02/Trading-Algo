"""Phase-1 REAL-COST run: charge the verified recorded Darwinex NDX cost and a
slippage sweep on the surviving long-only deep-dip revert configs.

Cost model (all RECORDED / live-probed, see FINDINGS):
  * round-trip spread = per-entry-bar recorded M1 `spread` (MT5 points) x POINT(0.1)
    -> ~0.8-0.9 idx pts RTH; charged once per round trip (= cross half each leg).
  * slippage swept 0..5 idx pts PER SIDE (market buy on the dip / market sell on revert).
    The EA assumed 2 ticks = 0.2 pt (optimistic); we bracket realistic 0.5-3 pt.
  * commission 0; RTH-only -> no overnight swap.

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.costs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E

SYM, Y0, Y1 = "NDX", 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"

CANDIDATES = {
    "M5  thr2.0% revert":  dict(tf="M5",  threshold=0.020, entry_mode="rel"),
    "M5  thr1.75% revert": dict(tf="M5",  threshold=0.0175, entry_mode="rel"),
    "M15 thr2.5% revert":  dict(tf="M15", threshold=0.025, entry_mode="rel"),
    "M15 thr2.25% revert": dict(tf="M15", threshold=0.0225, entry_mode="rel"),
    "M5  z3.5 lb200 revert": dict(tf="M5", entry_mode="z", z_thr=3.5, z_lookback=200),
}
BASE = dict(session="rth", exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False)


def _net_sharpe_and_cost(name, kw, slip):
    p = E.Params(**{**BASE, **kw, "use_recorded_spread": True, "slip_pts": slip})
    m, tr = E.run(SYM, Y0, Y1, p, name)
    if tr.empty:
        return None
    avg_cost = tr["cost_pts"].mean()
    avg_gross = tr["gross_pts"].mean()
    return m, avg_cost, avg_gross, tr


def slippage_sweep() -> pd.DataFrame:
    rows = []
    for name, kw in CANDIDATES.items():
        # frictionless reference
        pg = E.Params(**{**BASE, **kw})
        mg, _ = E.run(SYM, Y0, Y1, pg, name)
        gross_sh = mg["sharpe"]
        for slip in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            res = _net_sharpe_and_cost(name, kw, slip)
            if res is None:
                continue
            m, avg_cost, avg_gross, _ = res
            rows.append({
                "config": name, "slip_pts": slip, "gross_sharpe": gross_sh,
                "net_sharpe": m["sharpe"], "net_ann%": m["ann_ret%"], "maxDD%": m["maxDD%"],
                "trades": m["trades"], "tr/yr": m["tr/yr"], "win%": m["win%"], "PF": m["PF"],
                "avg_cost_pts": round(avg_cost, 2), "avg_gross_pts": round(avg_gross, 1),
                "cost/gross%": round(100 * avg_cost / abs(avg_gross), 1) if avg_gross else np.nan,
            })
    return pd.DataFrame(rows)


def sizing_compare(slip=1.0) -> pd.DataFrame:
    """fixed 1-unit vs per-day vol-target sizing at a realistic 1pt/side slippage."""
    rows = []
    for name, kw in CANDIDATES.items():
        for sizing in ("fixed", "vol_target"):
            p = E.Params(**{**BASE, **kw, "use_recorded_spread": True, "slip_pts": slip,
                            "sizing": sizing, "vol_target": 0.015, "lev_cap": 4.0})
            m, _ = E.run(SYM, Y0, Y1, p, f"{name} [{sizing}]")
            rows.append({"config": name, "sizing": sizing, "net_sharpe": m["sharpe"],
                         "net_ann%": m["ann_ret%"], "vol%": m["vol%"], "maxDD%": m["maxDD%"],
                         "trades": m["trades"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

    print("=" * 100)
    print("PHASE 1 — REAL recorded Darwinex spread + slippage sweep (long-only deep-dip revert)")
    print("=" * 100)
    s = slippage_sweep()
    s.to_csv(f"{OUT}/cost_slippage_sweep.csv", index=False)
    for name in CANDIDATES:
        sub = s[s["config"] == name]
        print(f"\n--- {name} ---  (gross Sharpe {sub['gross_sharpe'].iloc[0]:.2f})")
        print(sub[["slip_pts", "net_sharpe", "net_ann%", "maxDD%", "trades", "tr/yr",
                   "win%", "PF", "avg_cost_pts", "avg_gross_pts", "cost/gross%"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("SIZING — fixed vs vol-target @ 1pt/side slippage + recorded spread")
    print("=" * 100)
    z = sizing_compare(slip=1.0)
    z.to_csv(f"{OUT}/cost_sizing_compare.csv", index=False)
    print(z.to_string(index=False))
