"""Phase-0 FRICTIONLESS sweeps for the KAMA fair-value MR idea on NDX.

Question: does ANY configuration of the core idea (KAMA deviation -> MR entry) have
a real frictionless edge, before we spend anything on cost/realism? We sweep the
levers the user flagged: exit method (the EA's tiny fixed target is suspect), RR,
entry threshold, entry mode (fixed-% vs vol-normalised z), direction, stop width.

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.sweeps
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E

SYM, Y0, Y1 = "NDX", 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"


def _rows(grid, base_kw, label_fn):
    rows = []
    for combo in grid:
        kw = {**base_kw, **combo}
        p = E.Params(**kw)
        m, _ = E.run(SYM, Y0, Y1, p, label_fn(combo))
        m.update(combo)
        rows.append(m)
    return rows


def sweep_exit_rr_dir() -> pd.DataFrame:
    """Exit method x RR x direction, fixed thr=0.5%, M5 RTH, frictionless."""
    rows = []
    dirs = {"both": dict(longs=True, shorts=True),
            "long": dict(longs=True, shorts=False),
            "short": dict(longs=False, shorts=True)}
    # bracket at several RR
    for dname, dkw in dirs.items():
        for rr in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            p = E.Params(tf="M5", session="rth", entry_mode="rel", threshold=0.005,
                         exit_mode="bracket", stop_atr_mult=3.5, target_rr=rr, **dkw)
            m, _ = E.run(SYM, Y0, Y1, p, f"bracket RR={rr} {dname}")
            m.update(dict(exit="bracket", rr=rr, dir=dname)); rows.append(m)
        # revert + close exits (no fixed target)
        for ex in ("revert", "close"):
            p = E.Params(tf="M5", session="rth", entry_mode="rel", threshold=0.005,
                         exit_mode=ex, stop_atr_mult=3.5, **dkw)
            m, _ = E.run(SYM, Y0, Y1, p, f"{ex} {dname}")
            m.update(dict(exit=ex, rr=np.nan, dir=dname)); rows.append(m)
    return pd.DataFrame(rows)


def sweep_threshold(exit_mode: str, rr: float) -> pd.DataFrame:
    """Entry threshold plateau for a chosen exit, both directions, M5 RTH."""
    rows = []
    for thr in (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02):
        kw = dict(tf="M5", session="rth", entry_mode="rel", threshold=thr,
                  exit_mode=exit_mode, stop_atr_mult=3.5)
        if exit_mode == "bracket":
            kw["target_rr"] = rr
        for dname, dkw in (("both", dict()), ("long", dict(shorts=False)),
                           ("short", dict(longs=False))):
            p = E.Params(**{**kw, **dkw})
            m, _ = E.run(SYM, Y0, Y1, p, f"thr={thr} {dname}")
            m.update(dict(thr=thr, dir=dname)); rows.append(m)
    return pd.DataFrame(rows)


def sweep_zscore(exit_mode: str, rr: float) -> pd.DataFrame:
    """Vol-normalised z(delta) entry: z threshold x lookback x direction."""
    rows = []
    for zthr, zlb in itertools.product((1.0, 1.5, 2.0, 2.5, 3.0), (50, 100, 200)):
        kw = dict(tf="M5", session="rth", entry_mode="z", z_thr=zthr, z_lookback=zlb,
                  exit_mode=exit_mode, stop_atr_mult=3.5)
        if exit_mode == "bracket":
            kw["target_rr"] = rr
        for dname, dkw in (("both", dict()), ("long", dict(shorts=False)),
                           ("short", dict(longs=False))):
            p = E.Params(**{**kw, **dkw})
            m, _ = E.run(SYM, Y0, Y1, p, f"z={zthr} lb={zlb} {dname}")
            m.update(dict(zthr=zthr, zlb=zlb, dir=dname)); rows.append(m)
    return pd.DataFrame(rows)


def _show(df: pd.DataFrame, by="sharpe", cols=None, n=20):
    cols = cols or ["sharpe", "ann_ret%", "vol%", "maxDD%", "trades", "tr/yr",
                    "win%", "avgR", "PF", "avg_hold", "long%", "tp%", "sl%", "rev%", "eod%"]
    keep = [c for c in df.columns if c not in cols and c != "cfg"]
    return df.sort_values(by, ascending=False)[keep + cols].head(n).to_string(index=False)


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)

    print("=" * 100)
    print("SWEEP 1 — exit method x RR x direction  (thr=0.5%, M5 RTH, FRICTIONLESS)")
    print("=" * 100)
    s1 = sweep_exit_rr_dir()
    s1.to_csv(f"{OUT}/sweep1_exit_rr_dir.csv", index=False)
    print(_show(s1, n=30))

    print("\n" + "=" * 100)
    print("SWEEP 2 — entry-threshold plateau, revert exit, by direction (M5 RTH, FRICTIONLESS)")
    print("=" * 100)
    s2 = sweep_threshold("revert", rr=np.nan)
    s2.to_csv(f"{OUT}/sweep2_threshold_revert.csv", index=False)
    print(_show(s2, n=30))

    print("\n" + "=" * 100)
    print("SWEEP 3 — vol-normalised z(delta) entry, revert exit (M5 RTH, FRICTIONLESS)")
    print("=" * 100)
    s3 = sweep_zscore("revert", rr=np.nan)
    s3.to_csv(f"{OUT}/sweep3_zscore_revert.csv", index=False)
    print(_show(s3, n=30))
