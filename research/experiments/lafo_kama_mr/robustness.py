"""Phase-0/1 robustness: KAMA-param plateau, SP500 generalization, 24h session,
train/test split — does the long-only deep-dip revert edge hold off the chosen cell?

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.robustness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E

Y0, Y1 = 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"

WIN_M15 = dict(tf="M15", session="rth", entry_mode="rel", threshold=0.025,
               exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False)
WIN_M5 = dict(tf="M5", session="rth", entry_mode="rel", threshold=0.020,
              exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False)


def kama_param_plateau() -> pd.DataFrame:
    rows = []
    for period in (10, 14, 20, 30, 40):
        for fast in (3, 5, 8):
            for slow in (20, 30, 50):
                p = E.Params(**{**WIN_M15, "kama_period": period, "kama_fast": fast,
                                "kama_slow": slow})
                m, _ = E.run("NDX", Y0, Y1, p)
                # net at 1pt slip
                pn = E.Params(**{**WIN_M15, "kama_period": period, "kama_fast": fast,
                                 "kama_slow": slow, "use_recorded_spread": True, "slip_pts": 1.0})
                mn, _ = E.run("NDX", Y0, Y1, pn)
                rows.append({"period": period, "fast": fast, "slow": slow,
                             "gross_sharpe": m["sharpe"], "net_sharpe": mn["sharpe"],
                             "trades": m["trades"], "ann%": mn["ann_ret%"], "maxDD%": mn["maxDD%"]})
    return pd.DataFrame(rows)


def sp500_generalization() -> pd.DataFrame:
    """Same long-only deep-dip revert mechanism on SP500 (frictionless), thr sweep.

    A 2.5% intraday dip is rarer on SP500 (lower vol) so we sweep the threshold; the
    test is whether the long-only deep-dip plateau is POSITIVE and rises with depth,
    i.e. the MECHANISM generalizes off NDX. Frictionless (POINT-independent)."""
    rows = []
    for sym in ("SP500", "NDX"):
        for tf in ("M5", "M15"):
            for thr in (0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025):
                p = E.Params(tf=tf, session="rth", entry_mode="rel", threshold=thr,
                             exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False)
                m, _ = E.run(sym, Y0, Y1, p)
                rows.append({"sym": sym, "tf": tf, "thr": thr, "sharpe": m["sharpe"],
                             "ann%": m["ann_ret%"], "maxDD%": m["maxDD%"], "trades": m["trades"],
                             "tr/yr": m["tr/yr"], "win%": m["win%"], "PF": m["PF"]})
    return pd.DataFrame(rows)


def session_compare() -> pd.DataFrame:
    rows = []
    for name, kw in (("M15 2.5%", WIN_M15), ("M5 2.0%", WIN_M5)):
        for sess in ("rth", "h24"):
            p = E.Params(**{**kw, "session": sess, "use_recorded_spread": True, "slip_pts": 1.0})
            m, _ = E.run("NDX", Y0, Y1, p)
            rows.append({"config": name, "session": sess, "net_sharpe": m["sharpe"],
                         "ann%": m["ann_ret%"], "maxDD%": m["maxDD%"], "trades": m["trades"],
                         "tr/yr": m["tr/yr"]})
    return pd.DataFrame(rows)


def train_test_split() -> pd.DataFrame:
    """Split-sample (net @1pt): 2018-2021 vs 2022-2026, on configs with enough trades."""
    rows = []
    configs = {
        "M5 1.75%": dict(tf="M5", threshold=0.0175),
        "M5 2.0%": dict(tf="M5", threshold=0.020),
        "M15 2.25%": dict(tf="M15", threshold=0.0225),
        "M15 2.5%": dict(tf="M15", threshold=0.025),
    }
    base = dict(session="rth", entry_mode="rel", exit_mode="revert", stop_atr_mult=3.5,
                longs=True, shorts=False, use_recorded_spread=True, slip_pts=1.0)
    for name, kw in configs.items():
        for lbl, ya, yb in (("train 18-21", 2018, 2021), ("test 22-26", 2022, 2026)):
            p = E.Params(**{**base, **kw})
            m, _ = E.run("NDX", ya, yb, p)
            rows.append({"config": name, "split": lbl, "net_sharpe": m["sharpe"],
                         "ann%": m["ann_ret%"], "maxDD%": m["maxDD%"], "trades": m["trades"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

    print("=" * 90)
    print("KAMA PARAM PLATEAU — M15 2.5% long revert (base = period20 fast5 slow30)")
    print("=" * 90)
    k = kama_param_plateau()
    k.to_csv(f"{OUT}/rob_kama_params.csv", index=False)
    print(f"gross Sharpe: median={k['gross_sharpe'].median():.2f} min={k['gross_sharpe'].min():.2f} "
          f"max={k['gross_sharpe'].max():.2f}  | net@1pt: median={k['net_sharpe'].median():.2f} "
          f"min={k['net_sharpe'].min():.2f}  ({(k['net_sharpe']>0.5).mean()*100:.0f}% of cells net>0.5)")
    print(k.sort_values("net_sharpe", ascending=False).head(8).to_string(index=False))
    print("... worst cells:")
    print(k.sort_values("net_sharpe").head(5).to_string(index=False))

    print("\n" + "=" * 90)
    print("SP500 GENERALIZATION vs NDX — long deep-dip revert, frictionless, thr sweep")
    print("=" * 90)
    g = sp500_generalization()
    g.to_csv(f"{OUT}/rob_sp500_generalization.csv", index=False)
    print(g.to_string(index=False))

    print("\n" + "=" * 90)
    print("SESSION: RTH vs 24h (net @1pt slip + recorded spread)")
    print("=" * 90)
    sc = session_compare()
    sc.to_csv(f"{OUT}/rob_session.csv", index=False)
    print(sc.to_string(index=False))

    print("\n" + "=" * 90)
    print("TRAIN/TEST SPLIT — 2018-2021 vs 2022-2026 (net @1pt slip)")
    print("=" * 90)
    tt = train_test_split()
    tt.to_csv(f"{OUT}/rob_train_test.csv", index=False)
    print(tt.to_string(index=False))
