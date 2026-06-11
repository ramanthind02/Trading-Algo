"""Combine the screen + buy-hold benchmark into an annotated table, a robust-survivor
list, and per-family verdicts.

A config is a "robust survivor" only if it clears ALL of:
  - net_sharpe_1.0x >= 0.5                     (meaningful after measured spread)
  - both sub-period Sharpes > 0                (2018-2021 AND 2022-2026)
  - gross Sharpe > drift-stripped shuffle null (timing above the null)
  - plateau: >= half the param-axis cells at that (family,dir,instr,tf) are net>0
  - beta filter (long-biased only): gross Sharpe - buy_hold Sharpe >= 0.20
    (a long-only book must beat simply holding the instrument, else it is drift/beta)

Run: .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.analyze
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    df = pd.read_csv(OUT / "screen_master.csv")
    bh = pd.read_csv(OUT / "benchmark.csv")[["instrument", "tf", "bh_sharpe", "bh_max_dd",
                                             "bh_sharpe_2018_2021", "bh_sharpe_2022_2026"]]
    df = df.merge(bh, on=["instrument", "tf"], how="left")

    df["lift_vs_bh"] = df["sharpe"] - df["bh_sharpe"]               # risk-adj lift over holding
    df["null_margin"] = df["sharpe"] - df["sharpe_shuffle_null"]    # timing above drift-stripped null
    df["cost_decay"] = df["sharpe"] - df["net_sharpe_1.0x"]         # Sharpe lost to spread @1x
    df["both_subperiods_pos"] = (df["sharpe_2018_2021"] > 0) & (df["sharpe_2022_2026"] > 0)

    # plateau: fraction of the param axis (fixed family/dir/instr/tf) with net@1x > 0
    grp = ["family", "direction", "instrument", "tf"]
    df["plateau_frac_net_pos"] = df.groupby(grp)["net_sharpe_1.0x"].transform(lambda s: (s > 0).mean())
    df["plateau_median_net"] = df.groupby(grp)["net_sharpe_1.0x"].transform("median")

    # Beta filter applied to BOTH directions: any book (long OR long_short) that does not
    # beat simply holding the instrument by >= 0.20 Sharpe is not a worthwhile edge over the
    # trivial alternative. (For long_short this matters because the screen's long_short books
    # are net-directional, not market-neutral — a negative lift means the short leg degrades
    # passive beta rather than adding alpha.)
    df["survives_2x_cost"] = df["net_sharpe_2.0x"] >= 0.5
    df["robust"] = (
        (df["net_sharpe_1.0x"] >= 0.5)
        & df["both_subperiods_pos"]
        & (df["sharpe"] > df["sharpe_shuffle_null"])
        & (df["plateau_frac_net_pos"] >= 0.5)
        & (df["lift_vs_bh"] >= 0.20)
    )

    df.to_csv(OUT / "screen_annotated.csv", index=False)

    surv = df[df["robust"]].sort_values("net_sharpe_1.0x", ascending=False)
    surv.to_csv(OUT / "survivors.csv", index=False)

    # per-family verdict
    fam = df.groupby(["family", "direction"]).agg(
        n_configs=("sharpe", "size"),
        n_robust=("robust", "sum"),
        best_net=("net_sharpe_1.0x", "max"),
        best_gross=("sharpe", "max"),
        best_lift=("lift_vs_bh", "max"),
        median_net=("net_sharpe_1.0x", "median"),
    ).reset_index().sort_values("best_net", ascending=False)
    fam.to_csv(OUT / "family_verdict.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== PER-FAMILY VERDICT (sorted by best net@1x) ===")
    print(fam.round(2).to_string(index=False))
    print(f"\n=== ROBUST SURVIVORS: {len(surv)} of {len(df)} configs ===")
    cols = ["family", "direction", "instrument", "tf", "axis_value", "sharpe", "net_sharpe_1.0x",
            "net_sharpe_2.0x", "lift_vs_bh", "bh_sharpe", "sharpe_2018_2021", "sharpe_2022_2026",
            "sharpe_shuffle_null", "turnover", "max_dd", "plateau_frac_net_pos"]
    if len(surv):
        print(surv[cols].head(40).round(2).to_string(index=False))
    else:
        print("  (none cleared all robustness gates)")

    # how many survivors per family/instrument (breadth of the edge)
    if len(surv):
        print("\n=== survivor breadth (family,direction x instrument): count ===")
        print(surv.pivot_table(index=["family", "direction"], columns="instrument",
                               values="robust", aggfunc="size", fill_value=0).to_string())
    print(f"\nwrote screen_annotated.csv, survivors.csv, family_verdict.csv to {OUT}")


if __name__ == "__main__":
    main()
