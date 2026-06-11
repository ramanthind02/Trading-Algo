"""Smoke test: SPY paper-overlap window, 3 escalations + independent minute-grid PnL check."""
from __future__ import annotations
import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E

SYM, Y0, Y1 = "SPY", 2010, 2024


def minute_grid_unlev(win: pd.DataFrame, dt: pd.DataFrame, p: E.Params, marks_et, year: int) -> tuple[float, float]:
    """Independent from-scratch LOG-return derivation of unlevered daily PnL on the FULL
    minute grid vs the engine's mark-grid. Log returns telescope EXACTLY, so any residual
    difference is a position-alignment (lookahead) bug, not a compounding artifact."""
    mark_s = E.marks_broker_sec(marks_et)
    nmk = len(mark_s)
    sig = {k: dt[f"move_{k}"].shift(1).rolling(p.vol_lookback_tod).mean() for k in range(nmk)}
    eng_total, min_total = 0.0, 0.0
    wy = win[win["day"].dt.year == year]
    for day, g in wy.groupby("day", sort=True):
        if day not in dt.index:
            continue
        row = dt.loc[day]
        s = np.array([sig[k].get(day, np.nan) for k in range(nmk)])
        if not np.isfinite(s).all() or not np.isfinite(row["prev_close"]):
            continue
        base = max(row["open"], row["prev_close"]); basel = min(row["open"], row["prev_close"])
        ub = base * (1 + p.vm * s); lb = basel * (1 - p.vm * s)
        vw = np.array([row[f"vwap_{k}"] for k in range(nmk)])
        pxk = np.array([row[f"px_{k}"] for k in range(nmk)])
        if not np.isfinite(pxk).all():
            continue
        # engine mark-grid positions (identical code path to engine._step)
        pos = 0; positions = []
        for k in range(nmk):
            pos = E._step(pos, pxk[k], ub[k], lb[k], vw[k], p.stop_mode)
            positions.append(pos)
        positions = np.array(positions, dtype=float)
        sec = g["sec"].to_numpy(); c = g["close"].to_numpy()
        # only compare on days where every mark bar exists exactly (so the engine anchor
        # bars == the minute-grid boundary bars and log telescoping is EXACT). Missing-bar
        # days differ only by a <=1min boundary convention and are excluded from the assert.
        if not np.isin(mark_s, sec).all():
            continue
        grid = np.concatenate([pxk, [row["close"]]])           # nmk marks + forced close
        eng_total += float(np.dot(positions, np.log(grid[1:] / grid[:-1])))  # LOG segments
        keep = (sec >= mark_s[0]) & (sec <= E.CLOSE_S)
        cc = c[keep]; ss = sec[keep]
        ki = np.clip(np.searchsorted(mark_s, ss, side="right") - 1, 0, nmk - 1)
        minpos = positions[ki]
        min_total += float(np.dot(minpos[:-1], np.log(cc[1:] / cc[:-1])))
    return eng_total, min_total


def main():
    win = E.load_sessions(SYM, Y0, Y1)
    dt = E.build_day_table(win, E.MARKS_ET_SEMI)
    print(f"{SYM} {Y0}-{Y1}: {dt.index.min().date()}..{dt.index.max().date()}  "
          f"{len(dt)} day-rows, {len(win):,} RTH bars")
    print(f"first RTH bar sec={win['sec'].min()} ({win['sec'].min()//3600}:{(win['sec'].min()%3600)//60:02d} broker), "
          f"last sec={win['sec'].max()}\n")

    configs = [
        ("base opp-band (binary)", E.Params(stop_mode="opp", sizing="binary")),
        ("curr+VWAP (binary)", E.Params(stop_mode="curr_vwap", sizing="binary")),
        ("curr+VWAP vol-target", E.Params(stop_mode="curr_vwap", sizing="vol_target")),
    ]
    rows = []
    for label, p in configs:
        res = E.simulate(dt, p, len(E.MARKS_ET_SEMI))
        col = "lev_ret" if p.sizing == "vol_target" else "unlev_ret"
        m = E.metrics(res, col); m["cfg"] = label
        rows.append(m)
    print(pd.DataFrame(rows).set_index("cfg")[
        ["days", "sharpe", "ann_ret%", "vol%", "maxDD%", "hit%", "skew", "worst%", "best%", "tr/day", "lev_med"]
    ].to_string())

    # independent minute-grid check on 2022
    p = E.Params(stop_mode="curr_vwap", sizing="binary")
    eng, mn = minute_grid_unlev(win, dt, p, E.MARKS_ET_SEMI, 2022)
    print(f"\n[lookahead/alignment] 2022 unlev LOG-PnL  engine={eng:.6f}  minute-grid={mn:.6f}  "
          f"diff={abs(eng-mn):.2e}")
    assert abs(eng - mn) < 1e-9, "mark-grid vs minute-grid LOG PnL mismatch -> position-alignment/lookahead bug"
    print("PASS: position k earns exactly the mark_k->mark_{k+1} forward return (no lookahead).")


if __name__ == "__main__":
    main()
