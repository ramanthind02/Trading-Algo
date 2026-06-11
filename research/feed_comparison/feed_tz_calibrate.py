r"""Calibrate the MT5 (Darwinex) bar-timestamp timezone and characterize each
CFD/ETF trading session from M1 data.

MT5 `copy_rates` returns *broker server* time (Darwinex ~ EET, UTC+2/+3); the
scraper stored it tagged utc=True, so the stored 'UTC' is offset from true UTC.
We (a) infer the session from the minute-of-day bar histogram + maintenance gap,
and (b) find the stored-time snapshot that MAXIMIZES daily-return correlation to
the matching Norgate series (that snapshot == the instrument's true daily close).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MT5 = REPO / "data" / "mt5_data"
MS = REPO / "data" / "norgate" / "market_series"
STK = REPO / "data" / "stock_data"


def load_m1(sym: str, years=range(2021, 2027)) -> pd.DataFrame:
    parts = []
    for y in years:
        p = MT5 / sym / "bars_M1" / f"year={y}" / "part.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["time", "close"]))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)  # stored tag (may be mislabeled)
    return df


def norgate_close(kind: str, sym: str) -> pd.Series:
    if kind == "cash":
        folder = "us_indices" if sym in {"SPX", "NDX", "DJI"} else "cash_commodities"
        p = MS / folder / f"{sym}.parquet"
    else:  # etf
        letter, name = sym
        p = STK / letter / name / f"D_CAP_{name}.parquet"
        sym = name
    df = pd.read_parquet(p)
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).normalize()
    return s


def session_profile(sym: str, df: pd.DataFrame) -> None:
    t = df["time"]
    mod = t.dt.hour * 60 + t.dt.minute
    present = np.zeros(1440, dtype=int)
    vc = mod.value_counts()
    for k, v in vc.items():
        present[int(k)] = v
    active = np.where(present > 0)[0]
    # find largest contiguous gap (the daily maintenance break / overnight close)
    if len(active) == 0:
        print(f"  {sym}: no bars"); return
    # weekday coverage
    wd = t.dt.dayofweek.value_counts().sort_index()
    wd_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    wdstr = " ".join(f"{wd_map[d]}={wd.get(d,0)}" for d in range(7))
    # gaps in minute-of-day coverage
    full = present > 0
    # rotate to find longest run of empty minutes
    empties = np.where(~full)[0]
    def hm(x): return f"{x//60:02d}:{x%60:02d}"
    # longest empty run (cyclic)
    runs = []
    if len(empties):
        start = empties[0]; prev = empties[0]
        for e in empties[1:]:
            if e == prev + 1:
                prev = e
            else:
                runs.append((start, prev)); start = e; prev = e
        runs.append((start, prev))
        # handle wrap
        if runs and runs[0][0] == 0 and runs[-1][1] == 1439:
            s0, e0 = runs.pop(0); s1, e1 = runs.pop(-1)
            runs.append((s1, e0 + 1440))
        longest = max(runs, key=lambda r: r[1] - r[0])
        gap_str = f"daily-gap(storedUTC) {hm(longest[0]%1440)}..{hm(longest[1]%1440)} ({longest[1]-longest[0]+1} min)"
    else:
        gap_str = "no daily gap (24h)"
    print(f"  {sym:<10} stored-UTC bars {hm(active.min())}..{hm(active.max())} | {gap_str}")
    print(f"             weekday counts: {wdstr}")


def calibrate(sym: str, kind: str, ref: str | tuple, df: pd.DataFrame) -> None:
    ref_close = norgate_close(kind, ref)
    rref = np.log(ref_close).diff()
    best = (-2, None)
    print(f"  snapshot sweep (stored-UTC hour -> corr vs {ref if isinstance(ref,str) else ref[1]}):")
    row = []
    for hh in range(10, 24):
        cutoff = hh * 60 + 30  # use HH:30 stored
        t = df["time"]
        mod = t.dt.hour * 60 + t.dt.minute
        keep = mod <= cutoff
        d = pd.DataFrame({"date": t[keep].dt.tz_convert(None).dt.normalize(),
                          "close": df["close"].values[keep.values]})
        daily = d.groupby("date")["close"].last()
        r = np.log(daily.astype(float)).diff()
        j = pd.concat([r.rename("c"), rref.rename("r")], axis=1).dropna()
        if len(j) > 100:
            c = j["c"].corr(j["r"])
            row.append((hh, c))
            if c > best[0]:
                best = (c, hh)
    print("    " + "  ".join(f"{hh:02d}:30={c:.3f}" for hh, c in row))
    print(f"    >>> BEST stored-UTC snapshot ~ {best[1]:02d}:30  corr={best[0]:.4f}")


SETS = [
    ("SP500", "cash", "SPX"),
    ("SPY", "etf", ("S", "SPY")),
    ("QQQ", "etf", ("Q", "QQQ")),
    ("WS30", "cash", "DJI"),
    ("XAUUSD", "cash", "GC"),
    ("GLD", "etf", ("G", "GLD")),
    ("XTIUSD", "cash", "WTI"),
]

print("=" * 90)
print("SESSION PROFILES (from M1, 2021-2026, times in STORED-UTC tag)")
print("=" * 90)
for sym, _, _ in SETS:
    df = load_m1(sym)
    if not df.empty:
        session_profile(sym, df)

print("\n" + "=" * 90)
print("SNAPSHOT-TIME CALIBRATION (corr-maximizing close)")
print("=" * 90)
for sym, kind, ref in SETS:
    df = load_m1(sym)
    if df.empty:
        print(f"  {sym}: no M1"); continue
    print(f"\n[{sym}]")
    calibrate(sym, kind, ref, df)
