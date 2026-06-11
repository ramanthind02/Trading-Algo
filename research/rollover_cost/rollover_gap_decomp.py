r"""Decompose CFD daily returns into the part you CAPTURE when flat over the
17:00-ET rollover (open-to-close of the rollover-bounded session) vs the rollover
GAP you skip (prior close -> reopen). Motivates the research return convention:
backtest P&L on OPEN-TO-CLOSE, not close-to-close, because we flatten over the
rollover and never hold the gap.

Rollover-bounded daily bar (stored/broker tz; dead zone = stored 00:00-00:59):
  open  = first M1 close at/after stored 01:00  (~18:00 ET reopen, post dead-zone)
  close = last  M1 close at/before stored 23:59 (~16:59 ET, pre-rollover)

Run: .\.venv\Scripts\python.exe -m research.rollover_cost.rollover_gap_decomp
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANN = np.sqrt(252.0)


def rollover_bars(sym: str, y0: int = 2018) -> pd.DataFrame:
    """Build rollover-bounded daily open/close from M1 (broker-tz)."""
    parts = sorted((REPO / "data/mt5_data" / sym / "bars_M1").glob("year=*/part.parquet"))
    o, c = {}, {}
    for p in parts:
        if int(p.parent.name.split("=")[1]) < y0:
            continue
        df = pd.read_parquet(p, columns=["time", "close"])
        t = pd.to_datetime(df["time"], utc=True)
        mod = (t.dt.hour * 60 + t.dt.minute).values
        date = t.dt.tz_convert(None).dt.normalize().values
        d = pd.DataFrame({"date": date, "mod": mod, "close": df["close"].values})
        op = d[d["mod"] >= 60].sort_values("mod").groupby("date")["close"].first()
        cl = d[d["mod"] <= 23 * 60 + 59].sort_values("mod").groupby("date")["close"].last()
        o.update(op.to_dict())
        c.update(cl.to_dict())
    out = pd.concat([pd.Series(o).rename("open"), pd.Series(c).rename("close")], axis=1)
    return out.dropna().astype(float).sort_index()


def main() -> None:
    print(f"{'sym':<8}{'c2c_vol':>9}{'o2c_vol':>9}{'gap_vol':>9}"
          f"{'gap/var%':>9}{'gap_drift/yr':>13}{'corr_o2c_c2c':>14}")
    for sym in ["SP500", "NDX", "XAUUSD", "XAGUSD"]:
        b = rollover_bars(sym)
        o2c = np.log(b["close"] / b["open"])                 # captured (flat over rollover)
        c2c = np.log(b["close"] / b["close"].shift(1))       # full hold-through
        gap = np.log(b["open"] / b["close"].shift(1))        # skipped rollover gap
        d = pd.concat([o2c.rename("o"), c2c.rename("c"), gap.rename("g")], axis=1).dropna()
        print(f"{sym:<8}{d['c'].std()*ANN*100:>8.2f}%{d['o'].std()*ANN*100:>8.2f}%"
              f"{d['g'].std()*ANN*100:>8.2f}%{d['g'].var()/d['c'].var()*100:>8.1f}%"
              f"{d['g'].mean()*252*100:>+12.2f}%{d['o'].corr(d['c']):>14.3f}")
    print("\nlegend: c2c=close-to-close(hold through); o2c=open-to-close(captured, flat over rollover);")
    print("gap=close->reopen move skipped; gap/var%=share of daily variance in the gap;")
    print("gap_drift=systematic overnight drift (metals show large +drift => re-check before flattening).")


if __name__ == "__main__":
    main()
