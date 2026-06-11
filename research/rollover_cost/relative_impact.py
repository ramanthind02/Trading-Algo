r"""Swap & overlay impact RELATIVE to the actual strategy return series.

The right yardstick is Sharpe / % of returns, not absolute %/yr — the live strategy is
LOW vol (~3.7 %/yr) so a 2.5 %/yr swap is huge in relative terms. This takes the real
strategy daily returns (the backtest, which is swap-free) and the real signed positions,
charges the actual nightly swap, and the carry-aware overlay recovery, then reports Sharpe
and annual return for:

  GROSS  — the backtest as-is (no financing).
  LIVE   — GROSS minus the actual swap (what you really earn holding through the rollover).
  OVERLAY— LIVE plus the carry-aware overlay recovery on the index legs (market exec).

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.relative_impact
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.rollover_cost.config import SPECS

SNAP = _REPO / "tests/parity/snapshots/portfolio_research"
MT5 = _REPO / "data" / "mt5_data"
OUT = _REPO / "research" / "rollover_cost" / "outputs"
VAULT_TO_CFD = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD"}
OVERLAY_LEGS = ("ES", "NQ")          # overlay the tight-spread index legs; HOLD GC (wide spread)
ANN = 252
# FTMO long-swap is worse than Darwinex by these factors (research/rollover_cost/ftmo_spec_probe.py, 2026-06-06).
FTMO_RATIO = {"ES": 1.43, "NQ": 1.37, "GC": 1.36}


def daily_close(cfd: str) -> pd.Series:
    parts = sorted((MT5 / cfd / "bars_M1").glob("year=*/part.parquet"))
    df = pd.concat([pd.read_parquet(p, columns=["time", "close"]) for p in parts], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_localize(None).dt.normalize()
    return df.groupby("date")["close"].last()


def _triple_py(d: int) -> int:
    return {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[d]


def _exec_market_bps(cfd: str) -> float:
    ev = pd.read_parquet(OUT / f"events_{cfd}.parquet")
    mkt = ((ev["exit_spread_t1"] / 2 / ev["exit_mid_t1"]) + (ev["open_spread"] / 2 / ev["open_mid"])) * 1e4
    return float(mkt.mean())


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {"ann_return_pct": s.mean() * ANN * 100,
            "ann_vol_pct": s.std() * np.sqrt(ANN) * 100,
            "sharpe": s.mean() / s.std() * np.sqrt(ANN) if s.std() else float("nan")}


def main() -> None:
    gross = pd.read_parquet(SNAP / "portfolio_test_default__test__strategy_returns.parquet")["strategy_return"]
    gross.index = pd.to_datetime(gross.index).normalize()
    pos = pd.read_parquet(SNAP / "portfolio_test_default__test__positions.parquet")
    pos["datetime"] = pd.to_datetime(pos["datetime"]).dt.normalize()

    swap_d = pd.Series(0.0, index=gross.index)            # Darwinex swap (signed, negative=cost)
    swap_f = pd.Series(0.0, index=gross.index)            # FTMO swap
    overlay_d = pd.Series(0.0, index=gross.index)         # Darwinex overlay recovery (positive)
    overlay_f = pd.Series(0.0, index=gross.index)         # FTMO overlay recovery
    exec_bps = {c: _exec_market_bps(c) for c in VAULT_TO_CFD.values()}

    for tkr, cfd in VAULT_TO_CFD.items():
        spec = SPECS[cfd]
        f = pos[pos["ticker"].astype(str) == tkr].set_index("datetime")["position_fraction"].reindex(gross.index).fillna(0.0)
        price = daily_close(cfd).reindex(gross.index, method="ffill")
        is_long = f > 0
        swap_pts = np.where(is_long, spec.swap_long_pts, spec.swap_short_pts)
        mult = np.where(np.asarray(gross.index.weekday) == _triple_py(spec.triple_weekday), 3.0, 1.0)
        swap_bps = swap_pts * spec.point / price.to_numpy() * 1e4 * mult     # signed (Darwinex)
        ratio = FTMO_RATIO.get(tkr, 1.4)
        absf = f.abs().to_numpy()
        swap_d += pd.Series(absf * swap_bps / 1e4, index=gross.index).fillna(0.0)
        swap_f += pd.Series(absf * swap_bps * ratio / 1e4, index=gross.index).fillna(0.0)
        if tkr in OVERLAY_LEGS:                                              # recover on negative-carry nights
            neg = swap_bps < 0
            overlay_d += pd.Series(np.where(neg, absf * (-swap_bps - exec_bps[cfd]) / 1e4, 0.0), index=gross.index).fillna(0.0)
            overlay_f += pd.Series(np.where(neg, absf * (-swap_bps * ratio - exec_bps[cfd]) / 1e4, 0.0), index=gross.index).fillna(0.0)

    rows = {
        "GROSS (backtest, no financing)": _stats(gross),
        "LIVE Darwinex (- swap)": _stats(gross + swap_d),
        "OVERLAY Darwinex (+ recovery, index legs)": _stats(gross + swap_d + overlay_d),
        "LIVE FTMO (- swap, ~1.4x worse)": _stats(gross + swap_f),
        "OVERLAY FTMO (+ recovery, index legs)": _stats(gross + swap_f + overlay_f),
    }
    live = gross + swap_d
    overlay = gross + swap_d + overlay_d
    print(f"Period: {gross.index.min().date()}..{gross.index.max().date()}  ({len(gross)} days)\n")
    print(f"{'scenario':56s} {'ann_ret%':>9s} {'ann_vol%':>9s} {'Sharpe':>7s}")
    base_sh = rows["GROSS (backtest, no financing)"]["sharpe"]
    for name, st in rows.items():
        print(f"{name:56s} {st['ann_return_pct']:9.2f} {st['ann_vol_pct']:9.2f} {st['sharpe']:7.3f}")
    sw = rows["LIVE Darwinex (- swap)"]; ov = rows["OVERLAY Darwinex (+ recovery, index legs)"]
    print("\nRelative impact (Darwinex, on the ACTUAL strategy):")
    print(f"  Swap drag        : {sw['ann_return_pct']-rows['GROSS (backtest, no financing)']['ann_return_pct']:+.2f} %/yr "
          f"({(sw['ann_return_pct']/rows['GROSS (backtest, no financing)']['ann_return_pct']-1)*100:+.0f}% of gross return)  "
          f"dSharpe {sw['sharpe']-base_sh:+.3f}")
    print(f"  Overlay recovery : {ov['ann_return_pct']-sw['ann_return_pct']:+.2f} %/yr "
          f"({(ov['ann_return_pct']/sw['ann_return_pct']-1)*100:+.0f}% of LIVE return)  "
          f"dSharpe {ov['sharpe']-sw['sharpe']:+.3f}  (recovers {(ov['sharpe']-sw['sharpe'])/(base_sh-sw['sharpe'])*100:.0f}% of the swap's Sharpe hit)")
    pd.DataFrame(rows).T.to_csv(OUT / "G_relative_impact.csv")
    print(f"\nWrote {OUT/'G_relative_impact.csv'}")


if __name__ == "__main__":
    main()
