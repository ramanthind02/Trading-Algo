r"""Measure the REAL limit-capture fraction net of adverse selection (tick-replay).

For each rollover re-entry, replays the real entry-window ticks and compares two policies,
both marked to the same fair settled price (mid at +59min) — so the difference is the realized
execution edge INCLUDING adverse selection (the same thing the Nautilus matching engine prices):

  MARKET : buy at the reopen ask (t=0), hold to settled.        pnl = settled - open_ask
  LIMIT  : rest a buy-limit at the reopen bid (the touch); fills when a later ask <= bid (a
           maker fill); hold to settled. If it never fills in the window, SKIP (flat, pnl=0)
           — matching the Nautilus pure-passive (cross_after=1.0) measurement.

Reports, per leg: maker fill rate, market vs limit realized pnl (bps), and the capture
(limit - market) in bps and as a fraction of the half-spread. Positive capture => the limit
market-making genuinely beats market orders after adverse selection.

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.capture_sim --symbols NDX SP500 XAUUSD
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_platform.providers.mt5.tick_cache import ensure_ticks
from research.rollover_cost.build_events import rollover_dates, _ns
from research.rollover_cost.config import DEADZONE_MIN, ENTRY_WINDOW_MIN, FETCH_PAD_MIN

OUT = _REPO / "research" / "rollover_cost" / "outputs"


def replay_event(df: pd.DataFrame, roll: pd.Timestamp) -> dict | None:
    """One re-entry event: market vs limit-at-touch realized pnl (price units)."""
    entry = df[df["time"] >= roll + pd.Timedelta(minutes=DEADZONE_MIN)].reset_index(drop=True)
    if len(entry) < 20:
        return None
    open_bid = float(entry["bid"].iloc[0]); open_ask = float(entry["ask"].iloc[0])
    open_mid = 0.5 * (open_bid + open_ask)
    # settled (fair) mid ~ +59min after reopen
    open_t = entry["time"].iloc[0]
    ns = _ns(entry)
    si = int(np.searchsorted(ns, (open_t + pd.Timedelta(minutes=59)).value, side="right")) - 1
    if si < 1:
        return None
    settled = 0.5 * (float(entry["bid"].iloc[si]) + float(entry["ask"].iloc[si]))
    # LIMIT at the touch (rest a buy at open_bid): fills when a later ask <= open_bid
    asks = entry["ask"].to_numpy()
    hit = np.where(asks[1:] <= open_bid)[0]
    filled = len(hit) > 0
    mkt_pnl = settled - open_ask
    lim_pnl = (settled - open_bid) if filled else 0.0
    return {"open_mid": open_mid, "half_spread": 0.5 * (open_ask - open_bid),
            "filled": filled, "mkt_pnl": mkt_pnl, "lim_pnl": lim_pnl,
            "settled_minus_open": settled - open_mid}


def run_leg(sym: str, days: int) -> dict:
    rows = []
    for roll in rollover_dates(days):
        fs = roll + timedelta(minutes=DEADZONE_MIN - FETCH_PAD_MIN)
        fe = roll + timedelta(minutes=DEADZONE_MIN + ENTRY_WINDOW_MIN + FETCH_PAD_MIN)
        try:
            df = ensure_ticks(sym, fs, fe)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        r = replay_event(df, pd.Timestamp(roll))
        if r:
            rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    mid = d["open_mid"].mean()
    bps = lambda x: x / mid * 1e4
    return {
        "n": len(d), "fill_rate": d["filled"].mean(),
        "half_spread_bps": bps(d["half_spread"].mean()),
        "mkt_pnl_bps": bps(d["mkt_pnl"].mean()), "lim_pnl_bps": bps(d["lim_pnl"].mean()),
        "capture_bps": bps((d["lim_pnl"] - d["mkt_pnl"]).mean()),
        "capture_frac_of_halfspread": bps((d["lim_pnl"] - d["mkt_pnl"]).mean()) / bps(d["half_spread"].mean()),
        "post_reopen_drift_bps": bps(d["settled_minus_open"].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=["NDX", "SP500", "XAUUSD"])
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()
    print(f"{'sym':7s} {'n':>4s} {'fill%':>6s} {'half_spr':>8s} {'mkt_pnl':>8s} {'lim_pnl':>8s} "
          f"{'capture':>8s} {'cap/half':>8s} {'reopen_drift':>12s}  (bps)")
    res = {}
    for sym in args.symbols:
        r = run_leg(sym, args.days)
        res[sym] = r
        if r:
            print(f"{sym:7s} {r['n']:4d} {r['fill_rate']*100:5.1f}% {r['half_spread_bps']:8.2f} "
                  f"{r['mkt_pnl_bps']:8.2f} {r['lim_pnl_bps']:8.2f} {r['capture_bps']:8.2f} "
                  f"{r['capture_frac_of_halfspread']:8.2f} {r['post_reopen_drift_bps']:12.2f}")
    pd.DataFrame(res).T.to_csv(OUT / "I_capture_sim.csv")
    print("\ncapture>0 => limit market-making beats market orders net of adverse selection.")
    print("cap/half ~ how many half-spreads captured per round-trip (vs paying ~1 with market).")
    print(f"Wrote {OUT/'I_capture_sim.csv'}")


if __name__ == "__main__":
    main()
