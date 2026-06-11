r"""How much are swaps actually costing the live strategies?

Uses the REAL signed daily positions from a vault portfolio backtest
(tests/parity/snapshots/portfolio_research/portfolio_test_default__test__positions.parquet:
CL/ES/GC/NQ, 2023-01..2026-05) — NOT a buy/hold proxy — and charges each night's swap on the
actual position (size AND direction: longs pay swap_long, shorts earn swap_short), at the real
daily price (swap is a fixed points charge, so its bps cost moves with price). Then estimates
the carry-aware overlay's value: on the negative-carry nights, swap avoided − execution cost,
both scaled by the actual position size.

Maps vault tickers to the Darwinex CFDs we have swap data for: ES->SP500, NQ->NDX, GC->XAUUSD.
CL->XTIUSD is excluded (mode-3 swap; and CL is ~70% flat with |frac|~0.002 → negligible).

Run:
  .\.venv\Scripts\python.exe -m research.rollover_cost.actual_swap_cost
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

POSITIONS = _REPO / "tests/parity/snapshots/portfolio_research/portfolio_test_default__test__positions.parquet"
MT5 = _REPO / "data" / "mt5_data"
OUT = _REPO / "research" / "rollover_cost" / "outputs"

# vault ticker -> (CFD symbol with swap data, MT5 triple weekday python)
VAULT_TO_CFD = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD"}  # CL excluded (see module docstring)


def daily_close(cfd: str) -> pd.Series:
    """Daily close price (broker date -> close) from stored M1 bars."""
    base = MT5 / cfd / "bars_M1"
    parts = sorted(base.glob("year=*/part.parquet"))
    frames = [pd.read_parquet(p, columns=["time", "close"]) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_localize(None).dt.normalize()   # broker date
    return df.groupby("date")["close"].last()


def _triple_py(mt5_rollover3days: int) -> int:
    return {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[mt5_rollover3days]


def main() -> None:
    pos = pd.read_parquet(POSITIONS)
    pos["datetime"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    years = (pos["datetime"].max() - pos["datetime"].min()).days / 365.25
    print(f"Positions: {pos['datetime'].min().date()} .. {pos['datetime'].max().date()} "
          f"({years:.2f} yr), tickers {sorted(pos['ticker'].astype(str).unique())}")

    # per-instrument representative execution cost (bps) from the events study (means)
    exec_cost = {}
    for cfd in VAULT_TO_CFD.values():
        ev = pd.read_parquet(OUT / f"events_{cfd}.parquet")
        spec = SPECS[cfd]
        mkt = ((ev["exit_spread_t1"] / 2 / ev["exit_mid_t1"]) + (ev["open_spread"] / 2 / ev["open_mid"])) * 1e4
        filled = ev.get("exit_sell_limit_mid_fill_min").notna()
        lim_exit = np.where(filled, 0.0, (ev["exit_spread_t1"] / 2 / ev["exit_mid_t1"]) * 1e4)
        f30 = ev.get("entry_buylim_t5_filled_30m", pd.Series(False, index=ev.index)).astype(bool)
        improve = (5 * spec.tick_size) / ev["open_mid"] * 1e4
        chase = ((ev["entry_spread_p30"] / 2 / ev["entry_mid_p30"]) + (ev["entry_mid_p30"] - ev["open_mid"]) / ev["open_mid"]) * 1e4
        lim_entry = np.where(f30, -improve, chase)
        exec_cost[cfd] = {"market": float(mkt.mean()), "limit": float((lim_exit + lim_entry).mean())}

    total = {"swap_actual": 0.0, "overlay_mkt_net": 0.0, "overlay_lim_net": 0.0}
    print(f"\n{'tkr':4s}->{'cfd':7s} {'%long':>6s} {'|frac|':>7s} {'swap_drag%/yr':>13s} "
          f"{'overlay_save(mkt)':>17s} {'overlay_save(lim)':>17s}")
    rows = []
    for tkr, cfd in VAULT_TO_CFD.items():
        spec = SPECS[cfd]
        g = pos[pos["ticker"].astype(str) == tkr].copy().set_index("datetime")["position_fraction"]
        close = daily_close(cfd).reindex(g.index, method="ffill")
        f = g.to_numpy()
        absf = np.abs(f)
        price = close.to_numpy()
        # swap bps for the held direction (long pays<0, short earns>0), x3 on triple weekday
        is_long = f > 0
        swap_pts = np.where(is_long, spec.swap_long_pts, spec.swap_short_pts)
        triple = np.asarray(g.index.weekday) == _triple_py(spec.triple_weekday)
        mult = np.where(triple, 3.0, 1.0)
        swap_bps = swap_pts * spec.point / price * 1e4 * mult        # signed
        swap_ret = absf * swap_bps / 1e4                             # daily return contribution (signed)
        swap_drag_pct_yr = swap_ret.sum() / years * 100

        # carry-aware overlay: on NEGATIVE-carry nights (longs here), avoid the swap but pay exec.
        neg = swap_bps < 0
        swap_avoided = absf * (-swap_bps) / 1e4                      # positive (what we stop paying)
        exec_ret_mkt = absf * exec_cost[cfd]["market"] / 1e4
        exec_ret_lim = absf * exec_cost[cfd]["limit"] / 1e4
        overlay_mkt = np.where(neg, swap_avoided - exec_ret_mkt, 0.0).sum() / years * 100
        overlay_lim = np.where(neg, swap_avoided - exec_ret_lim, 0.0).sum() / years * 100

        total["swap_actual"] += swap_ret.sum() / years * 100
        total["overlay_mkt_net"] += overlay_mkt
        total["overlay_lim_net"] += overlay_lim
        print(f"{tkr:4s}->{cfd:7s} {(is_long.mean()*100):6.1f} {absf.mean():7.4f} "
              f"{swap_drag_pct_yr:13.3f} {overlay_mkt:17.3f} {overlay_lim:17.3f}")
        rows.append({"ticker": tkr, "cfd": cfd, "pct_long": is_long.mean()*100,
                     "abs_frac_mean": absf.mean(), "swap_drag_pct_yr": swap_drag_pct_yr,
                     "overlay_save_market_pct_yr": overlay_mkt, "overlay_save_limit_pct_yr": overlay_lim})

    print(f"\nPORTFOLIO actual swap drag: {total['swap_actual']:+.3f} %/yr")
    print(f"Carry-aware overlay net benefit (market exec): {total['overlay_mkt_net']:+.3f} %/yr")
    print(f"Carry-aware overlay net benefit (limit  exec): {total['overlay_lim_net']:+.3f} %/yr")
    print("\n(Position sizes are the strategy's actual fractions — both swap and execution scale with them.)")
    pd.DataFrame(rows).to_csv(OUT / "F_actual_swap_cost.csv", index=False)
    print(f"Wrote {OUT/'F_actual_swap_cost.csv'}")


if __name__ == "__main__":
    main()
