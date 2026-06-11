r"""FTMO vs Darwinex: real-tick rollover economics + the overlay Sharpe ladder.

Uses REAL scraped ticks from BOTH brokers (events_{cfd}.parquet = Darwinex,
events_ftmo_{canon}.parquet = FTMO) on the SAME actual vault position path, so the only
differences are the broker's spreads, fill rates (missed trades), and swap. Reports:

  * spreads (exit + reopen, bps)            — FTMO is wider
  * limit fill rates (exit + entry)         — FTMO misses more (wider touch = harder fill)
  * swap (bps/night)                        — FTMO ~1.4x worse
  * the Sharpe ladder (hold / market overlay / limit overlay) per broker

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.ftmo_compare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.rollover_cost.config import SPECS, SymbolSpec

SNAP = _REPO / "tests/parity/snapshots/portfolio_research"
MT5 = _REPO / "data" / "mt5_data"
OUT = _REPO / "research" / "rollover_cost" / "outputs"
ANN = 252
CAPTURE_FRAC = 0.5      # conservative maker-capture haircut (matches the Darwinex sim's middle case)

# canon -> (Darwinex CFD symbol, Darwinex bars dir)
DWX = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD"}
# FTMO specs (live probe 2026-06-06): point, tick, contract, swap_long_pts, swap_short_pts
FTMO_SPEC = {
    "ES": SymbolSpec("ES", 0.01, 0.01, 1.0, -154.67, -3.09, 5),
    "NQ": SymbolSpec("NQ", 0.01, 0.01, 1.0, -619.48, -12.41, 5),
    "GC": SymbolSpec("GC", 0.01, 0.01, 100.0, -86.40, -15.30, 3),
}


def _triple_py(d: int) -> int:
    return {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[d]


def daily_close(cfd: str) -> pd.Series:
    parts = sorted((MT5 / cfd / "bars_M1").glob("year=*/part.parquet"))
    df = pd.concat([pd.read_parquet(p, columns=["time", "close"]) for p in parts], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_localize(None).dt.normalize()
    return df.groupby("date")["close"].last()


def leg_econ(ev: pd.DataFrame) -> dict:
    """Spreads, fill rates and round-trip costs (bps) from one broker's events for one leg.

    Reopen spread is taken at +5min after each broker's reopen (NOT the first tick): FTMO
    reopens ~5min later than Darwinex, so first-tick spreads compare a decayed spread vs the
    spike — unfair. The +5min decayed spread is the realistic re-entry on BOTH brokers.
    """
    exit_half = (ev["exit_spread_t15"] / 2 / ev["exit_mid_t15"] * 1e4).mean()
    reopen_half = (ev["entry_spread_p5"] / 2 / ev["entry_mid_p5"] * 1e4).mean()
    fr_exit = ev["exit_sell_limit_mid_fill_min"].notna().mean()
    fr_entry = ev["entry_buylim_t5_filled_60m"].mean()
    open_lag = ev["open_lag_min"].median()
    market_rt = exit_half + reopen_half
    cf = CAPTURE_FRAC
    limit_rt = (fr_exit * (-cf * exit_half) + (1 - fr_exit) * exit_half) + (fr_entry * (-cf * reopen_half))
    return {"exit_half_bps": exit_half, "reopen_half_bps": reopen_half,
            "fr_exit": fr_exit, "fr_entry": fr_entry, "open_lag_min": open_lag,
            "market_rt_bps": market_rt, "limit_rt_bps": limit_rt}


def _stats(s: pd.Series) -> float:
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(ANN) if s.std() else float("nan")


def ladder(econ: dict, spec_map: dict) -> dict:
    """Sharpe ladder on the actual position path given per-leg econ + swap specs."""
    gross = pd.read_parquet(SNAP / "portfolio_test_default__test__strategy_returns.parquet")["strategy_return"]
    gross.index = pd.to_datetime(gross.index).normalize()
    pos = pd.read_parquet(SNAP / "portfolio_test_default__test__positions.parquet")
    pos["datetime"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    idx = gross.index
    hold = pd.Series(0.0, index=idx); ov_mkt = pd.Series(0.0, index=idx); ov_lim = pd.Series(0.0, index=idx)
    for canon in econ:
        spec = spec_map[canon]
        f = pos[pos["ticker"].astype(str) == canon].set_index("datetime")["position_fraction"].reindex(idx).fillna(0.0)
        price = daily_close(DWX[canon]).reindex(idx, method="ffill").to_numpy()
        absf = f.abs().to_numpy(); is_long = (f > 0).to_numpy()
        swap_pts = np.where(is_long, spec.swap_long_pts, spec.swap_short_pts)
        mult = np.where(np.asarray(idx.weekday) == _triple_py(spec.triple_weekday), 3.0, 1.0)
        swap_drag = absf * np.abs(swap_pts * spec.point / price * 1e4 * mult) / 1e4
        S = lambda a: pd.Series(a, index=idx).fillna(0.0)
        hold += S(swap_drag)
        ov_mkt += S(absf * econ[canon]["market_rt_bps"] / 1e4)
        ov_lim += S(absf * econ[canon]["limit_rt_bps"] / 1e4)
    return {"hold": _stats(gross - hold), "ov_mkt": _stats(gross - ov_mkt),
            "ov_lim": _stats(gross - ov_lim), "gross": _stats(gross)}


def main() -> None:
    dwx_econ, ftmo_econ = {}, {}
    for canon, cfd in DWX.items():
        dwx_econ[canon] = leg_econ(pd.read_parquet(OUT / f"events_{cfd}.parquet"))
        fp = OUT / f"events_ftmo_{canon}.parquet"
        if fp.exists():
            ftmo_econ[canon] = leg_econ(pd.read_parquet(fp))

    print("=== Per-leg real-tick economics: Darwinex vs FTMO (reopen spread @ +5min, fair) ===")
    print(f"{'leg':4s} {'broker':8s} {'exit_half':>9s} {'reopen_half':>11s} {'reopen_lag':>10s} "
          f"{'fr_exit':>8s} {'fr_entry':>8s} {'swap/ngt':>8s}")
    for canon in DWX:
        for label, econ, spc in (("Darwinex", dwx_econ, SPECS[DWX[canon]]), ("FTMO", ftmo_econ, FTMO_SPEC[canon])):
            if canon in econ:
                e = econ[canon]
                # representative swap bps/night at a recent price
                ref = {"ES": 6200, "NQ": 28000, "GC": 3300}[canon]
                swap_bps = abs(spc.swap_long_pts) * spc.point / ref * 1e4
                print(f"{canon:4s} {label:8s} {e['exit_half_bps']:9.2f} {e['reopen_half_bps']:11.2f} "
                      f"{e['open_lag_min']:9.1f}m {e['fr_exit']*100:7.1f}% {e['fr_entry']*100:7.1f}% {swap_bps:8.2f}")

    print("\n=== Sharpe ladder on the ACTUAL strategy (real positions) ===")
    print(f"{'broker':10s} {'GROSS':>6s} {'HOLD':>6s} {'OV_market':>10s} {'OV_limit(50%cap)':>16s}")
    dl = ladder(dwx_econ, SPECS_BY_CANON())
    print(f"{'Darwinex':10s} {dl['gross']:6.2f} {dl['hold']:6.2f} {dl['ov_mkt']:10.2f} {dl['ov_lim']:16.2f}")
    if ftmo_econ:
        fl = ladder(ftmo_econ, FTMO_SPEC)
        print(f"{'FTMO':10s} {fl['gross']:6.2f} {fl['hold']:6.2f} {fl['ov_mkt']:10.2f} {fl['ov_lim']:16.2f}")
    else:
        print("FTMO events not built yet (run build_events_ftmo).")


def SPECS_BY_CANON() -> dict:
    return {c: SPECS[cfd] for c, cfd in DWX.items()}


if __name__ == "__main__":
    main()
