r"""Three-venue rollover comparison: Darwinex vs FTMO vs FundedNext.

Same experiments as the two-broker study, extended to a third prop venue. Reports, on the
SAME actual vault position path (ES/NQ/GC, 2023-2026):

  1. SWAPS   — exact for every venue and horizon (contractual POINTS rate from symbol_info),
               so this is a true 3-year comparison regardless of tick reach.
  2. SPREADS — per-leg exit/reopen half-spread + fill rates from each venue's own scraped
               ticks (events_{}.parquet / events_ftmo_{} / events_fundednext_{}). Sample
               length differs by venue (FundedNext serves only ~3 weeks of tick history).
  3. LADDER  — the full-edge Sharpe ladder (GROSS / HOLD / OV-market / OV-full) per venue,
               = 3-year exact swap + per-leg representative spread + drift-timing, with the
               settled market-order execution policy (exit T-15, enter post-decay).

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.broker_compare
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
from research.rollover_cost.actual_swap_cost import daily_close, _triple_py

SNAP = _REPO / "tests/parity/snapshots/portfolio_research"
OUT = _REPO / "research" / "rollover_cost" / "outputs"
ANN = 252
CANON = ("ES", "NQ", "GC")
DWX_SYM = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD"}

# --- per-venue swap specs (POINTS/lot/day, long/short) from live symbol_info probes ----------
FTMO_SPEC = {
    "ES": SymbolSpec("ES", 0.01, 0.01, 1.0, -154.67, -3.09, 5),
    "NQ": SymbolSpec("NQ", 0.01, 0.01, 1.0, -619.48, -12.41, 5),
    "GC": SymbolSpec("GC", 0.01, 0.01, 100.0, -86.40, -15.30, 3),
}
# FundedNext probe 2026-06-06 (SPX500/NDX100/XAUUSD): point, tick, csize, swap_long, swap_short, triple_wd
FUNDEDNEXT_SPEC = {
    "ES": SymbolSpec("ES", 0.01, 0.01, 10.0, -104.64, 19.96, 5),
    "NQ": SymbolSpec("NQ", 0.01, 0.01, 10.0, -329.38, 62.82, 5),
    "GC": SymbolSpec("GC", 0.01, 0.01, 100.0, -57.94, -15.70, 3),
}
VENUES = {
    "Darwinex":   (lambda c: OUT / f"events_{DWX_SYM[c]}.parquet",   {c: SPECS[DWX_SYM[c]] for c in CANON}),
    "FTMO":       (lambda c: OUT / f"events_ftmo_{c}.parquet",       FTMO_SPEC),
    "FundedNext": (lambda c: OUT / f"events_fundednext_{c}.parquet", FUNDEDNEXT_SPEC),
}

# representative recent price per leg (for the per-night swap-bps headline only)
REF_PX = {"ES": 6200.0, "NQ": 28000.0, "GC": 3300.0}


def _load(evfile, canon):
    p = evfile(canon)
    return pd.read_parquet(p) if p.exists() else None


def _sharpe(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(ANN) if s.std() else float("nan")


def per_leg_econ(evfile, specmap):
    rows = {}
    for c in CANON:
        ev = _load(evfile, c)
        spec = specmap[c]
        swap_bps = abs(spec.swap_long_pts) * spec.point / REF_PX[c] * 1e4
        if ev is None or ev.empty:
            rows[c] = dict(n=0, exit_half=np.nan, reopen_half=np.nan,
                           fr_exit=np.nan, fr_entry=np.nan, swap_bps=swap_bps)
            continue
        exit_half = (ev["exit_spread_t15"] / 2 / ev["exit_mid_t15"] * 1e4).mean()
        reopen_half = (ev["entry_spread_p5"] / 2 / ev["entry_mid_p5"] * 1e4).mean()
        fr_exit = ev["exit_sell_limit_mid_fill_min"].notna().mean() if "exit_sell_limit_mid_fill_min" in ev else np.nan
        fr_entry = ev["entry_buylim_t5_filled_60m"].mean() if "entry_buylim_t5_filled_60m" in ev else np.nan
        rows[c] = dict(n=len(ev), exit_half=exit_half, reopen_half=reopen_half,
                       fr_exit=fr_exit, fr_entry=fr_entry, swap_bps=swap_bps)
    return rows


def ladder(evfile, specmap, gross, pos, idx):
    """Full-edge ladder over the 3yr position path: exact swap + per-leg spread + drift-timing."""
    swap = pd.Series(0.0, index=idx); execm = pd.Series(0.0, index=idx); timing = pd.Series(0.0, index=idx)
    for c in CANON:
        spec = specmap[c]; ev = _load(evfile, c)
        f = pos[pos["ticker"].astype(str) == c].set_index("datetime")["position_fraction"].reindex(idx).fillna(0.0)
        price = daily_close(DWX_SYM[c]).reindex(idx, method="ffill").to_numpy()
        mult = np.where(np.asarray(idx.weekday) == _triple_py(spec.triple_weekday), 3.0, 1.0)
        swap += pd.Series(f.abs().to_numpy() * np.abs(spec.swap_long_pts * spec.point / price * 1e4 * mult) / 1e4, index=idx)
        if ev is None or ev.empty:
            continue
        ev = ev.copy()
        ev["pdate"] = pd.to_datetime(ev["rollover"]).dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)
        ev["sp"] = ev["exit_spread_t15"] / 2 / ev["exit_mid_t15"] + ev["entry_spread_p5"] / 2 / ev["entry_mid_p5"]
        ev["mv"] = (ev["open_mid"] - ev["exit_mid_t15"]) / ev["exit_mid_t15"]
        em = ev.set_index("pdate")
        sp = em["sp"].reindex(idx); med = em["sp"].median()
        execm += pd.Series(f.abs().to_numpy() * sp.fillna(med).to_numpy(), index=idx)
        timing += pd.Series(f.to_numpy() * em["mv"].reindex(idx).fillna(0).to_numpy(), index=idx)
    return {"GROSS": _sharpe(gross), "HOLD": _sharpe(gross - swap),
            "OV-market": _sharpe(gross - execm), "OV-full": _sharpe(gross - execm - timing),
            "swap_yr": swap.sum(), "exec_yr": execm.sum(), "timing_yr": timing.sum()}


def main() -> None:
    gross = pd.read_parquet(SNAP / "portfolio_test_default__test__strategy_returns.parquet")["strategy_return"]
    gross.index = pd.to_datetime(gross.index).normalize()
    pos = pd.read_parquet(SNAP / "portfolio_test_default__test__positions.parquet")
    pos["datetime"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    idx = gross.index
    yrs = (idx.max() - idx.min()).days / 365.25

    print("=" * 78)
    print("1) SWAP per night (bps, longs pay) - EXACT, contractual; lower magnitude = cheaper")
    print(f"{'leg':4s} {'Darwinex':>10s} {'FTMO':>10s} {'FundedNext':>11s}")
    for c in CANON:
        d = abs(SPECS[DWX_SYM[c]].swap_long_pts) * SPECS[DWX_SYM[c]].point / REF_PX[c] * 1e4
        f = abs(FTMO_SPEC[c].swap_long_pts) * FTMO_SPEC[c].point / REF_PX[c] * 1e4
        n = abs(FUNDEDNEXT_SPEC[c].swap_long_pts) * FUNDEDNEXT_SPEC[c].point / REF_PX[c] * 1e4
        print(f"{c:4s} {d:10.2f} {f:10.2f} {n:11.2f}")

    print("\n" + "=" * 78)
    print("2) SPREADS + fills from each venue's own scraped ticks (sample n differs)")
    print(f"{'venue':11s} {'leg':4s} {'n':>4s} {'exit_half':>9s} {'reopen_half':>11s} {'fr_exit':>8s} {'fr_entry':>8s}")
    econ = {}
    for name, (evf, spc) in VENUES.items():
        econ[name] = per_leg_econ(evf, spc)
        for c in CANON:
            e = econ[name][c]
            print(f"{name:11s} {c:4s} {e['n']:4d} {e['exit_half']:9.2f} {e['reopen_half']:11.2f} "
                  f"{(e['fr_exit'] or float('nan'))*100:7.1f}% {(e['fr_entry'] or float('nan'))*100:7.1f}%")

    print("\n" + "=" * 78)
    print("3) FULL-EDGE Sharpe ladder on the ACTUAL 3yr vault book (market-order policy)")
    print(f"{'venue':11s} {'GROSS':>6s} {'HOLD':>6s} {'OV-market':>10s} {'OV-full':>8s} | "
          f"{'swap/yr':>8s} {'exec/yr':>8s} {'timing/yr':>9s}")
    for name, (evf, spc) in VENUES.items():
        L = ladder(evf, spc, gross, pos, idx)
        print(f"{name:11s} {L['GROSS']:6.2f} {L['HOLD']:6.2f} {L['OV-market']:10.2f} {L['OV-full']:8.2f} | "
              f"{L['swap_yr']/yrs*100:7.2f}% {-L['exec_yr']/yrs*100:7.2f}% {-L['timing_yr']/yrs*100:8.2f}%")


if __name__ == "__main__":
    main()
