r"""Integrated end-to-end simulation of the vault portfolio's financing/execution cost.

Replaces the layered approximation with one simulation on the ACTUAL position path:

  * GROSS alpha = the vault backtest's daily strategy return (frictionless).
  * Real per-trade SPREAD from the Darwinex M1 `spread` field at the actual trade minute
    (settlement ≈ 23:58 broker for the daily rebalance / overlay exit; reopen ≈ 01:01 broker
    for the overlay re-entry) — the genuine broker spread that day, not a constant.
  * Real nightly SWAP (contractual POINTS rate) on the held position at the real daily price.
  * The OVERLAY: on the chosen legs, flatten the full position before the rollover and re-enter
    the target after it (paying the real round-trip spread) instead of holding through (swap).

Costs are additive to the portfolio P&L, so net = gross − Σ_legs costs. Scenarios:

  GROSS         backtest, no friction.
  LIVE_HOLD     − daily-rebalance spread − nightly swap (hold every leg through the rollover).
  OVERLAY       overlay ES/NQ (round-trip spread, no swap) + hold GC/CL (rebalance spread + swap).

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.integrated_backtest
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
VAULT_TO_CFD = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD"}   # CL omitted (mode-3 swap, ~flat)
OVERLAY_LEGS = ("ES", "NQ")
FTMO_RATIO = {"ES": 1.43, "NQ": 1.37, "GC": 1.36}
ANN = 252


def _triple_py(d: int) -> int:
    return {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[d]


def daily_bars(cfd: str) -> pd.DataFrame:
    """Per broker-date: settlement close + spread (23:58 bar) + reopen spread (01:01 bar)."""
    parts = sorted((MT5 / cfd / "bars_M1").glob("year=*/part.parquet"))
    df = pd.concat([pd.read_parquet(p, columns=["time", "close", "spread"]) for p in parts], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_localize(None).dt.normalize()
    df["hhmm"] = df["time"].dt.strftime("%H:%M")
    g = df.groupby("date")
    close = g["close"].last()
    # spread at the last bar before the rollover (settlement/exit) and first after reopen
    def at(hhmm_options, which):
        sub = df[df["hhmm"].isin(hhmm_options)]
        s = sub.groupby("date")["spread"]
        return (s.last() if which == "last" else s.first())
    settle_sp = at(["23:55", "23:56", "23:57", "23:58", "23:59"], "last")
    reopen_sp = at(["01:00", "01:01", "01:02", "01:03", "01:04", "01:05"], "first")
    out = pd.DataFrame({"close": close, "settle_sp": settle_sp, "reopen_sp": reopen_sp})
    out["settle_sp"] = out["settle_sp"].fillna(out["settle_sp"].median())
    out["reopen_sp"] = out["reopen_sp"].fillna(out["reopen_sp"].median())
    return out


def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {"ann_return_pct": s.mean() * ANN * 100, "ann_vol_pct": s.std() * np.sqrt(ANN) * 100,
            "sharpe": s.mean() / s.std() * np.sqrt(ANN) if s.std() else float("nan")}


def _event_roundtrip_bps(capture_frac: float = 1.0) -> dict:
    """Per-instrument overlay round-trip cost (bps) from the REAL rollover TICKS.

    market : cross both legs — pay half the exit spread + half the (wide) reopen spread.
    limit  : post passive at the touch on both legs — CAPTURE the half-spread when filled
             (a maker fill, validated by the Nautilus engine), market-out the exit on a miss,
             SKIP the entry on a miss (the next rebalance reconciles). A negative value means
             the round-trip *earns* the spread (market-making). This is the at-touch *capture*
             model — the earlier flawed model credited only 5 ticks (~0.01 bps) and charged a
             30-min chase drift, which wrongly erased the benefit.
    cap    : ``capture_frac`` haircuts the maker capture for exit-leg adverse selection
             (1.0 = full capture / upper bound; 0.5 = half given back). Returns {cfd:(mkt,lim)}.
    """
    out = {}
    for cfd in VAULT_TO_CFD.values():
        ev = pd.read_parquet(OUT / f"events_{cfd}.parquet")
        hs_exit = float((ev["exit_spread_t15"] / 2 / ev["exit_mid_t15"] * 1e4).mean())
        hs_reopen = float((ev["open_spread"] / 2 / ev["open_mid"] * 1e4).mean())
        fr_e = float(ev["exit_sell_limit_mid_fill_min"].notna().mean())
        fr_n = float(ev["entry_buylim_t5_filled_60m"].mean())
        market = hs_exit + hs_reopen
        cf = capture_frac
        # exit: fill→capture cf*hs (else market-out pays hs); entry: fill→capture cf*hs (else skip 0)
        limit = (fr_e * (-cf * hs_exit) + (1 - fr_e) * hs_exit) + (fr_n * (-cf * hs_reopen))
        out[cfd] = (market, limit)
    return out


def build(swap_ratio: dict | None = None, capture_frac: float = 1.0) -> dict:
    gross = pd.read_parquet(SNAP / "portfolio_test_default__test__strategy_returns.parquet")["strategy_return"]
    gross.index = pd.to_datetime(gross.index).normalize()
    pos = pd.read_parquet(SNAP / "portfolio_test_default__test__positions.parquet")
    pos["datetime"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    idx = gross.index
    rt = _event_roundtrip_bps(capture_frac)   # tick-based round-trip cost per instrument

    # cost series per scenario (positive = drag on return)
    hold = pd.Series(0.0, index=idx)        # hold all, pay swap
    ov_mkt = pd.Series(0.0, index=idx)      # overlay ES/NQ market, hold GC
    ov_mkt_all = pd.Series(0.0, index=idx)  # overlay ALL legs, market orders
    ov_lim_all = pd.Series(0.0, index=idx)  # overlay ALL legs, LIMIT market-making
    ov_perfect = pd.Series(0.0, index=idx)  # overlay ALL legs, zero exec cost (ceiling)
    for tkr, cfd in VAULT_TO_CFD.items():
        spec = SPECS[cfd]
        f = pos[pos["ticker"].astype(str) == tkr].set_index("datetime")["position_fraction"].reindex(idx).fillna(0.0)
        bars = daily_bars(cfd).reindex(idx, method="ffill")
        price = bars["close"].to_numpy()
        hs_settle = bars["settle_sp"].to_numpy() * spec.point / 2.0 / price * 1e4
        absf = f.abs().to_numpy()
        dturn = f.diff().abs().fillna(f.abs()).to_numpy()
        is_long = (f > 0).to_numpy()
        swap_pts = np.where(is_long, spec.swap_long_pts, spec.swap_short_pts)
        mult = np.where(np.asarray(idx.weekday) == _triple_py(spec.triple_weekday), 3.0, 1.0)
        r = (swap_ratio or {}).get(tkr, 1.0)
        swap_cost = absf * np.abs(swap_pts * spec.point / price * 1e4 * mult * r) / 1e4   # >0 when paying
        # short legs EARN swap; keep sign by using the signed value where short:
        swap_signed = absf * (swap_pts * spec.point / price * 1e4 * mult * r) / 1e4       # signed
        swap_drag = -swap_signed                                                          # >0 = cost
        rebal = dturn * hs_settle / 1e4                                                   # daily-rebalance spread
        mkt_rt = absf * rt[cfd][0] / 1e4                                                  # overlay round-trip, market
        lim_rt = absf * rt[cfd][1] / 1e4                                                  # overlay round-trip, limit
        S = lambda a: pd.Series(a, index=idx).fillna(0.0)

        hold += S(rebal + swap_drag)
        ov_mkt += S(mkt_rt) if tkr in OVERLAY_LEGS else S(rebal + swap_drag)
        ov_mkt_all += S(mkt_rt)
        ov_lim_all += S(lim_rt)
        ov_perfect += S(np.zeros_like(absf))

    return {
        "gross": gross,
        "live": gross - hold,
        "overlay": gross - ov_mkt,
        "overlay_mkt_all": gross - ov_mkt_all,
        "overlay_lim_all": gross - ov_lim_all,
        "overlay_perfect": gross - ov_perfect,
    }


def main() -> None:
    full = build(swap_ratio=None, capture_frac=1.0)     # full maker capture (upper bound)
    half = build(swap_ratio=None, capture_frac=0.5)     # half capture (exit adverse-selection haircut)
    rows = {
        "GROSS (frictionless backtest)": _stats(full["gross"]),
        "LIVE_HOLD (hold all, pay swap)": _stats(full["live"]),
        "OVERLAY market, ES/NQ only (hold GC)": _stats(full["overlay"]),
        "OVERLAY market, ALL legs incl GC": _stats(full["overlay_mkt_all"]),
        "OVERLAY LIMIT, ALL legs (50% capture, conservative)": _stats(half["overlay_lim_all"]),
        "OVERLAY LIMIT, ALL legs (100% capture, upper bound)": _stats(full["overlay_lim_all"]),
    }
    print(f"Integrated end-to-end sim (Darwinex, real spreads + real swap), "
          f"{full['gross'].index.min().date()}..{full['gross'].index.max().date()}\n")
    print(f"{'scenario':52s} {'ann_ret%':>9s} {'ann_vol%':>9s} {'Sharpe':>7s}")
    g = rows["GROSS (frictionless backtest)"]
    for name, st in rows.items():
        d = st["sharpe"] - g["sharpe"]
        print(f"{name:52s} {st['ann_return_pct']:9.2f} {st['ann_vol_pct']:9.2f} {st['sharpe']:7.3f}   (dSh {d:+.3f})")
    print("\nLIMIT 'capture' = posting at the touch and earning the half-spread as a maker (validated")
    print("by the Nautilus engine). Round-trip is NEGATIVE (you earn the spread) so the limit overlay")
    print("can EXCEED the frictionless backtest. 100% = full capture; 50% haircuts exit adverse selection.")
    pd.DataFrame(rows).T.to_csv(OUT / "H_integrated_backtest.csv")
    print(f"\nWrote {OUT/'H_integrated_backtest.csv'}")


if __name__ == "__main__":
    main()
