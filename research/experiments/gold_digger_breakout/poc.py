"""
Phase-0 frictionless vectorized POC for the "Gold Digger" / Gen_Breakout EA.

Strategy (Asian-session range breakout, faithful to the MQL5 EA):
  - Range = [high, low] of M1 bars in the broker-time window [rs_h:00, re_h:00).
    Defaults 03:00-06:00 broker (EET/EEST).
  - After the window closes, arm stop orders at both edges. The FIRST post-window
    M1 bar to trade through an edge enters that side at the edge price.
  - Stop-loss = the OPPOSITE edge  =>  loss is exactly -1R (R = range width).
  - No take-profit. Force-flat at the daily close (close_h:00, default 18:00 broker).
  - At most ONE filled trade per session (opposite pending is cancelled on fill).

P&L unit = R-multiples (the EA's percent-risk sizing makes risk-per-trade constant;
range-width-dependent lot scaling normalizes away => every trade is measured in R).

Lookahead discipline:
  - Range uses ONLY bars with sec in [RS, RE).
  - Entry/SL/time-exit use ONLY bars with sec in [RE, CL).
  - Frictionless losses MUST equal -1R exactly (asserted) -> catches geometry bugs.
  - Total R is re-derived two independent ways (per-trade sum vs daily series) and
    asserted equal.

Broker-time note: MT5 stamps are tz-tagged UTC but are actually broker wall-clock
(EET/EEST). We strip the tz; the EA's hour thresholds are in that same broker clock,
so no shift is needed. Real dense M1 starts 2018 -> floor there.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

MT5_ROOT = Path("data/mt5_data")
INTRADAY_FLOOR_YEAR = 2018


# ----------------------------------------------------------------------------- #
#  Data                                                                         #
# ----------------------------------------------------------------------------- #
def load_m1(symbol: str, y0: int = 2018, y1: int = 2026) -> pd.DataFrame:
    files = sorted(MT5_ROOT.joinpath(symbol, "bars_M1").glob("year=*/part.parquet"))
    files = [f for f in files if y0 <= int(f.parent.name.split("=")[1]) <= y1]
    if not files:
        raise FileNotFoundError(f"no M1 for {symbol} in {y0}-{y1}")
    df = pd.concat(
        [pd.read_parquet(f, columns=["time", "open", "high", "low", "close", "spread"])
         for f in files],
        ignore_index=True,
    )
    t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)  # broker wall-clock
    out = pd.DataFrame({
        "dt": t.values,
        "sec": (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(np.int32).values,
        "open": df["open"].astype(np.float64).values,
        "high": df["high"].astype(np.float64).values,
        "low": df["low"].astype(np.float64).values,
        "close": df["close"].astype(np.float64).values,
        "spread_pts": df["spread"].astype(np.float64).values,
    })
    out = out.sort_values("dt").reset_index(drop=True)
    out["day"] = out["dt"].values.astype("datetime64[D]")
    return out


def prep_days(df: pd.DataFrame) -> list[dict]:
    """Pre-split into per-day numpy arrays once, so config sweeps are cheap."""
    days = []
    for day, g in df.groupby("day", sort=True):
        days.append({
            "day": day,
            "sec": g["sec"].values,
            "open": g["open"].values,
            "high": g["high"].values,
            "low": g["low"].values,
            "close": g["close"].values,
            "spread_pts": g["spread_pts"].values,
        })
    return days


# ----------------------------------------------------------------------------- #
#  Single-day resolver                                                          #
# ----------------------------------------------------------------------------- #
def resolve_day(rhigh, rlow, h_op, h_hi, h_lo, h_cl, gap_fill, be_R, h_sp=None):
    """Resolve one session: find breakout entry, then SL / BE / time exit.

    Returns a dict with R-multiple outcome, or None if no breakout occurred.
    Conventions:
      - SL has intrabar priority over the BE-arm on the same bar (pessimistic).
      - straddle (one bar engulfs the whole range on the entry bar) -> -1R, side=0.
    """
    width = rhigh - rlow
    n = len(h_hi)

    # --- entry: first hold bar to trade through an edge ---
    entry_idx = -1
    side = 0
    for i in range(n):
        lt = h_hi[i] >= rhigh
        st = h_lo[i] <= rlow
        if lt or st:
            entry_idx = i
            if lt and st:  # straddle: ambiguous tick path -> conservative -1R
                return dict(side=0, reason="straddle", R=-1.0,
                            entry=np.nan, exit=np.nan, width=width, entry_idx=i,
                            entry_sp=(h_sp[i] if h_sp is not None else np.nan))
            side = 1 if lt else -1
            break
    if entry_idx < 0:
        return None  # no breakout -> pendings expire, flat day

    entry_sp = h_sp[entry_idx] if h_sp is not None else np.nan
    if side == 1:
        entry = max(rhigh, h_op[entry_idx]) if gap_fill else rhigh
        sl = rlow
    else:
        entry = min(rlow, h_op[entry_idx]) if gap_fill else rlow
        sl = rhigh

    be_armed = False
    for j in range(entry_idx, n):
        if side == 1:
            if h_lo[j] <= sl:  # stop hit (SL, or BE if already armed to entry)
                exit_px = (min(sl, h_op[j]) if gap_fill else sl)
                return dict(side=side, reason=("BE" if be_armed else "SL"),
                            R=(exit_px - entry) / width, entry=entry, exit=exit_px,
                            width=width, entry_idx=entry_idx, entry_sp=entry_sp)
            if be_R is not None and not be_armed and h_hi[j] >= entry + be_R * width:
                be_armed = True
                sl = entry
                if h_lo[j] <= sl:  # pessimistic: BE could be tapped same bar after arming
                    return dict(side=side, reason="BE", R=(sl - entry) / width,
                                entry=entry, exit=sl, width=width,
                                entry_idx=entry_idx, entry_sp=entry_sp)
        else:
            if h_hi[j] >= sl:
                exit_px = (max(sl, h_op[j]) if gap_fill else sl)
                return dict(side=side, reason=("BE" if be_armed else "SL"),
                            R=(entry - exit_px) / width, entry=entry, exit=exit_px,
                            width=width, entry_idx=entry_idx, entry_sp=entry_sp)
            if be_R is not None and not be_armed and h_lo[j] <= entry - be_R * width:
                be_armed = True
                sl = entry
                if h_hi[j] >= sl:  # pessimistic: BE could be tapped same bar after arming
                    return dict(side=side, reason="BE", R=(entry - sl) / width,
                                entry=entry, exit=sl, width=width,
                                entry_idx=entry_idx, entry_sp=entry_sp)

    # --- time exit at the last hold bar's close (~daily close) ---
    exit_px = h_cl[-1]
    return dict(side=side, reason="time", R=(exit_px - entry) / width * side,
                entry=entry, exit=exit_px, width=width, entry_idx=entry_idx,
                entry_sp=entry_sp)


def build_trades(days: list[dict], rs_h=3, re_h=6, close_h=18,
                 gap_fill=False, be_R=None, min_width=0.0) -> pd.DataFrame:
    RS, RE, CL = rs_h * 3600, re_h * 3600, close_h * 3600
    rows = []
    for d in days:
        sec = d["sec"]
        rmask = (sec >= RS) & (sec < RE)
        if not rmask.any():
            continue
        rhigh = d["high"][rmask].max()
        rlow = d["low"][rmask].min()
        if rhigh - rlow <= min_width:
            continue
        hmask = (sec >= RE) & (sec < CL)
        if not hmask.any():
            continue
        res = resolve_day(rhigh, rlow, d["open"][hmask], d["high"][hmask],
                          d["low"][hmask], d["close"][hmask], gap_fill, be_R,
                          h_sp=d["spread_pts"][hmask])
        if res is None:
            continue
        res["day"] = d["day"]
        rows.append(res)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
#  Stats + verification                                                         #
# ----------------------------------------------------------------------------- #
def verify(tr: pd.DataFrame, gap_fill: bool, be_R) -> None:
    """Geometry/consistency asserts (NOT a lookahead check).

    These constrain the P&L arithmetic — frictionless losses are exactly -1R,
    the daily and per-trade totals reconcile, and R == (exit-entry)/width*side.
    They do NOT prove temporal correctness: a lookahead bug would still pass.
    The lookahead guarantee comes from (a) the structural window masks
    ([RS,RE) for the range, [RE,CL) for entries/exits) and (b) the independent
    clean-room re-derivation, which reproduced these numbers to 5-6 decimals.
    """
    R = tr["R"].values
    # (1) frictionless losses (SL / straddle, no BE) must be exactly -1R
    if not gap_fill:
        loss_mask = tr["reason"].isin(["SL", "straddle"]).values
        if loss_mask.any():
            assert np.allclose(R[loss_mask], -1.0, atol=1e-9), \
                "frictionless SL/straddle losses are not exactly -1R"
    # (2) total R two ways: per-trade sum vs daily-series sum
    daily = tr.groupby("day")["R"].sum()
    assert abs(daily.sum() - R.sum()) < 1e-6, "daily-vs-pertrade total R mismatch"
    # (3) R recomputed from entry/exit/side for non-straddle trades
    m = tr["side"] != 0
    rr = (tr.loc[m, "exit"] - tr.loc[m, "entry"]) / tr.loc[m, "width"] * tr.loc[m, "side"]
    assert np.allclose(rr.values, tr.loc[m, "R"].values, atol=1e-9), \
        "R != (exit-entry)/width*side"


def stats(tr: pd.DataFrame, all_days: np.ndarray) -> dict:
    R = tr["R"].values
    n = len(R)
    wins = R[R > 0]
    losses = R[R < 0]
    gross_w = wins.sum()
    gross_l = -losses.sum()
    pf = gross_w / gross_l if gross_l > 0 else np.inf
    # daily R series over ALL trading days present in the data (0 on no-trade days)
    dseries = pd.Series(0.0, index=pd.Index(np.unique(all_days)))
    dseries.loc[tr["day"].values] = tr.groupby("day")["R"].sum()
    sharpe_d = (dseries.mean() / dseries.std() * np.sqrt(252)) if dseries.std() > 0 else 0.0
    # equity & maxDD in R units (per-trade ordered by day)
    eq = np.cumsum(tr.sort_values("day")["R"].values)
    dd = eq - np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    n_years = (np.unique(all_days).max() - np.unique(all_days).min()).astype("timedelta64[D]").astype(int) / 365.25
    return dict(
        n_trades=n,
        n_straddle=int((tr["side"] == 0).sum()),
        win_rate=len(wins) / n if n else 0.0,
        avg_R=R.mean() if n else 0.0,
        median_R=np.median(R) if n else 0.0,
        total_R=R.sum(),
        R_per_year=R.sum() / n_years if n_years else 0.0,
        profit_factor=pf,
        sharpe_daily=sharpe_d,
        maxdd_R=dd.min() if len(dd) else 0.0,
        n_time_exits=int((tr["reason"] == "time").sum()),
        n_sl=int((tr["reason"] == "SL").sum()),
        n_be=int((tr["reason"] == "BE").sum()),
    )


def by_year(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.copy()
    tr["year"] = pd.to_datetime(tr["day"]).dt.year
    rows = []
    for y, g in tr.groupby("year"):
        R = g["R"].values
        gw = R[R > 0].sum()
        gl = -R[R < 0].sum()
        rows.append(dict(
            year=y, n=len(R), win_rate=round((R > 0).mean(), 3),
            total_R=round(R.sum(), 2), avg_R=round(R.mean(), 3),
            pf=round(gw / gl, 2) if gl > 0 else np.inf,
        ))
    return pd.DataFrame(rows)


def run(symbol, rs_h=3, re_h=6, close_h=18, gap_fill=False, be_R=None,
        y0=2018, y1=2026, df=None):
    if df is None:
        df = load_m1(symbol, y0, y1)
    days = prep_days(df)
    all_days = df["day"].values
    tr = build_trades(days, rs_h, re_h, close_h, gap_fill, be_R)
    verify(tr, gap_fill, be_R)
    s = stats(tr, all_days)
    return tr, s, by_year(tr), df


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    for sym in ["XAUUSD", "USDJPY"]:
        print(f"\n{'='*70}\n{sym}  |  range 03:00-06:00 broker, flat 18:00, frictionless\n{'='*70}")
        tr, s, yr, _ = run(sym)
        for k, v in s.items():
            print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
        print(yr.to_string(index=False))
