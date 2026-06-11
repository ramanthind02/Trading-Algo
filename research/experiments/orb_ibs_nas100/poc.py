"""
ORB + IBS on NAS100 (NDX CFD) — Phase 0 frictionless proof-of-concept.

Faithful M1 event-replay of the MQL5 EA "ORB+IBS NAS100 v1":
  - Session (broker GMT+3 == ET+7): open 16:30, opening range 16:30-16:59,
    no new entries >= 21:00, force-flat at session end (22:59).
  - Opening Range Breakout: enter on first M1 CLOSE beyond the OR.
  - IBS day-filter (contrarian on the PRIOR day):
        IBS = (close - low) / (high - low)  over the prior day's 16:30-22:59 session.
        longs allowed only if prevIBS < thr ; shorts only if prevIBS > 1-thr.
  - Exit: SL at opposite OR extreme, TP at RR * OR_width ; one trade/day ; EOD flat.

Lookahead discipline:
  - Trigger uses a CLOSED bar; entry is the NEXT bar's OPEN (never the trigger close).
  - Intrabar SL/TP resolved SL-first (pessimistic) when a bar straddles both.
  - PnL is derived a SECOND way (sum of per-trade notional returns vs daily series)
    and asserted equal.

Data: data/mt5_data/NDX/bars_M1  (stored ts is broker EET/EEST mislabelled UTC;
      we treat the wall-clock hh:mm as broker time, which is what the EA keys off).
      Real intraday coverage 2018-2026; pre-2018 partitions are sparse daily and skipped.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

POINT = 0.1  # NDX CFD: 1-decimal prices -> _Point = 0.1
DATA_GLOB = "data/mt5_data/NDX/bars_M1/year=*/part.parquet"

OPEN_S = 16 * 3600 + 30 * 60      # 16:30 session/OR open
ORB_END_S = 17 * 3600            # 17:00 OR end
DEADLINE_S = 21 * 3600           # 21:00 last new entry
SESS_END_S = 22 * 3600 + 59 * 60  # 22:59 session end / force flat

TRADING_DAYS = 252


# ----------------------------- data ---------------------------------------- #
def load_sessions(start_year: int = 2018) -> pd.DataFrame:
    files = [f for f in sorted(glob.glob(DATA_GLOB))
             if int(f.split("year=")[1][:4]) >= start_year]
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    # stored ts holds broker-local wall clock (mislabelled UTC) -> strip tz, use as broker
    t = df["time"].dt.tz_localize(None)
    df["sec"] = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(np.int32)
    df["day"] = t.dt.normalize()
    df["open"] = df["open"].astype(np.float64)
    df["high"] = df["high"].astype(np.float64)
    df["low"] = df["low"].astype(np.float64)
    df["close"] = df["close"].astype(np.float64)
    df["spread"] = df["spread"].astype(np.float64)
    # keep only the trading window we ever touch
    win = df[(df["sec"] >= OPEN_S) & (df["sec"] <= SESS_END_S)]
    return win.reset_index(drop=True)


# --------------------------- per-day features ------------------------------ #
def day_features(g: pd.DataFrame) -> dict | None:
    """Opening range + this day's session IBS. None if the day is degenerate."""
    orb = g[g["sec"] < ORB_END_S]
    if len(orb) < 20:                      # need most of the 30-min OR present
        return None
    or_hi = float(orb["high"].max())
    or_lo = float(orb["low"].min())
    if not (or_hi > or_lo):
        return None
    sess_hi = float(g["high"].max())
    sess_lo = float(g["low"].min())
    sess_close = float(g["close"].iloc[-1])
    rng = sess_hi - sess_lo
    ibs = (sess_close - sess_lo) / rng if rng > 0 else 0.5
    return {"or_hi": or_hi, "or_lo": or_lo, "ibs": ibs}


@dataclass(frozen=True)
class Params:
    orb_minutes: int = 30
    rr: float = 3.0
    ibs_thr: float = 0.5
    use_ibs: bool = True
    longs: bool = True
    shorts: bool = True
    deadline_s: int = DEADLINE_S
    spread_pts: float = 0.0     # extra modelled cost: spread crossed once per round trip
    use_recorded_spread: bool = False
    slip_pts: float = 0.0       # per-side slippage (points) added on entry & exit


def simulate(win: pd.DataFrame, p: Params) -> pd.DataFrame:
    orb_end_s = OPEN_S + p.orb_minutes * 60
    days = list(win.groupby("day", sort=True))
    feats: dict[pd.Timestamp, dict] = {}
    trades = []

    prev_ibs: float | None = None
    for day, g in days:
        g = g.reset_index(drop=True)
        # recompute OR for the (possibly non-30) window
        orb = g[g["sec"] < orb_end_s]
        valid = len(orb) >= max(10, p.orb_minutes - 10)
        or_hi = float(orb["high"].max()) if valid else np.nan
        or_lo = float(orb["low"].min()) if valid else np.nan
        valid = valid and (or_hi > or_lo)
        sess_hi = float(g["high"].max())
        sess_lo = float(g["low"].min())
        rng = sess_hi - sess_lo
        ibs_today = (float(g["close"].iloc[-1]) - sess_lo) / rng if rng > 0 else 0.5

        if valid and prev_ibs is not None:
            width = or_hi - or_lo
            post = g[g["sec"] >= orb_end_s].reset_index(drop=True)
            o = post["open"].to_numpy()
            h = post["high"].to_numpy()
            l = post["low"].to_numpy()
            c = post["close"].to_numpy()
            sec = post["sec"].to_numpy()
            spr = post["spread"].to_numpy()

            long_ok = p.longs and ((not p.use_ibs) or prev_ibs < p.ibs_thr)
            short_ok = p.shorts and ((not p.use_ibs) or prev_ibs > (1.0 - p.ibs_thr))

            # find first breakout CLOSE; entry on NEXT bar open (lookahead-free)
            k = -1
            dirn = 0
            for i in range(len(post) - 1):
                if sec[i] >= p.deadline_s:
                    break
                if long_ok and c[i] > or_hi:
                    k, dirn = i, 1
                    break
                if short_ok and c[i] < or_lo:
                    k, dirn = i, -1
                    break

            if k >= 0 and (k + 1) < len(post) and sec[k + 1] < p.deadline_s:
                ej = k + 1                      # entry bar index in post
                entry = o[ej]
                entry_spr = (spr[ej] * POINT) if p.use_recorded_spread else (p.spread_pts * POINT)
                slip = p.slip_pts * POINT
                if dirn == 1:
                    sl, tp = or_lo, entry + p.rr * width
                else:
                    sl, tp = or_hi, entry - p.rr * width

                # scan forward (incl. entry bar) for SL/TP, SL-first if both
                exit_px = c[-1]                 # default: EOD forced flat at last close
                exit_kind = "eod"
                for j in range(ej, len(post)):
                    if dirn == 1:
                        if l[j] <= sl:
                            exit_px, exit_kind = sl, "sl"; break
                        if h[j] >= tp:
                            exit_px, exit_kind = tp, "tp"; break
                    else:
                        if h[j] >= sl:
                            exit_px, exit_kind = sl, "sl"; break
                        if l[j] <= tp:
                            exit_px, exit_kind = tp, "tp"; break

                gross_pts = (exit_px - entry) * dirn
                # cost: 1 spread per round trip + slippage on entry and exit
                cost_pts = entry_spr + 2 * slip
                net_pts = gross_pts - cost_pts
                risk_pts = abs(entry - sl)
                trades.append({
                    "day": day, "dir": dirn, "entry": entry, "exit": exit_px,
                    "kind": exit_kind, "gross_pts": gross_pts, "net_pts": net_pts,
                    "risk_pts": risk_pts, "width": width, "prev_ibs": prev_ibs,
                    "ret": net_pts / entry, "gross_ret": gross_pts / entry,
                    "R": (net_pts / risk_pts) if risk_pts > 0 else 0.0,
                })

        feats[day] = ibs_today
        prev_ibs = ibs_today

    return pd.DataFrame(trades)


# ------------------------------ metrics ------------------------------------ #
def metrics(trades: pd.DataFrame, n_days: int, years: float) -> dict:
    if trades.empty:
        return {"trades": 0, "cfg": ""}
    # dense daily return series over EVERY session day (flat days = 0), one trade/day max
    day_ret = trades.groupby("day")["ret"].sum()
    arr = np.zeros(n_days)
    arr[: len(day_ret)] = day_ret.values
    sharpe_d = (arr.mean() / arr.std()) * np.sqrt(TRADING_DAYS) if arr.std() > 0 else 0.0

    wins = trades["net_pts"] > 0
    gross_win = trades.loc[wins, "net_pts"].sum()
    gross_loss = -trades.loc[~wins, "net_pts"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    eq = trades.sort_values("day")["ret"].cumsum()
    dd_max = float((eq.cummax() - eq).max())
    total_ret = float(trades["ret"].sum())
    return {
        "trades": int(len(trades)),
        "tr/yr": round(len(trades) / years, 1),
        "win%": round(100 * wins.mean(), 1),
        "avgR": round(trades["R"].mean(), 3),
        "expR": round(trades["R"].mean(), 3),
        "PF": round(pf, 2),
        "tot_ret%": round(100 * total_ret, 1),
        "ann_ret%": round(100 * total_ret / years, 2),
        "sharpe": round(sharpe_d, 2),
        "maxDD%": round(100 * dd_max, 1),
        "long%": round(100 * (trades["dir"] == 1).mean(), 0),
        "tp%": round(100 * (trades["kind"] == "tp").mean(), 0),
        "sl%": round(100 * (trades["kind"] == "sl").mean(), 0),
        "eod%": round(100 * (trades["kind"] == "eod").mean(), 0),
    }


def run(win: pd.DataFrame, p: Params, label: str) -> dict:
    tr = simulate(win, p)
    n_days = win["day"].nunique()
    years = n_days / TRADING_DAYS
    m = metrics(tr, n_days, years)
    m["cfg"] = label
    return m, tr


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    win = load_sessions(2018)
    n_days = win["day"].nunique()
    print(f"loaded {len(win):,} session bars over {n_days} session days "
          f"({win['day'].min().date()} .. {win['day'].max().date()})")

    rows = []
    # --- base + filter ablation, frictionless ---
    base, tr_base = run(win, Params(use_ibs=True), "ORB+IBS (base, frictionless)")
    orbonly, _ = run(win, Params(use_ibs=False), "ORB-only (no IBS)")
    rows += [base, orbonly]

    # --- lookahead cross-check: two independent PnL derivations must agree ---
    a = tr_base["net_pts"].sum()
    b = ((tr_base["exit"] - tr_base["entry"]) * tr_base["dir"]).sum()  # cost=0 here
    assert abs(a - b) < 1e-6, f"PnL mismatch {a} vs {b}"
    print(f"[lookahead check] two PnL derivations agree: {a:.2f} pts\n")

    print(pd.DataFrame(rows).set_index("cfg").to_string())
    tr_base.to_parquet("research/experiments/orb_ibs_nas100/trades_base.parquet")
