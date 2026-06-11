"""
Concretum / Zarattini-Aziz-Barbon intraday momentum ("Beat the Market") — Phase 0
frictionless faithful replication + robustness engine.

Strategy (per the paper):
  * Noise Area: time-of-day boundaries from the average |move-from-open| over the
    previous `vol_lookback_tod` days, per time-of-day mark.
        move_{t-i,HH:MM} = | close_{t-i,HH:MM} / open_{t-i,9:30} - 1 |
        sigma_{t,HH:MM}  = mean_{i=1..N} move_{t-i,HH:MM}
        UB_{t,HH:MM} = max(open_{t,9:30}, close_{t-1,16:00}) * (1 + VM * sigma)
        LB_{t,HH:MM} = min(open_{t,9:30}, close_{t-1,16:00}) * (1 - VM * sigma)
  * Decisions ONLY at semi-hourly marks (10:00 .. 15:30 ET). Force-flat at 16:00.
  * Entry: flat & price>UB -> long ; flat & price<LB -> short.
  * Exit / stop (3 variants):
      - 'opp'       : hold until opposite band -> reverse (base model, Table 1).
      - 'curr'      : exit to flat when price re-enters current band; reverse on opp.
      - 'curr_vwap' : trailing stop = max(UB,VWAP) long / min(LB,VWAP) short;
                      reverse on opposite-band crossover (headline model, Table 2/3).
  * Vol-target sizing (Table 3): leverage_t = min(cap, sigma_target / sigma_daily,t),
    sigma_daily,t = stdev of last `vol_lookback_day` daily close-to-close returns.

Lookahead discipline (Phase-0 mandatory):
  * sigma_tod and prev_close use STRICTLY past days (shift(1) before rolling).
  * Decision at mark m uses price/VWAP/bands known AT m; the position chosen at m
    earns the return from m -> m+1 (never the same-bar return that informed it).
  * sigma_daily / leverage use daily returns up to t-1 (shift(1)).
  * Daily PnL is derived TWO ways (segment-sum vs position-path) and asserted equal.

Data: data/mt5_data/<SYM>/bars_M1/year=*/part.parquet  (broker EET/EEST mislabelled
UTC; ET = stored - 7h, constant year-round). We keep the broker wall-clock and map
the US cash session 09:30-16:00 ET -> 16:30-23:00 broker.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Broker-time grid (ET + 7h, constant)
# --------------------------------------------------------------------------- #
def _bsec(et_h: int, et_m: int) -> int:
    """ET hh:mm -> broker seconds-of-day."""
    return (et_h + 7) * 3600 + et_m * 60

OPEN_S = _bsec(9, 30)      # 16:30 broker = 09:30 ET cash open
CLOSE_S = _bsec(16, 0)     # 23:00 broker = 16:00 ET cash close

# semi-hourly decision marks 10:00..15:30 ET (first tradable mark is 10:00)
MARKS_ET_SEMI = [(h, m) for h in range(10, 16) for m in (0, 30)]      # 12 marks
MARKS_ET_HOUR = [(h, 0) for h in range(10, 16)]                       # 6 marks
MARKS_ET_Q = [(h, m) for h in range(10, 16) for m in (0, 15, 30, 45)]  # 15-min, 24
MARKS_ET_LAST30 = [(15, 30)]                                          # academic baseline

TRADING_DAYS = 252


def marks_broker_sec(marks_et) -> list[int]:
    return [_bsec(h, m) for (h, m) in marks_et]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_sessions(sym: str, y0: int, y1: int) -> pd.DataFrame:
    """Load RTH M1 bars for one symbol across [y0, y1] inclusive."""
    files = [
        f for f in sorted(glob.glob(f"data/mt5_data/{sym}/bars_M1/year=*/part.parquet"))
        if y0 <= int(f.split("year=")[1][:4]) <= y1
    ]
    if not files:
        raise FileNotFoundError(f"no M1 partitions for {sym} in {y0}-{y1}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    t = df["time"]
    try:
        t = t.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    sec = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(np.int32)
    out = pd.DataFrame({
        "day": t.dt.normalize(),
        "sec": sec,
        "open": df["open"].astype(np.float64),
        "high": df["high"].astype(np.float64),
        "low": df["low"].astype(np.float64),
        "close": df["close"].astype(np.float64),
        "vol": df["tick_volume"].astype(np.float64),
        "spread": df["spread"].astype(np.float64) if "spread" in df else 0.0,
    })
    win = out[(out["sec"] >= OPEN_S) & (out["sec"] <= CLOSE_S)]
    return win.sort_values(["day", "sec"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-day feature table (open, prev_close, mark prices, mark VWAPs, daily ret)
# --------------------------------------------------------------------------- #
def build_day_table(win: pd.DataFrame, marks_et) -> pd.DataFrame:
    """One row per session day with everything the simulator needs.

    Columns: open, close, daily_ret, and for each mark k: px_k, vwap_k, move_k,
    plus spread_med (median recorded spread in points across RTH, for cost models).
    """
    mark_s = marks_broker_sec(marks_et)
    nmk = len(mark_s)
    rows = []
    for day, g in win.groupby("day", sort=True):
        sec = g["sec"].to_numpy()
        o = g["open"].to_numpy()
        h = g["high"].to_numpy()
        lo = g["low"].to_numpy()
        c = g["close"].to_numpy()
        vol = g["vol"].to_numpy()
        spr = g["spread"].to_numpy()
        if len(g) < 30:
            continue
        open_px = float(o[0])               # open of the 09:30 bar
        if open_px <= 0:
            continue
        # cumulative VWAP over the day (typical price * volume)
        typ = (h + lo + c) / 3.0
        cum_pv = np.cumsum(typ * vol)
        cum_v = np.cumsum(vol)
        # close = last bar at/<= CLOSE_S (already filtered <= CLOSE_S)
        close_px = float(c[-1])
        rec = {"day": day, "open": open_px, "close": close_px,
               "spread_med": float(np.median(spr)) if len(spr) else 0.0}
        for k, ms in enumerate(mark_s):
            idx = np.searchsorted(sec, ms, side="right") - 1  # last bar at/<= mark
            if idx < 0:
                rec[f"px_{k}"] = np.nan
                rec[f"vwap_{k}"] = np.nan
                rec[f"move_{k}"] = np.nan
            else:
                px = float(c[idx])
                rec[f"px_{k}"] = px
                rec[f"vwap_{k}"] = float(cum_pv[idx] / cum_v[idx]) if cum_v[idx] > 0 else px
                rec[f"move_{k}"] = abs(px / open_px - 1.0)
        rows.append(rec)
    dt = pd.DataFrame(rows).set_index("day").sort_index()
    dt["prev_close"] = dt["close"].shift(1)
    dt["daily_ret"] = dt["close"] / dt["close"].shift(1) - 1.0
    return dt


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    vm: float = 1.0                  # volatility multiplier (band width)
    vol_lookback_tod: int = 14       # days in the time-of-day sigma
    stop_mode: str = "curr_vwap"     # 'opp' | 'curr' | 'curr_vwap'
    longs: bool = True
    shorts: bool = True
    # vol-target sizing
    sizing: str = "vol_target"       # 'binary' (lev=1) | 'vol_target'
    vol_target: float = 0.02         # daily target vol
    vol_lookback_day: int = 14
    lev_cap: float = 4.0
    # costs (Phase-1 preview): bps of notional per ONE-WAY unit traded
    cost_bps_per_turn: float = 0.0
    use_recorded_spread: bool = False  # add half the recorded spread each side as cost
    point_size: float = 0.01           # price per 1 unit of the MT5 `spread` column


# --------------------------------------------------------------------------- #
# State machine (one mark)
# --------------------------------------------------------------------------- #
def _step(pos: int, price: float, ub: float, lb: float, vwap: float, mode: str) -> int:
    if pos != 0:
        if mode == "opp":
            if pos == 1 and price < lb:
                return -1
            if pos == -1 and price > ub:
                return 1
            return pos
        # curr / curr_vwap: opposite-band crossover reverses; tighter stop -> flat
        if mode == "curr":
            ls, ss = ub, lb
        else:  # curr_vwap
            ls, ss = max(ub, vwap), min(lb, vwap)
        if pos == 1:
            if price < lb:
                return -1
            if price <= ls:
                return 0
            return 1
        else:  # pos == -1
            if price > ub:
                return 1
            if price >= ss:
                return 0
            return -1
    # flat -> entry
    if price > ub:
        return 1
    if price < lb:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Simulate
# --------------------------------------------------------------------------- #
def simulate(dt: pd.DataFrame, p: Params, nmk: int) -> pd.DataFrame:
    """Return a per-day frame: unlev / lev / net returns, turnover, leverage."""
    # time-of-day sigma: strictly-past rolling mean of move_k
    sig = {}
    for k in range(nmk):
        sig[k] = dt[f"move_{k}"].shift(1).rolling(p.vol_lookback_tod).mean()
    # daily vol for sizing (returns up to t-1)
    sigma_day = dt["daily_ret"].shift(1).rolling(p.vol_lookback_day).std(ddof=1)
    if p.sizing == "vol_target":
        lev = np.minimum(p.lev_cap, p.vol_target / sigma_day)
    else:
        lev = pd.Series(1.0, index=dt.index)
    lev = lev.replace([np.inf, -np.inf], p.lev_cap).clip(lower=0.0).fillna(0.0)

    days = dt.index.to_numpy()
    open_px = dt["open"].to_numpy()
    prev_close = dt["prev_close"].to_numpy()
    close_px = dt["close"].to_numpy()
    px = np.column_stack([dt[f"px_{k}"].to_numpy() for k in range(nmk)])      # (D, nmk)
    vwap = np.column_stack([dt[f"vwap_{k}"].to_numpy() for k in range(nmk)])
    sig_arr = np.column_stack([sig[k].to_numpy() for k in range(nmk)])
    lev_arr = lev.to_numpy()
    spread_med = dt["spread_med"].to_numpy()

    out_unlev = np.zeros(len(dt))
    out_turn = np.zeros(len(dt))
    out_ntr = np.zeros(len(dt))
    valid = np.zeros(len(dt), dtype=bool)

    for d in range(len(dt)):
        base = max(open_px[d], prev_close[d])
        basel = min(open_px[d], prev_close[d])
        if not np.isfinite(base) or not np.isfinite(basel) or base <= 0:
            continue
        s = sig_arr[d]
        if not np.isfinite(s).all():
            continue           # warmup
        pxd = px[d]
        if not np.isfinite(pxd).all() or not np.isfinite(close_px[d]):
            continue
        valid[d] = True
        ub = base * (1.0 + p.vm * s)
        lb = basel * (1.0 - p.vm * s)
        vw = vwap[d]
        # walk the marks; price grid has nmk marks + the forced-close price
        pos = 0
        positions = np.empty(nmk, dtype=np.int8)
        for k in range(nmk):
            np_pos = _step(pos, pxd[k], ub[k], lb[k], vw[k], p.stop_mode)
            if np_pos == 1 and not p.longs:
                np_pos = 0
            if np_pos == -1 and not p.shorts:
                np_pos = 0
            pos = np_pos
            positions[k] = pos
        # segment returns: mark k -> mark k+1 ; last mark -> close
        grid = np.concatenate([pxd, [close_px[d]]])          # nmk+1 points
        seg = grid[1:] / grid[:-1] - 1.0                     # nmk segments
        unlev = float(np.dot(positions.astype(np.float64), seg))
        # turnover: 0 -> p0 -> p1 -> ... -> p_{nmk-1} -> 0 (forced close)
        path = np.concatenate([[0], positions, [0]]).astype(np.float64)
        turn = float(np.abs(np.diff(path)).sum())            # one-way units traded
        ntr = float((np.abs(np.diff(path)) > 0).sum())
        out_unlev[d] = unlev
        out_turn[d] = turn
        out_ntr[d] = ntr

    res = pd.DataFrame(index=dt.index)
    res["valid"] = valid
    res["lev"] = np.where(valid, lev_arr, 0.0)
    res["unlev_ret"] = out_unlev
    res["turn"] = out_turn
    res["n_tr"] = out_ntr
    # cost per one-way unit (bps) + optional recorded half-spread each side
    cost_bps = p.cost_bps_per_turn
    spr_cost = np.zeros(len(dt))
    if p.use_recorded_spread:
        # half the recorded spread (points -> price) / open price, per one-way unit
        spr_cost = (0.5 * spread_med * p.point_size) / np.where(open_px > 0, open_px, np.nan)
        spr_cost = np.nan_to_num(spr_cost)
    res["cost_unlev"] = out_turn * (cost_bps * 1e-4) + out_turn * spr_cost
    res["net_unlev"] = res["unlev_ret"] - res["cost_unlev"]
    res["lev_ret"] = res["lev"] * res["unlev_ret"]
    res["net_lev"] = res["lev"] * res["net_unlev"]
    return res


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0


def _maxdd(r: np.ndarray) -> float:
    eq = np.cumsum(r)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


def metrics(res: pd.DataFrame, col: str = "net_lev") -> dict:
    sub = res[res["valid"]]
    if sub.empty:
        return {"days": 0}
    r = sub[col].to_numpy()
    r = np.where(np.isfinite(r), r, 0.0)      # untradeable/NaN-leverage days -> flat
    years = len(sub) / TRADING_DAYS
    eq = np.cumsum(r)
    rr = r[np.isfinite(r)]
    sk = float(pd.Series(rr).skew()) if len(rr) > 2 else 0.0
    pos = rr[rr > 0].sum()
    neg = -rr[rr < 0].sum()
    return {
        "days": int(len(sub)),
        "years": round(years, 2),
        "sharpe": round(_sharpe(r), 3),
        "ann_ret%": round(100 * r.mean() * TRADING_DAYS, 2),
        "vol%": round(100 * r.std() * np.sqrt(TRADING_DAYS), 2),
        "maxDD%": round(100 * _maxdd(r), 2),
        "tot_ret%": round(100 * eq[-1], 1),
        "hit%": round(100 * (rr > 0).mean(), 1),
        "trade_days%": round(100 * (sub["n_tr"] > 0).mean(), 1),
        "skew": round(sk, 2),
        "worst%": round(100 * rr.min(), 2),
        "best%": round(100 * rr.max(), 2),
        "pf": round(pos / neg, 2) if neg > 0 else float("inf"),
        "tr/day": round(sub["n_tr"].mean(), 2),
        "turn/day": round(sub["turn"].mean(), 2),
        "lev_med": round(float(sub["lev"].median()), 2),
    }


# --------------------------------------------------------------------------- #
# Convenience driver: load -> table -> simulate -> metrics
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def run(sym: str, y0: int, y1: int, p: Params, marks_et=MARKS_ET_SEMI,
        col: str = "net_lev") -> tuple[dict, pd.DataFrame]:
    key = (sym, y0, y1, tuple(marks_et))
    if key not in _CACHE:
        win = load_sessions(sym, y0, y1)
        _CACHE[key] = build_day_table(win, marks_et)
    dt = _CACHE[key]
    res = simulate(dt, p, len(marks_et))
    m = metrics(res, col)
    return m, res
