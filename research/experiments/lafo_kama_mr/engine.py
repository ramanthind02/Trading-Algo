"""
LAFO / KAMA fair-value mean-reversion — Phase 0 frictionless engine + Phase 1 cost.

Faithful port of the NinjaTrader EA "LAFO_MeanReversion_Share" (a KAMA-based
instantiation of Xu et al. 2025 UCL "Advanced Signal Filtering for Mean Reversion"):

  fair_value = KAMA(period, fast, slow)            # Kaufman adaptive MA
  delta      = (close - fair_value) / fair_value   # relative mispricing
  LONG  when delta < -thr   (price below fair value -> expect reversion up)
  SHORT when delta > +thr   (price above fair value -> expect reversion down)
  EXIT: ATR stop + fixed R:R target  (+ forced flat at session close)

Beyond the literal EA we parameterise the levers the user asked to experiment with:
  * entry_mode : 'rel'  (fixed % deviation, the EA)  |  'z' (vol-normalised z of delta)
  * exit_mode  : 'bracket' (ATR stop + RR target, the EA)
               | 'revert'  (exit when price crosses back through fair value; ATR stop protects)
               | 'time'    (exit after N bars; ATR stop protects)
               | 'close'   (hold to session close; ATR stop protects)
  * sizing     : 'fixed' (1 unit) | 'vol_target' (leverage = min(cap, vt/sigma_day))
  * session    : 'rth'  (09:30-16:00 ET = 16:30-23:00 broker; zero overnight swap)
               | 'h24'  (full ~23h CFD session, flat at the 23:45 broker pre-rollover)
  * direction  : longs / shorts independently

Lookahead discipline (Phase-0 mandatory):
  * KAMA/ATR/delta at bar t use closes <= t (streaming, causal).
  * The signal is read on a CLOSED bar t; entry fills at the NEXT bar's OPEN (t+1),
    never the trigger close. Re-entry only once flat.
  * Intrabar stop/target resolved STOP-FIRST (pessimistic) when a bar straddles both.
  * PnL is re-derived a second, independent way (sum net_pts vs sum (exit-entry)*dir
    - costs) and asserted equal on every run.

Costs (Phase 1) are the REAL recorded Darwinex NDX figures (verified from 54.2M ticks):
  * NDX price increment POINT = 0.1  (the Nautilus catalog's 0.01 is wrong; using 0.01
    under-charges spread 10x — the source of a prior "spread~=0" artifact).
  * recorded round-trip spread ~0.9 idx pts RTH (the M1 `spread` column, integer MT5
    points x POINT, is the full bid/ask gap = crossed once per round trip).
  * commission 0; swap only matters if held across 00:00 broker (RTH avoids it).

Data: data/mt5_data/NDX/bars_M1/year=*/part.parquet  (broker EET/EEST mislabelled UTC;
      ET = stored - 7h, constant; real 1-min data 2018+). Self-contained scratch.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

POINT = 0.1            # NDX CFD price increment (live-probed; catalog 0.01 is WRONG)
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Broker-time grid (ET + 7h, constant year-round)
# --------------------------------------------------------------------------- #
def _bsec(et_h: int, et_m: int) -> int:
    """ET hh:mm -> broker seconds-of-day."""
    return (et_h + 7) * 3600 + et_m * 60


RTH_OPEN_S = _bsec(9, 30)      # 16:30 broker = 09:30 ET cash open
RTH_CLOSE_S = _bsec(16, 0)     # 23:00 broker = 16:00 ET cash close
H24_OPEN_S = 1 * 3600 + 5 * 60       # 01:05 broker (just after the 00:00-01:00 dead zone)
H24_CLOSE_S = 23 * 3600 + 45 * 60    # 23:45 broker = T-15 pre-rollover flat (swap-dodge)

_RESAMPLE = {"M5": "5min", "M15": "15min", "M30": "30min", "M1": "1min"}


# --------------------------------------------------------------------------- #
# Data: load M1, window to session, resample
# --------------------------------------------------------------------------- #
def load_m1(sym: str, y0: int, y1: int, session: str) -> pd.DataFrame:
    """Raw M1 bars for *sym* over [y0,y1], windowed to the session, broker wall-clock.

    Returns columns [datetime, sec, day, open, high, low, close, volume, spread].
    For h24 the first intraday bar of each session is de-staled (the 01:00 carried
    open spike) to its own close, mirroring cfd_candles / vault_intraday.data_io.
    """
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
        "datetime": t,
        "sec": sec,
        "day": t.dt.normalize(),
        "open": df["open"].astype(np.float64),
        "high": df["high"].astype(np.float64),
        "low": df["low"].astype(np.float64),
        "close": df["close"].astype(np.float64),
        "volume": df["tick_volume"].astype(np.float64),
        "spread": (df["spread"].astype(np.float64) if "spread" in df else 0.0),
    })
    if session == "rth":
        o_s, c_s = RTH_OPEN_S, RTH_CLOSE_S
    elif session == "h24":
        o_s, c_s = H24_OPEN_S, H24_CLOSE_S
    else:
        raise ValueError(session)
    win = out[(out["sec"] >= o_s) & (out["sec"] < c_s)].sort_values("datetime").reset_index(drop=True)
    if session == "h24":
        # de-stale the carried session-open bar (first bar of each dense session)
        bars_per_day = win.groupby("day")["close"].transform("size")
        first_idx = win.groupby("day", sort=False).head(1).index
        dense_first = first_idx[bars_per_day.loc[first_idx].to_numpy() > 100]
        cv = win.loc[dense_first, "close"].to_numpy()
        for col in ("open", "high", "low"):
            win.loc[dense_first, col] = cv
    return win


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample windowed M1 to *tf* (left-closed/left-labelled, causal close).

    Overnight/weekend bins are empty and dropped (dropna on open), so a resampled
    bar never spans the session gap. `spread` -> median (per-bar half-spread proxy).
    """
    if tf == "M1":
        return m1.reset_index(drop=True)
    agg = (
        m1.set_index("datetime")
        .resample(_RESAMPLE[tf], label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), spread=("spread", "median"))
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["sec"] = (agg["datetime"].dt.hour * 3600 + agg["datetime"].dt.minute * 60
                  + agg["datetime"].dt.second).astype(np.int32)
    agg["day"] = agg["datetime"].dt.normalize()
    return agg


# --------------------------------------------------------------------------- #
# Indicators: KAMA, ATR(Wilder), delta, z(delta)
# --------------------------------------------------------------------------- #
def kama(close: np.ndarray, period: int, fast: int, slow: int) -> np.ndarray:
    """Kaufman Adaptive MA (NinjaTrader-faithful), computed continuously over the series.

    ER = |close[i]-close[i-period]| / sum_{i-period+1..i}|close[j]-close[j-1]|
    sc = (ER*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2
    kama[i] = kama[i-1] + sc*(close[i]-kama[i-1]) ; seeded kama[i]=close[i] for i<period.
    """
    n = close.shape[0]
    out = close.copy()
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    absdiff = np.abs(np.diff(close, prepend=close[0]))  # |close[j]-close[j-1]|, [0]=0
    for i in range(1, n):
        if i < period:
            out[i] = close[i]
            continue
        change = abs(close[i] - close[i - period])
        volatility = absdiff[i - period + 1: i + 1].sum()
        er = (change / volatility) if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR via EWMA(alpha=1/period) of true range; causal (uses prev close)."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    s = pd.Series(tr)
    return s.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def add_indicators(bars: pd.DataFrame, kama_period: int, fast: int, slow: int,
                   atr_period: int, z_lookback: int) -> pd.DataFrame:
    c = bars["close"].to_numpy()
    fv = kama(c, kama_period, fast, slow)
    delta = (c - fv) / fv
    out = bars.copy()
    out["kama"] = fv
    out["atr"] = atr_wilder(bars["high"].to_numpy(), bars["low"].to_numpy(), c, atr_period)
    out["delta"] = delta
    # vol-normalised deviation; rolling std known at close t (causal), min_periods to warm
    out["delta_z"] = delta / pd.Series(delta).rolling(z_lookback, min_periods=z_lookback).std(ddof=1).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    tf: str = "M5"
    session: str = "rth"
    # KAMA fair value
    kama_period: int = 20
    kama_fast: int = 5
    kama_slow: int = 30
    # entry
    entry_mode: str = "rel"       # 'rel' | 'z'
    threshold: float = 0.01       # rel: fractional deviation (0.01 = 1%)
    z_thr: float = 2.0            # z:   |z(delta)| trigger
    z_lookback: int = 100
    longs: bool = True
    shorts: bool = True
    # exit / risk
    exit_mode: str = "bracket"    # 'bracket' | 'revert' | 'time' | 'close'
    atr_period: int = 14
    stop_atr_mult: float = 3.5
    target_rr: float = 0.5
    time_stop_bars: int = 12
    # sizing
    sizing: str = "fixed"         # 'fixed' | 'vol_target'
    vol_target: float = 0.015     # daily target vol (vol_target sizing)
    vol_lookback_day: int = 14
    lev_cap: float = 4.0
    # costs (Phase 1)
    use_recorded_spread: bool = False
    spread_pts: float = 0.0       # override round-trip spread in idx points (if not recorded)
    slip_pts: float = 0.0         # per-side slippage in idx points
    swap_bps_long: float = 0.0    # per-overnight-held charge (h24), applied if held across rollover
    swap_bps_short: float = 0.0


# --------------------------------------------------------------------------- #
# Simulate (single pass, multi-trade-per-session state machine)
# --------------------------------------------------------------------------- #
def simulate(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Event replay -> per-trade ledger. Entry at next-bar open; intrabar stop-first."""
    n = len(bars)
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    spr = bars["spread"].to_numpy()
    day = bars["day"].to_numpy()
    sec = bars["sec"].to_numpy()
    atr = bars["atr"].to_numpy()
    delta = bars["delta"].to_numpy()
    dz = bars["delta_z"].to_numpy()
    kama_v = bars["kama"].to_numpy()

    warmup = max(p.kama_period + 10, p.atr_period + 1, (p.z_lookback if p.entry_mode == "z" else 0))

    def signal(i: int) -> int:
        if p.entry_mode == "rel":
            val, thr = delta[i], p.threshold
        else:
            val, thr = dz[i], p.z_thr
        if not np.isfinite(val) or not np.isfinite(atr[i]) or atr[i] <= 0:
            return 0
        if p.longs and val < -thr:
            return 1
        if p.shorts and val > thr:
            return -1
        return 0

    trades = []
    i = warmup
    while i < n - 1:
        # only decide on a non-last-bar-of-day (need a same-day next-bar open to fill)
        if day[i + 1] != day[i]:
            i += 1
            continue
        s = signal(i)
        if s == 0:
            i += 1
            continue
        ej = i + 1                         # entry fills at next bar open (lookahead-free)
        entry = o[ej]
        stop_dist = p.stop_atr_mult * atr[i]
        if stop_dist <= 0:
            i += 1
            continue
        tgt_dist = stop_dist * p.target_rr
        if s == 1:
            sl, tp = entry - stop_dist, entry + tgt_dist
        else:
            sl, tp = entry + stop_dist, entry - tgt_dist

        exit_px, exit_kind, xb = c[-1], "eod", n - 1
        held = 0
        j = ej
        while True:
            last_of_day = (j == n - 1) or (day[j + 1] != day[j])
            # ---- intrabar protective stop / bracket target (stop-first) ----
            if s == 1:
                if lo[j] <= sl:
                    exit_px, exit_kind, xb = sl, "sl", j; break
                if p.exit_mode == "bracket" and h[j] >= tp:
                    exit_px, exit_kind, xb = tp, "tp", j; break
            else:
                if h[j] >= sl:
                    exit_px, exit_kind, xb = sl, "sl", j; break
                if p.exit_mode == "bracket" and lo[j] <= tp:
                    exit_px, exit_kind, xb = tp, "tp", j; break
            # ---- bar-close signal exits (known only at close[j]) ----
            if p.exit_mode == "revert":
                # exit when price has crossed back through fair value
                if (s == 1 and c[j] >= kama_v[j]) or (s == -1 and c[j] <= kama_v[j]):
                    exit_px, exit_kind, xb = c[j], "revert", j; break
            if p.exit_mode == "time" and held >= p.time_stop_bars:
                exit_px, exit_kind, xb = c[j], "time", j; break
            if last_of_day:
                exit_px, exit_kind, xb = c[j], "eod", j; break
            j += 1
            held += 1

        gross_pts = (exit_px - entry) * s
        risk_pts = abs(entry - sl)
        # cost uses the TRIGGER bar's recorded spread (known at the decision), not the
        # fill bar's — strictly causal (the fill-bar spread is unknown when we commit).
        rt_spread = (spr[i] * POINT) if p.use_recorded_spread else p.spread_pts
        held_overnight = day[xb] != day[ej]   # only possible in h24
        swap = 0.0
        if held_overnight:
            swap_bps = p.swap_bps_long if s == 1 else p.swap_bps_short
            swap = -swap_bps * 1e-4 * entry   # signed bps of notional -> idx pts (long pays if neg)
        cost_pts = rt_spread + 2 * p.slip_pts - swap  # swap already signed as a cost
        net_pts = gross_pts - cost_pts
        trades.append({
            "day": day[ej], "entry_sec": int(sec[ej]), "dir": s, "entry": entry,
            "exit": exit_px, "kind": exit_kind,
            "gross_pts": gross_pts, "net_pts": net_pts, "cost_pts": cost_pts,
            "risk_pts": risk_pts, "held": held, "delta_in": delta[i], "atr_in": atr[i],
            "gross_ret": gross_pts / entry, "ret": net_pts / entry,
            "R": (net_pts / risk_pts) if risk_pts > 0 else 0.0,
        })
        # flat at close[xb]; EA decides entry on this same (now-flat) close -> next bar
        i = xb
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# Vol-target overlay (size-at-entry, per day; no intraday rebalance)
# --------------------------------------------------------------------------- #
def apply_vol_target(trades: pd.DataFrame, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Scale each trade's return by a per-day leverage = min(cap, vt/sigma_day).

    sigma_day from strictly-past daily close-to-close returns (shift(1).rolling).
    """
    if trades.empty or p.sizing != "vol_target":
        trades = trades.copy()
        trades["lev"] = 1.0
        trades["ret_sized"] = trades["ret"]
        trades["gross_ret_sized"] = trades["gross_ret"]
        return trades
    daily_close = bars.groupby("day")["close"].last()
    daily_ret = daily_close.pct_change()
    sigma = daily_ret.shift(1).rolling(p.vol_lookback_day).std(ddof=1)
    lev = np.minimum(p.lev_cap, p.vol_target / sigma).clip(lower=0.0)
    lev = lev.replace([np.inf, -np.inf], p.lev_cap).fillna(0.0)
    t = trades.copy()
    t["lev"] = t["day"].map(lev).fillna(0.0).to_numpy()
    t["ret_sized"] = t["ret"] * t["lev"]
    t["gross_ret_sized"] = t["gross_ret"] * t["lev"]
    return t


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _sharpe(arr: np.ndarray) -> float:
    return float(arr.mean() / arr.std() * np.sqrt(TRADING_DAYS)) if arr.std() > 0 else 0.0


def metrics(trades: pd.DataFrame, session_days: np.ndarray, ret_col: str = "ret") -> dict:
    """Daily-aggregated metrics over EVERY session day (flat days = 0)."""
    n_days = len(session_days)
    years = max(n_days / TRADING_DAYS, 1e-9)
    if trades.empty:
        return {"trades": 0, "sharpe": 0.0, "ann_ret%": 0.0, "maxDD%": 0.0, "tr/yr": 0.0}
    day_ret = trades.groupby("day")[ret_col].sum()
    grid = pd.Series(0.0, index=pd.Index(session_days, name="day"))
    grid.loc[day_ret.index] = day_ret.values
    arr = grid.to_numpy()
    eq = np.cumsum(arr)
    maxdd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    wins = trades["net_pts"] > 0
    gw = trades.loc[wins, "net_pts"].sum()
    gl = -trades.loc[~wins, "net_pts"].sum()
    tot = float(arr.sum())
    return {
        "trades": int(len(trades)),
        "tr/yr": round(len(trades) / years, 0),
        "win%": round(100 * wins.mean(), 1),
        "avgR": round(trades["R"].mean(), 3),
        "PF": round(gw / gl, 2) if gl > 0 else float("inf"),
        "avg_hold": round(trades["held"].mean(), 1),
        "sharpe": round(_sharpe(arr), 2),
        "ann_ret%": round(100 * tot / years, 2),
        "vol%": round(100 * arr.std() * np.sqrt(TRADING_DAYS), 2),
        "maxDD%": round(100 * maxdd, 1),
        "tot_ret%": round(100 * tot, 1),
        "long%": round(100 * (trades["dir"] == 1).mean(), 0),
        "tp%": round(100 * (trades["kind"] == "tp").mean(), 0),
        "sl%": round(100 * (trades["kind"] == "sl").mean(), 0),
        "rev%": round(100 * (trades["kind"] == "revert").mean(), 0),
        "eod%": round(100 * (trades["kind"] == "eod").mean(), 0),
    }


def assert_no_lookahead(trades: pd.DataFrame) -> None:
    """Two independent PnL derivations must agree (catches alignment/cost bugs)."""
    if trades.empty:
        return
    a = trades["net_pts"].sum()
    b = ((trades["exit"] - trades["entry"]) * trades["dir"] - trades["cost_pts"]).sum()
    assert abs(a - b) < 1e-6, f"PnL re-derivation mismatch: {a} vs {b}"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_BAR_CACHE: dict = {}


def prep_bars(sym: str, y0: int, y1: int, p: Params) -> pd.DataFrame:
    key = (sym, y0, y1, p.tf, p.session, p.kama_period, p.kama_fast, p.kama_slow,
           p.atr_period, p.z_lookback)
    if key not in _BAR_CACHE:
        m1 = load_m1(sym, y0, y1, p.session)
        bars = resample(m1, p.tf)
        _BAR_CACHE[key] = add_indicators(bars, p.kama_period, p.kama_fast, p.kama_slow,
                                         p.atr_period, p.z_lookback)
    return _BAR_CACHE[key]


def run(sym: str, y0: int, y1: int, p: Params, label: str = "") -> tuple[dict, pd.DataFrame]:
    bars = prep_bars(sym, y0, y1, p)
    trades = simulate(bars, p)
    assert_no_lookahead(trades)
    trades = apply_vol_target(trades, bars, p)
    session_days = np.sort(bars["day"].unique())
    ret_col = "ret_sized" if p.sizing == "vol_target" else "ret"
    m = metrics(trades, session_days, ret_col=ret_col)
    m["cfg"] = label
    return m, trades


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    SYM, Y0, Y1 = "NDX", 2018, 2026

    # literal EA config, FRICTIONLESS
    base = Params(tf="M5", session="rth", entry_mode="rel", threshold=0.01,
                  exit_mode="bracket", stop_atr_mult=3.5, target_rr=0.5)
    m, tr = run(SYM, Y0, Y1, base, "EA literal (M5 RTH, thr=1%, 3.5xATR, 0.5R) frictionless")
    bars = prep_bars(SYM, Y0, Y1, base)
    print(f"loaded {len(bars):,} {base.tf} {base.session} bars over "
          f"{bars['day'].nunique()} session days "
          f"({bars['day'].min().date()} .. {bars['day'].max().date()})")
    print(f"[lookahead check] passed (2 PnL derivations agree)\n")
    print(pd.DataFrame([m]).set_index("cfg").to_string())
