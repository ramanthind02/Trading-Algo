r"""Shared helpers for the gold/silver future-vs-CFD study (Tasks 3 & 4).

Provides:
  * daily OHLC loaders per execution feed (FUT_adj, FUT_real, CFD@settle,
    CFD@close, ETF) — the breakout/ATR sleeves need high/low, not just close,
    so the MT5 loader aggregates M1 bars into a daily OHLC bar.
  * a node-streaming harness that runs the ACTUAL production node classes
    (DonchianBreakoutSignal, SmaRegimeSignalNode, RobustTrendBreakout) over a
    daily OHLC frame, returning the signed signal series. Streaming candles is
    clean: _compute_candle computes from streamed bars only and never consults
    the on-disk bias-node cache (that cache is read only by the vectorized
    get_cached_* API, which we do not call here).

Read-only with respect to repo state; writes nothing.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OHLC = REPO / "data" / "ohlc_data"
MT5 = REPO / "data" / "mt5_data"
STK = REPO / "data" / "stock_data"

# MT5 bars are stored in broker server tz (EET/EEST = UTC+2/+3). The repo's
# calibrated rule: ET = stored - 7h. We keep bars in stored tz and group by the
# stored calendar date, which equals the ET date for the active session.

# Daily OHLC of each future feed -------------------------------------------------
def fut_ohlc(ticker: str, *, adjusted: bool) -> pd.DataFrame:
    """Daily OHLC for a Norgate future. adjusted=True -> additive back-adjusted
    D_{T}.parquet; False -> D_{T}_unadj.parquet (real % returns)."""
    name = f"D_{ticker}.parquet" if adjusted else f"D_{ticker}_unadj.parquet"
    p = OHLC / ticker / name
    df = pd.read_parquet(p)[["open", "high", "low", "close"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def etf_ohlc(letter: str, sym: str, adj: str = "CAP") -> pd.DataFrame:
    p = STK / letter / sym / f"D_{adj}_{sym}.parquet"
    df = pd.read_parquet(p)[["open", "high", "low", "close"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def mt5_daily_ohlc(sym: str, *, snap_stored_hour: int | None = None) -> pd.DataFrame:
    """Daily OHLC built from MT5 M1 bars, grouped by the stored calendar date.

    snap_stored_hour:
      * None  -> aggregate the WHOLE stored day (open=first, high=max, low=min,
                 close=last). Used for the daily-mark close convention.
      * int h -> only M1 bars with stored hour <= h contribute (cut at a fixed
                 clock so 'close' lands at a specific ET time, e.g. 20 == 13:00
                 ET gold settle, 23 == 16:00 ET cash close).
    """
    parts = sorted(glob.glob(str(MT5 / sym / "bars_M1" / "year=*" / "part.parquet")))
    frames: list[pd.DataFrame] = []
    for p in parts:
        df = pd.read_parquet(p, columns=["time", "open", "high", "low", "close"])
        t = pd.to_datetime(df["time"], utc=True)  # stored tag kept as-is
        if snap_stored_hour is not None:
            keep = (t.dt.hour <= snap_stored_hour).values
            if not keep.any():
                continue
            df = df.loc[keep]
            t = t[keep]
        df = df.assign(date=t.dt.tz_convert(None).dt.normalize())
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    allm = pd.concat(frames, ignore_index=True)
    g = allm.groupby("date")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).sort_index()
    return out


# Node-streaming harness ---------------------------------------------------------
def _make_node(kind: str, ticker_name: str):
    from lib.core.enums import Ticker, TimeFrame, PositionMode

    tk = Ticker[ticker_name]
    if kind == "donchian":
        from nodes.breakout.donchian.donchian_breakout_signal import DonchianBreakoutSignal

        return DonchianBreakoutSignal(tk, TimeFrame.D, lookback=20, exit_bars=5)
    if kind == "sma_regime":
        from nodes.regime.sma.sma_regime_signal import SmaRegimeSignalNode

        return SmaRegimeSignalNode(tk, TimeFrame.D, period=252, mode=PositionMode.LONG_SHORT)
    if kind == "robust_trend":
        from nodes.breakout.donchian.robust_trend_breakout import RobustTrendBreakout

        return RobustTrendBreakout(
            tk, TimeFrame.D, lookback=20, ema_period=126, atr_len=14,
            atr_mult=1.5, atr_vol_filter=0, cooldown_bars=0, strategy_mode="long",
        )
    raise ValueError(kind)


def signal_for(ohlc: pd.DataFrame, kind: str, ticker_name: str) -> pd.Series:
    """Stream the daily OHLC frame through the real node and return the signed
    signal series indexed by date. A fresh node instance is built per call so no
    state leaks across feeds."""
    from lib.core.enums import Ticker, TimeFrame
    from lib.core.models import Candle

    node = _make_node(kind, ticker_name)
    tk = Ticker[ticker_name]
    out = {}
    for dt, row in ohlc.iterrows():
        c = Candle(
            datetime=dt.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=0.0,
            ticker=tk,
            tf=TimeFrame.D,
        )
        res = node.add_candle(c)
        out[dt] = float(res[0])
    return pd.Series(out, name=f"{kind}:{ticker_name}").sort_index()


def daily_logret_from_close(close: pd.Series) -> pd.Series:
    s = close.astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return np.log(s).diff()
