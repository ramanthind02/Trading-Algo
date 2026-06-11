r"""Load Darwinex CFD candles keyed by the canonical (vault) ``Ticker``.

Maps a vault ticker (ES/NQ/GC/SI/CL/...) to its Darwinex MT5 symbol via
``brokers.resolve('darwinex', ...)`` and returns OHLCV in the project candle
contract so the CFD feed is a drop-in for the Norgate futures feed.

Daily bars
----------
Preferred source is the broker's own daily store
``data/mt5_data/{SYM}/bars_D1/part.parquet`` (correct broker sessions). When that
is absent (only some symbols have been daily-scraped) daily bars are **derived
from the M1 store** ``bars_M1/year=*/part.parquet`` by grouping on the broker
session date. MT5 timestamps are broker time mislabelled UTC and the daily
financing rollover is 00:00 broker, so the UTC calendar date IS the session date
(see ``docs/library/Data/mt5_timezones.md``).

Weekly / Monthly
----------------
Resampled from the daily series with the **identical** convention the Norgate
futures store uses (``W-SUN`` week-ending-Sunday, ``ME`` month-end; OHLCV agg =
first/max/min/last/sum) — see ``data_platform.providers.norgate.migrate._aggregate``
— so CFD W/M labels line up with futures W/M and nothing downstream drifts.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from data_platform.providers.mt5 import brokers
from lib.core.enums import Ticker, TimeFrame

# Pandas resample rules matching the futures store (norgate.migrate._aggregate).
_RESAMPLE_RULE: dict[str, str] = {"W": "W-SUN", "M": "ME"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
_M1_COLS = ["time", "open", "high", "low", "close", "tick_volume"]

# The CFD M1 store has three historical density regimes (see
# docs/library/Data/mt5_timezones.md and the splice notes below):
#   * pre-2016  : ~1 bar/day (daily candles stored as M1) — bar 1 == the day
#   * 2016-2017 : ~8-24 bars/day (hourly candles stored as M1)
#   * 2018+     : ~1366-1378 bars/day (true 1-minute bars)
# The de-stale step (overwrite the session OPEN with the first M1 bar's CLOSE to
# step over the 00:00-01:00 financing dead zone) is only valid on *true intraday*
# days: when M1 is sparse, "first M1 close" is the daily CLOSE, so the overwrite
# collapses open==close and zeroes the intraday return. We therefore de-stale a
# session ONLY when it carries more than MIN_INTRADAY_BARS_FOR_DESTALE M1 bars.
# 60 sits an order of magnitude above the hourly regime's ceiling (~24 bars/day)
# and far below the true-minute regime's norm (~1366), so 2018+ days qualify and
# the pre-2016 / 2016-17 sparse regimes are left with their genuine session open.
MIN_INTRADAY_BARS_FOR_DESTALE = 60

# Spliced-feed cutover: the CFD M1 store only became true 1-minute data (and thus
# only carries de-staleable, real intraday opens) from ~2018. Before this date the
# spliced feed uses the faithful (%-preserving) futures RATIO series instead of the
# sparse/broken-open CFD segment; on/after it uses the clean true-minute CFD.
CFD_INTRADAY_CUTOVER = date(2018, 1, 1)


def cfd_symbol_for(ticker: Ticker | str) -> str:
    """Canonical (vault) ticker -> Darwinex MT5 symbol (ES->SP500, GC->XAUUSD...).

    Accepts EITHER a canonical/vault ticker (``ES``, ``NQ``, ``GC``) or a Darwinex
    CFD symbol already (``SP500``, ``NDX``, ``XAUUSD``): an unmapped name that is
    itself a known broker symbol passes through unchanged, so callers may pass
    whichever they hold.
    """
    name = ticker.name if hasattr(ticker, "name") else str(ticker)
    try:
        return brokers.resolve(brokers.DEFAULT_BROKER, name)
    except KeyError:
        # Not a canonical ticker — maybe it's already a Darwinex CFD symbol.
        try:
            brokers.canonical_for(brokers.DEFAULT_BROKER, name)
        except KeyError:
            raise KeyError(
                f"{name!r} is neither a canonical ticker nor a Darwinex CFD symbol."
            ) from None
        return name


def _sym_dir(symbol: str) -> Path:
    return brokers.mt5_data_root() / symbol


def _read_d1(symbol: str) -> pd.DataFrame | None:
    path = _sym_dir(symbol) / "bars_D1" / "part.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["time", "open", "high", "low", "close", "tick_volume"])
    return df if not df.empty else None


def _read_m1(symbol: str) -> pd.DataFrame:
    base = _sym_dir(symbol) / "bars_M1"
    parts = sorted(base.glob("year=*/part.parquet"))
    if not parts:
        return pd.DataFrame(columns=_M1_COLS)
    return pd.concat(
        [pd.read_parquet(p, columns=_M1_COLS) for p in parts], ignore_index=True
    )


def _m1_to_daily(m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M1 bars to daily by broker session date (UTC calendar date)."""
    t = pd.to_datetime(m1["time"], utc=True).dt.tz_localize(None)
    grouped = m1.assign(_date=t.dt.normalize()).groupby("_date", sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("tick_volume", "sum"),
    ).reset_index().rename(columns={"_date": "datetime"})
    return out


def _first_tradeable_open(symbol: str) -> pd.DataFrame | None:
    """Per broker session date -> first *tradeable* open + that day's M1 bar count.

    The MT5 CFD feed's first M1 bar of each session (01:00 broker) OPENS at the
    previous session's close, carried across the 00:00-01:00 financing dead zone —
    a stale, untradeable price — then reprices to the live level within that minute
    (the whole overnight/weekend gap lands inside bar 1; e.g. NDX 2025-02-03 opens
    at the Friday close 21464.8 and closes the minute at 20888.3). The genuinely
    tradeable session open is therefore that first bar's CLOSE, not its open, so
    ``log(close/open)`` is a real intraday return that reconciles with the
    execution lane (which fills the entry at the first bar's close).

    This logic is only valid on *true intraday* days. The returned ``bars`` count
    lets the caller guard the de-stale (see :data:`MIN_INTRADAY_BARS_FOR_DESTALE`):
    on sparse days (pre-2016 daily-as-M1, 2016-17 hourly-as-M1) "first close" is
    the daily close, so overwriting the open would zero the intraday return.

    Returns a frame indexed by session date with columns ``first_close`` (float64)
    and ``bars`` (int), or None when the symbol has no M1 store (no de-stale).
    """
    parts = sorted((_sym_dir(symbol) / "bars_M1").glob("year=*/part.parquet"))
    if not parts:
        return None
    m1 = pd.concat(
        [pd.read_parquet(p, columns=["time", "close"]) for p in parts],
        ignore_index=True,
    )
    if m1.empty:
        return None
    t = pd.to_datetime(m1["time"], utc=True).dt.tz_localize(None)
    grouped = m1.assign(_date=t.dt.normalize()).sort_values("time").groupby("_date")
    return pd.DataFrame(
        {
            "first_close": grouped["close"].first().astype("float64"),
            "bars": grouped.size().astype("int64"),
        }
    )


@lru_cache(maxsize=64)
def _cfd_daily_frame(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for a Darwinex symbol: bars_D1 if present, else derived from M1.

    Returns columns [datetime, open, high, low, close, volume] (tz-naive midnight
    date label, float64 OHLC, int64 volume), sorted, deduped on datetime. The
    session ``open`` is de-staled to the first *tradeable* price (see
    :func:`_first_tradeable_open`) — both the broker D1 store and the M1
    ``open=first`` aggregation otherwise carry the prior close across the
    00:00-01:00 dead zone as an untradeable open.
    """
    d1 = _read_d1(symbol)
    if d1 is not None:
        df = d1.rename(columns={"tick_volume": "volume"}).copy()
        df["datetime"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
        df = df.drop(columns=["time"])
    else:
        m1 = _read_m1(symbol)
        if m1.empty:
            raise FileNotFoundError(
                f"No CFD daily (bars_D1) or M1 (bars_M1) data for symbol {symbol!r} "
                f"under {_sym_dir(symbol)}"
            )
        df = _m1_to_daily(m1)

    # De-stale the session open: replace the stale dead-zone open with the first
    # M1 bar's close (the first settled, tradeable price). Applied ONLY on true
    # intraday days (> MIN_INTRADAY_BARS_FOR_DESTALE M1 bars); on sparse days the
    # bar's own real open is kept, otherwise open would collapse onto close. No-op
    # without M1.
    tradeable = _first_tradeable_open(symbol)
    if tradeable is not None:
        dense = tradeable[tradeable["bars"] > MIN_INTRADAY_BARS_FOR_DESTALE]["first_close"]
        df["open"] = df["datetime"].map(dense).fillna(df["open"])

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = (
        df[["datetime", "open", "high", "low", "close", "volume"]]
        .sort_values("datetime")
        .loc[lambda d: ~d["datetime"].duplicated(keep="last")]
        .reset_index(drop=True)
    )
    return df


def _resample(daily: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
    rule = _RESAMPLE_RULE[timeframe.name]
    agg = (
        daily.set_index("datetime")
        .resample(rule)
        .agg(_AGG)
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["volume"] = agg["volume"].astype("int64")
    return agg


# Default splice boundary = the intraday cutover: faithful futures RATIO before
# (covers the pre-2018 sparse/broken-open CFD era), clean true-minute CFD after.
_SPLICE_DEFAULT = pd.Timestamp(CFD_INTRADAY_CUTOVER)


def _ratio_parquet_path(ticker_name: str, timeframe: TimeFrame) -> Path:
    """On-disk path of the faithful (%-preserving) RATIO parquet for a timeframe.

    ``data/ohlc_data/{T}/{TF}_{T}_ratio.parquet`` — D/W/M each have their own file
    (mirrors the additive store's per-timeframe layout).
    """
    return (
        brokers.mt5_data_root().parent
        / "ohlc_data"
        / ticker_name
        / f"{timeframe.name}_{ticker_name}_ratio.parquet"
    )


def _read_ratio_parquet(path: Path) -> pd.DataFrame:
    """Read one RATIO parquet into [datetime, open, high, low, close, volume] (tz-naive)."""
    df = pd.read_parquet(path, engine="fastparquet")
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    else:
        reset = df.reset_index()
        dt = pd.to_datetime(reset[reset.columns[0]])
        df = reset
    out = pd.DataFrame({"datetime": pd.DatetimeIndex(dt).tz_localize(None).normalize()})
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy().astype("float64")
    out["volume"] = (df["volume"].to_numpy() if "volume" in df.columns else 0)
    out["volume"] = out["volume"].astype("int64")
    return out.sort_values("datetime").reset_index(drop=True)


def load_futures_ratio_candles_raw(ticker: Ticker | str, timeframe: TimeFrame) -> pd.DataFrame:
    """Faithful (%-preserving) RATIO futures OHLCV for a ticker — NO date filter, NO ticker col.

    Reads ``data/ohlc_data/{T}/{TF}_{T}_ratio.parquet`` (the ratio-back-adjusted
    series; close/open ratios are the true contract returns, unlike the additive
    ``_CCB`` store whose absolute offset distorts percentage returns). This is the
    return feed for the "futures" result lane. Keyed by the canonical ticker NAME
    (e.g. ``ES``/``NQ``/``GC``), NOT a Darwinex CFD symbol. Columns:
    ``[datetime, open, high, low, close, volume]`` (tz-naive midnight date label).

    Raises ``FileNotFoundError`` when the ratio parquet for that ticker/timeframe is
    absent (date filtering / column decoration is the caller's job, mirroring
    ``load_cfd_candles_raw``).
    """
    name = ticker.name if hasattr(ticker, "name") else str(ticker)
    path = _ratio_parquet_path(name, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No futures-ratio parquet for ticker={name!r} timeframe={timeframe.name} at {path}"
        )
    return _read_ratio_parquet(path)


def has_futures_ratio(ticker: Ticker | str, timeframe: TimeFrame = TimeFrame.D) -> bool:
    """Whether a faithful RATIO futures parquet exists for this ticker/timeframe.

    True for back-adjusted futures (ES/NQ/GC/CL/SI/…), where the additive ``_CCB`` series
    sign-inverts deep history so research must use the ratio series. False for pure-CFD
    instruments — forex (AUDNZD/EURUSD/…) have no contract roll and therefore no ratio series;
    they research directly on the CFD feed. Lets the executor route DAILY research per asset class.
    """
    name = ticker.name if hasattr(ticker, "name") else str(ticker)
    return _ratio_parquet_path(name, timeframe).exists()


def _futures_ratio_daily(ticker_name: str) -> pd.DataFrame | None:
    """Faithful (%-preserving) RATIO back-adjusted futures daily, or None if absent.

    Columns [datetime, open, high, low, close, volume] (tz-naive midnight, float64/int64).
    """
    path = _ratio_parquet_path(ticker_name, TimeFrame.D)
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="fastparquet")
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    else:
        reset = df.reset_index()
        dt = pd.to_datetime(reset[reset.columns[0]])
        df = reset
    out = pd.DataFrame({"datetime": pd.DatetimeIndex(dt).tz_localize(None).normalize()})
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy().astype("float64")
    out["volume"] = (df["volume"].to_numpy() if "volume" in df.columns else 0)
    out["volume"] = out["volume"].astype("int64")
    return out.sort_values("datetime").reset_index(drop=True)


def load_spliced_candles_raw(
    ticker: Ticker | str, timeframe: TimeFrame, split: pd.Timestamp = _SPLICE_DEFAULT
) -> pd.DataFrame:
    """Long-history feed: faithful futures RATIO history before ``split``, CFD after.

    ``split`` defaults to :data:`CFD_INTRADAY_CUTOVER` (2018-01-01) — the point at
    which the CFD M1 store became true 1-minute data with de-staleable real opens.
    Before it the CFD segment is sparse (daily/hourly candles stored as M1) and its
    de-staled opens collapse onto the close, so the faithful (%-preserving) futures
    RATIO series is used instead.

    The pre-``split`` futures-ratio segment is **proportionally rescaled** so its
    level matches the CFD close at the join (continuous %-returns across the seam),
    giving one daily series back to the futures history start with faithful returns
    throughout. Falls back to CFD-only when the ratio series is missing OR does not
    overlap the join (no ``fut_pre`` / ``cfd_post``); in that fallback a ticker
    without futures-ratio coverage before the cutover keeps its (sparse) CFD daily.
    W/M resampled from daily.
    """
    daily_cfd = _cfd_daily_frame(cfd_symbol_for(ticker))
    name = ticker.name if hasattr(ticker, "name") else str(ticker)
    fut = _futures_ratio_daily(name)
    daily = daily_cfd
    if fut is not None and not fut.empty:
        split_ts = pd.Timestamp(split)
        cfd_post = daily_cfd[daily_cfd["datetime"] >= split_ts]
        fut_pre = fut[fut["datetime"] < split_ts]
        if not cfd_post.empty and not fut_pre.empty:
            cfd_anchor = float(cfd_post["close"].iloc[0])
            fut_anchor = float(fut_pre["close"].iloc[-1])
            k = (cfd_anchor / fut_anchor) if fut_anchor else 1.0
            scaled = fut_pre.copy()
            for col in ("open", "high", "low", "close"):
                scaled[col] = scaled[col] * k
            daily = (
                pd.concat([scaled, cfd_post], ignore_index=True)
                .sort_values("datetime")
                .reset_index(drop=True)
            )
    if timeframe == TimeFrame.D:
        return daily.copy()
    if timeframe.name in _RESAMPLE_RULE:
        return _resample(daily, timeframe)
    raise ValueError(f"Unsupported timeframe for spliced candles: {timeframe!r}")


def load_research_candles_raw(
    ticker: Ticker | str, timeframe: TimeFrame, feed: str
) -> pd.DataFrame:
    """Dispatch a non-futures research feed to its raw OHLCV loader."""
    if feed == "cfd":
        return load_cfd_candles_raw(ticker, timeframe)
    if feed == "spliced":
        return load_spliced_candles_raw(ticker, timeframe)
    if feed == "futures_ratio":
        return load_futures_ratio_candles_raw(ticker, timeframe)
    raise ValueError(f"load_research_candles_raw: unsupported feed {feed!r}")


def load_cfd_candles_raw(ticker: Ticker | str, timeframe: TimeFrame) -> pd.DataFrame:
    """CFD OHLCV for a vault ticker at a timeframe — NO date filter, NO ticker col.

    Columns: [datetime, open, high, low, close, volume]. D from the daily frame;
    W/M resampled from daily with the futures convention. Date filtering / column
    decoration is the caller's job (mirrors the futures ``load_data`` split).
    """
    symbol = cfd_symbol_for(ticker)
    daily = _cfd_daily_frame(symbol)
    if timeframe == TimeFrame.D:
        return daily.copy()
    if timeframe.name in _RESAMPLE_RULE:
        return _resample(daily, timeframe)
    raise ValueError(f"Unsupported timeframe for CFD candles: {timeframe!r}")
