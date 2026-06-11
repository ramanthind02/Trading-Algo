"""M1 load + intraday resample for the vault-intraday screen.

The MT5 M1 store lives at ``data/mt5_data/<SYMBOL>/bars_M1/year=*/part.parquet``
with columns ``[time, open, high, low, close, tick_volume, spread, real_volume]``.
``time`` is broker EET/EEST mislabelled UTC (ET = stored - 7h); we keep the broker
wall-clock (tz_localize(None)) because intraday bar boundaries should sit on a
consistent clock and these signals are not time-of-day gated.

True 1-minute data only exists from ~2018 (``CFD_INTRADAY_CUTOVER``); before that
the M1 store is sparse daily/hourly-as-M1, so we hard-floor the intraday window at
2018-01-01. The first M1 bar of each broker session (~01:00) opens at the prior
session's close carried across the 00:00-01:00 financing dead zone (a fabricated
open/high/low that reprices within the minute); we de-stale it to its own close
(the first settled price) before resampling, mirroring ``cfd_candles``. The close
series is untouched, so close-to-close returns are unaffected.

Self-contained (experiment scratch) — reads parquet directly so the ``spread``
column (needed for the cost haircut) is preserved.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# data/mt5_data/<SYM>/bars_M1/year=*/part.parquet — repo root is 3 parents up
# (research/experiments/vault_intraday/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MT5_ROOT = _REPO_ROOT / "data" / "mt5_data"
_CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "_bars_cache"

INTRADAY_FLOOR = pd.Timestamp("2018-01-01")

# A session with more than this many M1 bars is "true intraday" (2018+ minute data
# averages ~1366 bars/session); on sparse legacy sessions the first bar's own open is
# genuine, so we only de-stale dense sessions (mirrors cfd_candles MIN_INTRADAY_BARS).
_MIN_INTRADAY_BARS = 100

# pandas resample rules for the requested intraday timeframes (left-closed,
# left-labelled: a bar stamped t covers [t, t+rule); its close is the last M1
# close in the window, known at the bar's end -> causal when positions shift(1)).
RESAMPLE_RULE: dict[str, str] = {"H4": "4h", "H2": "2h", "H1": "1h", "M30": "30min"}

_M1_COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "spread": "median",
}


def _sym_dir(symbol: str) -> Path:
    return _MT5_ROOT / symbol


def read_m1(symbol: str, *, floor: pd.Timestamp = INTRADAY_FLOOR) -> pd.DataFrame:
    """Concatenated M1 frame for *symbol*, broker-naive ``datetime`` index-free.

    Returns columns ``[datetime, open, high, low, close, volume, spread]`` (float
    OHLC, int volume, int spread-in-points), filtered to ``datetime >= floor`` and
    with the 00:00-01:00 broker dead zone removed.
    """
    parts = sorted((_sym_dir(symbol) / "bars_M1").glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"No M1 parquet for symbol {symbol!r} under {_sym_dir(symbol)}")
    # Year partitions are named year=YYYY — only read partitions that can contain
    # data >= floor (cheap pruning of the pre-2018 sparse era).
    keep = [p for p in parts if int(p.parent.name.split("=")[1]) >= floor.year]
    frame = pd.concat(
        [pd.read_parquet(p, columns=_M1_COLS) for p in keep], ignore_index=True
    )
    dt = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": frame["open"].astype("float64"),
            "high": frame["high"].astype("float64"),
            "low": frame["low"].astype("float64"),
            "close": frame["close"].astype("float64"),
            "volume": frame["tick_volume"].astype("int64"),
            "spread": frame["spread"].astype("float64"),
        }
    )
    out = out[out["datetime"] >= floor].sort_values("datetime").reset_index(drop=True)
    # De-stale the session-open bar: collapse the first M1 bar of each true-intraday
    # session to its own close (the first settled tradeable price), removing the stale
    # carried-open spike before it reaches the OHLC-consuming nodes. Close is untouched.
    sess = out["datetime"].dt.normalize()
    bars_per_day = out.groupby(sess)["close"].transform("size")
    first_idx = out.groupby(sess, sort=False).head(1).index
    dense_first = first_idx[bars_per_day.loc[first_idx].to_numpy() > _MIN_INTRADAY_BARS]
    close_vals = out.loc[dense_first, "close"].to_numpy()
    for col in ("open", "high", "low"):
        out.loc[dense_first, col] = close_vals
    return out.reset_index(drop=True)


def resample_intraday(m1: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """Resample a cleaned M1 frame to an intraday timeframe.

    OHLC = first/max/min/last; volume = sum; spread = median (per-bar typical
    half-spread proxy, in points). Empty bins (weekend/overnight gaps) are dropped.
    """
    rule = RESAMPLE_RULE[tf_name]
    agg = (
        m1.set_index("datetime")
        .resample(rule, label="left", closed="left")
        .agg(_AGG)
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["volume"] = agg["volume"].fillna(0).astype("int64")
    return agg


def load_bars(symbol: str, tf_name: str, *, use_cache: bool = True) -> pd.DataFrame:
    """Resampled intraday bars for (symbol, tf_name), cached to outputs/_bars_cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{symbol}_{tf_name}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    bars = resample_intraday(read_m1(symbol), tf_name)
    bars.to_parquet(cache, index=False)
    return bars


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Realized bar frequency = n_bars / calendar-years-spanned.

    Robust to overnight/weekend gaps (the true intraday cadence, not a nominal
    24x7 count) — the right annualization factor for a gappy intraday series.
    """
    if len(index) < 2:
        return 1.0
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    years = max(span_days / 365.25, 1e-9)
    return len(index) / years
