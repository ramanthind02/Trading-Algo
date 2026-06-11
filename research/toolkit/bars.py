"""M1 data I/O, session filtering, and intraday resampling.

Promotion sources:
  - research/experiments/vault_intraday/data_io.py   — read_m1, resample logic,
    year-partition pruning, tz strip, de-stale, bars_per_year.
  - research/experiments/lafo_kama_mr/engine.py      — load_m1: sec/day columns,
    year-range bound, session filter, h24 de-stale; resample: tf rules, sec/day
    recomputation after resample.

Superset of their fixes applied here:
  - sec column (int32 broker seconds-of-day) and day column (date-floor) are
    always present on the output frame (lafo approach).
  - spread column retained as float64 (both experiments use it for cost models).
  - OHLC upcast to float64 (both experiments require this).
  - Year-partition pruning at both ends via floor.year and year_end (test bounding).
  - De-stale the ~01:00 session-open bar unconditionally in read_m1 before any
    session filter (vault_intraday approach, correct for both RTH and H24).

Datastore layout::

    data/mt5_data/<SYMBOL>/bars_M1/year=<YYYY>/part.parquet
    columns: time · open · high · low · close · tick_volume · spread · real_volume

``time`` is broker EET/EEST stored as a UTC-labelled Unix epoch (the scraper calls
``pd.to_datetime(raw, unit='s', utc=True)`` then writes to parquet — the UTC label
is a lie).  We strip it with ``.tz_localize(None)`` and keep broker wall-clock.
See docs/library/Data/mt5_timezones.md for the full story.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.toolkit.sessions import Session, session_window

# Repo root resolved from this file's location:  research/toolkit/ → research/ → repo root
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_MT5_ROOT: Path = _REPO_ROOT / "data" / "mt5_data"

INTRADAY_FLOOR: pd.Timestamp = pd.Timestamp("2018-01-01")
"""Default floor: real dense M1 data exists from 2018-01-01 onwards."""

# Sessions with more M1 bars than this per calendar day are "true intraday"
# (2018+ minute data averages ~1 366 bars/session).  Sparse legacy sessions
# (pre-2018 daily-as-M1 back-fill) retain their own genuine open; only dense
# sessions have the stale carried-open spike at 01:00 that needs de-staling.
_MIN_INTRADAY_BARS: int = 100

RESAMPLE_RULES: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H2":  "2h",
    "H4":  "4h",
}

_M1_COLS: list[str] = ["time", "open", "high", "low", "close", "tick_volume", "spread"]

_RESAMPLE_AGG: dict[str, tuple[str, str]] = {
    "open":   ("open",   "first"),
    "high":   ("high",   "max"),
    "low":    ("low",    "min"),
    "close":  ("close",  "last"),
    "volume": ("volume", "sum"),
    "spread": ("spread", "median"),
}


def _sym_dir(symbol: str) -> Path:
    return _MT5_ROOT / symbol


def read_m1(
    symbol: str,
    *,
    floor: pd.Timestamp = INTRADAY_FLOOR,
    year_end: int | None = None,
) -> pd.DataFrame:
    """Concatenated M1 frame for *symbol* in broker wall-clock (naive datetime).

    Output columns: ``[datetime, sec, day, open, high, low, close, volume, spread]``

    ``floor`` prunes year partitions before floor.year (default 2018, the start
    of true minute-level data).  ``year_end`` caps the upper partition (inclusive)
    — useful for test-bounded reads that avoid loading multi-year datasets.

    The first M1 bar of each true-intraday session (~01:00 broker, the first tick
    after the 00:00–01:00 financing dead zone) carries a stale fabricated open from
    the prior session.  We collapse its open/high/low to its own close before
    returning, matching the ``cfd_candles`` de-stale in the daily pipeline.
    The close series is untouched, so close-to-close returns are unaffected.
    """
    parts = sorted((_sym_dir(symbol) / "bars_M1").glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No M1 parquet for symbol {symbol!r} under {_sym_dir(symbol)}"
        )

    def _year(p: Path) -> int:
        return int(p.parent.name.split("=")[1])

    keep = [
        p for p in parts
        if _year(p) >= floor.year and (year_end is None or _year(p) <= year_end)
    ]
    if not keep:
        raise FileNotFoundError(
            f"No M1 partitions for {symbol!r} in range "
            f"[{floor.year}, {year_end or '∞'}]"
        )

    raw = pd.concat(
        [pd.read_parquet(p, columns=_M1_COLS) for p in keep], ignore_index=True
    )
    # Strip the false UTC label → broker wall-clock (EET/EEST); ET = broker − 7h.
    dt = pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)
    sec = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).astype(np.int32)
    day = dt.dt.normalize()

    out = pd.DataFrame(
        {
            "datetime": dt,
            "sec":      sec,
            "day":      day,
            "open":     raw["open"].astype(np.float64),
            "high":     raw["high"].astype(np.float64),
            "low":      raw["low"].astype(np.float64),
            "close":    raw["close"].astype(np.float64),
            "volume":   raw["tick_volume"].astype(np.int64),
            "spread":   raw["spread"].astype(np.float64),
        }
    )
    out = out[out["datetime"] >= floor].sort_values("datetime").reset_index(drop=True)

    # De-stale: set open=high=low=close for the first bar of each dense session.
    # Groups by calendar date; the first M1 bar at ~01:00 broker is the stale bar.
    bars_per_day = out.groupby("day")["close"].transform("size")
    first_idx = out.groupby("day", sort=False).head(1).index
    dense_first = first_idx[bars_per_day.loc[first_idx].to_numpy() > _MIN_INTRADAY_BARS]
    close_vals = out.loc[dense_first, "close"].to_numpy()
    for col in ("open", "high", "low"):
        out.loc[dense_first, col] = close_vals

    return out.reset_index(drop=True)


def filter_session(m1: pd.DataFrame, session: Session) -> pd.DataFrame:
    """Filter a broker-wall-clock M1 frame to the given session window.

    Uses the ``sec`` column (broker seconds-of-day, produced by ``read_m1``) to
    select rows in ``[open_s, close_s)``.  Empty bins (dead zone, weekends) drop
    naturally from the window filter.  Returns a sorted, re-indexed frame.
    """
    open_s, close_s = session_window(session)
    mask = (m1["sec"] >= open_s) & (m1["sec"] < close_s)
    return m1.loc[mask].sort_values("datetime").reset_index(drop=True)


def resample_bars(m1: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """Resample a (session-filtered) M1 frame to *tf_name*.

    Aggregation: OHLC = first/max/min/last; volume = sum; spread = median (per-bar
    typical half-spread proxy, in MT5 points).  Empty bins (overnight/weekend gaps)
    are dropped (``dropna`` on ``open``).

    Resampling is **left-closed / left-labelled**: a bar stamped *t* covers
    ``[t, t + rule)``; its close is the last M1 close in the window, known only at
    bar-end → causal when signals use ``shift(1)``.

    Derived columns ``sec`` and ``day`` are recomputed from the resampled
    ``datetime``.  Output column order matches ``read_m1`` (minus ``volume`` →
    ``volume`` is retained, just aggregated by sum).
    """
    if tf_name not in RESAMPLE_RULES:
        raise ValueError(
            f"Unknown tf_name {tf_name!r}; known: {sorted(RESAMPLE_RULES)}"
        )
    if tf_name == "M1":
        return m1.reset_index(drop=True)

    rule = RESAMPLE_RULES[tf_name]
    agg = (
        m1.set_index("datetime")
        .resample(rule, label="left", closed="left")
        .agg(**{k: v for k, v in _RESAMPLE_AGG.items()})
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["volume"] = agg["volume"].fillna(0).astype(np.int64)
    agg["sec"] = (
        agg["datetime"].dt.hour * 3600
        + agg["datetime"].dt.minute * 60
        + agg["datetime"].dt.second
    ).astype(np.int32)
    agg["day"] = agg["datetime"].dt.normalize()

    cols = ["datetime", "sec", "day", "open", "high", "low", "close", "volume", "spread"]
    return agg[cols].reset_index(drop=True)


def load_bars(
    symbol: str,
    tf_name: str,
    session: Session = Session.H24,
    *,
    floor: pd.Timestamp = INTRADAY_FLOOR,
    year_end: int | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: ``read_m1`` → ``filter_session`` → ``resample_bars``, cached.

    Cache stored as parquet under *cache_dir* (defaults to
    ``data/.bars_cache/`` at the repo root).  The cache key encodes symbol,
    tf_name, session, floor year, and year_end so stale entries are never
    silently reused.
    """
    if cache_dir is None:
        cache_dir = _REPO_ROOT / "data" / ".bars_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ye_tag = str(year_end) if year_end is not None else "latest"
    cache_key = f"{symbol}_{tf_name}_{session.value}_{floor.year}_{ye_tag}.parquet"
    cache = cache_dir / cache_key

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    bars = resample_bars(
        filter_session(
            read_m1(symbol, floor=floor, year_end=year_end),
            session,
        ),
        tf_name,
    )
    bars.to_parquet(cache, index=False)
    return bars


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Realised bar frequency = n_bars / calendar-years-spanned.

    Robust to overnight/weekend gaps (the true intraday cadence, not a nominal
    24×7 count) — the correct annualisation factor for a gappy intraday series.
    """
    if len(index) < 2:
        return 1.0
    span_days = (index[-1] - index[0]).total_seconds() / 86_400.0
    years = max(span_days / 365.25, 1e-9)
    return len(index) / years
