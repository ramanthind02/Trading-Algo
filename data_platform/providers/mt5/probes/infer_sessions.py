"""Infer a symbol's trading-session schedule from its stored M1 bars.

MT5's Python API exposes no session-hours function (see mt5_data_scraper.md), but
the schedule is empirically encoded in *where M1 bars exist*: within a session
bars are ~1 minute apart; between sessions there is a gap (daily maintenance
break, weekend, holiday). This tool reads data/mt5_data/{SYMBOL}/bars_M1/ and
derives the recurring weekly open/close pattern from those gaps.

Method
------
1. Load all M1 bar timestamps for the symbol (UTC, from the year=* parquet parts).
2. A gap > ``gap_minutes`` between consecutive bars ends a session.
3. For each session, record (weekday, open_time, close_time) in the bars' tz.
4. Aggregate across all weeks: the modal open/close per weekday is the schedule.

This is *descriptive* (what actually traded) — more reliable than a static spec
sheet, and it captures broker-specific quirks (Darwinex maintenance windows,
early closes) automatically.

Run:
    python -m data_platform.providers.mt5.probes.infer_sessions --symbol EURUSD
    python -m data_platform.providers.mt5.probes.infer_sessions --symbol AAPL --tz America/New_York
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[4],
    )


def _bars_dir(symbol: str) -> Path:
    return _repo_root() / "data" / "mt5_data" / symbol / "bars_M1"


def load_m1_times(symbol: str) -> pd.DatetimeIndex:
    """Load all M1 bar timestamps (UTC) for a symbol from its parquet parts."""
    bdir = _bars_dir(symbol)
    parts = sorted(bdir.glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No M1 bars for {symbol} at {bdir} — run the MT5 scraper first."
        )
    frames = [pd.read_parquet(p, columns=["time"]) for p in parts]
    times = pd.to_datetime(pd.concat(frames)["time"], utc=True)
    return pd.DatetimeIndex(times.sort_values().unique())


def infer_sessions(
    times: pd.DatetimeIndex,
    *,
    tz: str = "UTC",
    gap_minutes: int = 5,
    min_weeks: int = 4,
) -> pd.DataFrame:
    """Return a per-weekday session table (modal open/close) in timezone ``tz``."""
    local = times.tz_convert(tz)
    # Gap detection: a gap > gap_minutes ends a session.
    deltas = local.to_series().diff()
    boundary = deltas > pd.Timedelta(minutes=gap_minutes)
    session_id = boundary.cumsum()

    df = pd.DataFrame({"ts": local, "session": session_id.to_numpy()})
    grp = df.groupby("session")["ts"]
    opens = grp.min()
    closes = grp.max()

    rows = []
    for sid in opens.index:
        o, c = opens[sid], closes[sid]
        rows.append({
            "weekday": o.weekday(),
            "open": o.strftime("%H:%M"),
            "close": c.strftime("%H:%M"),
            "close_weekday": c.weekday(),
        })
    sess = pd.DataFrame(rows)
    if sess.empty:
        return sess

    # Modal open/close per weekday (robust to holidays / early closes).
    out = []
    for wd in range(7):
        wd_sess = sess[sess["weekday"] == wd]
        if len(wd_sess) < min_weeks:
            continue
        open_mode = Counter(wd_sess["open"]).most_common(1)[0]
        close_mode = Counter(wd_sess["close"]).most_common(1)[0]
        out.append({
            "weekday": _WEEKDAYS[wd],
            "open": open_mode[0],
            "close": close_mode[0],
            "sessions_observed": len(wd_sess),
            "open_consistency": round(open_mode[1] / len(wd_sess), 2),
            "close_consistency": round(close_mode[1] / len(wd_sess), 2),
        })
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer MT5 session hours from stored M1 bars.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tz", default="UTC", help="Timezone for the reported hours (e.g. America/New_York).")
    p.add_argument("--gap-minutes", type=int, default=5, help="Min inter-bar gap that ends a session.")
    args = p.parse_args()

    times = load_m1_times(args.symbol)
    print(f"{args.symbol}: {len(times):,} M1 bars, "
          f"{times.min().date()} -> {times.max().date()} ({args.tz})")
    table = infer_sessions(times, tz=args.tz, gap_minutes=args.gap_minutes)
    if table.empty:
        print("Not enough data to infer a stable schedule.")
        return
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
