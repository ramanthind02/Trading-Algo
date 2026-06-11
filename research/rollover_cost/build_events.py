r"""Fetch + feature-ize rollover events for the cost study.

For each weeknight rollover (00:00 broker time on Tue–Fri) over the lookback window,
fetch the bracket [rollover-65m, rollover+125m] via the verified on-demand tick cache
(ensure_ticks), split into the exit window [rollover-60m, rollover) and entry window
[rollover+60m, rollover+120m), and pre-compute the execution features each experiment
needs (spread/liquidity profile, limit fill outcomes). One row per (symbol, rollover date).

Output: research/rollover_cost/outputs/events_{SYMBOL}.parquet

Run:
  .\.venv\Scripts\python.exe -m research.rollover_cost.build_events --days 365
  .\.venv\Scripts\python.exe -m research.rollover_cost.build_events --symbols XAUUSD --days 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_platform.providers.mt5.tick_cache import ensure_ticks
from lib.core.logger import get_logger
from research.rollover_cost.config import (
    SPECS, SYMBOLS, EXIT_WINDOW_MIN, ENTRY_WINDOW_MIN, DEADZONE_MIN, FETCH_PAD_MIN,
    EXIT_OFFSETS_MIN, ARRIVAL_OFFSET_MIN, ENTRY_LIMIT_TICKS, ENTRY_SNAPSHOT_MIN,
    FILL_HORIZONS_MIN,
)

logger = get_logger(__name__)
UTC = timezone.utc
OUT_DIR = _REPO / "research" / "rollover_cost" / "outputs"


# ---------------------------------------------------------------------------
# Event enumeration
# ---------------------------------------------------------------------------

def rollover_dates(days_back: int) -> list[datetime]:
    """Broker-time rollover instants (00:00 on Tue–Fri) over the last `days_back` days.

    A rollover at 00:00 on weekday X separates the session ending on X-1 evening from
    the session starting X morning. We use Tue–Fri so the prior evening (exit window)
    is always a weekday; Monday 00:00 is the Sunday weekend reopen (different regime).
    Newest first so partial runs still capture recent events.
    """
    today = datetime.now(UTC).date()
    out: list[datetime] = []
    for d in range(days_back):
        day = today - timedelta(days=d)
        # weekday(): Mon=0 .. Sun=6 → rollover days Tue(1)..Fri(4)
        if 1 <= day.weekday() <= 4:
            out.append(datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC))
    return out


# ---------------------------------------------------------------------------
# Tick helpers (operate on broker-labelled-UTC tick frames)
# ---------------------------------------------------------------------------

def _mid(df: pd.DataFrame) -> np.ndarray:
    return (df["bid"].to_numpy() + df["ask"].to_numpy()) / 2.0


def _ns(df: pd.DataFrame) -> np.ndarray:
    """int64 nanoseconds-since-epoch for the (tz-aware) time column.

    Force ns resolution: the tick store is timestamp('s'), whose `.astype(int64)`
    yields seconds/ms — not comparable with `pd.Timestamp.value` (always ns).
    """
    return df["time"].to_numpy(dtype="datetime64[ns]").astype("int64")


def _snapshot_at(df: pd.DataFrame, t: pd.Timestamp) -> dict | None:
    """Last tick at or before instant t. None if no tick precedes t."""
    idx = int(np.searchsorted(_ns(df), t.value, side="right")) - 1
    if idx < 0:
        return None
    row = df.iloc[idx]
    return {"bid": float(row["bid"]), "ask": float(row["ask"]),
            "mid": (float(row["bid"]) + float(row["ask"])) / 2.0,
            "spread": float(row["ask"]) - float(row["bid"])}


def _first_after(df: pd.DataFrame, t: pd.Timestamp) -> dict | None:
    """First tick at or after instant t."""
    idx = int(np.searchsorted(_ns(df), t.value, side="left"))
    if idx >= len(df):
        return None
    row = df.iloc[idx]
    return {"bid": float(row["bid"]), "ask": float(row["ask"]),
            "mid": (float(row["bid"]) + float(row["ask"])) / 2.0,
            "spread": float(row["ask"]) - float(row["bid"]),
            "time": row["time"]}


def _buy_limit_fill_min(df: pd.DataFrame, limit: float, open_time: pd.Timestamp) -> float:
    """Minutes from open_time until a BUY limit at `limit` fills (ask <= limit), else NaN."""
    asks = df["ask"].to_numpy()
    hit = asks <= limit
    if not hit.any():
        return float("nan")
    i = int(np.argmax(hit))
    dt = (df["time"].iloc[i] - open_time).total_seconds() / 60.0
    return float(dt)


def _sell_limit_fill_min(df: pd.DataFrame, limit: float, start_time: pd.Timestamp) -> float:
    """Minutes from start until a SELL limit at `limit` fills (bid >= limit), else NaN."""
    bids = df["bid"].to_numpy()
    hit = bids >= limit
    if not hit.any():
        return float("nan")
    i = int(np.argmax(hit))
    dt = (df["time"].iloc[i] - start_time).total_seconds() / 60.0
    return float(dt)


# ---------------------------------------------------------------------------
# Per-event features
# ---------------------------------------------------------------------------

def event_features(symbol: str, rollover: datetime, exit_df: pd.DataFrame,
                   entry_df: pd.DataFrame, spec=None) -> dict | None:
    spec = spec if spec is not None else SPECS[symbol]
    roll_ts = pd.Timestamp(rollover)
    feat: dict = {
        "symbol": symbol,
        "rollover": roll_ts,
        "weekday": rollover.weekday(),          # 1=Tue..4=Fri (broker date)
        "is_triple": int(rollover.weekday() == _triple_to_pyweekday(spec.triple_weekday)),
        "exit_n_ticks": int(len(exit_df)),
        "entry_n_ticks": int(len(entry_df)),
    }
    if len(exit_df) < 20 or len(entry_df) < 20:
        return None  # holiday / thin — drop in analysis

    # ---- EXIT: spread + liquidity profile by minutes-before-rollover ----
    last_exit = exit_df.iloc[-1]
    feat["exit_last_mid"] = (float(last_exit["bid"]) + float(last_exit["ask"])) / 2.0
    feat["exit_last_spread"] = float(last_exit["ask"]) - float(last_exit["bid"])
    feat["exit_last_gap_min"] = (roll_ts - exit_df["time"].iloc[-1]).total_seconds() / 60.0

    arrival = _snapshot_at(exit_df, roll_ts - pd.Timedelta(minutes=ARRIVAL_OFFSET_MIN))
    feat["arrival_mid"] = arrival["mid"] if arrival else np.nan

    for k in EXIT_OFFSETS_MIN:
        snap = _snapshot_at(exit_df, roll_ts - pd.Timedelta(minutes=k))
        if snap is None:
            feat[f"exit_mid_t{k}"] = np.nan
            feat[f"exit_spread_t{k}"] = np.nan
            feat[f"exit_rate_t{k}"] = np.nan
            continue
        feat[f"exit_mid_t{k}"] = snap["mid"]
        feat[f"exit_spread_t{k}"] = snap["spread"]
        # tick rate in the 1-min bin ending at T-k (liquidity proxy)
        lo = roll_ts - pd.Timedelta(minutes=k + 1)
        hi = roll_ts - pd.Timedelta(minutes=k)
        feat[f"exit_rate_t{k}"] = int(((exit_df["time"] >= lo) & (exit_df["time"] < hi)).sum())

    # ---- EXIT passive-limit test (place at arrival mid at T-15, must fill before rollover) ----
    # SELL limit at arrival mid (capture ~half spread). Fill when bid >= arrival_mid.
    if arrival is not None:
        after15 = exit_df[exit_df["time"] >= roll_ts - pd.Timedelta(minutes=ARRIVAL_OFFSET_MIN)]
        feat["exit_sell_limit_mid_fill_min"] = _sell_limit_fill_min(
            after15, arrival["mid"], roll_ts - pd.Timedelta(minutes=ARRIVAL_OFFSET_MIN))

    # ---- ENTRY: open, spread decay, mid path, limit fills ----
    open_snap = _first_after(entry_df, roll_ts + pd.Timedelta(minutes=DEADZONE_MIN))
    if open_snap is None:
        return None
    open_time = pd.Timestamp(open_snap["time"])
    open_mid = open_snap["mid"]
    feat["open_time"] = open_time
    feat["open_mid"] = open_mid
    feat["open_bid"] = open_snap["bid"]
    feat["open_ask"] = open_snap["ask"]
    feat["open_spread"] = open_snap["spread"]
    feat["open_lag_min"] = (open_time - (roll_ts + pd.Timedelta(minutes=DEADZONE_MIN))).total_seconds() / 60.0

    for m in ENTRY_SNAPSHOT_MIN:
        snap = _snapshot_at(entry_df, open_time + pd.Timedelta(minutes=m))
        feat[f"entry_mid_p{m}"] = snap["mid"] if snap else np.nan
        feat[f"entry_spread_p{m}"] = snap["spread"] if snap else np.nan

    # settled mid = mid at +60 (proxy for the 'fair' post-reopen price)
    settled = _snapshot_at(entry_df, open_time + pd.Timedelta(minutes=59))
    feat["entry_settled_mid"] = settled["mid"] if settled else np.nan

    # BUY limit at open_mid - n ticks: fill time + price improvement + adverse selection.
    for n in ENTRY_LIMIT_TICKS:
        limit = open_mid - n * spec.tick_size
        fill_min = _buy_limit_fill_min(entry_df, limit, open_time)
        feat[f"entry_buylim_t{n}_fillmin"] = fill_min
        for h in FILL_HORIZONS_MIN:
            feat[f"entry_buylim_t{n}_filled_{h}m"] = int(not np.isnan(fill_min) and fill_min <= h)

    return feat


def _triple_to_pyweekday(mt5_rollover3days: int) -> int:
    """MT5 swap_rollover3days (0=Sun..6=Sat) -> python weekday (Mon=0..Sun=6)."""
    # MT5: 0=Sun,1=Mon,...,6=Sat  →  python: Mon=0..Sun=6
    return {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[mt5_rollover3days]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_symbol(symbol: str, days_back: int) -> pd.DataFrame:
    rolls = rollover_dates(days_back)
    rows: list[dict] = []
    n = len(rolls)
    for i, roll in enumerate(rolls):
        fetch_start = roll - timedelta(minutes=EXIT_WINDOW_MIN + FETCH_PAD_MIN)
        fetch_end = roll + timedelta(minutes=DEADZONE_MIN + ENTRY_WINDOW_MIN + FETCH_PAD_MIN)
        try:
            df = ensure_ticks(symbol, fetch_start, fetch_end)
        except Exception as exc:
            logger.warning("%s %s: fetch failed: %s", symbol, roll.date(), str(exc)[:80])
            continue
        if df.empty:
            continue
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        exit_df = df[df["time"] < pd.Timestamp(roll)].reset_index(drop=True)
        entry_df = df[df["time"] >= pd.Timestamp(roll) + pd.Timedelta(minutes=DEADZONE_MIN)].reset_index(drop=True)
        feat = event_features(symbol, roll, exit_df, entry_df)
        if feat is not None:
            rows.append(feat)
        if (i + 1) % 25 == 0:
            logger.info("  %s: %d/%d events (%d kept)", symbol, i + 1, n, len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Build rollover-cost event features")
    p.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in args.symbols:
        logger.info("Building events for %s (lookback %dd)", sym, args.days)
        dfo = build_symbol(sym, args.days)
        if dfo.empty:
            logger.warning("  %s: no events", sym); continue
        out = OUT_DIR / f"events_{sym}.parquet"
        dfo.to_parquet(out, index=False)
        logger.info("  %s: wrote %d events -> %s", sym, len(dfo), out)
    print("Done.")


if __name__ == "__main__":
    main()
