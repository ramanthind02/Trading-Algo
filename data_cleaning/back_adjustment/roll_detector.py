"""Detect roll events in historical futures data.

Two detection modes:
1. Norgate-guided (preferred): Use Norgate's Delivery Month to identify
   roll dates, then measure the gap in the target data.
2. Threshold-based (fallback): Find abnormal close-to-close gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data_cleaning.back_adjustment.roll_rules import RollRule


@dataclass(frozen=True)
class RollEvent:
    roll_date: datetime
    old_contract_close: float
    new_contract_close: float
    gap_points: float


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample any-frequency DataFrame to daily OHLC bars."""
    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp["datetime"])
    tmp["_date"] = tmp["_dt"].dt.normalize()

    daily = tmp.groupby("_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index().rename(columns={"_date": "date"})

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def detect_roll_dates_from_norgate(
    df: pd.DataFrame,
    norgate_unadjusted: pd.DataFrame,
    search_window_days: int = 5,
) -> List[RollEvent]:
    """Detect rolls using Norgate Delivery Month as reference.

    1. Find dates where Norgate's Delivery Month changes (= roll dates).
    2. Resample the target data (Kibot) to daily.
    3. For each Norgate roll date, find the closest date in the target
       data and measure the close-to-close gap.

    Parameters
    ----------
    df : pd.DataFrame
        Target data (Kibot) with columns: datetime, open, high, low, close.
    norgate_unadjusted : pd.DataFrame
        Norgate unadjusted continuous futures with columns: Date, Close,
        Delivery Month.
    search_window_days : int
        How many days around each Norgate roll date to search for the
        corresponding gap in the target data.
    """
    if df.empty or norgate_unadjusted.empty:
        return []

    if "Delivery Month" not in norgate_unadjusted.columns:
        return []

    # Find Norgate roll dates
    norgate = norgate_unadjusted.copy()
    norgate["_date"] = pd.to_datetime(norgate["Date"]).dt.normalize()
    dm = norgate["Delivery Month"]
    roll_mask = dm != dm.shift(1)
    norgate_roll_dates = norgate["_date"][roll_mask].iloc[1:].tolist()

    if not norgate_roll_dates:
        return []

    # Resample target data to daily
    daily = _resample_to_daily(df)
    daily_dates = pd.to_datetime(daily["date"])
    closes = daily["close"].values.astype(np.float64)

    events: list[RollEvent] = []

    for nrd in norgate_roll_dates:
        nrd_ts = pd.Timestamp(nrd)
        # Find the closest daily bar on or just before the Norgate roll date
        window_start = nrd_ts - timedelta(days=search_window_days)
        window_end = nrd_ts + timedelta(days=search_window_days)

        mask = (daily_dates >= window_start) & (daily_dates <= window_end)
        window_idx = daily.index[mask].tolist()

        if len(window_idx) < 2:
            continue

        # Find the biggest absolute gap within the window
        best_gap = 0.0
        best_i = window_idx[0]
        for idx in window_idx:
            if idx == 0:
                continue
            gap = closes[idx] - closes[idx - 1]
            if abs(gap) > abs(best_gap):
                best_gap = gap
                best_i = idx

        if abs(best_gap) < 1e-6:
            continue

        roll_date = pd.Timestamp(daily_dates.iloc[best_i]).to_pydatetime()
        old_close = float(closes[best_i - 1])
        new_close = float(closes[best_i])

        events.append(RollEvent(
            roll_date=roll_date,
            old_contract_close=old_close,
            new_contract_close=new_close,
            gap_points=best_gap,
        ))

    events.sort(key=lambda e: e.roll_date)
    return events


def detect_roll_dates(
    df: pd.DataFrame,
    rule: RollRule,
    trailing_window: int = 252,
    sigma_multiplier: float = 3.0,
    pct_floor: float = 0.01,
    norgate_unadjusted: Optional[pd.DataFrame] = None,
) -> List[RollEvent]:
    """Detect roll dates in historical data.

    If norgate_unadjusted is provided, uses Norgate Delivery Month for
    reliable detection. Otherwise falls back to threshold-based gap
    detection.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: datetime, open, high, low, close.
    rule : RollRule
        Roll rule for this ticker.
    norgate_unadjusted : pd.DataFrame, optional
        Norgate unadjusted data with Delivery Month column.
    """
    # Prefer Norgate-guided detection
    if norgate_unadjusted is not None:
        return detect_roll_dates_from_norgate(df, norgate_unadjusted)

    # Fallback: threshold-based detection
    if df.empty:
        return []

    daily = _resample_to_daily(df)
    if len(daily) < 2:
        return []

    closes = daily["close"].values.astype(np.float64)
    dates = pd.to_datetime(daily["date"]).values

    changes = np.diff(closes)
    abs_changes = np.abs(changes)

    events: list[RollEvent] = []

    for i in range(len(changes)):
        start_idx = max(0, i - trailing_window)
        window = abs_changes[start_idx:i] if i > 0 else abs_changes[:1]
        trailing_std = float(np.std(window)) if len(window) > 1 else 0.0

        prev_close = closes[i]
        threshold = max(pct_floor * abs(prev_close), sigma_multiplier * trailing_std)

        if threshold > 0 and abs(changes[i]) >= threshold:
            roll_date = pd.Timestamp(dates[i + 1]).to_pydatetime()
            old_close = float(closes[i])
            new_close = float(closes[i + 1])
            gap = new_close - old_close

            events.append(RollEvent(
                roll_date=roll_date,
                old_contract_close=old_close,
                new_contract_close=new_close,
                gap_points=gap,
            ))

    events.sort(key=lambda e: e.roll_date)
    return events
