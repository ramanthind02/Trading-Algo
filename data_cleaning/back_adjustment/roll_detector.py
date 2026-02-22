"""Detect roll events in historical futures data by finding price gaps."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

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


def detect_roll_dates(
    df: pd.DataFrame,
    rule: RollRule,
    trailing_window: int = 252,
    sigma_multiplier: float = 3.0,
    pct_floor: float = 0.01,
) -> List[RollEvent]:
    """Detect roll dates by finding abnormal close-to-close gaps.

    Algorithm:
    1. Resample to daily close prices.
    2. Compute daily close-to-close changes.
    3. Threshold = max(pct_floor * prev_close, sigma_multiplier * trailing_std).
    4. Flag days where abs(change) >= threshold.
    """
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
